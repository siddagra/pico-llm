# starter code by matus & o1-pro
#
# ################################################################################
# OPTIMIZED VERSION by Gemini
#
# Based on pico-llm.pdf [cite: 1-84] and TinyStories paper (2305.07759v2.pdf) 
#
# Core Architectural Changes for Optimal Performance:
# 1. Replaced nn.Embedding positional encoding with Rotary Positional Embeddings (RoPE)
#    - Added RotaryEmbedding class
#    - Added apply_rotary_emb function
#    - Updated CausalMultiHeadAttention to use RoPE
# 2. Replaced standard FeedForward MLP with SwiGLU FFN (from Llama/modern practice)
#    - Added SwiGLU_FFN class
# 3. Switched from optim.Adam to optim.AdamW
# 4. Added Gradient Clipping (grad_clip)
# 5. Kept Pre-Normalization (RMSNorm) as it's a best practice 
#
# Core Tasks (from starter) are preserved:
# 1. KGramMLPSeqModel (Core Task 2) [cite: 35]
# 2. nucleus_sampling (Core Task 3) [cite: 40]
# 3. RMSNorm (Core Task 4) [cite: 46]
# 4. TransformerModel (Core Task 4) [cite: 44]
# ################################################################################

import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import glob
from datetime import datetime
try:
    import yaml
except Exception:
    yaml = None
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    # --- Load config file first (if specified) to set defaults ---
    # This parser just looks for the --config argument
    conf_parser = argparse.ArgumentParser(
        description="Pico-LLM Training Config",
        add_help=False  # Disables help to avoid conflicts with the main parser
    )
    conf_parser.add_argument("--config", type=str, default=None,
                             help="Path to a YAML config file to set defaults.")
    args, remaining_argv = conf_parser.parse_known_args()

    config_defaults = {}
    if args.config and yaml:
        if os.path.exists(args.config):
            print(f"Loading config from {args.config}...")
            with open(args.config, "r") as f:
                config_defaults = yaml.safe_load(f)
        else:
            print(f"Warning: Config file {args.config} not found. Using defaults.")
    elif args.config:
        print("Warning: --config specified but 'yaml' module not found. Using defaults.")
    
    # --- Main parser ---
    parser = argparse.ArgumentParser(
        description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.",
        parents=[conf_parser] # Inherit the --config argument
    )
    
    # Dataset args
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources.")
    parser.add_argument("--tinystories_weight", type=float,
                        help="Probability of sampling from TinyStories. [default: 0.5]")
    
    # Model args (K-Gram)
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. [default: 3]")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. [default: 1]")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. [default: 1]")

    # Model args (Transformer / Shared)
    parser.add_argument("--block_size", type=int, help="Maximum sequence length. [default: 256]")
    parser.add_argument("--embed_size", type=int, help="Embedding dimension. [default: 256]")
    parser.add_argument("--n_heads", type=int, help="Number of attention heads. [default: 4]")
    parser.add_argument("--n_blocks", type=int, help="Number of transformer blocks. [default: 4]")

    # Training args
    parser.add_argument("--batch_size", type=int, help="Batch size. [default: 64]")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs. [default: 5]")
    parser.add_argument("--learning_rate", type=float, help="Base learning rate. [default: 3e-4]")
    parser.add_argument("--max_lr", type=float, default=None, help="Max LR for OneCycle (if not set, derived from lr).")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for AdamW. [default: 0.1]")
    parser.add_argument("--grad_clip", type=float, help="Gradient clipping value (0 to disable). [default: 1.0]")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps.")

    # Device & Performance args
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0').")
    parser.add_argument("--use_bf16", action="store_true",
                        help="Use bfloat16 autocast on CUDA.")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Enable fused/flash attention.")

    # Logging & Checkpointing args
    parser.add_argument("--ckpt_dir", type=str, help="Directory to save checkpoints. [default: 'ckpt']")
    parser.add_argument("--save_ckpt_steps", type=int,
                        help="Save model checkpoint every N global steps (0 to disable). [default: 500]")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint in ckpt_dir.")
    parser.add_argument("--use_wandb", action="store_true", help="Log training to Weights & Biases.")
    parser.add_argument("--log_dir", type=str, help="Directory for TensorBoard logs. [default: 'runs']")
    
    # Generation args
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")
    
    # Monosemantic (disabled)
    parser.add_argument("--monosemantic_enabled", action="store_true", help="(DISABLED BY DEFAULT)")
    
    # Set defaults from config file, then from hardcoded values
    parser.set_defaults(
        **config_defaults,
        # Set hardcoded defaults for any values *not* in the config
        tinystories_weight=config_defaults.get("tinystories_weight", 0.5),
        block_size=config_defaults.get("block_size", 256),
        embed_size=config_defaults.get("embed_size", 256),
        n_heads=config_defaults.get("n_heads", 4),
        n_blocks=config_defaults.get("n_blocks", 4),
        batch_size=config_defaults.get("batch_size", 64),
        num_epochs=config_defaults.get("num_epochs", 5),
        learning_rate=config_defaults.get("learning_rate", 3e-4),
        weight_decay=config_defaults.get("weight_decay", 0.1),
        grad_clip=config_defaults.get("grad_clip", 1.0),
        ckpt_dir=config_defaults.get("ckpt_dir", "ckpt"),
        save_ckpt_steps=config_defaults.get("save_ckpt_steps", 500),
        log_dir=config_defaults.get("log_dir", "runs"),
        use_wandb=config_defaults.get("use_wandb", False),
        use_bf16=config_defaults.get("use_bf16", False),
        use_flash_attn=config_defaults.get("use_flash_attn", False),
        resume=config_defaults.get("resume", False),
        monosemantic_enabled=False # Always disable
    )

    # Parse the remaining arguments (those not used by conf_parser)
    final_args = parser.parse_args(remaining_argv)
    
    # If wandb is set in config or CLI, override the default
    final_args.use_wandb = config_defaults.get("use_wandb", False) or final_args.use_wandb
    
    return final_args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach (Core Task 2)
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # fill in [cite: 35]
        # The forward() loop in the starter code uses F.one_hot and .flatten(),
        # so the input dimension to our network is (k * vocab_size).
        input_dim = self.k * self.vocab_size
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, self.embed_size))
        layers.append(nn.SiLU()) # Using SiLU as seen in args

        # Inner layers
        for _ in range(self.num_inner_layers):
            layers.append(nn.Linear(self.embed_size, self.embed_size))
            layers.append(nn.SiLU())

        # Output layer
        layers.append(nn.Linear(self.embed_size, self.vocab_size))
        
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        
        THIS IS THE PROVIDED FORWARD() ROUTINE [cite: 36]
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Transformer Implementation (Core Task 4 & Optimal Upgrades)
#    - RMSNorm [cite: 46]
#    - Pre-Normalization 
#    - RoPE (Rotary Positional Embeddings) 
#    - SwiGLU Feed-Forward Network
################################################################################

class RMSNorm(nn.Module):
    """Implement RMSNorm [cite: 46]"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed

# --- RoPE Implementation ---

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE) 
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute theta values
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._freqs_cis_cached = None

    def forward(self, seq_len: int, device: torch.device):
        # Cache freqs_cis for efficiency
        if seq_len == self._seq_len_cached:
            return self._freqs_cis_cached.to(device)

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Convert to complex numbers: cis(theta) = cos(theta) + i*sin(theta)
        # (seq_len, dim/2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        self._seq_len_cached = seq_len
        self._freqs_cis_cached = freqs_cis
        return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    xq, xk: (seq_len, batch, n_heads, d_head)
    freqs_cis: (seq_len, d_head/2)
    """
    # (seq_len, batch, n_heads, d_head) -> (seq_len, batch, n_heads, d_head/2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # (seq_len, batch, n_heads, d_head/2)
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)
    
    # (seq_len, d_head/2) -> (seq_len, 1, 1, d_head/2)
    freqs_cis = freqs_cis.unsqueeze(1).unsqueeze(1)
    
    # Apply rotation
    # (seq_len, batch, n_heads, d_head/2)
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# --- SwiGLU FFN Implementation ---

def _calc_hidden_dim(d_model, multiple_of=256):
    """Calculate hidden dim for SwiGLU, usually 2/3 of 4*d_model, rounded to multiple_of."""
    hidden_dim = int(2 * (4 * d_model) / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

class SwiGLU_FFN(nn.Module):
    """
    SwiGLU Feed-forward network, (silu(w1(x)) * w3(x)) @ w2
    A modern best practice for transformer FFNs.
    """
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = _calc_hidden_dim(d_model)
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) # The 'gate'
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False) # The 'down' projection

    def forward(self, x):
        # (S, B, D) -> (S, B, H)
        silu_x = F.silu(self.w1(x))
        # (S, B, D) -> (S, B, H)
        gate_x = self.w3(x)
        
        # (S, B, H) * (S, B, H) -> (S, B, H)
        gated_x = silu_x * gate_x
        
        # (S, B, H) -> (S, B, D)
        return self.w2(gated_x)


# --- Transformer Components ---

class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with combined QKV projection, causal masking, and RoPE.
    [cite: 50, 51]
    """
    def __init__(self, d_model, n_heads, use_flash: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, freqs_cis):
        """
        Input x: (seq_len, batch_size, d_model)
        freqs_cis: (seq_len, d_head/2)
        """
        seq_len, batch_size, _ = x.shape

        # (S, B, 3*D)
        qkv = self.W_qkv(x)
        
        # (S, B, D) each
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (S, B, D) -> (S, B, n_heads, d_head)
        q = q.view(seq_len, batch_size, self.n_heads, self.d_head)
        k = k.view(seq_len, batch_size, self.n_heads, self.d_head)
        v = v.view(seq_len, batch_size, self.n_heads, self.d_head)

        # Apply RoPE positional embeddings
        # (S, B, n_heads, d_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Permute for built-in attention
        # (S, B, n_heads, d_head) -> (B, n_heads, S, d_head)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Use built-in scaled_dot_product_attention with causal masking
        attn_output = F.scaled_dot_product_attention(q, k, v, 
                                                   is_causal=True, 
                                                   dropout_p=0.0) # No dropout for pico models

        # Reshape back to (S, B, D)
        # (B, n_heads, S, d_head) -> (S, B, n_heads, d_head) -> (S, B, D)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.d_model)

        return self.W_o(attn_output)


class TransformerBlock(nn.Module):
    """
    A single Transformer block.
    Using Pre-Normalization  and SwiGLU FFN.
    """
    def __init__(self, d_model, n_heads, use_flash: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, use_flash=use_flash)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU_FFN(d_model)

    def forward(self, x, freqs_cis):
        # Pre-Normalization: Norm -> Op -> Add
        # [cite: 50, 53]
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    """
    Causal Decoder-Only Transformer [cite: 44]
    
    Architectural Upgrades:
    - RoPE instead of learned positional embeddings 
    - SwiGLU FFN instead of standard MLP
    - RMSNorm (Pre-Normalization) [cite: 46, 81]
    """
    def __init__(self, vocab_size=50257, d_model=256, n_heads=4, n_blocks=4, block_size=256, use_flash: bool = False):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Token Embedding [cite: 47]
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Rotary Positional Embedding (RoPE)
        self.rope = RotaryEmbedding(self.d_head)

        # Transformer Blocks [cite: 49]
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, use_flash=use_flash) for _ in range(n_blocks)
        ])

        # Final normalization and unembedding
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # [cite: 58]

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        """
        seq_len, batch_size = tokens_seq.shape
        assert seq_len <= self.block_size, f"Sequence length {seq_len} exceeds block size {self.block_size}"

        device = tokens_seq.device
        
        # (S, B, D)
        x = self.token_embedding(tokens_seq)
        
        # (S, D_head/2)
        freqs_cis = self.rope(seq_len, device=device)

        # Pass through blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Final norm
        x = self.norm_final(x)
        
        # (S, B, VocabSize)
        logits = self.lm_head(x)
        return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################

def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    Core Task 3: Implement Nucleus Sampling (Top-p) [cite: 40]
    logits: (vocab_size,)
    """
    if p >= 1.0:
        probs = F.softmax(logits, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  use_bf16: bool = False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            if hasattr(model, 'block_size'):
                context_to_feed = context_tokens[-model.block_size:]
            else:
                context_to_feed = context_tokens
                
            seq_tensor = torch.tensor(context_to_feed, dtype=torch.long, device=device).unsqueeze(1)

            if use_bf16 and str(device).startswith("cuda"):
                autocast_ctx = torch.autocast if hasattr(torch, 'autocast') else torch.cuda.amp.autocast
                with autocast_ctx(device_type='cuda', dtype=torch.bfloat16):
                    logits_seq = model(seq_tensor)
            else:
                logits_seq = model(seq_tensor)
            
            next_logits = logits_seq[-1, 0, :]

            if top_p is None:
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    
    newly_generated_annotations = annotation_list
    
    for (tid, neighs) in newly_generated_annotations:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return annotated_text


################################################################################
# 8. Training
################################################################################

def find_latest_checkpoint(ckpt_dir, model_name):
    pattern = os.path.join(ckpt_dir, f"{model_name}_step*.pt")
    candidates = glob.glob(pattern)
    final_path = os.path.join(ckpt_dir, f"{model_name}_final.pt")
    if os.path.exists(final_path):
        candidates.append(final_path)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]

def compute_accuracy(logits, tokens):
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return 0.0
    preds = logits[:-1].argmax(dim=-1)
    gold = tokens[1:]
    correct = (preds == gold).float().sum().item()
    total = float((seq_len - 1) * batch_size)
    return correct / total if total > 0 else 0.0

def train_one_model(model,
                    loader,
                    val_loader, # Added for Optional Task 2 [cite: 67]
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    weight_decay=0.1, # <-- Added for AdamW
                    grad_clip=1.0,    # <-- Added for Gradient Clipping
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    use_bf16: bool = False,
                    ckpt_dir: str = None,
                    save_ckpt_steps: int = 0,
                    resume: bool = False,
                    use_wandb: bool = False,
                    log_dir: str = "runs",
                    max_lr: float = None):
    
    # --- OPTIMIZER: Use AdamW instead of Adam ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max_steps_per_epoch if max_steps_per_epoch is not None else max(1, len(loader))
    total_steps = steps_per_epoch * epochs
    scheduler = None
    try:
        if total_steps > 0:
            _max_lr = max_lr if max_lr is not None else max(lr * 10, lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=_max_lr, total_steps=total_steps)
    except Exception as e:
        print(f"[{model_name}] Warning: failed to create OneCycleLR scheduler: {e}")
        scheduler = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logdir = os.path.join(log_dir, f"{model_name}_{timestamp}")
    writer = SummaryWriter(tb_logdir)
    if use_wandb and _HAS_WANDB:
        wandb.init(project="pico-llm", name=f"{model_name}_{timestamp}", config={"model": model_name, "lr": lr})
    elif use_wandb:
        print(f"[{model_name}] Warning: wandb not installed; skipping W&B logging.")

    start_epoch = 1
    global_step = 0
    resume_skip_batch_idx = None
    if ckpt_dir and resume:
        latest = find_latest_checkpoint(ckpt_dir, model_name)
        if latest:
            try:
                checkpoint = torch.load(latest, map_location=device)
                model.load_state_dict(checkpoint.get('model_state_dict', {}))
                opt_state = checkpoint.get('optimizer_state_dict', None)
                if opt_state is not None:
                    optimizer.load_state_dict(opt_state)
                sched_state = checkpoint.get('scheduler_state_dict', None)
                if scheduler is not None and sched_state is not None:
                    try:
                        scheduler.load_state_dict(sched_state)
                    except Exception as e:
                        print(f"[{model_name}] Warning: failed to load scheduler state: {e}")
                global_step = checkpoint.get('global_step', 0)
                start_epoch = checkpoint.get('epoch', 1)
                resume_skip_batch_idx = checkpoint.get('batch_idx', None)
                print(f"[{model_name}] Resumed from {latest} (epoch={start_epoch}, global_step={global_step}, batch_idx={resume_skip_batch_idx})")
            except Exception as e:
                print(f"[{model_name}] Failed to resume from checkpoint {latest}: {e}")

    start_time = time.time()
    next_sample_time = start_time

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            if resume_skip_batch_idx is not None and epoch == start_epoch and batch_idx <= resume_skip_batch_idx:
                continue

            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)

            if use_bf16 and str(device).startswith("cuda"):
                autocast_ctx = torch.autocast if hasattr(torch, 'autocast') else torch.cuda.amp.autocast
                with autocast_ctx(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(batch_tokens)
                loss = compute_next_token_loss(logits.float(), batch_tokens)
            else:
                logits = model(batch_tokens)
                loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            
            # --- GRADIENT CLIPPING ---
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

            # -------------------- CHECKPOINTING --------------------
            if ckpt_dir and save_ckpt_steps and save_ckpt_steps > 0 and (global_step % save_ckpt_steps == 0):
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_step{global_step}.pt")
                    val_loss, val_acc = None, None
                    if val_loader:
                        model.eval()
                        with torch.no_grad():
                            v_total, v_acc_sum, v_count = 0.0, 0.0, 0
                            for v_batch in val_loader:
                                v_batch = v_batch.to(device)
                                v_logits = model(v_batch)
                                v_total += compute_next_token_loss(v_logits, v_batch).item()
                                v_acc_sum += compute_accuracy(v_logits, v_batch)
                                v_count += 1
                            if v_count > 0:
                                val_loss, val_acc = v_total / v_count, v_acc_sum / v_count
                        model.train()

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'global_step': global_step, 'epoch': epoch, 'batch_idx': batch_idx,
                        'val_loss': val_loss, 'val_acc': val_acc,
                    }, ckpt_path)
                    print(f"[{model_name}] Saved checkpoint: {ckpt_path}")
                except Exception as e:
                    print(f"[{model_name}] Failed to save checkpoint at step {global_step}: {e}")
            # -------------------------------------------------------

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                lr_now = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups)>0 else lr
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f} lr={lr_now:.6e}")
                writer.add_scalar("train/partial_loss", avg_part_loss, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                if use_wandb and _HAS_WANDB:
                    wandb.log({"train/partial_loss": avg_part_loss, "train/lr": lr_now, "global_step": global_step}, step=global_step)
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        do_monosemantic=False,
                        use_bf16=use_bf16
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}\n")
                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        # End of epoch logs & validation
        avg_loss = total_loss / step_in_epoch if step_in_epoch>0 else 0.0
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Train Loss: {avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        if use_wandb and _HAS_WANDB:
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch}, step=global_step)

        # Validation loop (Optional Task 2) [cite: 67]
        avg_val_loss = None
        avg_val_acc = None
        if val_loader:
            print(f"[{model_name}] Running validation for epoch {epoch}...")
            model.eval()
            total_val_loss, total_val_acc, val_steps = 0.0, 0.0, 0
            with torch.no_grad():
                for val_batch_idx, val_batch_tokens in enumerate(val_loader, start=1):
                    val_batch_tokens = val_batch_tokens.to(device)
                    val_logits = model(val_batch_tokens)
                    total_val_loss += compute_next_token_loss(val_logits, val_batch_tokens).item()
                    total_val_acc += compute_accuracy(val_logits, val_batch_tokens)
                    val_steps += 1
            
            if val_steps > 0:
                avg_val_loss = total_val_loss / val_steps
                avg_val_acc = total_val_acc / val_steps
                print(f"[{model_name}] *** Epoch {epoch} Validation Loss ***: {avg_val_loss:.4f}  Acc: {avg_val_acc:.4f}")
                writer.add_scalar("val/loss", avg_val_loss, epoch)
                writer.add_scalar("val/acc", avg_val_acc, epoch)
                if use_wandb and _HAS_WANDB:
                    wandb.log({"val/loss": avg_val_loss, "val/acc": avg_val_acc, "epoch": epoch}, step=global_step)
            model.train()
        print("--------------------------------------------------")

    if ckpt_dir:
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            final_path = os.path.join(ckpt_dir, f"{model_name}_final.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'global_step': global_step, 'epoch': epoch, 'final': True,
                'train_loss': avg_loss, 'val_loss': avg_val_loss, 'val_acc': avg_val_acc
            }, final_path)
            print(f"[{model_name}] Saved final checkpoint: {final_path}")
        except Exception as e:
            print(f"[{model_name}] Failed to save final checkpoint: {e}")

    writer.close()
    if use_wandb and _HAS_WANDB:
        wandb.finish()


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # --- Local variables from args ---
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size
    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    block_size = args.block_size
    
    # Transformer params
    n_heads = args.n_heads
    n_blocks = args.n_blocks
    
    log_interval_steps = 100
    sample_interval_seconds = 60
    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers
    val_split_ratio = 0.1 # 10% for validation 

    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, embed_size={embed_size}")
    print(f"Transformer params: n_heads={n_heads}, n_blocks={n_blocks}")
    use_bf16 = args.use_bf16
    use_flash = args.use_flash_attn

    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []
    
    # Use a smaller subset for faster local testing if max_steps is set
    train_subset_size = 50000 
    if max_steps_per_epoch is not None and max_steps_per_epoch < 500:
         train_subset_size = max_steps_per_epoch * batch_size * 2 # Just enough for a few epochs

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        try:
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            if len(dataset) > train_subset_size:
                 dataset = dataset.select(range(train_subset_size))
        except Exception as e:
            print(f"Failed to load TinyStories: {e}. Exiting.")
            return
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")

    # --- Train/Test Split (Optional Task 2) [cite: 67] ---
    random.shuffle(tinystories_seqs)
    random.shuffle(other_seqs)
    
    ts_split_idx = int(len(tinystories_seqs) * (1.0 - val_split_ratio))
    train_tinystories_seqs = tinystories_seqs[:ts_split_idx]
    val_tinystories_seqs = tinystories_seqs[ts_split_idx:]
    
    other_split_idx = int(len(other_seqs) * (1.0 - val_split_ratio))
    train_other_seqs = other_seqs[:other_split_idx]
    val_other_seqs = other_seqs[other_split_idx:]
    
    print(f"Data split: {len(train_tinystories_seqs) + len(train_other_seqs)} train, {len(val_tinystories_seqs) + len(val_other_seqs)} val")
    # --- End Split ---

    p_tiny = args.tinystories_weight
    
    if len(train_tinystories_seqs) == 0 and len(train_other_seqs) == 0:
        print("Error: No training data loaded. Check --input_files or --tinystories_weight.")
        return
        
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=train_tinystories_seqs,
        other_seqs=train_other_seqs, p_tiny=p_tiny
    )
    
    val_dataset = MixedSequenceDataset(
        tinystories_seqs=val_tinystories_seqs,
        other_seqs=val_other_seqs, p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=seq_collate_fn
    )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=seq_collate_fn
        )
    else:
        print("No validation data. Validation steps will be skipped.")


    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size, k=k, embed_size=embed_size,
        num_inner_layers=num_inner_layers, chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size, embed_size=embed_size, hidden_size=embed_size
    ).to(device)

    transformer_model = TransformerModel(
        vocab_size=vocab_size, d_model=embed_size, n_heads=n_heads,
        n_blocks=n_blocks, block_size=block_size, use_flash=use_flash
    ).to(device)

    # Cast to bfloat16 if requested (after moving to device)
    if use_bf16 and str(device).startswith("cuda"):
        print("Casting models to bfloat16...")
        try:
            kgram_model.to(dtype=torch.bfloat16)
            lstm_model.to(dtype=torch.bfloat16)
            transformer_model.to(dtype=torch.bfloat16)
        except Exception as e:
            print(f"Warning: failed to cast models to bfloat16: {e}. Continuing in float32.")
            use_bf16 = False # Disable if it fails


    models = {
       "kgram_mlp_seq": kgram_model,
       "lstm_seq": lstm_model,
       "transformer": transformer_model,
    }

    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
        
        train_one_model(
            model=model,
            loader=train_loader,
            val_loader=val_loader, # Pass val_loader
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            weight_decay=args.weight_decay, # <--- Pass AdamW hparam
            grad_clip=args.grad_clip,       # <--- Pass grad clip hparam
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,
            use_bf16=use_bf16,
            ckpt_dir=args.ckpt_dir,
            save_ckpt_steps=args.save_ckpt_steps,
            resume=args.resume,
            use_wandb=args.use_wandb,
            log_dir=args.log_dir,
            max_lr=args.max_lr
        )

        print(f"\n--- Final Generation for {model_name} ---")
        with torch.no_grad():
            text_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None, use_bf16=use_bf16,
            )
            text_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95, use_bf16=use_bf16,
            )
            text_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0, use_bf16=use_bf16,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"\n[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"\n[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print("--------------------------------------------------")

    print("\n*** All models trained. ***")


if __name__ == "__main__":
    main()