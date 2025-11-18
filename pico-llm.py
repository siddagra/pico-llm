# starter code by matus & o1-pro
#
# Completed implementation by Gemini, based on pico-llm.pdf and user requests.
#
# Core Tasks Completed:
# 1. KGramMLPSeqModel (Core Task 2) [cite: 35]
# 2. nucleus_sampling (Core Task 3) [cite: 40]
# 3. RMSNorm (Core Task 4) 
# 4. TransformerModel (Core Task 4) 
#
# Optional Tasks (as requested by user):
# 1. Train/Test Split (Optional Task 2) 
# 2. Positional Embedding (Optional Task 4) 
# 3. Pre-Normalization (Optional Task 7) 
#

import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os  # <--- added for checkpoint directory handling
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

# Optional detailed model summary if torchinfo is available
try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken


def load_env_file(env_path=".env"):
    """Load environment variables from a .env file.
    Prefer python-dotenv if available, otherwise fall back to a simple parser.
    Returns True if file existed and was processed, False otherwise.
    """
    try:
        if not os.path.exists(env_path):
            return False
    except Exception:
        return False

    # Try python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return True
    except Exception:
        # Manual fallback
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Remove surrounding quotes if present
                    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                        v = v[1:-1]
                    # Only set if not already in environment
                    os.environ.setdefault(k, v)
            return True
        except Exception:
            return False

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=256,
                        help="Maximum sequence length for each example. Default=256.")

    # Additional arguments:
    parser.add_argument("--embed_size", type=int, default=256,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=256.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    # Performance / precision options
    parser.add_argument("--use_bf16", action="store_true",
                        help="Use bfloat16 autocast on CUDA (requires supported hardware/PyTorch).")
    # AMP (automatic mixed precision, float16) - prefer this for most GPUs
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable automatic mixed precision (AMP) using torch.cuda.amp (FP16).")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Enable fused/flash attention when available (PyTorch may pick optimized kernels automatically).")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and validation DataLoaders (default=64).")

    # Checkpointing args (ADDED)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt",
                        help="Directory to save checkpoints (default='./ckpt').")
    parser.add_argument("--save_ckpt_steps", type=int, default=500,
                        help="Save model checkpoint every N global steps (default=500). Set 0 to disable.")

    # Resume & logging flags (ADDED)
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint in ckpt_dir.")
    parser.add_argument("--use_wandb", action="store_true", help="Log training to Weights & Biases (requires wandb installed).")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory for TensorBoard logs.")
    parser.add_argument("--max_lr", type=float, default=None, help="Max LR for OneCycle scheduler (if not set, derived from lr).")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Optional path to a YAML config file to override defaults (CLI overrides config).")
    parser.add_argument("--weight_tying", action="store_true",
                        help="Enable weight tying between token embedding and lm_head (CLI overrides config).")

    # New CLI flags requested by user:
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train each model (default=5).")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for optimizers (default=3e-4).")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads for Transformer.")
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of Transformer blocks.")
    parser.add_argument("--model_type", type=str, default="all",
                        choices=["all", "kgram", "lstm", "transformer"],
                        help="Which model to train: all/kgram/lstm/transformer (default=all).")
    parser.add_argument("--norm", type=str, default="pre", choices=["pre", "post"],
                        help="Transformer normalization style: pre (pre-norm) or post (post-norm). Default=pre.")
    parser.add_argument("--kgram_use_embedding", action="store_true",
                        help="Use nn.Embedding + concat method inside KGramMLPSeqModel (efficient method).")
    # EMA args
    parser.add_argument("--ema_decay", type=float, default=1.0,
                        help="Exponential Moving Average decay for model weights (default=disabled). Set <1.0 to enable EMA (e.g. 0.999).")
    # Optimizer selection + weight decay
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"],
                        help="Optimizer to use: 'adam' (default) or 'adamw' (Adam with weight decay).")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay to use with AdamW optimizer (ignored for plain Adam).")
    args = parser.parse_args()
    return args


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
    For each position t in [0..seq_len-1], gather the last k tokens => embeddings OR one-hot => MLP => logits.
    When use_embedding=True we use nn.Embedding(vocab_size, embed_size) and concatenate k embeddings
    to produce an input vector of size (k * embed_size) which is fed to the MLP. This is the efficient design.
    """
    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1, use_embedding=False):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
        self.use_embedding = use_embedding

        if self.use_embedding:
            # Efficient embedding-based design
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
            input_dim = self.k * self.embed_size
        else:
            # One-hot flattened input (original heavy method)
            input_dim = self.k * self.vocab_size

        layers = []
        layers.append(nn.Linear(input_dim, self.embed_size))
        layers.append(nn.SiLU())

        for _ in range(self.num_inner_layers):
            layers.append(nn.Linear(self.embed_size, self.embed_size))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(self.embed_size, self.vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        Two code-paths:
         - use_embedding=False: fallback to original per-position one-hot implementation (keeps chunking)
         - use_embedding=True: efficient vectorized embedding lookup + concat for all positions
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device

        if self.use_embedding:
            # Pad with zeros at the top so that for t<k we have zero-padding tokens
            pad = torch.zeros(self.k - 1, batch_size, dtype=torch.long, device=device)
            padded = torch.cat([pad, tokens_seq], dim=0)  # (seq_len + k -1, batch)
            # Use unfold to create sliding windows: (seq_len, k, batch)
            windows = padded.unfold(dimension=0, size=self.k, step=1)  # (seq_len, k, batch)
            # Rearrange to (seq_len, batch, k)
            windows = windows.permute(0, 2, 1).contiguous()  # (S, B, k)
            S, B, K = windows.shape
            # Flatten to (S*B, k) for embedding lookup
            windows_flat = windows.view(-1, K)  # (S*B, k)
            # Embed -> (S*B, k, embed)
            emb = self.embedding(windows_flat)  # (S*B, k, embed)
            # Flatten per-context to (S*B, k*embed)
            emb_flat = emb.view(emb.size(0), -1)  # (S*B, k*embed)
            logits_flat = self.net(emb_flat)      # (S*B, vocab)
            logits = logits_flat.view(S, B, -1)  # (S, B, vocab)
            return logits
        else:
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
# 5. Transformer Implementation (Core Task 4 & Optional Tasks)
#    - RMSNorm [cite: 46]
#    - Pre-Normalization 
#    - Positional Embedding 
################################################################################

class RMSNorm(nn.Module):
    """Implement RMSNorm """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate the root mean square
        # E[x^2]
        variance = x.pow(2).mean(-1, keepdim=True)
        # x / sqrt(E[x^2] + eps)
        x_normed = x * torch.rsqrt(variance + self.eps)
        # scale with gamma
        return self.weight * x_normed

class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with combined QKV projection and causal masking.
    We use F.scaled_dot_product_attention for efficiency, which has
    causal masking built-in. [cite: 50, 51]
    """
    def __init__(self, d_model, n_heads, use_flash: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash
        
        # Combined projection for Q, K, V
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Input x: (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, _ = x.shape

        # (S, B, 3*D)
        qkv = self.W_qkv(x)
        
        # (S, B, D) each
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (S, B, D) -> (S, B, n_heads, d_head) -> (B, n_heads, S, d_head)
        q = q.view(seq_len, batch_size, self.n_heads, self.d_head).permute(1, 2, 0, 3)
        k = k.view(seq_len, batch_size, self.n_heads, self.d_head).permute(1, 2, 0, 3)
        v = v.view(seq_len, batch_size, self.n_heads, self.d_head).permute(1, 2, 0, 3)

        # Use built-in scaled_dot_product_attention with causal masking
        # (B, n_heads, S, d_head)
        # PyTorch may dispatch optimized/fused kernels (flash attention) when inputs
        # have suitable dtypes and CUDA support. We expose a flag `use_flash` to
        # allow the model to request that behavior; however actual use depends on
        # the PyTorch/CUDA build and hardware.
        if self.use_flash:
            # Prefer calling the fused op; in modern PyTorch this is the same
            # function but may pick faster kernels internally.
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back to (S, B, D)
        # (B, n_heads, S, d_head) -> (S, B, n_heads, d_head) -> (S, B, D)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.d_model)

        # Output projection
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    """A simple 2-layer MLP, as described in [cite: 52]"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.silu = nn.SiLU() # Using SiLU as seen elsewhere
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.silu(self.linear1(x)))

class TransformerBlock(nn.Module):
    """
    A single Transformer block.
    Supports pre-norm (default) and post-norm (optional) behaviour controlled by norm_type.
    """
    def __init__(self, d_model, n_heads, d_ff, use_flash: bool = False, norm_type: str = "pre"):
        super().__init__()
        assert norm_type in ("pre", "post")
        self.norm_type = norm_type
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, use_flash=use_flash)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        if self.norm_type == "pre":
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x
        else:
            # post-norm style: apply op then normalize after residual
            y = self.attn(x)
            x = self.norm1(x + y)
            z = self.ff(x)
            x = self.norm2(x + z)
            return x

class TransformerModel(nn.Module):
    """
    Causal Decoder-Only Transformer 
    """
    def __init__(self, vocab_size=50257, d_model=256, n_heads=4, n_blocks=4, block_size=256, use_flash: bool = False, weight_tying: bool = False, norm_type: str = "pre"):
        super().__init__()
        self.block_size = block_size

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)

        d_ff = d_model * 4
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_flash=use_flash, norm_type=norm_type) for _ in range(n_blocks)
        ])

        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.weight_tying = weight_tying
        if self.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        """
        seq_len, batch_size = tokens_seq.shape
        assert seq_len <= self.block_size, f"Sequence length {seq_len} exceeds block size {self.block_size}"

        # (seq_len, batch)
        device = tokens_seq.device
        
        # (S, 1) -> (S)
        pos = torch.arange(0, seq_len, device=device)
        
        # (S, B, D)
        tok_emb = self.token_embedding(tokens_seq)
        # (S, D) -> (S, 1, D)
        pos_emb = self.pos_embedding(pos).unsqueeze(1)
        
        x = tok_emb + pos_emb

        # Pass through blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm_final(x)
        
        # Project to vocabulary logits via lm_head
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
    Core Task 3: Implement Nucleus Sampling (Top-p) 
    logits: (vocab_size,)
    """
    if p >= 1.0:
        # p=1.0 is equivalent to sampling from the full distribution
        probs = F.softmax(logits, dim=-1)
    else:
        # 1. Sort probabilities
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 2. Find the smallest k such that cumulative mass >= p
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask to zero out tokens beyond the nucleus
        # We find the *first* index where cumulative_probs > p
        # and keep all tokens *before* and *including* that one.
        sorted_indices_to_remove = cumulative_probs > p
        
        # Shift mask to the right so we keep the token that *just* crossed p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0 # Always keep the most likely token
        
        # 3. Truncate the tail
        # Map sorted mask back to original indices
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        
        # 4. Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # 5. Sample from the renormalized distribution
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  use_bf16: bool = False,
                  use_amp: bool = False):
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
            # Truncate context if it exceeds model's block size (for Transformer)
            if hasattr(model, 'block_size'):
                context_to_feed = context_tokens[-model.block_size:]
            else:
                context_to_feed = context_tokens
                
            seq_tensor = torch.tensor(context_to_feed, dtype=torch.long, device=device).unsqueeze(1)
            # Choose autocast: AMP (FP16) if requested, else bfloat16 if requested, else default
            if use_amp and str(device).startswith("cuda"):
                with torch.cuda.amp.autocast():
                    logits_seq = model(seq_tensor)
            elif use_bf16 and str(device).startswith("cuda"):
                autocast_ctx = torch.autocast if hasattr(torch, 'autocast') else torch.cuda.amp.autocast
                with autocast_ctx(device_type='cuda', dtype=torch.bfloat16):
                    logits_seq = model(seq_tensor)
            else:
                logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
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

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    
    # Use annotation_list which now only contains the *newly* generated tokens
    newly_generated_annotations = annotation_list # This was correct in starter code
    
    for (tid, neighs) in newly_generated_annotations:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


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
    """
    logits: (seq_len, batch, vocab) ; tokens: (seq_len, batch)
    compute next-token accuracy by shifting as in loss.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return 0.0
    preds = logits[:-1].argmax(dim=-1)   # (seq_len-1, batch)
    gold = tokens[1:]                    # (seq_len-1, batch)
    correct = (preds == gold).float().sum().item()
    total = float((seq_len - 1) * batch_size)
    return correct / total if total > 0 else 0.0

def train_one_model(model,
					loader,
					val_loader, # Added for Optional Task 2 
					epochs,
					model_name,
					device,
					lr=1e-3,
					log_steps=100,
					sample_interval=30,
					max_steps_per_epoch=None,
					enc=None,
					monosemantic_info=None,
					prompt="Once upon a",
					use_bf16: bool = False,
					use_amp: bool = False,
					ckpt_dir: str = None,                # <--- added
					save_ckpt_steps: int = 0,           # <--- added (kept but treated as epoch-based now)
					resume: bool = False,               # <--- added
					use_wandb: bool = False,            # <--- added
					log_dir: str = "runs",              # <--- added
					max_lr: float = None,               # <--- added
					optimizer_name: str = "adam",       # "adam" or "adamw"
					weight_decay: float = 0.0,
					ema_decay: float = 1.0):
	"""
	We add `prompt` as an explicit argument so we can pass it down from main().
	"""
	# Choose optimizer (Adam or AdamW with weight decay)
	if optimizer_name.lower() == "adamw":
		optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		optimizer = optim.Adam(model.parameters(), lr=lr)

	# --- EMA initialization (FIX) ---
	# Ensure ema_state and ema_active always exist. EMA is active only when ema_decay < 1.0
	ema_state = {}
	ema_active = (ema_decay is not None and ema_decay < 1.0)
	if ema_active:
		print(f"[{model_name}] EMA enabled with decay={ema_decay}")
	else:
		# normalize ema_decay to None when EMA disabled to make saves explicit
		ema_decay = None

	# OneCycle LR scheduler
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

	# Logging setup
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	tb_logdir = os.path.join(log_dir, f"{model_name}_{timestamp}")
	writer = SummaryWriter(tb_logdir)
	if use_wandb and _HAS_WANDB:
		wandb.init(project="pico-llm", name=f"{model_name}_{timestamp}", config={"model": model_name, "lr": lr})
	elif use_wandb:
		print(f"[{model_name}] Warning: wandb not installed; skipping W&B logging.")

	# TRACK top-K best models (lower metric better). Keep tuples (metric, path, epoch)
	TOP_K = 5
	best_models = []  # will hold (metric_value, path, epoch)

	# resume logic: try to find latest ckpt and load states
	start_epoch = 1
	global_step = 0
	resume_skip_batch_idx = None
	# AMP scaler (if requested and CUDA available)
	scaler = None
	if use_amp and str(device).startswith("cuda"):
		try:
			scaler = torch.cuda.amp.GradScaler()
		except Exception:
			scaler = None
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

				# --- Restore EMA state if present in checkpoint (FIX) ---
				ckpt_ema_state = checkpoint.get('ema_state_dict', None)
				ckpt_ema_decay = checkpoint.get('ema_decay', None)
				if ckpt_ema_state:
					# checkpoint saved EMA tensors on CPU. Move them to model device for in-memory usage.
					try:
						# determine model device (use first param)
						model_device = None
						for p in model.parameters():
							model_device = p.device
							break
						if model_device is None:
							model_device = device
						ema_state = {}
						for k, v in ckpt_ema_state.items():
							if isinstance(v, torch.Tensor):
								ema_state[k] = v.to(device=model_device)
							else:
								ema_state[k] = v
						# Set ema_active based on checkpoint's decay if present
						if ckpt_ema_decay is not None and ckpt_ema_decay < 1.0:
							ema_active = True
							ema_decay = ckpt_ema_decay
						else:
							ema_active = False
							ema_decay = None
					except Exception as e:
						print(f"[{model_name}] Warning: failed to restore EMA tensors from checkpoint: {e}")

				# --- Restore AMP scaler state if present in checkpoint ---
				ckpt_scaler_state = checkpoint.get('scaler_state_dict', None)
				if ckpt_scaler_state is not None and scaler is not None:
					try:
						scaler.load_state_dict(ckpt_scaler_state)
					except Exception as e:
						print(f"[{model_name}] Warning: failed to load AMP scaler state: {e}")
				# If user enabled AMP but checkpoint has no scaler, we'll continue with a fresh scaler (already created above)

				print(f"[{model_name}] Resumed from {latest} (epoch={start_epoch}, global_step={global_step}, batch_idx={resume_skip_batch_idx})")
			except Exception as e:
				print(f"[{model_name}] Failed to resume from checkpoint {latest}: {e}")

	start_time = time.time()
	next_sample_time = start_time

	# Track best model (prefer validation loss when available; fallback to training epoch loss)
	best_metric = float("inf")
	best_saved_path = None

	# Main training loop supports resume by starting from start_epoch.
	for epoch in range(start_epoch, epochs + 1):
		model.train()
		total_loss = 0.0
		partial_loss = 0.0
		partial_count = 0

		step_in_epoch = 0
		for batch_idx, batch_tokens in enumerate(loader, start=1):
			# If resuming, skip already-processed batches in the resumed epoch
			if resume_skip_batch_idx is not None and epoch == start_epoch and batch_idx <= resume_skip_batch_idx:
				continue

			step_in_epoch += 1
			global_step += 1

			batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

			# Forward/backward with AMP or BF16 or plain FP32
			if use_amp and str(device).startswith("cuda"):
				with torch.cuda.amp.autocast():
					logits = model(batch_tokens)
					loss = compute_next_token_loss(logits, batch_tokens)
				optimizer.zero_grad()
				# scale gradients & step
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			elif use_bf16 and str(device).startswith("cuda"):
				autocast_ctx = torch.autocast if hasattr(torch, 'autocast') else torch.cuda.amp.autocast
				with autocast_ctx(device_type='cuda', dtype=torch.bfloat16):
					logits = model(batch_tokens)
				# Compute loss in float32 for numerical stability
				loss = compute_next_token_loss(logits.float(), batch_tokens)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			else:
				logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
				loss = compute_next_token_loss(logits, batch_tokens)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			if scheduler is not None:
				try:
					scheduler.step()
				except Exception:
					pass

			# Update EMA after optimizer step — keep EMA tensors on same device as model parameters (GPU if available)
			if ema_active:
				with torch.no_grad():
					for name, param in model.named_parameters():
						if not param.requires_grad:
							continue
						param_tensor = param.detach()
						# Initialize EMA entry lazily if missing — place on param.device and match dtype
						if name not in ema_state or ema_state.get(name) is None:
							ema_state[name] = param_tensor.clone().to(device=param_tensor.device, dtype=param_tensor.dtype)
							continue
						ema_t = ema_state[name]
						# Move/cast ema tensor to param device/dtype if needed
						if isinstance(ema_t, torch.Tensor) and (ema_t.device != param_tensor.device or ema_t.dtype != param_tensor.dtype):
							ema_t = ema_t.to(device=param_tensor.device, dtype=param_tensor.dtype)
							ema_state[name] = ema_t
						# In-place EMA update on same device
						ema_t.mul_(ema_decay)
						ema_t.add_(param_tensor * (1.0 - ema_decay))
						ema_state[name] = ema_t

			# Accumulate losses for epoch stats (no per-step checkpointing/sampling)
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

			if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
				print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
				break

		# End of epoch logs & validation (now standardized to happen here)
		avg_loss = total_loss / step_in_epoch if step_in_epoch>0 else 0.0
		print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Train Loss: {avg_loss:.4f}")
		writer.add_scalar("train/epoch_loss", avg_loss, epoch)
		if use_wandb and _HAS_WANDB:
			wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch}, step=global_step)

		# Validation loop (use EMA weights if available)
		avg_val_loss = None
		avg_val_acc = None
		if val_loader:
			print(f"[{model_name}] Running validation for epoch {epoch}...")
			# Optionally swap to EMA weights for validation
			orig_state = None
			if ema_active and ema_state:
				orig_state = model.state_dict()
				# build a state dict where parameter tensors are replaced by ema tensors (casted to the target dtype/device)
				ema_sd = orig_state.copy()
				for n, orig_val in orig_state.items():
					if n in ema_state and ema_state[n] is not None:
						ema_val = ema_state[n]
						if isinstance(ema_val, torch.Tensor):
							try:
								# cast EMA value to same device/dtype as orig param tensor
								ema_sd[n] = ema_val.to(device=orig_val.device, dtype=orig_val.dtype).clone()
							except Exception:
								# best-effort: cast to orig dtype
								ema_sd[n] = ema_val.to(dtype=orig_val.dtype).clone()
				try:
					model.load_state_dict(ema_sd)
				except Exception as e:
					print(f"[{model_name}] Warning: failed to load EMA state for validation: {e}")
			model.eval() # Set model to evaluation mode
			total_val_loss = 0.0
			total_val_acc = 0.0
			val_steps = 0
			with torch.no_grad():
				for val_batch_idx, val_batch_tokens in enumerate(val_loader, start=1):
					val_batch_tokens = val_batch_tokens.to(device)
					val_logits = model(val_batch_tokens)
					val_loss = compute_next_token_loss(val_logits, val_batch_tokens)
					val_acc = compute_accuracy(val_logits, val_batch_tokens)
					total_val_loss += val_loss.item()
					total_val_acc += val_acc
					val_steps += 1
            
			if val_steps > 0:
				avg_val_loss = total_val_loss / val_steps
				avg_val_acc = total_val_acc / val_steps
				print(f"[{model_name}] *** Epoch {epoch} Validation Loss ***: {avg_val_loss:.4f}  Acc: {avg_val_acc:.4f}")
				writer.add_scalar("val/loss", avg_val_loss, epoch)
				writer.add_scalar("val/acc", avg_val_acc, epoch)
				if use_wandb and _HAS_WANDB:
					wandb.log({"val/loss": avg_val_loss, "val/acc": avg_val_acc, "epoch": epoch}, step=global_step)
			else:
				print(f"[{model_name}] Validation loader is empty, skipping.")
			model.train()

			# restore original weights if we swapped in EMA
			if orig_state is not None:
				try:
					model.load_state_dict(orig_state)
				except Exception as e:
					print(f"[{model_name}] Warning: failed to restore original model state after EMA validation: {e}")
		else:
			print(f"[{model_name}] No validation data. Validation steps will be skipped.")
		print("--------------------------------------------------")

		# Decide which metric to use for "best" model selection
		metric_to_check = None
		metric_name = "train_loss"
		if val_loader and avg_val_loss is not None:
			metric_to_check = avg_val_loss
			metric_name = "val_loss"
		else:
			metric_to_check = avg_loss
			metric_name = "train_loss"

		# -------------------- Save epoch checkpoint --------------------
		if ckpt_dir:
			try:
				os.makedirs(ckpt_dir, exist_ok=True)
				epoch_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch{epoch}.pt")
				# include EMA state: convert EMA tensors to CPU for checkpointing
				ema_to_save = None
				if ema_active and ema_state:
					ema_to_save = {}
					for k, v in ema_state.items():
						if isinstance(v, torch.Tensor):
							try:
								ema_to_save[k] = v.detach().cpu()
							except Exception:
								# fallback: store v as-is
								ema_to_save[k] = v
						else:
							ema_to_save[k] = v
				save_dict = {
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
					'global_step': global_step,
					'epoch': epoch,
					'train_loss': avg_loss,
					'val_loss': avg_val_loss,
					'val_acc': avg_val_acc,
					'ema_state_dict': ema_to_save,
					'ema_decay': ema_decay if ema_active else None,
				}
				# include scaler state if AMP used
				if scaler is not None:
					try:
						save_dict['scaler_state_dict'] = scaler.state_dict()
					except Exception:
						save_dict['scaler_state_dict'] = None
				torch.save(save_dict, epoch_ckpt_path)
				print(f"[{model_name}] Saved epoch checkpoint: {epoch_ckpt_path}")
			except Exception as e:
				print(f"[{model_name}] Failed to save epoch checkpoint: {e}")
		# ---------------------------------------------------------------

		# -------------------- Maintain Top-K best models --------------------
		try:
			if metric_to_check is not None:
				# add current epoch candidate
				if ckpt_dir:
					candidate_path = os.path.join(ckpt_dir, f"{model_name}_epoch{epoch}.pt")
				else:
					candidate_path = f"{model_name}_epoch{epoch}.pt"
				best_models.append((metric_to_check, candidate_path, epoch))
				# sort ascending by metric (lower is better)
				best_models.sort(key=lambda x: x[0])
				# trim to TOP_K
				best_models = best_models[:TOP_K]

				# Save copies named by rank for clarity (overwrite previous rank files)
				for rank, (mval, path, e) in enumerate(best_models, start=1):
					rank_path = os.path.join(ckpt_dir if ckpt_dir else ".", f"{model_name}_top{rank}.pt")
					try:
						# If this entry corresponds to the current epoch, the EMA is in-memory; else try to load ema from the saved file
						if e == epoch:
							ema_for_save = {k: v.detach().cpu() for k, v in ema_state.items()} if ema_active else None
							model_sd_to_save = model.state_dict()
						else:
							loaded = {}
							try:
								loaded = torch.load(path, map_location=device)
								model_sd_to_save = loaded.get('model_state_dict', None)
								ema_for_save = loaded.get('ema_state_dict', None)
							except Exception:
								# fallback: use current model (best-effort)
								model_sd_to_save = model.state_dict()
								ema_for_save = {k: v.detach().cpu() for k, v in ema_state.items()} if ema_active else None

						torch.save({
							'model_state_dict': model_sd_to_save,
							'optimizer_state_dict': optimizer.state_dict(),
							'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
							'global_step': global_step,
							'epoch': e,
							'metric_name': metric_name,
							'metric_value': mval,
							'train_loss': avg_loss if e == epoch else None,
							'val_loss': avg_val_loss if e == epoch else None,
							'ema_state_dict': ema_for_save,
							'ema_decay': ema_decay if ema_active else None,
						}, rank_path)
					except Exception as ex:
						print(f"[{model_name}] Failed to write ranked checkpoint {rank_path}: {ex}")

				print(f"[{model_name}] Updated Top-{TOP_K} models: {[ (round(x[0],4), x[2]) for x in best_models ]}")
		except Exception as e:
			print(f"[{model_name}] Error maintaining top-{TOP_K} models: {e}")
		# ----------------------------------------------------------------

		# -------------------- Sampling at end of epoch --------------------
		try:
			samples = []
			# For sampling we prefer EMA weights if available
			orig_state = None
			if ema_active and ema_state:
				orig_state = model.state_dict()
				ema_sd = orig_state.copy()
				for n in ema_state:
					if n in ema_sd and ema_state[n] is not None:
						# ema_state values are stored on CPU to make checkpoints small; cast to model dtype
						ema_val = ema_state[n]
						if isinstance(ema_val, torch.Tensor):
							ema_sd[n] = ema_val.to(ema_sd[n].dtype).clone()
				model.load_state_dict(ema_sd)

			# greedy
			g_text, _ = generate_text(model, enc, prompt, max_new_tokens=20, device=device, top_p=None, use_bf16=use_bf16, use_amp=use_amp)
			samples.append(("greedy", None, g_text))

			# explicit top-p sampling
			top_p_values = [0.1, 0.3, 0.5, 0.7, 0.95, 1.0]
			with torch.no_grad():
				for p in top_p_values:
					s_text, _ = generate_text(model, enc, prompt, max_new_tokens=20, device=device, top_p=p, use_bf16=use_bf16, use_amp=use_amp)
					samples.append(("nucleus", p, s_text))

			# restore original weights if swapped
			if orig_state is not None:
				model.load_state_dict(orig_state)

			# Print and log samples
			print(f"[{model_name}] Samples after epoch {epoch}:")
			sample_file_path = None
			if ckpt_dir:
				try:
					sample_file_path = os.path.join(ckpt_dir, f"{model_name}_epoch{epoch}_samples.txt")
					with open(sample_file_path, "w", encoding="utf-8") as sf:
						sf.write(f"Samples for {model_name} epoch {epoch}\n\n")
						for kind, p, text in samples:
							header = f"{kind} p={p}" if p is not None else f"{kind}"
							sf.write(f"--- {header} ---\n{text}\n\n")
				except Exception as e:
					print(f"[{model_name}] Failed to write samples to file: {e}")

			for kind, p, text in samples:
				header = f"{kind} p={p}" if p is not None else f"{kind}"
				print(f"[{model_name}] {header}:\n{text}\n")
				if use_wandb and _HAS_WANDB:
					try:
						# create a one-column table named "text"
						tbl = wandb.Table(data=[[text]], columns=["text"])
						wandb.log({f"samples/{model_name}/{header}": tbl}, step=global_step)
					except Exception as e:
						# fallback: log raw text if Table fails for any reason
						try:
							wandb.log({f"samples/{model_name}/{header}": text}, step=global_step)
						except Exception:
							# swallow to avoid breaking training/logging
							pass
				writer.add_text(f"samples/{model_name}/{header}", text, epoch)
		except Exception as e:
			print(f"[{model_name}] Sampling at epoch end failed: {e}")
		# ----------------------------------------------------------------

	# Final checkpoint at end of training (if requested)
	if ckpt_dir:
		try:
			os.makedirs(ckpt_dir, exist_ok=True)
			final_path = os.path.join(ckpt_dir, f"{model_name}_final.pt")
			# convert EMA tensors to CPU for saving (if present)
			ema_to_save = None
			if ema_active and ema_state:
				ema_to_save = {}
				for k, v in ema_state.items():
					if isinstance(v, torch.Tensor):
						ema_to_save[k] = v.detach().cpu()
					else:
						ema_to_save[k] = v
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
				'global_step': global_step,
				'epoch': epoch,
				'final': True,
				'train_loss': avg_loss,
				'val_loss': avg_val_loss,
				'val_acc': avg_val_acc,
				'ema_state_dict': ema_to_save,
				'ema_decay': ema_decay if ema_active else None,
			}, final_path)
			print(f"[{model_name}] Saved final checkpoint: {final_path}")
		except Exception as e:
			print(f"[{model_name}] Failed to save final checkpoint: {e}")
	# Ensure best model files exist (already maintained during epochs)
	try:
		save_dir = ckpt_dir if ckpt_dir else "."
		if ckpt_dir:
			os.makedirs(save_dir, exist_ok=True)
		# If there are best models, ensure top1 exists or save current as top1
		if best_models:
			# best_models already saved above to rank files; just report
			print(f"[{model_name}] Final Top-{TOP_K} models: {[ (round(x[0],4), x[2]) for x in best_models ]}")
		else:
			# Fallback: save current model as top1
			fallback_path = os.path.join(save_dir, f"{model_name}_top1.pt")
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
				'global_step': global_step,
				'epoch': epoch,
				'best_metric': best_metric if 'best_metric' in locals() else None,
			}, fallback_path)
			print(f"[{model_name}] Saved fallback top1 at: {fallback_path}")
	except Exception as e:
		print(f"[{model_name}] Failed to finalize best model save: {e}")

	writer.close()
	if use_wandb and _HAS_WANDB:
		wandb.finish()


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Try to load config.yaml if provided via --config (simple behaviour)
    if args.config:
        config_path = args.config
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}...")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            # Override any args from config
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: config key '{key}' not found in argparse args.")
        else:
            print(f"Config file {config_path} not found. Using default args.")

    # Load environment variables from a .env file (if present). This enables
    # automatic wandb login via an API key stored in the .env (e.g. WANDB_API_KEY).
    env_loaded = load_env_file(".env")
    if env_loaded:
        print("Loaded environment variables from .env")

    # If the user requested wandb logging and wandb is installed, attempt to
    # login automatically using common env var names.
    if args.use_wandb and _HAS_WANDB:
        possible_keys = ["WANDB_API_KEY", "WANDB_KEY", "WANDB_TOKEN", "WANDB_API_TOKEN"]
        wandb_key = None
        for kk in possible_keys:
            val = os.environ.get(kk)
            if val:
                wandb_key = val
                break
        if wandb_key:
            try:
                # wandb.login accepts `key` parameter
                wandb.login(key=wandb_key)
                print("wandb: logged in using key from environment")
            except Exception as e:
                print(f"wandb: automatic login failed: {e}")
        else:
            print("wandb: --use_wandb set but no WANDB key found in environment; will attempt anonymous or skip login")

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    # --- Hyperparameters ---
    embed_size = args.embed_size     # 256
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    block_size = args.block_size     # 256
    train_subset_size = 50000        # Get a bit more data
    
    # Transformer-specific params (now from args)
    n_heads = args.n_heads
    n_blocks = args.n_blocks
    norm_type = args.norm  # "pre" or "post"
    # -------------------------

    log_interval_steps = 100
    sample_interval_seconds = 60 # Less frequent sampling

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers
    val_split_ratio = 0.1 # 10% for validation 

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")
    print(f"Transformer params: n_heads={n_heads}, n_blocks={n_blocks}, norm={norm_type}")
    use_bf16 = args.use_bf16
    use_flash = args.use_flash_attn

    # Enable some CUDA cudnn tuning for better perf on repeated shapes
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
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
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    # --- Train/Test Split (Optional Task 2)  ---
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
    
    if len(train_tinystories_seqs) == 0 and p_tiny > 0:
        print("Warning: TinyStories train set is empty but tinystories_weight>0.")
        
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=train_tinystories_seqs,
        other_seqs=train_other_seqs,
        p_tiny=p_tiny
    )
    
    val_dataset = MixedSequenceDataset(
        tinystories_seqs=val_tinystories_seqs,
        other_seqs=val_other_seqs,
        p_tiny=p_tiny # Use same prob for val to keep distribution
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation set
            num_workers=0,
            collate_fn=seq_collate_fn
        )
    else:
        print("No validation data. Validation steps will be skipped.")


    ############################################################################
    # Models
    ############################################################################
    models = {}

    if args.model_type in ("all", "kgram"):
        kgram_model = KGramMLPSeqModel(
            vocab_size=vocab_size,
            k=k,
            embed_size=embed_size,
            num_inner_layers=num_inner_layers,
            chunk_size=chunk_size,
            use_embedding=args.kgram_use_embedding
        ).to(device)

        if use_bf16 and str(device).startswith("cuda"):
            try:
                kgram_model.to(dtype=torch.bfloat16)
            except Exception:
                print("Warning: failed to cast kgram_model to bfloat16; continuing in float32.")
        models["kgram_mlp_seq"] = kgram_model

    if args.model_type in ("all", "lstm"):
        lstm_model = LSTMSeqModel(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=embed_size
        ).to(device)

        if use_bf16 and str(device).startswith("cuda"):
            try:
                lstm_model.to(dtype=torch.bfloat16)
            except Exception:
                print("Warning: failed to cast lstm_model to bfloat16; continuing in float32.")
        models["lstm_seq"] = lstm_model

    if args.model_type in ("all", "transformer"):
        transformer_model = TransformerModel(
            vocab_size=vocab_size,
            d_model=embed_size,
            n_heads=n_heads,
            n_blocks=n_blocks,
            block_size=block_size,
            use_flash=use_flash,
            weight_tying=args.weight_tying,
            norm_type=norm_type
        ).to(device)

        if use_bf16 and str(device).startswith("cuda"):
            try:
                transformer_model.to(dtype=torch.bfloat16)
            except Exception:
                print("Warning: failed to cast transformer_model to bfloat16; continuing in float32.")
        models["transformer"] = transformer_model

    # Print model summaries (torchinfo if available, otherwise compact fallback)
    print("\n=== Model summaries ===")
    for model_name, model in models.items():
        if model_name == "transformer":
            print("\n--- Transformer Model Detailed Parameter Breakdown ---")
            total_transformer_params = 0

            def count_params(module):
                return sum(p.numel() for p in module.parameters() if p.requires_grad)

            # Embedding layers
            tok_emb_params = count_params(model.token_embedding)
            pos_emb_params = count_params(model.pos_embedding)
            print(f"  Token Embedding: {tok_emb_params:,} parameters")
            print(f"  Positional Embedding: {pos_emb_params:,} parameters")
            total_transformer_params += tok_emb_params + pos_emb_params

            # Transformer Blocks
            for i, block in enumerate(model.blocks):
                block_params = count_params(block)
                print(f"  Block {i+1} (Total): {block_params:,} parameters")
                print(f"    RMSNorm 1: {count_params(block.norm1):,} parameters")
                print(f"    CausalMultiHeadAttention: {count_params(block.attn):,} parameters")
                print(f"    RMSNorm 2: {count_params(block.norm2):,} parameters")
                print(f"    FeedForward: {count_params(block.ff):,} parameters")
                total_transformer_params += block_params

            # Final Normalization and LM Head
            norm_final_params = count_params(model.norm_final)
            lm_head_params = count_params(model.lm_head)
            print(f"  Final RMSNorm: {norm_final_params:,} parameters")
            print(f"  LM Head: {lm_head_params:,} parameters")
            total_transformer_params += norm_final_params + lm_head_params

            print(f"  Total Transformer Trainable Parameters: {total_transformer_params:,} ({total_transformer_params/1e6:.2f}M)")
            # --- End detailed summary for TransformerModel ---
            

    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        
        # Count parameters
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
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            use_bf16=use_bf16,
            use_amp=args.use_amp,
            ckpt_dir=args.ckpt_dir,               # <--- pass ckpt dir
            save_ckpt_steps=args.save_ckpt_steps, # <--- pass checkpoint frequency
            resume=args.resume,                    # <--- resume flag
            use_wandb=args.use_wandb,              # <--- wandb flag
            log_dir=args.log_dir,                  # <--- tb log dir
            max_lr=args.max_lr,                    # <--- max lr for scheduler
            optimizer_name=args.optimizer,         # pass optimizer choice from config/CLI
            weight_decay=args.weight_decay,        # pass weight decay for AdamW
            ema_decay=args.ema_decay               # ensure EMA arg propagates
        )

        # Final generation from the user-provided prompt (args.prompt).
        print(f"\n--- Final Generation for {model_name} ---")
        with torch.no_grad():
            # 1) Greedy (top_p=None)
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

            # 2) top-p = 0.1 (very low randomness)
            text_topp01, ann_topp01 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.1,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

            # 3) top-p = 0.3 (low randomness)
            text_topp03, ann_topp03 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.3,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

            # 4) top-p = 0.5 (medium randomness)
            text_topp05, ann_topp05 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.5,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

            # 5) top-p = 0.7 (high randomness)
            text_topp07, ann_topp07 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.7,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

            # 6) top-p = 0.95 (very high randomness)
            text_topp095, ann_topp095 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

            # 7) top-p = 1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
                use_bf16=use_bf16,
                use_amp=args.use_amp,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)

        print(f"\n[{model_name}] Final sample (top-p=0.1) from prompt: '{args.prompt}'")
        print(text_topp01)

        print(f"\n[{model_name}] Final sample (top-p=0.3) from prompt: '{args.prompt}'")
        print(text_topp03)

        print(f"\n[{model_name}] Final sample (top-p=0.5) from prompt: '{args.prompt}'")
        print(text_topp05)

        print(f"\n[{model_name}] Final sample (top-p=0.7) from prompt: '{args.prompt}'")
        print(text_topp07)

        print(f"\n[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp095)

        print(f"\n[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)

        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")

if __name__ == "__main__":
    main()