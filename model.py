"""
Transformer Denoiser for Class-Conditional Diffusion Models
Supports ablations: depth, width, positional encoding, conditioning injection
Flash Attention via torch.nn.functional.scaled_dot_product_attention (PyTorch >= 2.0)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_flash_attn_available():
    """
    PyTorch 2.0+ has Flash Attention built-in via F.scaled_dot_product_attention.
    It automatically uses the fastest kernel available (Flash / Memory-Efficient / Math).
    No separate package needed on CUDA; falls back gracefully on CPU/MPS.
    """
    return int(torch.__version__.split(".")[0]) >= 2


# ─────────────────────────────────────────────
# Timestep Embedding
# ─────────────────────────────────────────────
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────
# FiLM / AdaLN Conditioning
# ─────────────────────────────────────────────
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale and shift from conditioning vector."""
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x, cond):
        # x: (B, N, d_model), cond: (B, cond_dim)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)  # (B, d_model) each
        x = self.norm(x)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ─────────────────────────────────────────────
# Flash Attention (SDPA wrapper)
# ─────────────────────────────────────────────
class FlashSelfAttention(nn.Module):
    """
    Multi-head self-attention using F.scaled_dot_product_attention.
    On CUDA with PyTorch >= 2.0 this dispatches to Flash Attention automatically.
    Falls back to standard math attention on CPU/MPS (correct but slower).
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.dropout  = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        # Project to Q, K, V
        qkv = self.qkv_proj(x)                           # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                   # each (B, N, D)

        # Reshape to (B, n_heads, N, d_head)
        def reshape(t):
            return t.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # Flash Attention (or fallback)
        # enable_flash=True is the default in PyTorch >= 2.0 on CUDA
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                  # (B, n_heads, N, d_head)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(attn_out)


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, cond_dim, cond_type="add", use_flash_attn=True):
        """
        cond_type:      "add" (additive to tokens) | "film" (FiLM/AdaLN)
        use_flash_attn: use F.scaled_dot_product_attention (Flash Attn on CUDA)
        """
        super().__init__()
        self.cond_type = cond_type

        if use_flash_attn and check_flash_attn_available():
            self.attn = FlashSelfAttention(d_model, n_heads)
            self.use_flash = True
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.use_flash = False

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        if cond_type == "film":
            self.film1 = FiLMLayer(d_model, cond_dim)
            self.film2 = FiLMLayer(d_model, cond_dim)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def _attn(self, x):
        if self.use_flash:
            return self.attn(x)
        else:
            out, _ = self.attn(x, x, x)
            return out

    def forward(self, x, cond):
        if self.cond_type == "film":
            h = self.film1(x, cond)
            x = x + self._attn(h)
            h = self.film2(x, cond)
            x = x + self.ff(h)
        else:
            h = self.norm1(x)
            x = x + self._attn(h)
            h = self.norm2(x)
            x = x + self.ff(h)
        return x


# ─────────────────────────────────────────────
# Patch-based Transformer Denoiser
# ─────────────────────────────────────────────
class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_channels=1,
        d_model=256,
        n_heads=4,
        depth=4,
        num_classes=10,
        use_pe=True,       # positional encoding ablation
        cond_type="add",   # "add" | "film"
        use_flash_attn=True,  # Flash Attention via SDPA (PyTorch >= 2.0)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.use_pe = use_pe
        self.use_flash_attn = use_flash_attn

        n_patches_side = img_size // patch_size
        self.n_patches = n_patches_side ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Positional encoding (2D absolute, learned)
        if use_pe:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Timestep embedding
        self.time_emb = SinusoidalTimestepEmbedding(d_model)

        # Class embedding (for CFG, class=num_classes means "unconditional")
        self.class_emb = nn.Embedding(num_classes + 1, d_model)

        # Combined condition projection
        cond_dim = d_model
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, cond_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, cond_dim, cond_type, use_flash_attn)
            for _ in range(depth)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, patch_dim)

    def patchify(self, x):
        """(B, C, H, W) -> (B, N, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)       # (B, Hg, Wg, C, p, p)
        x = x.reshape(B, -1, C * p * p)
        return x

    def unpatchify(self, x, img_size):
        """(B, N, patch_dim) -> (B, C, H, W)"""
        B, N, _ = x.shape
        p = self.patch_size
        C = self.in_channels
        g = img_size // p
        x = x.reshape(B, g, g, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)       # (B, C, g, p, g, p)
        x = x.reshape(B, C, img_size, img_size)
        return x

    def forward(self, x_t, t, class_labels, cfg_dropout_prob=0.0):
        """
        x_t: (B, C, H, W)
        t:   (B,) int timesteps
        class_labels: (B,) int class indices
        cfg_dropout_prob: drop class condition for CFG training
        """
        B, C, H, W = x_t.shape

        # Drop class labels for CFG training
        if cfg_dropout_prob > 0.0 and self.training:
            mask = torch.rand(B, device=x_t.device) < cfg_dropout_prob
            class_labels = torch.where(mask, torch.full_like(class_labels, self.class_emb.num_embeddings - 1), class_labels)

        # Tokenize
        tokens = self.patchify(x_t)               # (B, N, patch_dim)
        tokens = self.patch_embed(tokens)          # (B, N, d_model)

        # Positional encoding
        if self.use_pe:
            tokens = tokens + self.pos_emb

        # Conditioning: timestep + class
        t_emb = self.time_emb(t)                  # (B, d_model)
        c_emb = self.class_emb(class_labels)       # (B, d_model)
        cond = self.cond_proj(torch.cat([t_emb, c_emb], dim=-1))  # (B, cond_dim)

        # Add condition to tokens (for "add" type, also injected per block)
        tokens = tokens + cond.unsqueeze(1)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, cond)

        # Output
        tokens = self.norm_out(tokens)
        tokens = self.out_proj(tokens)             # (B, N, patch_dim)
        out = self.unpatchify(tokens, H)           # (B, C, H, W)
        return out


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    flash_available = check_flash_attn_available()
    print(f"Flash Attention available: {flash_available} (PyTorch {torch.__version__})")

    model = TransformerDenoiser(img_size=28, patch_size=4, d_model=256, depth=4,
                                cond_type="film", use_flash_attn=True)
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,))
    out = model(x, t, y)
    print(f"Output shape: {out.shape}")   # (4, 1, 28, 28)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.2f}M")

    # Benchmark flash vs standard (CUDA only)
    if torch.cuda.is_available():
        import time
        device = torch.device("cuda")
        model_flash = TransformerDenoiser(img_size=28, patch_size=4, d_model=256,
                                          depth=4, use_flash_attn=True).to(device)
        model_std   = TransformerDenoiser(img_size=28, patch_size=4, d_model=256,
                                          depth=4, use_flash_attn=False).to(device)
        x   = torch.randn(64, 1, 28, 28, device=device)
        t   = torch.randint(0, 1000, (64,), device=device)
        y   = torch.randint(0, 10,   (64,), device=device)

        # Warm-up
        for _ in range(10):
            model_flash(x, t, y); model_std(x, t, y)
        torch.cuda.synchronize()

        N = 100
        t0 = time.time()
        for _ in range(N): model_flash(x, t, y)
        torch.cuda.synchronize()
        flash_ms = (time.time() - t0) / N * 1000

        t0 = time.time()
        for _ in range(N): model_std(x, t, y)
        torch.cuda.synchronize()
        std_ms = (time.time() - t0) / N * 1000

        print(f"\nFlash Attn: {flash_ms:.2f} ms/fwd")
        print(f"Standard:   {std_ms:.2f} ms/fwd")
        print(f"Speedup:    {std_ms/flash_ms:.2f}x")
