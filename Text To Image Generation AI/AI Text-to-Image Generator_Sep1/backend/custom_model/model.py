"""
model.py
--------
STEP 5: MODEL ARCHITECTURE

Defines the custom, from-scratch (no pretrained weights) neural network that
will later (Step 6) be trained as a conditional DDPM: given a noisy image
x_t, a diffusion timestep t, and a text prompt c, it predicts the noise
epsilon added to x_t. This module only DEFINES and INSPECTS the network --
it does not train it. Weight updates, loss computation, and optimizers are
Step 6.

COMPONENTS
    TextEncoder   -- token ids -> pooled text embedding c  (Step 1: "Learned
                     Token Embeddings + Cross-Projection", text_embed_dim=128)
    TimeEmbedding -- integer timestep t -> sinusoidal + MLP embedding
                     (time_embed_dim=128)
    ConditionalUNet -- the image generation/decoder component. A small
                     encoder-decoder U-Net over 64x64x3 images, base_channels
                     64, channel_multipliers (1, 2, 4) -> stage widths
                     [64, 128, 256], 2 residual blocks per stage (matches
                     Step 1's ModelArchitectureConfig exactly). Every
                     residual block is FiLM-conditioned (scale & shift) on
                     the combined time+text embedding -- the natural choice
                     for a *pooled* conditioning vector (as opposed to
                     cross-attention, which needs a per-token
                     key/value sequence).
    TextToImageDiffusionModel -- wraps the three pieces above and exposes:
        forward(images, timesteps, token_ids)  -> predicted noise [B,3,64,64]
            (the function Step 6 will train, epsilon_theta(x_t, t, c))
        generate(token_ids, num_steps=...)     -> image [B,3,64,64] in [-1,1]
            (runs the full reverse-diffusion sampling loop through the
            UNTRAINED network -- this is the literal text -> model -> image
            path; with random weights the output is noise-shaped garbage,
            not a real picture. It exists to prove the architecture is
            wired correctly end-to-end, not to produce useful images.)

WEIGHT INITIALIZATION
    Conv2d / Linear : Kaiming-normal (fan_in, nonlinearity="relu") -- a
        reasonable match for the SiLU activations used throughout.
    Embedding       : Normal(mean=0, std=0.02) -- standard small-init so the
        untrained text embedding starts close to zero, not saturated.
    GroupNorm / LayerNorm : weight=1, bias=0 (identity at init).
    Final output conv : explicitly zero-initialized (weight AND bias) --
        a standard DDPM stabilization trick, so the untrained network's
        first-ever prediction is exactly zero noise instead of a large
        random tensor, which keeps early (Step 6) training numerically
        stable.
    All biases elsewhere: zero.

VOCAB SIZE
    The text embedding table is sized from Step 3's ACTUAL fitted
    vocabulary (data/processed/vocab.json, 53 tokens) rather than Step 1's
    upper-bound placeholder (vocab_size=2000) -- the embedding table must
    match the tokenizer that actually produced the token_ids this model
    consumes, or token ids from Step 3/4 would index out of range or into
    meaningless rows.

COMPATIBILITY WITH STEP 4's 399/50/50 SPLITS
    This module is dataset-size-agnostic (it consumes one sample/batch at a
    time), so "compatible with the splits" means: it must correctly consume
    the exact tensor contract Step 3/4 produced -- image [3, 64, 64]
    float32 in [-1, 1], token_ids [32] int64 -- for samples drawn from
    train, val, AND test. `verify_split_compatibility()` below loads a real
    batch from each of the three split files (via their `tensor_file`
    pointers into data/processed/tensors/) and runs a forward pass on each,
    to prove that end-to-end rather than assert it.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    DEFAULT_ARCH_CONFIG,
    DEFAULT_PROBLEM_SPEC,
    LOGS_DIR,
    PROCESSED_TENSORS_DIR,
    TEST_METADATA_PATH,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    VOCAB_PATH,
    ModelArchitectureConfig,
    ProblemSpecification,
)

ARCHITECTURE_REPORT_PATH = LOGS_DIR / "model_architecture_report.json"

# Implementation-level constants not tracked in ModelArchitectureConfig
# (that dataclass fixes *widths*; these fix small structural choices).
TEXT_ENCODER_NHEAD = 4
TEXT_ENCODER_FF_DIM = 256
GROUPNORM_GROUPS = 8
PAD_TOKEN_ID = 0


def load_vocab_size(default: int) -> int:
    """Reads the real fitted vocabulary size from Step 3's vocab.json, so
    the embedding table matches the tokenizer that produced the token ids
    this model will actually see. Falls back to `default` (Step 1's
    placeholder) only if Step 3 hasn't been run yet."""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["vocab_size"]
    return default


# ---------------------------------------------------------------------------
# Text encoder / embedding component
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """Token ids [B, L] -> pooled conditioning embedding [B, embed_dim].

    Embedding + learned positional embedding, fed through a small
    Transformer encoder (num_layers from Step 1's config), then mean-pooled
    over non-<pad> positions. <pad> tokens are excluded both from attention
    (key_padding_mask) and from the pooling average.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_length: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=TEXT_ENCODER_NHEAD,
            dim_feedforward=TEXT_ENCODER_FF_DIM,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # enable_nested_tensor=False: disables a PyTorch inference fast-path
        # (torch._nested_tensor_from_mask_left_aligned) that triggers
        # whenever this encoder is called under eval()+no_grad() with a
        # src_key_padding_mask -- exactly Step 6's validation path -- and
        # is not implemented on the MPS backend. Training-mode forward
        # passes are unaffected either way.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        B, L = token_ids.shape
        pad_mask = token_ids == PAD_TOKEN_ID  # [B, L], True = ignore

        x = self.token_embedding(token_ids) + self.pos_embedding[:, :L, :]
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, embed_dim]

        valid = (~pad_mask).unsqueeze(-1).float()  # [B, L, 1]
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return pooled  # [B, embed_dim]


# ---------------------------------------------------------------------------
# Diffusion timestep embedding
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Scalar timestep t -> fixed sinusoidal embedding [B, dim] (standard
    Transformer/DDPM positional-encoding formula, no trainable parameters)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding -> 2-layer MLP, so the (fixed) sinusoidal
    encoding becomes a learned, trainable conditioning signal."""

    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(timesteps))


# ---------------------------------------------------------------------------
# Image generation / decoder component: a small conditional U-Net
# ---------------------------------------------------------------------------
class FiLMResBlock(nn.Module):
    """Residual block: GroupNorm -> SiLU -> Conv -> FiLM(scale, shift from
    conditioning) -> GroupNorm -> SiLU -> Dropout -> Conv, plus a skip
    connection (1x1 conv only when channel counts differ)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(GROUPNORM_GROUPS, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.film = nn.Linear(cond_dim, out_ch * 2)  # -> (scale, shift)

        self.norm2 = nn.GroupNorm(min(GROUPNORM_GROUPS, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ConditionalUNet(nn.Module):
    """The image generation/decoder component: x_t [B,3,64,64] + cond
    [B,cond_dim] -> predicted noise [B,3,64,64].

    Stage widths = base_channels * channel_multipliers, e.g. 64 * (1,2,4)
    = [64, 128, 256] over resolutions [64, 32, 16], with an 8x8 bottleneck
    -- exactly the Step 1 architecture budget. One skip connection per
    stage (concatenated into the first up-block of that stage) keeps the
    parameter count down while still giving the decoder access to
    high-resolution detail lost during downsampling.
    """

    def __init__(
        self,
        image_channels: int,
        base_channels: int,
        channel_multipliers: Tuple[int, ...],
        num_res_blocks: int,
        cond_dim: int,
        dropout: float,
    ):
        super().__init__()
        stage_channels = [base_channels * m for m in channel_multipliers]  # e.g. [64, 128, 256]

        self.init_conv = nn.Conv2d(image_channels, stage_channels[0], kernel_size=3, padding=1)

        # ---- Encoder (down) path ----
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = stage_channels[0]
        for i, out_ch in enumerate(stage_channels):
            blocks = nn.ModuleList()
            for b in range(num_res_blocks):
                blocks.append(FiLMResBlock(in_ch if b == 0 else out_ch, out_ch, cond_dim, dropout))
            self.down_blocks.append(blocks)
            in_ch = out_ch
            is_last = i == len(stage_channels) - 1
            self.downsamplers.append(Downsample(out_ch) if not is_last else nn.Identity())

        # ---- Bottleneck ----
        bottleneck_ch = stage_channels[-1]
        self.bottleneck1 = FiLMResBlock(bottleneck_ch, bottleneck_ch, cond_dim, dropout)
        self.bottleneck2 = FiLMResBlock(bottleneck_ch, bottleneck_ch, cond_dim, dropout)

        # ---- Decoder (up) path (mirrors the encoder) ----
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        rev_stage_channels = list(reversed(stage_channels))
        in_ch = bottleneck_ch
        for i, out_ch in enumerate(rev_stage_channels):
            blocks = nn.ModuleList()
            for b in range(num_res_blocks):
                # First block of the stage receives the concatenated skip
                # connection from the matching encoder stage.
                block_in = (in_ch + out_ch) if b == 0 else out_ch
                blocks.append(FiLMResBlock(block_in, out_ch, cond_dim, dropout))
            self.up_blocks.append(blocks)
            in_ch = out_ch
            is_last = i == len(rev_stage_channels) - 1
            self.upsamplers.append(Upsample(out_ch) if not is_last else nn.Identity())

        # ---- Output head ----
        self.out_norm = nn.GroupNorm(min(GROUPNORM_GROUPS, stage_channels[0]), stage_channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(stage_channels[0], image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.init_conv(x)

        skips: List[torch.Tensor] = []
        for blocks, downsample in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            h = downsample(h)

        h = self.bottleneck1(h, cond)
        h = self.bottleneck2(h, cond)

        for stage_idx, (blocks, upsample) in enumerate(zip(self.up_blocks, self.upsamplers)):
            skip = skips[-(stage_idx + 1)]
            for b, block in enumerate(blocks):
                h = block(torch.cat([h, skip], dim=1), cond) if b == 0 else block(h, cond)
            h = upsample(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# ---------------------------------------------------------------------------
# Top-level model: text -> conditioning -> U-Net -> image
# ---------------------------------------------------------------------------
class TextToImageDiffusionModel(nn.Module):
    """The complete from-scratch model: wraps TextEncoder + TimeEmbedding +
    ConditionalUNet and defines the full text -> model -> image path."""

    def __init__(
        self,
        vocab_size: int,
        arch_cfg: ModelArchitectureConfig = DEFAULT_ARCH_CONFIG,
        problem_cfg: ProblemSpecification = DEFAULT_PROBLEM_SPEC,
    ):
        super().__init__()
        self.arch_cfg = arch_cfg
        self.problem_cfg = problem_cfg

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=arch_cfg.text_embed_dim,
            max_length=problem_cfg.max_text_length,
            num_layers=arch_cfg.num_text_layers,
            dropout=arch_cfg.dropout,
        )
        self.time_embedding = TimeEmbedding(arch_cfg.time_embed_dim)

        cond_dim = arch_cfg.text_embed_dim + arch_cfg.time_embed_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        self.unet = ConditionalUNet(
            image_channels=problem_cfg.image_channels,
            base_channels=arch_cfg.unet_base_channels,
            channel_multipliers=arch_cfg.channel_multipliers,
            num_res_blocks=arch_cfg.num_res_blocks_per_stage,
            cond_dim=cond_dim,
            dropout=arch_cfg.dropout,
        )

        # Diffusion (beta) schedule -- fixed buffers, not trained, needed by
        # `generate()`'s reverse sampling loop.
        betas = torch.linspace(problem_cfg.beta_start, problem_cfg.beta_end, problem_cfg.diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.apply(self._init_weights)
        # Zero-init the final conv: untrained model starts by predicting
        # exactly zero noise (a standard DDPM stabilization trick), rather
        # than a large random tensor from generic Kaiming init.
        nn.init.zeros_(self.unet.out_conv.weight)
        nn.init.zeros_(self.unet.out_conv.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)

    def encode_condition(self, timesteps: torch.LongTensor, token_ids: torch.LongTensor) -> torch.Tensor:
        text_emb = self.text_encoder(token_ids)          # [B, text_embed_dim]
        time_emb = self.time_embedding(timesteps)         # [B, time_embed_dim]
        cond = torch.cat([text_emb, time_emb], dim=-1)     # [B, cond_dim]
        return self.cond_proj(cond)

    def forward(self, images: torch.FloatTensor, timesteps: torch.LongTensor, token_ids: torch.LongTensor) -> torch.Tensor:
        """The trainable forward pass: epsilon_theta(x_t, t, c).
        images:    [B, 3, 64, 64] float32, the noisy image x_t
        timesteps: [B] long, diffusion step index for each sample
        token_ids: [B, 32] long, tokenized prompt
        returns:   [B, 3, 64, 64] float32, predicted noise
        """
        cond = self.encode_condition(timesteps, token_ids)
        return self.unet(images, cond)

    @torch.no_grad()
    def generate(self, token_ids: torch.LongTensor, num_steps: Optional[int] = None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """The complete text -> model -> image path: starts from pure
        Gaussian noise and runs the full reverse-diffusion sampling loop
        through this (untrained, at this Step) network. `num_steps` lets a
        cheap architecture-check use far fewer steps than the full
        `diffusion_timesteps` schedule; the model is untrained either way,
        so the output is structurally valid but not a meaningful image."""
        self.eval()
        device = next(self.parameters()).device
        B = token_ids.shape[0]
        T = self.problem_cfg.diffusion_timesteps
        step_indices = list(range(T - 1, -1, -1))
        if num_steps is not None and num_steps < len(step_indices):
            stride = max(1, len(step_indices) // num_steps)
            step_indices = step_indices[::stride]

        x = torch.randn(
            B, self.problem_cfg.image_channels, self.problem_cfg.image_height, self.problem_cfg.image_width,
            device=device, generator=generator,
        )

        for t in step_indices:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            predicted_noise = self.forward(x, t_batch, token_ids)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            coef = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * predicted_noise)

            if t > 0:
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Parameter counting + model summary
# ---------------------------------------------------------------------------
def count_parameters(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def build_layer_table(model: nn.Module) -> List[Dict[str, object]]:
    """One row per parameterized leaf module: name, type, output-relevant
    dims, and its own parameter count."""
    rows = []
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # only leaf modules
        n_params = sum(p.numel() for p in module.parameters())
        if n_params == 0 and not isinstance(module, (nn.SiLU, nn.Dropout, nn.Identity)):
            continue
        row = {"name": name, "type": type(module).__name__, "params": n_params}
        if isinstance(module, nn.Conv2d):
            row["dims"] = f"in={module.in_channels}, out={module.out_channels}, k={module.kernel_size[0]}, stride={module.stride[0]}"
            row["activation"] = "-"
        elif isinstance(module, nn.Linear):
            row["dims"] = f"in={module.in_features}, out={module.out_features}"
            row["activation"] = "-"
        elif isinstance(module, nn.Embedding):
            row["dims"] = f"vocab={module.num_embeddings}, dim={module.embedding_dim}"
            row["activation"] = "-"
        elif isinstance(module, nn.GroupNorm):
            row["dims"] = f"groups={module.num_groups}, channels={module.num_channels}"
            row["activation"] = "-"
        elif isinstance(module, nn.SiLU):
            row["dims"] = "-"
            row["activation"] = "SiLU"
        else:
            row["dims"] = "-"
            row["activation"] = "-"
        rows.append(row)
    return rows


def print_model_summary(model: TextToImageDiffusionModel, vocab_size: int, batch_size: int = 4) -> Dict[str, object]:
    print("=" * 70)
    print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
    print("  STEP 5: MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)

    total_params, trainable_params = count_parameters(model)
    text_total, _ = count_parameters(model.text_encoder)
    time_total, _ = count_parameters(model.time_embedding)
    cond_total, _ = count_parameters(model.cond_proj)
    unet_total, _ = count_parameters(model.unet)

    print("\n1. COMPONENT PARAMETER BREAKDOWN:")
    print(f"   - TextEncoder (vocab={vocab_size}):      {text_total:>10,} params")
    print(f"   - TimeEmbedding:                     {time_total:>10,} params")
    print(f"   - Conditioning projection:           {cond_total:>10,} params")
    print(f"   - ConditionalUNet (decoder):         {unet_total:>10,} params")
    print(f"   - TOTAL:                             {total_params:>10,} params ({total_params / 1e6:.2f}M)")
    print(f"   - Trainable:                          {trainable_params:>10,} params")
    print(f"   - Target ceiling (Step 1 config):     {model.arch_cfg.target_param_limit_millions:.1f}M")
    print(f"   - Within budget: {total_params / 1e6 <= model.arch_cfg.target_param_limit_millions}")

    print("\n2. LAYER TABLE (leaf modules with parameters or activations):")
    rows = build_layer_table(model)
    print(f"   {'name':<45} {'type':<16} {'dims':<40} {'act':<6} {'params':>10}")
    for r in rows[:1000]:
        print(f"   {r['name']:<45} {r['type']:<16} {str(r['dims']):<40} {r['activation']:<6} {r['params']:>10,}")

    print("\n3. FORWARD PASS SHAPE TRACE (dummy batch, batch_size={}):".format(batch_size))
    device = next(model.parameters()).device
    dummy_images = torch.randn(batch_size, model.problem_cfg.image_channels, model.problem_cfg.image_height, model.problem_cfg.image_width, device=device)
    dummy_timesteps = torch.randint(0, model.problem_cfg.diffusion_timesteps, (batch_size,), device=device)
    dummy_tokens = torch.randint(0, vocab_size, (batch_size, model.problem_cfg.max_text_length), device=device)

    text_emb = model.text_encoder(dummy_tokens)
    time_emb = model.time_embedding(dummy_timesteps)
    cond = model.encode_condition(dummy_timesteps, dummy_tokens)
    output = model(dummy_images, dummy_timesteps, dummy_tokens)

    shape_trace = {
        "input_images": list(dummy_images.shape),
        "input_timesteps": list(dummy_timesteps.shape),
        "input_token_ids": list(dummy_tokens.shape),
        "text_embedding": list(text_emb.shape),
        "time_embedding": list(time_emb.shape),
        "conditioning_vector": list(cond.shape),
        "predicted_noise_output": list(output.shape),
        "output_dtype": str(output.dtype),
    }
    for k, v in shape_trace.items():
        print(f"   - {k:<24} {v}")

    assert tuple(output.shape) == tuple(dummy_images.shape), "Output shape must match input image shape."
    assert output.dtype == torch.float32, "Output must be float32."
    print("\n   Output shape matches input image shape [B, 3, 64, 64], dtype float32: OK")

    print("=" * 70)

    return {
        "component_params": {
            "text_encoder": text_total,
            "time_embedding": time_total,
            "conditioning_projection": cond_total,
            "conditional_unet": unet_total,
        },
        "total_params": total_params,
        "trainable_params": trainable_params,
        "target_param_limit_millions": model.arch_cfg.target_param_limit_millions,
        "within_budget": total_params / 1e6 <= model.arch_cfg.target_param_limit_millions,
        "layer_table": rows,
        "shape_trace": shape_trace,
    }


# ---------------------------------------------------------------------------
# Step 4 split compatibility check
# ---------------------------------------------------------------------------
def _load_split_batch(metadata_path: Path, n: int = 8) -> Optional[Dict[str, torch.Tensor]]:
    if not metadata_path.exists():
        return None
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    if not rows:
        return None

    images, token_ids, ids = [], [], []
    for r in rows:
        tensor_path = PROCESSED_TENSORS_DIR / r["tensor_file"]
        td = torch.load(tensor_path, weights_only=False)
        images.append(td["image"])
        token_ids.append(td["token_ids"])
        ids.append(r["id"])

    return {
        "images": torch.stack(images),          # [n, 3, 64, 64]
        "token_ids": torch.stack(token_ids),     # [n, 32]
        "ids": ids,
    }


def verify_split_compatibility(model: TextToImageDiffusionModel) -> Dict[str, object]:
    print("\n4. STEP 4 SPLIT COMPATIBILITY CHECK (real train/val/test batches):")
    results = {}
    for split_name, path in [("train", TRAIN_METADATA_PATH), ("val", VAL_METADATA_PATH), ("test", TEST_METADATA_PATH)]:
        batch = _load_split_batch(path)
        if batch is None:
            print(f"   - {split_name:<6} SKIPPED (split file not found)")
            results[split_name] = {"ok": False, "reason": "split file not found"}
            continue

        B = batch["images"].shape[0]
        timesteps = torch.randint(0, model.problem_cfg.diffusion_timesteps, (B,))
        with torch.no_grad():
            out = model(batch["images"], timesteps, batch["token_ids"])

        ok = tuple(out.shape) == tuple(batch["images"].shape) and out.dtype == torch.float32
        print(f"   - {split_name:<6} batch={B:<3} sample_ids={batch['ids']} -> output {list(out.shape)} ok={ok}")
        results[split_name] = {
            "ok": ok,
            "batch_size": B,
            "sample_ids": batch["ids"],
            "input_image_shape": list(batch["images"].shape),
            "output_shape": list(out.shape),
        }
    return results


# ---------------------------------------------------------------------------
# Report entry point
# ---------------------------------------------------------------------------
def run_step5_architecture_report() -> Dict[str, object]:
    vocab_size = load_vocab_size(default=DEFAULT_PROBLEM_SPEC.vocab_size)
    model = TextToImageDiffusionModel(vocab_size=vocab_size)
    model.eval()

    summary = print_model_summary(model, vocab_size=vocab_size)
    split_results = verify_split_compatibility(model)

    print("\n5. END-TO-END TEXT -> MODEL -> IMAGE DEMO (untrained, 20-step sample):")
    demo_tokens = torch.randint(0, vocab_size, (1, model.problem_cfg.max_text_length))
    generated = model.generate(demo_tokens, num_steps=20, generator=torch.Generator().manual_seed(42))
    print(f"   - Generated image shape: {list(generated.shape)}, dtype={generated.dtype}")
    print(f"   - Value range: [{generated.min().item():.4f}, {generated.max().item():.4f}] (expected within [-1, 1])")
    print("   - NOTE: weights are untrained (random init) -- output is structurally")
    print("     valid, not a meaningful image. Training happens in Step 6.")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "step": "Step 5: Model Architecture",
        "trained": False,
        "vocab_size_used": vocab_size,
        "vocab_size_source": "data/processed/vocab.json" if VOCAB_PATH.exists() else "ProblemSpecification default (placeholder)",
        "architecture_config": {
            "text_embed_dim": model.arch_cfg.text_embed_dim,
            "num_text_layers": model.arch_cfg.num_text_layers,
            "unet_base_channels": model.arch_cfg.unet_base_channels,
            "channel_multipliers": list(model.arch_cfg.channel_multipliers),
            "num_res_blocks_per_stage": model.arch_cfg.num_res_blocks_per_stage,
            "time_embed_dim": model.arch_cfg.time_embed_dim,
            "dropout": model.arch_cfg.dropout,
        },
        "parameters": {
            "total": summary["total_params"],
            "trainable": summary["trainable_params"],
            "target_ceiling_millions": summary["target_param_limit_millions"],
            "within_budget": summary["within_budget"],
            "by_component": summary["component_params"],
        },
        "shape_trace": summary["shape_trace"],
        "split_compatibility": split_results,
        "generation_demo": {
            "output_shape": list(generated.shape),
            "output_dtype": str(generated.dtype),
            "value_min": generated.min().item(),
            "value_max": generated.max().item(),
            "num_sampling_steps": 20,
        },
        "notes": "Step 5 architecture only -- no training, no saved weights/checkpoints. Model was NOT initialized from any pretrained checkpoint.",
    }
    with open(ARCHITECTURE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n6. Architecture report saved to: {ARCHITECTURE_REPORT_PATH}")
    print("=" * 70)
    print("Step 5 (Model Architecture) complete. No training performed;")
    print("no pretrained weights were used (Step 6 is training).")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_step5_architecture_report()
