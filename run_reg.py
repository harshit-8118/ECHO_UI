"""EF regression using the shared EchoJEPA encoder and official attentive probe."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_seg import (
    EMBED_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_FRAMES,
    VisionTransformer,
    load_torch_file,
)


class ProbeMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, channels)))


class ProbeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = Attention()
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = ProbeMLP(EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 16
        self.q = nn.Linear(EMBED_DIM, EMBED_DIM, bias=True)
        self.kv = nn.Linear(EMBED_DIM, EMBED_DIM * 2, bias=True)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, queries, channels = query.shape
        query = self.q(query).reshape(batch, queries, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        count = tokens.shape[1]
        kv = self.kv(tokens).reshape(batch, count, 2, self.num_heads, channels // self.num_heads)
        key, value = kv.permute(2, 0, 3, 1, 4)
        query = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return query.transpose(1, 2).reshape(batch, queries, channels)


class CrossAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.xattn = CrossAttention()
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = ProbeMLP(EMBED_DIM)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        query = query + self.xattn(query, self.norm1(tokens))
        return query + self.mlp(self.norm2(query))


class AttentivePooler(nn.Module):
    """Exact depth-4 official attentive pooler used by the regression run."""

    def __init__(self) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.cross_attention_block = CrossAttentionBlock()
        self.blocks = nn.ModuleList([ProbeBlock() for _ in range(3)])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        query = self.query_tokens.repeat(tokens.shape[0], 1, 1)
        return self.cross_attention_block(query, tokens)


class AttentiveRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pooler = AttentivePooler()
        self.regressor = nn.Linear(EMBED_DIM, 1, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.pooler(tokens).squeeze(1))


def load_regression_probe(checkpoint_path: Path, device: torch.device) -> AttentiveRegressor:
    start = time.perf_counter()
    checkpoint = load_torch_file(checkpoint_path)
    if "model" not in checkpoint:
        raise KeyError("Regression checkpoint has no 'model' state_dict")
    state = checkpoint["model"]
    probe_state = {
        key[len("probe.") :]: value for key, value in state.items() if key.startswith("probe.")
    }
    if not probe_state:
        raise RuntimeError("Regression checkpoint contains no probe.* weights")
    floating_tensor = next((value for value in probe_state.values() if torch.is_floating_point(value)), None)
    checkpoint_dtype = floating_tensor.dtype if floating_tensor is not None else torch.float32
    # Native FP16 execution is intended for CUDA. On CPU, retain the quantized
    # checkpoint values but upcast them to FP32 for operator compatibility.
    runtime_dtype = checkpoint_dtype if device.type == "cuda" else torch.float32
    probe = AttentiveRegressor().to(dtype=runtime_dtype)
    incompatible = probe.load_state_dict(probe_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Regression probe mismatch: {incompatible}")
    probe = probe.to(device).eval()
    print(
        f"[TIMING] Loaded EF probe (checkpoint={checkpoint_dtype}, runtime={runtime_dtype}) "
        f"from epoch {checkpoint.get('epoch', 'unknown')} "
        f"in {time.perf_counter() - start:.2f}s; metrics={checkpoint.get('metrics', {})}"
    )
    return probe


def regression_clip_indices(frame_count: int, randomized: bool, rng: random.Random) -> list[int]:
    """Training-style stratified random indices or deterministic validation indices."""
    if frame_count < 1:
        raise ValueError("Video has no frames")
    if randomized:
        bins = np.linspace(0, frame_count, NUM_FRAMES + 1)
        values = [rng.uniform(bins[i], max(bins[i] + 1, bins[i + 1])) for i in range(NUM_FRAMES)]
    else:
        values = np.linspace(0, frame_count - 1, NUM_FRAMES)
    return [min(max(int(round(value)), 0), frame_count - 1) for value in values]


def make_regression_clip(frames: list[np.ndarray], indices: list[int]) -> torch.Tensor:
    array = np.stack([frames[index] for index in indices]).astype(np.float32) / 255.0
    clip = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    return (clip - IMAGENET_MEAN) / IMAGENET_STD


@torch.inference_mode()
def predict_ef(
    encoder: VisionTransformer,
    probe: AttentiveRegressor,
    model_frames: list[np.ndarray],
    device: torch.device,
    num_clips: int = 1,
    seed: int = 7,
) -> dict[str, object]:
    """Predict EF; clip 0 is uniform and additional clips are stratified random."""
    rng = random.Random(seed)
    predictions, sampled_indices = [], []
    encoder_seconds = probe_seconds = 0.0
    for clip_number in range(num_clips):
        indices = regression_clip_indices(len(model_frames), randomized=clip_number > 0, rng=rng)
        encoder_dtype = next(encoder.parameters()).dtype
        clip = make_regression_clip(model_frames, indices).unsqueeze(0).to(
            device=device, dtype=encoder_dtype
        )
        phase_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            tokens = encoder(clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_seconds += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            prediction = probe(tokens)
        predictions.append(float(prediction.float().cpu().item() * 100.0))
        probe_seconds += time.perf_counter() - phase_start
        sampled_indices.append(indices)
    print(f"[TIMING] EF encoder: {encoder_seconds:.2f}s | probe: {probe_seconds:.2f}s")
    return {
        "ef_percent": float(np.mean(predictions)),
        "ef_std": float(np.std(predictions)),
        "clip_predictions": predictions,
        "clip_indices": sampled_indices,
    }
