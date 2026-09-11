"""Checkpoint-compatible EchoJEPA encoder and LV segmentation model."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# These settings must match frozen_target_context_3000/run_config.json.
RESOLUTION = 224
NUM_FRAMES = 16
PATCH_SIZE = 16
TUBELET_SIZE = 2
EMBED_DIM = 1024
DEPTH = 24
NUM_HEADS = 16
DECODER_DIM = 256
TARGET_TOKEN_INDEX = 4  # centered clip: frame offset 8 / tubelet size 2
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1, 1)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

DEFAULT_BACKBONE = MODELS_DIR / "vjepa21_vitl_mimic_pt117-005_target_fp16.pt"
DEFAULT_SEGMENTATION = MODELS_DIR / "seg_decoder_best.pt"

class PatchEmbed3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            3,
            EMBED_DIM,
            kernel_size=(TUBELET_SIZE, PATCH_SIZE, PATCH_SIZE),
            stride=(TUBELET_SIZE, PATCH_SIZE, PATCH_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def rotate_queries_or_keys(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    omega = torch.arange(head_dim // 2, dtype=x.dtype, device=x.device) / (head_dim / 2.0)
    frequency = torch.einsum("...,f->...f", positions, 1.0 / (10000**omega))
    sin = frequency.sin().squeeze(-1).repeat(1, 1, 1, 2)
    cos = frequency.cos().squeeze(-1).repeat(1, 1, 1, 2)
    pair = x.unflatten(-1, (-1, 2))
    first, second = pair.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        component_dim = 2 * ((self.head_dim // 3) // 2)
        self.d_dim = self.h_dim = self.w_dim = component_dim

    def forward(self, x: torch.Tensor, temporal: int, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, value = qkv[0], qkv[1], qkv[2]

        ids = torch.arange(temporal * height * width, device=x.device)
        frame_pos = ids // (height * width)
        height_pos = (ids - frame_pos * height * width) // width
        width_pos = ids - frame_pos * height * width - height_pos * width

        offset = 0
        q_parts, k_parts = [], []
        for size, positions in (
            (self.d_dim, frame_pos),
            (self.h_dim, height_pos),
            (self.w_dim, width_pos),
        ):
            q_parts.append(rotate_queries_or_keys(q[..., offset : offset + size], positions))
            k_parts.append(rotate_queries_or_keys(k[..., offset : offset + size], positions))
            offset += size
        if offset < self.head_dim:
            q_parts.append(q[..., offset:])
            k_parts.append(k[..., offset:])
        q, k = torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)

        attended = F.scaled_dot_product_attention(q, k, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(attended))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RoPEAttention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor, temporal: int, height: int, width: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), temporal, height, width)
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed3D()
        self.blocks = nn.ModuleList([Block(EMBED_DIM, NUM_HEADS) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=1e-6)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        temporal = video.shape[2] // TUBELET_SIZE
        height = video.shape[3] // PATCH_SIZE
        width = video.shape[4] // PATCH_SIZE
        x = self.patch_embed(video)
        for block in self.blocks:
            x = block(x, temporal, height, width)
        return self.norm(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class SimpleMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(EMBED_DIM * 2, DECODER_DIM, 1),
            nn.GroupNorm(8, DECODER_DIM),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(*[UpsampleBlock(DECODER_DIM) for _ in range(4)])
        self.out = nn.Conv2d(DECODER_DIM, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(self.proj(x))
        if x.shape[-2:] != (RESOLUTION, RESOLUTION):
            x = F.interpolate(x, (RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False)
        return self.out(x)


class EchoJepaSegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VisionTransformer()
        self.decoder = SimpleMaskDecoder()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(video)
        expected = (NUM_FRAMES // TUBELET_SIZE) * (RESOLUTION // PATCH_SIZE) ** 2
        if tokens.shape[1] != expected:
            raise RuntimeError(f"Expected {expected} encoder tokens, got {tokens.shape[1]}")
        grid = RESOLUTION // PATCH_SIZE
        features = tokens.view(video.shape[0], NUM_FRAMES // TUBELET_SIZE, grid, grid, EMBED_DIM)
        target = features[:, TARGET_TOKEN_INDEX]
        context = features.mean(dim=1)
        fused = torch.cat((target, context), dim=-1).permute(0, 3, 1, 2).contiguous()
        return self.decoder(fused)


def load_torch_file(path: Path, device: torch.device | str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    cleaned = OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = value
    return cleaned


def load_model(backbone_path: Path, segmentation_path: Path, device: torch.device) -> EchoJepaSegmentationModel:
    total_start = time.perf_counter()
    print(f"Loading backbone checkpoint: {backbone_path}")
    phase_start = time.perf_counter()
    backbone_checkpoint = load_torch_file(backbone_path)
    print(f"[TIMING] Read backbone checkpoint: {time.perf_counter() - phase_start:.2f}s")
    print("Backbone keys:", list(backbone_checkpoint.keys()))
    if "target_encoder" not in backbone_checkpoint:
        raise KeyError("Expected 'target_encoder' in the EchoJEPA checkpoint")

    phase_start = time.perf_counter()
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = EchoJepaSegmentationModel().to(dtype=model_dtype)
    print(f"[TIMING] Build model architecture: {time.perf_counter() - phase_start:.2f}s")
    phase_start = time.perf_counter()
    backbone_state = clean_state_dict(backbone_checkpoint["target_encoder"])
    incompatible = model.encoder.load_state_dict(backbone_state, strict=False)
    allowed_missing = {"norm.weight", "norm.bias"}
    actual_missing = set(incompatible.missing_keys)
    allowed_unexpected_prefixes = ("img_mod_embed", "video_mod_embed", "patch_embed_img.", "norms_block.")
    bad_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith(allowed_unexpected_prefixes)
    ]
    if actual_missing - allowed_missing or bad_unexpected:
        raise RuntimeError(
            f"Backbone mismatch. Missing={incompatible.missing_keys}; unexpected={incompatible.unexpected_keys}"
        )
    print(
        f"Loaded target_encoder ({len(backbone_state)} tensors); "
        f"allowed missing={incompatible.missing_keys}, allowed extra={len(incompatible.unexpected_keys)}"
    )
    print(f"[TIMING] Apply backbone weights: {time.perf_counter() - phase_start:.2f}s")
    del backbone_checkpoint, backbone_state

    print(f"Loading segmentation checkpoint: {segmentation_path}")
    phase_start = time.perf_counter()
    segmentation_checkpoint = load_torch_file(segmentation_path)
    if "model" not in segmentation_checkpoint:
        raise KeyError("Segmentation checkpoint has no 'model' state_dict")
    segmentation_state = segmentation_checkpoint["model"]
    if not any(key.startswith("decoder.") for key in segmentation_state):
        raise RuntimeError("Segmentation checkpoint contains no decoder weights")
    incompatible = model.load_state_dict(segmentation_state, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith("encoder.")]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Segmentation mismatch. Missing={bad_missing}; unexpected={incompatible.unexpected_keys}"
        )
    print(
        f"Loaded segmentation epoch {segmentation_checkpoint.get('epoch', 'unknown')} "
        f"with metrics {segmentation_checkpoint.get('metrics', {})}"
    )
    print(f"[TIMING] Read/apply decoder checkpoint: {time.perf_counter() - phase_start:.2f}s")
    phase_start = time.perf_counter()
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"[TIMING] Move model to {device} ({model_dtype}): {time.perf_counter() - phase_start:.2f}s")
    print(f"[TIMING] Total model loading: {time.perf_counter() - total_start:.2f}s")
    return model


def make_clip(frames: list[np.ndarray], target: int) -> torch.Tensor:
    start = target - NUM_FRAMES // 2
    indices = [min(max(start + i, 0), len(frames) - 1) for i in range(NUM_FRAMES)]
    array = np.stack([frames[i] for i in indices]).astype(np.float32) / 255.0
    video = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
    return (video - IMAGENET_MEAN) / IMAGENET_STD


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = frame.copy()
    red = np.zeros_like(result)
    red[..., 0] = 255
    result[mask] = (result[mask] * 0.55 + red[mask] * 0.45).astype(np.uint8)
    return result
