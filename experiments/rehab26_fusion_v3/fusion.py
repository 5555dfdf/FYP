"""Calibration-free, repetition-level fusion of MediaPipe world landmarks.

No triangulation or per-frame rigid fit. A constant hip-yaw rotation per view
preserves gravity-relative tilt and shoulder-versus-pelvis motion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/rehab26_v1"))
from pipeline import JOINTS, make_features, normalize_resample


def align_view(segment):
    xyz, visibility, scale = normalize_resample(segment, 96)
    hip_axis = xyz[:, 24] - xyz[:, 23]
    horizontal = hip_axis[:, [0, 2]]
    lengths = np.linalg.norm(horizontal, axis=1)
    valid = lengths > 1e-5
    if not valid.any():
        raise ValueError("Cannot estimate yaw from degenerate hip geometry")
    # Robust constant direction: each valid frame has unit directional vote.
    direction = np.median(horizontal[valid] / lengths[valid, None], axis=0)
    if np.linalg.norm(direction) < 1e-4:
        raise ValueError("Ambiguous hip direction over repetition")
    yaw = float(np.arctan2(direction[1], direction[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    result = xyz.copy()
    result[..., 0] = c * xyz[..., 0] + s * xyz[..., 2]
    result[..., 2] = -s * xyz[..., 0] + c * xyz[..., 2]
    visibility = np.clip(visibility, 0, 1)
    return result, visibility, {"yaw_radians": yaw, "torso_scale": scale}


def merge_aligned(xyz, visibility, method="visibility_squared"):
    xyz = np.asarray(xyz, dtype=np.float32)
    visibility = np.clip(np.asarray(visibility, dtype=np.float32), 0, 1)
    if xyz.shape != (3, 96, 33, 3) or visibility.shape != (3, 96, 33):
        raise ValueError("Expected three matched 96-frame, 33-joint views")
    if not np.isfinite(xyz).all() or not np.isfinite(visibility).all():
        raise ValueError("Nonfinite fusion input")
    if method == "visibility_squared":
        weights = visibility**2
        empty = weights.sum(0) < 1e-8
        weights[:, empty] = 1.0
    elif method == "equal":
        weights = np.ones_like(visibility)
        empty = np.zeros(visibility.shape[1:], dtype=bool)
    else:
        raise ValueError(method)
    weights /= weights.sum(0, keepdims=True)
    fused = (xyz * weights[..., None]).sum(0)
    # Descriptive visibility feature, not a calibrated confidence estimate.
    fused_visibility = (visibility * weights).sum(0)
    # Joint-specific weights can move the fused hip midpoint away from zero.
    segment = np.concatenate([fused, fused_visibility[..., None]], axis=-1)
    fused, fused_visibility, scale = normalize_resample(segment, 96)
    return fused, fused_visibility, {"fallback_joint_frames": int(empty.sum()), "fused_scale": scale}


def fuse_segments(segments):
    aligned = [align_view(segment) for segment in segments]
    xyz = np.stack([r[0] for r in aligned])
    visibility = np.stack([r[1] for r in aligned])
    fused, vis, diag = merge_aligned(xyz, visibility)
    equal, evis, _ = merge_aligned(xyz, visibility, "equal")
    # Spread is a consistency diagnostic only, not ground-truth reconstruction error.
    spread = np.linalg.norm(xyz - xyz.mean(0, keepdims=True), axis=-1)
    return {
        "xyz": fused, "visibility": vis,
        "x": make_features(fused, vis),
        "x_equal": make_features(equal, evis),
        "x_cam2": make_features(xyz[1], visibility[1]),
        "diagnostic": {**diag, "view_alignment": [r[2] for r in aligned],
                       "aligned_spread_mean": float(spread[:, :, JOINTS].mean()),
                       "aligned_spread_q95": float(np.quantile(spread[:, :, JOINTS], .95))},
    }
