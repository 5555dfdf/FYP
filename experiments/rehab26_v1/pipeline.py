"""Shared, deterministic preprocessing, metrics and lightweight TCN."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
ANGLE_NAMES = ["right_elbow", "left_elbow", "right_shoulder", "left_shoulder",
               "right_knee", "left_knee", "torso_vertical", "head_vs_shoulders",
               "shoulder_inclination", "right_arm_forward", "right_arm_lateral", "torso_yaw"]
XYZ_CHANNELS = len(JOINTS) * 3
ANGLE_SLICE = slice(XYZ_CHANNELS, XYZ_CHANNELS + len(ANGLE_NAMES))
VIS_SLICE = slice(ANGLE_SLICE.stop, ANGLE_SLICE.stop + len(JOINTS))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-7)


def angle(a, b, c):
    return np.arccos(np.clip((unit(a - b) * unit(c - b)).sum(axis=-1), -1, 1)) / np.pi


def angles(xyz: np.ndarray) -> np.ndarray:
    values = [angle(xyz[:, a], xyz[:, b], xyz[:, c]) for a, b, c in (
        (12, 14, 16), (11, 13, 15), (24, 12, 14), (23, 11, 13), (24, 26, 28), (23, 25, 27))]
    torso = unit((xyz[:, 11] + xyz[:, 12] - xyz[:, 23] - xyz[:, 24]) / 2)
    shoulder = unit(xyz[:, 12] - xyz[:, 11])
    arm = unit(xyz[:, 14] - xyz[:, 12])
    normal = unit(np.cross(shoulder, torso))
    values.extend([
        np.arccos(np.clip(-torso[:, 1], -1, 1)) / np.pi,
        np.arccos(np.clip((unit(xyz[:, 8] - xyz[:, 7]) * shoulder).sum(axis=-1), -1, 1)) / np.pi,
        np.arcsin(np.clip(shoulder[:, 1], -1, 1)) / (np.pi / 2),
        (arm * normal).sum(axis=-1), (arm * shoulder).sum(axis=-1),
        np.arctan2(shoulder[:, 2], shoulder[:, 0]) / np.pi,
    ])
    return np.stack(values, axis=-1).astype(np.float32)


def normalize_resample(segment: np.ndarray, length: int = 96):
    segment = np.asarray(segment, dtype=np.float32)
    if segment.ndim != 3 or segment.shape[1:] != (33, 4) or len(segment) < 2:
        raise ValueError("Expected at least two frames, 33 joints and xyz+visibility")
    if not np.isfinite(segment).all() or (segment[..., :3] == 0).all(axis=(1, 2)).any():
        raise ValueError("Nonfinite or full-zero input frame requires an explicit missing-data policy")
    xyz = segment[..., :3]
    hip = (xyz[:, 23] + xyz[:, 24]) / 2
    shoulder = (xyz[:, 11] + xyz[:, 12]) / 2
    scale = float(np.median(np.linalg.norm(shoulder - hip, axis=-1)))
    if scale < 1e-5:
        raise ValueError("Degenerate torso scale")
    xyz = (xyz - hip[:, None]) / scale
    position = np.linspace(0, len(segment) - 1, length)
    low = np.floor(position).astype(int)
    high = np.ceil(position).astype(int)
    weight = (position - low).astype(np.float32)[:, None, None]
    sampled = xyz[low] * (1 - weight) + xyz[high] * weight
    visibility = np.minimum(segment[low, :, 3], segment[high, :, 3])
    return sampled, visibility, scale


def make_features(xyz, visibility):
    return np.concatenate([xyz[:, JOINTS].reshape(len(xyz), -1), angles(xyz), visibility[:, JOINTS]], axis=-1).astype(np.float32)


def noisy_features(xyz, visibility, observation_id, seed=20261109):
    xyz, visibility = xyz.copy(), visibility.copy()
    digest = hashlib.sha256(f"{observation_id}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    xyz[32:64, [14, 16]] += rng.normal(0, .15, (32, 2, 3)).astype(np.float32)
    visibility[32:64, [14, 16]] = .1
    return make_features(xyz, visibility)


def angle_statistics(x):
    a = x[:, :, ANGLE_SLICE]
    return np.concatenate([a.mean(axis=1), a.std(axis=1), a.min(axis=1), a.max(axis=1),
                           np.quantile(a, .1, axis=1), np.quantile(a, .9, axis=1),
                           np.abs(np.diff(a, axis=1)).mean(axis=1)], axis=-1).astype(np.float32)


def feature_indices(variant):
    stop = {"tcn_xyz": XYZ_CHANNELS, "tcn_xyz_angles": ANGLE_SLICE.stop,
            "tcn_xyz_angles_visibility": VIS_SLICE.stop}[variant]
    return np.arange(stop)


def subject_folds(subjects):
    subjects = sorted(set(subjects))
    if len(subjects) != 5:
        raise ValueError("This protocol requires exactly five subjects")
    return [{"fold": i, "test": [person], "validation": [subjects[(i + 1) % 5]],
             "train": [p for p in subjects if p not in {person, subjects[(i + 1) % 5]}]}
            for i, person in enumerate(subjects)]


def thresholds_from_validation(y, scores):
    from sklearn.metrics import f1_score
    grid = sorted(np.round(np.arange(.15, .851, .05), 2), key=lambda v: (abs(v - .5), v))
    return np.asarray([max(grid, key=lambda t: f1_score(y[:, j], scores[:, j] >= t, zero_division=0))
                       for j in range(y.shape[1])], dtype=np.float32)


def reliability_margin(scores, thresholds):
    denominator = np.where(scores >= thresholds, 1 - thresholds, thresholds)
    return (np.abs(scores - thresholds) / denominator).min(axis=1)


def select_reliability_policy(y, scores, thresholds, grid):
    margin = reliability_margin(scores, thresholds)
    exact = ((scores >= thresholds) == y).all(axis=1)
    candidates = []
    for cutoff in grid:
        keep = margin >= cutoff
        coverage = float(keep.mean())
        accuracy = float(exact[keep].mean()) if keep.any() else None
        candidates.append({"cutoff": cutoff, "coverage": coverage, "exact_match": accuracy})
    qualifying = [r for r in candidates if r["coverage"] >= .5 and r["exact_match"] >= .8]
    chosen = max(qualifying, key=lambda r: r["coverage"]) if qualifying else None
    return {"qualified_on_validation": chosen is not None,
            "cutoff": chosen["cutoff"] if chosen else None, "validation_candidates": candidates}


def metrics(y, predicted, labels):
    from sklearn.metrics import precision_recall_fscore_support, f1_score
    y, predicted = np.asarray(y, dtype=bool), np.asarray(predicted, dtype=bool)
    if len(y) == 0:
        return {"n_observations": 0}
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, average=None, zero_division=0)
    correct = ~y.any(axis=1)
    alerted = predicted.any(axis=1)
    return {
        "n_observations": len(y), "macro_f1": float(f1.mean()),
        "micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "exact_match": float((y == predicted).all(axis=1).mean()),
        "incorrect_detection_f1": float(f1_score(~correct, alerted, zero_division=0)),
        "n_correct": int(correct.sum()), "n_incorrect": int((~correct).sum()),
        "false_alert_rate": float(alerted[correct].mean()) if correct.any() else None,
        "missed_alert_rate": float((~alerted[~correct]).mean()) if (~correct).any() else None,
        "per_error": [{"label": label, "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i]),
                       "false_positives": int((predicted[:, i] & ~y[:, i]).sum()),
                       "false_negatives": int((~predicted[:, i] & y[:, i]).sum())} for i, label in enumerate(labels)],
    }


def build_tcn(input_channels, config):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, incoming, width, dilation):
            super().__init__()
            padding = dilation * (config["kernel_size"] - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(incoming, width, config["kernel_size"], dilation=dilation, padding=padding),
                nn.GroupNorm(4, width), nn.ReLU(), nn.Dropout(config["dropout"]),
                nn.Conv1d(width, width, config["kernel_size"], dilation=dilation, padding=padding),
                nn.GroupNorm(4, width), nn.ReLU(), nn.Dropout(config["dropout"]),
            )
            self.skip = nn.Conv1d(incoming, width, 1) if incoming != width else nn.Identity()

        def forward(self, x):
            return torch.relu(self.net(x) + self.skip(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            width = config["width"]
            self.blocks = nn.Sequential(*[Block(input_channels if i == 0 else width, width, d)
                                          for i, d in enumerate(config["dilations"])])
            self.output = nn.Linear(width * 2, 5)

        def forward(self, x):
            x = self.blocks(x)
            return self.output(torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1))

    return TCN()
