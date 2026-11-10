"""Shared data, models and metrics for subject-disjoint REHAB26 experiments."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/REHAB26-ViewAngles/processed_v1"
RUN = ROOT / "runs/rehab26_transformer_v2"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_data():
    metadata = json.loads((DATA / "dataset.json").read_text(encoding="utf-8"))
    assert sha256(DATA / "samples.npz") == metadata["samples_sha256"], "Changed sample cache"
    assert sha256(ROOT / "experiments/rehab26_v1/pipeline.py") == metadata["pipeline_sha256"]
    observations = json.loads((DATA / "observations.json").read_text(encoding="utf-8"))
    with np.load(DATA / "samples.npz", allow_pickle=False) as arrays:
        x, noisy, y = arrays["x"], arrays["x_noisy"], arrays["y"].astype(np.float32)
    assert x.shape == (3333, 96, 88) and y.shape == (3333, 5)
    assert np.isfinite(x).all() and np.isfinite(noisy).all()
    assert set(np.unique(y)) == {0, 1}
    assert len(set(r["observation_id"] for r in observations)) == len(x)
    for i, row in enumerate(observations):
        assert [bool(v) for v in y[i]] == [label in row["error_labels"] for label in metadata["labels"][str(row["exercise_id"])]]
    return x, noisy, y, observations, metadata


def groups(observations, exercise, test, validation):
    subjects = sorted({r["person"] for r in observations})
    assert test != validation and test in subjects and validation in subjects
    split = {"train": [p for p in subjects if p not in (test, validation)], "validation": [validation], "test": [test]}
    indices = {name: np.asarray([i for i, r in enumerate(observations) if r["exercise_id"] == exercise and r["person"] in people]) for name, people in split.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(split[a]) & set(split[b])
        assert not {observations[i]["sample_id"] for i in indices[a]} & {observations[i]["sample_id"] for i in indices[b]}
    assert sum(map(len, indices.values())) == sum(r["exercise_id"] == exercise for r in observations)
    return indices, split


def feature_indices(config):
    return np.arange({"xyz": 57, "angles": 69, "visibility": 88}[config["features"]])


def fit_scaler(x, train, config):
    features = feature_indices(config)
    mean = np.zeros(len(features), dtype=np.float32)
    std = np.ones(len(features), dtype=np.float32)
    if config.get("standardize", True):
        stop = min(len(features), 69)
        # Index current inner-training rows before computing any statistic.
        selected = x[train, :, :stop]
        mean[:stop] = selected.mean((0, 1), dtype=np.float64)
        std[:stop] = np.maximum(selected.std((0, 1), dtype=np.float64), 1e-4)
    return mean, std


def transform(x, config, mean, std):
    return np.ascontiguousarray((x[..., feature_indices(config)] - mean) / std, dtype=np.float32)


def angle_statistics(x):
    a = x[:, :, 57:69]
    return np.concatenate([a.mean(1), a.std(1), a.min(1), a.max(1), np.quantile(a, .1, axis=1),
                           np.quantile(a, .9, axis=1), np.abs(np.diff(a, axis=1)).mean(1)], axis=-1).astype(np.float32)


def select_thresholds(y, scores, beta=1.0):
    grid = sorted(np.arange(1, 20) / 20, key=lambda t: (round(abs(t-.5), 8), t))
    result = []
    for j in range(5):
        best, chosen = -1., .5
        for threshold in grid:
            p = scores[:, j] >= threshold
            tp = np.logical_and(p, y[:, j]).sum()
            fp = np.logical_and(p, ~y[:, j].astype(bool)).sum()
            fn = np.logical_and(~p, y[:, j]).sum()
            value = (1+beta**2)*tp / max((1+beta**2)*tp + beta**2*fn + fp, 1)
            if value > best + 1e-12:
                best, chosen = value, threshold
        result.append(chosen)
    # Preserve the exact grid values used in selection, especially for float64
    # sklearn scores at a threshold boundary. Casting .8 to float32 can flip >=.
    return np.asarray(result, dtype=np.float64)


def metrics(y, scores, thresholds=None, decisions=None):
    y = np.asarray(y, dtype=bool)
    predicted = np.asarray(decisions, dtype=bool) if decisions is not None else scores >= np.asarray(.5 if thresholds is None else thresholds)
    p, r, f, support = precision_recall_fscore_support(y, predicted, average=None, zero_division=0)
    ap = np.asarray([average_precision_score(y[:, j], scores[:, j]) if y[:, j].any() else 0. for j in range(5)])
    correct, alert = ~y.any(1), predicted.any(1)
    return {"n": len(y), "macro_f1": float(f.mean()), "macro_ap": float(ap.mean()),
            "micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
            "exact_match": float((y == predicted).all(1).mean()),
            "false_alert_rate": float(alert[correct].mean()) if correct.any() else None,
            "missed_alert_rate": float((~alert[~correct]).mean()) if (~correct).any() else None,
            "per_error": [{"precision": float(p[j]), "recall": float(r[j]), "f1": float(f[j]), "ap": float(ap[j]),
                           "support": int(support[j]), "fp": int((predicted[:, j] & ~y[:, j]).sum()),
                           "fn": int((~predicted[:, j] & y[:, j]).sum())} for j in range(5)]}


class TemporalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        width, dropout = config.get("width", 64), config.get("dropout", .1)
        activation = nn.GELU if config.get("activation", "gelu") == "gelu" else nn.ReLU
        self.input = nn.Linear(len(feature_indices(config)), width)
        # Separate constructors ensure genuinely independent initialization of layers.
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, 4, width*2, dropout,
                                   activation=config.get("activation", "gelu"), batch_first=True,
                                   norm_first=config.get("pre_norm", True)) for _ in range(config.get("layers", 2))])
        self.final_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(width*2, 32), activation(), nn.Dropout(dropout), nn.Linear(32, 5))
        positions = torch.arange(96).float()[:, None]
        factors = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.)/width))
        encoding = torch.zeros(1, 96, width)
        encoding[0, :, 0::2] = torch.sin(positions*factors)
        encoding[0, :, 1::2] = torch.cos(positions*factors)
        self.register_buffer("position", encoding)

    def forward(self, x):
        x = self.dropout(self.input(x) + self.position)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(torch.cat([x.mean(1), x.amax(1)], 1))


class TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = 24
        incoming = len(feature_indices(config))
        self.blocks, self.skips = nn.ModuleList(), nn.ModuleList()
        for dilation in [1, 2, 4]:
            layers = []
            for source in [incoming, width]:
                layers += [nn.Conv1d(source, width, 5, padding=2*dilation, dilation=dilation),
                           nn.GroupNorm(4, width), nn.ReLU(), nn.Dropout(config.get("dropout", .1))]
            self.blocks.append(nn.Sequential(*layers))
            self.skips.append(nn.Conv1d(incoming, width, 1) if incoming != width else nn.Identity())
            incoming = width
        self.head = nn.Linear(2*width, 5)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block, skip in zip(self.blocks, self.skips):
            x = torch.relu(block(x) + skip(x))
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], 1))


def build_model(config):
    return TemporalTransformer(config) if config["family"] == "transformer" else TCN(config)


def score_model(model, x, device="cuda", batch_size=256):
    model.eval()
    scores = []
    with torch.inference_mode():
        for first in range(0, len(x), batch_size):
            tensor = x[first:first+batch_size]
            if not torch.is_tensor(tensor):
                tensor = torch.from_numpy(tensor)
            scores.append(model(tensor.to(device)).sigmoid().cpu().numpy())
    return np.concatenate(scores)


def configure_torch():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fused SDPA and eval fast paths can change numeric behavior by batch shape.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)


def source_identity():
    return {"samples": sha256(DATA / "samples.npz"), "observations": sha256(DATA / "observations.json"),
            "preprocessing": sha256(ROOT / "experiments/rehab26_v1/pipeline.py"),
            "core": sha256(HERE / "core.py"), "train": sha256(HERE / "train.py"),
            "suite": sha256(HERE / "suite.py"),
            "protocol": sha256(HERE / "protocol.json")}
