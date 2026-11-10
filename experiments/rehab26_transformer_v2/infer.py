"""Inference on a known 3D skeleton repetition using locally trained v2 artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from core import ROOT, configure_torch, save_json, sha256
from train import load_artifact, predict_artifact


def prepare_segment(segment):
    sys.path.insert(0, str(ROOT / "experiments/rehab26_v1"))
    from pipeline import make_features, normalize_resample
    xyz, visibility, scale = normalize_resample(segment)
    return make_features(xyz, visibility)[None], scale


def predict_segment(path, segment, precision=False):
    path = Path(path).resolve()
    if path.suffix == ".json":
        bundle = json.loads(path.read_text(encoding="utf-8"))
        members = [(path.parent / member).resolve() for member in bundle["members"]]
    else:
        members = [path if path.is_dir() else path.parent]
        bundle = {"status": "single_inner_model", "limitations": "Known exercise and repetition; no calibrated confidence or clinical claim"}
    x, scale = prepare_segment(segment)
    scores, thresholds = [], []
    for folder in members:
        artifact = load_artifact(folder)
        if artifact["identity"]["preprocessing"] != sha256(ROOT / "experiments/rehab26_v1/pipeline.py"):
            raise ValueError("Artifact uses different raw preprocessing")
        scores.append(predict_artifact(artifact, x)[0])
        thresholds.append(artifact["precision_thresholds" if precision else "thresholds"])
        exercise = artifact["exercise"]
        if "model" in artifact:
            del artifact["model"]
        del artifact
    scores, thresholds = np.mean(scores, 0), np.mean(thresholds, 0)
    metadata = json.loads((ROOT / "data/REHAB26-ViewAngles/processed_v1/dataset.json").read_text(encoding="utf-8"))
    labels = metadata["labels"][str(exercise)]
    return {"exercise": exercise, "members": len(members), "status": bundle["status"],
            "prediction_unit": "whole known repetition", "input_frames": len(segment), "torso_scale": scale,
            "operating_point": "validation_F0.5" if precision else "validation_F1",
            "predicted_errors": [label for label, p, t in zip(labels, scores, thresholds) if p >= t],
            "per_error": [{"label": label, "score": float(p), "threshold": float(t)} for label, p, t in zip(labels, scores, thresholds)],
            "scores_are_calibrated_probabilities": False,
            "no_alert_does_not_verify_correctness": True, "limitations": bundle["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Model directory, .pt/.joblib, or bundle.json")
    parser.add_argument("--skeleton", type=Path, required=True, help="MediaPipe world landmarks [L,33,4]")
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True, help="Inclusive last frame")
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_torch()
    array = np.load(args.skeleton, mmap_mode="r", allow_pickle=False)
    if not 0 <= args.first < args.last < len(array):
        raise ValueError("Invalid inclusive frame interval")
    result = predict_segment(args.artifact, array[args.first:args.last+1], args.precision)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
