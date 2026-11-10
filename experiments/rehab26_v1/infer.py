"""Run a saved model on one known repetition from a 3d .npy skeleton."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from pipeline import HERE, angle_statistics, build_tcn, make_features, normalize_resample, reliability_margin, save_json, sha256


def load_artifact(path):
    path = Path(path)
    if path.suffix == ".pt":
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".joblib":
        artifact = joblib.load(path)
    else:
        raise ValueError("Expected a locally generated .pt or .joblib artifact")
    if artifact["identity"]["pipeline_sha256"] != sha256(HERE / "pipeline.py"):
        raise ValueError("Artifact and preprocessing code hashes differ")
    if artifact["variant"].startswith("tcn_"):
        model = build_tcn(len(artifact["feature_indices"]), artifact["tcn_config"])
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        artifact["loaded_model"] = model
    return artifact


def score_features(artifact, features):
    if features.ndim == 2:
        features = features[None]
    if artifact["variant"].startswith("angle_"):
        stats = angle_statistics(features)
        scores = np.stack([m.predict_proba(stats)[:, 1] for m in artifact["estimators"]], axis=1)
    else:
        selected = features[:, :, artifact["feature_indices"]]
        standardized = (selected - np.asarray(artifact["mean"], dtype=np.float32)) / np.asarray(artifact["std"], dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(standardized.transpose(0, 2, 1)))
        with torch.no_grad():
            scores = torch.sigmoid(artifact["loaded_model"](tensor)).numpy()
    return scores


def predict_segment(artifact, segment):
    xyz, visibility, scale = normalize_resample(segment)
    scores = score_features(artifact, make_features(xyz, visibility))[0]
    thresholds = np.asarray(artifact["thresholds"])
    margin = float(reliability_margin(scores[None], thresholds)[0])
    policy = artifact["reliability_policy"]
    if not policy["qualified_on_validation"]:
        status = "reliability_policy_not_supported_on_validation"
    elif margin < policy["cutoff"]:
        status = "unable_to_judge_reliably"
    else:
        status = "retained_by_validation_policy_not_a_guarantee"
    return {"exercise": artifact["exercise"], "model": artifact["variant"],
            "prediction_unit": "whole repetition, not frame-level localization",
            "raw_predicted_errors": [code for code, score, threshold in zip(artifact["labels"], scores, thresholds) if score >= threshold],
            "reliability_status": status, "threshold_distance": margin,
            "scores_are_calibrated_probabilities": False,
            "per_error": [{"code": label, "score": float(score), "validation_threshold": float(threshold)}
                          for label, score, threshold in zip(artifact["labels"], scores, thresholds)],
            "torso_scale_m": scale, "input_frames": len(segment), "resampled_frames": 96,
            "test_person_for_this_fold": artifact["fold"]["test"][0],
            "limitations": "Known exercise and repetition boundaries required; no RGB, medical prescription or frame-level error interval inferred."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    skeleton = np.load(args.skeleton, mmap_mode="r", allow_pickle=False)
    if not (0 <= args.first <= args.last < len(skeleton)):
        raise ValueError("Frame interval outside the input array")
    artifact = load_artifact(args.artifact)
    result = predict_segment(artifact, skeleton[args.first:args.last + 1])
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
