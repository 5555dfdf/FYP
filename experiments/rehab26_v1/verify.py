"""Verify split isolation, geometric invariants and saved-model reproducibility."""
from __future__ import annotations

import json

import numpy as np
import torch

from infer import load_artifact, predict_segment, score_features
from pipeline import ROOT, HERE, angles, metrics, normalize_resample, sha256
from train import index_groups


def main():
    torch.set_num_threads(4)
    base = ROOT / "data/REHAB26-ViewAngles"
    processed = base / "processed_v1"
    observations = json.loads((processed / "observations.json").read_text(encoding="utf-8"))
    folds = json.loads((processed / "folds.json").read_text(encoding="utf-8"))
    metadata = json.loads((processed / "dataset.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    assert metadata["pipeline_sha256"] == sha256(HERE / "pipeline.py")
    assert metadata["samples_sha256"] == sha256(processed / "samples.npz")
    with np.load(processed / "samples.npz", allow_pickle=False) as data:
        x, y = data["x"], data["y"].astype(bool)
    for ex in protocol["exercises"]:
        all_test = []
        for fold in folds:
            groups = index_groups(observations, ex, fold)
            assert sum(map(len, groups.values())) == sum(r["exercise_id"] == ex for r in observations)
            all_test.extend(groups["test"].tolist())
            for group in groups.values():
                for sample_id in {observations[i]["sample_id"] for i in group}:
                    assert sum(observations[i]["sample_id"] == sample_id for i in group) == 3
        assert len(all_test) == len(set(all_test))
    row = observations[0]
    raw = np.load(base / "raw" / row["skeleton_file"], mmap_mode="r", allow_pickle=False)
    segment = raw[row["first_frame"]:row["last_frame"] + 1].copy()
    normalized, _, _ = normalize_resample(segment)
    transformed = segment.copy()
    transformed[..., :3] = segment[..., :3] * 1.7 + np.asarray([3, -2, 1])
    normalized_transformed, _, _ = normalize_resample(transformed)
    assert np.allclose(normalized, normalized_transformed, atol=2e-5)
    tilt = .35
    rotation = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    transformed[..., :3] = segment[..., :3] @ rotation.T
    tilted, _, _ = normalize_resample(transformed)
    assert not np.allclose(angles(normalized), angles(tilted), atol=1e-3)
    assert np.allclose(normalized[:, [23, 24]].mean(axis=1), 0, atol=1e-5)
    m = metrics(np.array([[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=np.uint8),
                np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]), list("abcde"))
    assert m["per_error"][0]["false_positives"] == 1
    assert m["per_error"][1]["false_negatives"] == 1
    run = ROOT / "runs/rehab26_v1_20261109_cpu"
    checked, largest_difference = 0, 0.0
    for ex in protocol["exercises"]:
        for fold in folds:
            for variant in protocol["models"]:
                directory = run / f'ex{ex}/fold{fold["fold"]}_{fold["test"][0]}/{variant}'
                extension = "joblib" if variant.startswith("angle_") else "pt"
                artifact = load_artifact(directory / f"model.{extension}")
                with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
                    ids, expected = predictions["test_indices"], predictions["test_scores"]
                    positions = np.asarray([0, len(ids) // 2, len(ids) - 1])
                    recomputed = score_features(artifact, x[ids[positions]])
                    difference = float(np.abs(recomputed - expected[positions]).max())
                    largest_difference = max(largest_difference, difference)
                    assert np.allclose(recomputed, expected[positions], atol=2e-6), (directory, difference)
                    assert np.array_equal(predictions["y_test"], y[ids])
                    observed = observations[int(ids[0])]
                    raw = np.load(base / "raw" / observed["skeleton_file"], mmap_mode="r", allow_pickle=False)
                    result = predict_segment(artifact, raw[observed["first_frame"]:observed["last_frame"] + 1])
                    raw_scores = np.asarray([r["score"] for r in result["per_error"]])
                    assert np.allclose(raw_scores, expected[0], atol=2e-6), directory
                checked += 1
    print(json.dumps({"status": "passed", "saved_models_reloaded": checked,
                      "cached_feature_predictions_checked": checked * 3, "raw_clip_inference_checked": checked,
                      "max_score_difference": largest_difference,
                      "other_checks": ["subject/view isolation", "one test appearance per observation", "translation/scale invariance", "torso tilt retained", "explicit multilabel FP/FN accounting"]}, indent=2))


if __name__ == "__main__":
    main()
