"""Create immutable single-view samples from audited repetition indices."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from pipeline import ROOT, HERE, ANGLE_NAMES, JOINTS, make_features, noisy_features, normalize_resample, save_json, sha256, subject_folds


def main():
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    base = ROOT / "data/REHAB26-ViewAngles"
    output = base / "processed_v1"
    if (output / "dataset.json").exists():
        raise SystemExit("processed_v1 already exists; retain it and choose a new version for changed preprocessing")
    audit = json.loads((base / "audit/summary.json").read_text(encoding="utf-8"))
    if audit["issues"]:
        raise ValueError("Resolve structural audit issues before sample preparation")
    with (base / "raw/Error_types.csv").open(encoding="utf-8-sig", newline="") as handle:
        errors = list(csv.DictReader(handle, delimiter=";"))
    labels = {str(ex): [r["error_code"] for r in errors if int(r["exercise_id"]) == ex] for ex in protocol["exercises"]}
    manifest = [json.loads(line) for line in (base / "audit/repetition_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = [r for r in manifest if r["exercise_id"] in protocol["exercises"]]
    grouped = defaultdict(list)
    for sample in manifest:
        grouped[sample["video_id"]].append(sample)
    features, stress, targets, observations = [], [], [], []
    for video_id, samples in sorted(grouped.items()):
        for camera in (1, 2, 3):
            path = samples[0]["views"][camera - 1]["skeleton_files"]["3d"]
            array = np.load(base / "raw" / path, mmap_mode="r", allow_pickle=False)
            for sample in samples:
                xyz, visibility, scale = normalize_resample(array[sample["first_frame"]:sample["last_frame"] + 1], protocol["preprocessing"]["sequence_length"])
                observation_id = f'{sample["sample_id"]}_cam{camera}'
                features.append(make_features(xyz, visibility))
                stress.append(noisy_features(xyz, visibility, observation_id, protocol["seed"]))
                targets.append([code in sample["error_labels"] for code in labels[str(sample["exercise_id"])]])
                observations.append({**{k: sample[k] for k in ["sample_id", "person", "exercise_id", "first_frame", "last_frame", "primary_error", "additional_errors", "error_labels"]},
                                     "observation_id": observation_id, "camera": camera,
                                     "view_angle_degrees": sample["views"][camera - 1]["view_angle_degrees"],
                                     "skeleton_file": path, "torso_scale_m": scale,
                                     "mean_visibility": float(visibility[:, JOINTS].mean()),
                                     "right_arm_mean_visibility": float(visibility[:, [12, 14, 16]].mean())})
        print(f"Prepared {video_id}", flush=True)
    x, x_noisy, y = np.stack(features), np.stack(stress), np.asarray(targets, dtype=np.uint8)
    assert np.isfinite(x).all() and np.isfinite(x_noisy).all()
    assert len({r["observation_id"] for r in observations}) == len(observations)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "samples.npz", x=x, x_noisy=x_noisy, y=y)
    save_json(output / "observations.json", observations)
    folds = subject_folds(r["person"] for r in observations)
    for fold in folds:
        groups = {group: {r["sample_id"] for r in observations if r["person"] in fold[group]}
                  for group in ["train", "validation", "test"]}
        assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    save_json(output / "folds.json", folds)
    save_json(output / "dataset.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_physical_repetitions": len(manifest), "n_observations": len(observations),
        "x_shape": list(x.shape), "y_shape": list(y.shape), "labels": labels,
        "angle_names": ANGLE_NAMES, "selected_joints": JOINTS,
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "pipeline_sha256": sha256(HERE / "pipeline.py"), "prepare_sha256": sha256(HERE / "prepare.py"),
        "manifest_sha256": sha256(base / "audit/repetition_manifest.jsonl"),
        "samples_sha256": sha256(output / "samples.npz"),
        "segmentation_sha256": sha256(base / "raw/Segmentation.csv"),
        "stress_test": protocol["stress_test"],
    })
    print(f"Saved {x.shape} to {output}", flush=True)


if __name__ == "__main__":
    main()
