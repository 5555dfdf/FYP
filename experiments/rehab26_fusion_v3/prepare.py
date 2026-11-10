"""Create one fused example per physical repetition without labels in fusion."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fusion import ROOT, fuse_segments
from pipeline import noisy_features, save_json, sha256

HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/REHAB26-ViewAngles/processed_fusion_v3"


def main():
    if (DATA / "dataset.json").exists():
        print("Existing immutable fused dataset retained.")
        return
    source = ROOT / "data/REHAB26-ViewAngles/processed_v1"
    meta = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    observations = json.loads((source / "observations.json").read_text(encoding="utf-8"))
    assert sha256(source / "samples.npz") == meta["samples_sha256"]
    groups = defaultdict(list)
    for row in observations:
        groups[row["sample_id"]].append(row)
    arrays = {key: [] for key in ["x", "x_equal", "x_cam2", "x_noisy", "y"]}
    output, diagnostics, opened = [], [], {}
    for sample_id, views in sorted(groups.items()):
        views.sort(key=lambda r: r["camera"])
        assert [r["camera"] for r in views] == [1, 2, 3]
        for key in ["person", "exercise_id", "first_frame", "last_frame", "error_labels"]:
            assert all(r[key] == views[0][key] for r in views)
        segments = []
        for view in views:
            path = view["skeleton_file"]
            if path not in opened:
                opened[path] = np.load(ROOT / "data/REHAB26-ViewAngles/raw" / path, mmap_mode="r", allow_pickle=False)
            segments.append(opened[path][view["first_frame"]:view["last_frame"] + 1])
        merged = fuse_segments(segments)
        for key in ["x", "x_equal", "x_cam2"]:
            arrays[key].append(merged[key])
        arrays["x_noisy"].append(noisy_features(merged["xyz"], merged["visibility"], sample_id, 20261109))
        labels = meta["labels"][str(views[0]["exercise_id"])]
        arrays["y"].append([label in views[0]["error_labels"] for label in labels])
        output.append({k: views[0][k] for k in ["sample_id", "person", "exercise_id", "first_frame", "last_frame", "primary_error", "additional_errors", "error_labels"]})
        output[-1].update(observation_id=sample_id + "_fused", source_views=[{k: r[k] for k in ["camera", "skeleton_file", "view_angle_degrees"]} for r in views])
        diagnostics.append({"sample_id": sample_id, **merged["diagnostic"]})
    arrays = {key: np.asarray(value, dtype=np.uint8 if key == "y" else np.float32) for key, value in arrays.items()}
    assert arrays["x"].shape == (1111, 96, 88)
    assert all(np.isfinite(a).all() for a in arrays.values())
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "samples.npz", **arrays)
    save_json(DATA / "observations.json", output)
    save_json(DATA / "fusion_diagnostics.json", diagnostics)
    save_json(DATA / "dataset.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_physical_repetitions": len(output), "n_observations": len(output),
        "x_shape": list(arrays["x"].shape), "labels": meta["labels"],
        "samples_sha256": sha256(DATA / "samples.npz"),
        "observations_sha256": sha256(DATA / "observations.json"),
        "parent_samples_sha256": meta["samples_sha256"],
        "fusion_sha256": sha256(HERE / "fusion.py"), "prepare_sha256": sha256(HERE / "prepare.py"),
        "all_low_visibility_joint_frames": sum(r["fallback_joint_frames"] for r in diagnostics),
        "spread_median": float(np.median([r["aligned_spread_mean"] for r in diagnostics])),
        "time_alignment": "inclusive annotation boundaries; same 0-to-1 repetition progress resampled to 96 frames; no timestamp correction",
        "rotation": "constant per-view per-repetition yaw from median unit right-minus-left hip horizontal direction",
        "weights": "visibility squared per joint/frame; all-zero weight fallback equal mean",
        "visibility_feature": "same-weight mean visibility, descriptive only",
        "controls": {"x_equal": "same aligned views, equal joint average", "x_cam2": "same preprocessing and yaw alignment, camera 2 only"},
        "scope": "approximate pose fusion in dimensionless torso units, no calibrated metric reconstruction",
    })
    print(json.dumps(json.loads((DATA / "dataset.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
