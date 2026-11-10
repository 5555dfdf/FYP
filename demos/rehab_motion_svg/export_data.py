"""Export labelled REHAB26 repetitions for a self-contained SVG animation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/rehab26_fusion_v3"))
from fusion import align_view, merge_aligned

LABELS = {
    "ERR_TORSO_FORWARD": ("躯干前倾", "后伸时躯干随手臂向前倾斜。", [11, 12, 23, 24]),
    "ERR_ELEVATED_SHOULDER": ("耸肩", "抬高肩部，出现肩带代偿。", [11, 12, 7, 8]),
    "ERR_TORSO_ROTATION": ("躯干旋转", "肩线相对骨盆发生转动。", [11, 12, 23, 24]),
    "ERR_BENT_ARM": ("屈肘", "运动过程中肘部弯曲。", [13, 14, 15, 16]),
    "ERR_ARM_TO_SIDE": ("手臂外偏", "手臂后伸时同时向身体外侧偏移。", [11, 12, 13, 14, 15, 16]),
    "ERR_ARM_FORWARD": ("手臂前偏", "手臂外展时偏向身体前方。", [11, 12, 13, 14, 15, 16]),
    "ERR_HEAD_TO_SIDE": ("头部侧倾", "抬臂时头部向一侧倾斜。", list(range(11))),
    "ERR_LOW_ABDUCTION": ("外展幅度不足", "手臂外展的最高位置偏低。", [11, 12, 13, 14, 15, 16]),
}
EXERCISES = {
    7: {"name": "肩后伸", "en": "Shoulder extension", "hint": "观察躯干稳定、肩部位置与肘部伸展。", "yaw": 65,
        "errors": ["ERR_TORSO_FORWARD", "ERR_BENT_ARM", "ERR_ELEVATED_SHOULDER", "ERR_TORSO_ROTATION", "ERR_ARM_TO_SIDE"]},
    1: {"name": "肩外展", "en": "Shoulder abduction", "hint": "观察外展轨迹、抬臂幅度与头肩姿态。", "yaw": 0,
        "errors": ["ERR_LOW_ABDUCTION", "ERR_BENT_ARM", "ERR_ELEVATED_SHOULDER", "ERR_ARM_FORWARD", "ERR_HEAD_TO_SIDE"]},
}


def main():
    folder = ROOT / "data/REHAB26-ViewAngles/processed_fusion_v3"
    rows = json.loads((folder / "observations.json").read_text(encoding="utf-8"))
    with np.load(folder / "samples.npz", allow_pickle=False) as arrays:
        features = arrays["x_equal"].copy()
    # Keep one participant throughout, selected by overall joint visibility.
    people = sorted({r["person"] for r in rows})
    visibility = {p: float(np.median(features[[i for i, r in enumerate(rows) if r["person"] == p], :, 69:])) for p in people}
    person = max(people, key=lambda p: (visibility[p], p))
    output = {"source": "REHAB26-ViewAngles", "person": person, "frames": 96, "duration": 5.0,
              "processing": "three-view equal fusion; fixed hip yaw; torso scale; display-only Gaussian smoothing sigma=1.2 frames",
              "selection": "one participant with greatest median visibility; within each single-label class, closest to median 12D geometry trajectory; no classifier outputs",
              "exercises": {}}
    manifest = []
    for ex, settings in EXERCISES.items():
        result = {**settings, "clips": {}}
        for code in ["correct"] + settings["errors"]:
            ids = [i for i, r in enumerate(rows) if r["person"] == person and r["exercise_id"] == ex
                   and r["error_labels"] == ([] if code == "correct" else [code])]
            assert ids, (person, ex, code)
            trajectories = features[ids, :, 57:69]
            median = np.median(trajectories, axis=0)
            deviation = np.maximum(np.std(trajectories, axis=(0, 1)), .03)
            distance = np.mean(((trajectories - median) / deviation)**2, axis=(1, 2))
            i = ids[int(np.argmin(distance))]
            row = rows[i]
            segments = [np.load(ROOT / "data/REHAB26-ViewAngles/raw" / v["skeleton_file"], mmap_mode="r")
                        [row["first_frame"]:row["last_frame"]+1] for v in sorted(row["source_views"], key=lambda v:v["camera"])]
            aligned = [align_view(s) for s in segments]
            xyz, vis, _ = merge_aligned(np.stack([v[0] for v in aligned]), np.stack([v[1] for v in aligned]), "equal")
            xyz = gaussian_filter1d(xyz, sigma=1.2, axis=0, mode="nearest")
            # Place feet on a common floor, retaining every relative joint position.
            floor = np.median(xyz[:, [27,28,29,30,31,32], 1])
            xyz[:, :, 1] -= floor
            title, hint, joints = ("标准动作", settings["hint"], []) if code == "correct" else LABELS[code]
            clip = {"id": row["sample_id"], "label": code, "name": title, "hint": hint, "highlight": joints,
                    "points": np.round(xyz, 5).tolist()}
            result["clips"][code] = clip
            manifest.append({"sample_id": row["sample_id"], "person": person, "exercise_id": ex,
                             "error_labels": row["error_labels"], "source_views": row["source_views"],
                             "first_frame": row["first_frame"], "last_frame": row["last_frame"]})
        output["exercises"][str(ex)] = result
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "motion_data.json").write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (HERE / "source_manifest.json").write_text(json.dumps({"selection": output["selection"], "visibility_by_person": visibility, "clips": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(person, len(manifest), (HERE / "motion_data.json").stat().st_size)


if __name__ == "__main__":
    main()
