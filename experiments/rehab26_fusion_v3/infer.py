"""Replay a three-camera repetition through its held-out-person ensemble."""
from __future__ import annotations

import argparse
import json
import numpy as np

from run import ROOT, RUN, DATA, core, train, load_json
from fusion import fuse_segments


def raw_features(row):
    views = sorted(row["source_views"], key=lambda v: v["camera"])
    assert [v["camera"] for v in views] == [1, 2, 3]
    segments = [np.load(ROOT / "data/REHAB26-ViewAngles/raw" / v["skeleton_file"], mmap_mode="r", allow_pickle=False)
                [row["first_frame"]:row["last_frame"] + 1] for v in views]
    return fuse_segments(segments)["x"]


def predict_repetition(row, seed=20261109):
    selection = load_json(RUN / "selection_final.json")
    ex, person = row["exercise_id"], row["person"]
    candidate = selection["selection"][str(ex)][person]["transformer"]
    features = raw_features(row)
    scores, thresholds = [], []
    for folder in sorted((RUN / f"fits/ex{ex}/test_{person}").glob(f"val_*/{candidate}/seed_{seed}")):
        artifact = train.load_artifact(folder)
        assert person not in artifact["split"]["train"] + artifact["split"]["validation"]
        scores.append(train.predict_artifact(artifact, features[None])[0])
        thresholds.append(artifact["thresholds"])
        del artifact
        core.torch.cuda.empty_cache()
    assert len(scores) == 4
    return features, np.mean(scores, 0), np.mean(thresholds, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    core.configure_torch()
    observations = load_json(DATA / "observations.json")
    row = next(r for r in observations if r["sample_id"] == args.sample_id)
    _, scores, thresholds = predict_repetition(row, args.seed)
    labels = load_json(DATA / "dataset.json")["labels"][str(row["exercise_id"])]
    print(json.dumps({"sample_id": args.sample_id, "held_out_person": row["person"], "seed": args.seed,
        "predictions": [{"label": label, "probability": float(score), "threshold": float(threshold), "alert": bool(score >= threshold)}
                        for label, score, threshold in zip(labels, scores, thresholds)]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
