"""Check every finished checkpoint and raw-to-held-out-ensemble replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import numpy as np

# Import the local inference module before run.py adds the shared v2 directory;
# v2 also contains a different infer.py.
from infer import predict_repetition
from run import RUN, core, train, initialize, read_data, load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-only", action="store_true", help="Check finished validation checkpoints while training continues; no outer-test inference")
    args = parser.parse_args()
    core.configure_torch()
    _, identity = initialize()
    arrays, observations, _ = read_data()
    if not args.checkpoints_only:
        gate = load_json(RUN / "test_release_gate.json")
        assert (RUN / "control_summary.json").exists()
    paths = sorted((RUN / "fits").rglob("result.json")) + sorted((RUN / "controls").rglob("result.json"))
    if not args.checkpoints_only:
        assert len(paths) == gate["fits_complete"] + gate["control_fits_complete"]
    cache_path = RUN / "checkpoint_verification_cache.json"
    cache = load_json(cache_path) if cache_path.exists() else {"identity": identity, "verified": {}}
    assert cache["identity"] == identity
    deltas, scaler_deltas = [], []
    for n, path in enumerate(paths, 1):
        model_path = path.parent / ("model.pt" if (path.parent / "model.pt").exists() else "model.joblib")
        hashes = {p.name: core.sha256(p) for p in [path, model_path, path.parent / "validation.npz"]}
        cache_key = str(path.relative_to(RUN))
        checked = cache["verified"].get(cache_key)
        if checked and checked["hashes"] == hashes:
            deltas.append(checked["score_delta"])
            if checked["scaler_delta"] is not None:
                scaler_deltas.append(checked["scaler_delta"])
            continue
        result = load_json(path)
        expected = {k: result[k] for k in ["identity", "config", "training", "seed", "exercise", "test"]}
        # Shared v2 result.json uses "validation" for metrics; the person is
        # retained in split, while the checkpoint/hash still use the person.
        expected["validation"] = result["split"]["validation"][0]
        assert result["identity"] == identity
        assert result["job_hash"] == core.digest(expected)
        relative = path.relative_to(RUN).parts
        key = relative[1] if relative[0] == "controls" else "x"
        ids, split = core.groups(observations, result["exercise"], result["test"], expected["validation"])
        assert result["split"] == split
        assert result["n_train"] == len(ids["train"]) and result["n_validation"] == len(ids["validation"])
        artifact = train.load_artifact(path.parent)
        for field, value in expected.items():
            assert artifact[field] == value
        with np.load(path.parent / "validation.npz", allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["ids"], ids["validation"])
            np.testing.assert_array_equal(cached["y"], arrays["y"][cached["ids"]])
            np.testing.assert_array_equal(cached["thresholds"], artifact["thresholds"])
            scores = train.predict_artifact(artifact, arrays[key][cached["ids"]])
            delta = float(np.abs(scores - cached["scores"]).max())
            assert delta < 2e-6, (str(path), delta)
            deltas.append(delta)
        scaler_delta = None
        if "model" in artifact:
            mean, std = core.fit_scaler(arrays[key], ids["train"], artifact["config"])
            difference = max(float(np.abs(mean-np.asarray(artifact["mean"])).max()), float(np.abs(std-np.asarray(artifact["std"])).max()))
            assert difference < 1e-7
            scaler_deltas.append(difference)
            scaler_delta = difference
        del artifact
        core.torch.cuda.empty_cache()
        cache["verified"][cache_key] = {"hashes": hashes, "score_delta": delta, "scaler_delta": scaler_delta}
        if n % 50 == 0:
            core.save_json(cache_path, cache)
        if n % 100 == 0:
            print(f"Verified {n}/{len(paths)} checkpoints; max score difference {max(deltas):.3g}", flush=True)
    core.save_json(cache_path, cache)
    if args.checkpoints_only:
        print(f"Validation-only verification complete: {len(paths)} checkpoints; no outer-test inference", flush=True)
        return
    replay = []
    for ex in [1, 7]:
        for person in sorted({r["person"] for r in observations}):
            i = next(i for i, r in enumerate(observations) if r["person"] == person and r["exercise_id"] == ex)
            row = observations[i]
            features, scores, thresholds = predict_repetition(row)
            feature_delta = float(np.abs(features - arrays["x"][i]).max())
            assert feature_delta < 2e-6
            with np.load(RUN / f"outer_predictions/ex{ex}/{person}/selected_transformer/seed_20261109/predictions.npz", allow_pickle=False) as cached:
                position = int(np.flatnonzero(cached["ids"] == i)[0])
                score_delta = float(np.abs(scores - cached["scores"][position]).max())
                assert score_delta < 2e-6
                np.testing.assert_allclose(thresholds, cached["thresholds"], atol=1e-12, rtol=0)
            replay.append({"sample_id": row["sample_id"], "feature_max_difference": feature_delta, "score_max_difference": score_delta})
    report = {"status": "passed", "at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_count": len(paths),
              "checkpoint_reload_max_score_difference": max(deltas), "training_only_scaler_max_difference": max(scaler_deltas),
              "checks": ["unchanged code/data/protocol hashes", "all checkpoint job hashes and artifacts", "subject-isolated split membership",
                         "validation labels/ids and thresholds", "all cached validation predictions reproduced", "training-only scaler recomputation",
                         "10 raw-three-view to held-out four-model ensemble replays"], "raw_replays": replay}
    core.save_json(RUN / "model_verification.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
