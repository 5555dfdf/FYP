"""Meaningful numerical, split, preprocessing and artifact checks; never selects a model."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from core import (HERE, ROOT, RUN, TemporalTransformer, configure_torch, feature_indices,
                  fit_scaler, groups, metrics, read_data, save_json, select_thresholds,
                  sha256, source_identity, transform)
from train import load_artifact, predict_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-artifacts", action="store_true")
    args = parser.parse_args()
    configure_torch()
    x, noisy, y, observations, metadata = read_data()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    assert source_identity() == json.loads((RUN / "identity.json").read_text())
    people = sorted({r["person"] for r in observations})
    checked_splits = 0
    for exercise in [1, 7]:
        for test in people:
            for val in people:
                if val == test:
                    continue
                ids, split = groups(observations, exercise, test, val)
                assert (y[ids["train"]].sum(0) > 0).all()
                for name, rows in ids.items():
                    sample_ids = {observations[i]["sample_id"] for i in rows}
                    assert len(rows) == 3*len(sample_ids), "All three cameras must stay together"
                    assert all(observations[i]["person"] in split[name] for i in rows)
                checked_splits += 1
    ids, _ = groups(observations, 1, people[0], people[1])
    config = protocol["candidates"]["tr_ref"]
    mean, std = fit_scaler(x, ids["train"], config)
    perturbed = x.copy()
    perturbed[np.concatenate([ids["test"], ids["validation"]]), :, :69] += 1e6
    mean2, std2 = fit_scaler(perturbed, ids["train"], config)
    assert np.array_equal(mean, mean2) and np.array_equal(std, std2), "Scaler leaks held-out rows"
    standardized = transform(x[ids["train"]], config, mean, std)
    assert np.allclose(standardized.mean((0, 1)), 0, atol=1e-4)
    vconfig = protocol["candidates"]["tr_visibility"]
    vm, vs = fit_scaler(x, ids["train"], vconfig)
    assert np.array_equal(transform(x[:3], vconfig, vm, vs)[..., 69:], x[:3, :, 69:])
    fixture_y = np.array([[0,0,0,0,0], [1,1,0,0,0], [1,0,1,0,0], [0,0,0,1,1]], dtype=bool)
    fixture_scores = np.array([[.8,.1,.1,.1,.1],[.8,.2,.1,.1,.1],[.1,.1,.8,.1,.1],[.1,.1,.1,.8,.8]])
    measured = metrics(fixture_y, fixture_scores)
    assert measured["false_alert_rate"] == 1.
    assert measured["per_error"][0]["fp"] == 1 and measured["per_error"][1]["fn"] == 1
    assert np.isclose(measured["macro_f1"], f1_score(fixture_y, fixture_scores>=.5, average="macro", zero_division=0))
    t = select_thresholds(fixture_y, fixture_scores)
    assert t.shape == (5,) and (t >= .05).all() and (t <= .95).all()
    for j in range(5):
        expected = max(f1_score(fixture_y[:, j], fixture_scores[:, j]>=k/20, zero_division=0) for k in range(1,20))
        assert np.isclose(f1_score(fixture_y[:, j], fixture_scores[:, j]>=t[j], zero_division=0), expected)
    # Extreme logits must remain finite, and an all-zero target must produce a gradient.
    logits = torch.tensor([[100.,-100.,0.,4.,-4.]], requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    model = TemporalTransformer(config)
    assert sum(p.numel() for p in model.parameters()) == 75845
    assert not torch.equal(model.blocks[0].linear1.weight, model.blocks[1].linear1.weight)
    assert not torch.equal(model.blocks[0].self_attn.in_proj_weight, model.blocks[1].self_attn.in_proj_weight)
    model.eval()
    xx = torch.from_numpy(standardized[:2])
    with torch.no_grad():
        assert torch.equal(model(xx), model(xx)), "eval dropout must be disabled"
    # Verify raw-to-feature regeneration on validation data, not outer tests during development.
    import sys
    sys.path.insert(0, str(ROOT / "experiments/rehab26_v1"))
    from pipeline import make_features, normalize_resample
    raw_checks = 0
    for i in ids["validation"][[0, len(ids["validation"])//2, -1]]:
        row = observations[int(i)]
        raw = np.load(ROOT / "data/REHAB26-ViewAngles/raw" / row["skeleton_file"], mmap_mode="r")
        segment = raw[row["first_frame"]:row["last_frame"]+1]
        xyz, visibility, _ = normalize_resample(segment)
        assert np.allclose(make_features(xyz, visibility), x[i], atol=2e-6)
        raw_checks += 1
    paths = sorted((RUN / "fits").rglob("result.json")) if args.all_artifacts else [RUN / "smoke/result.json"]
    largest, artifacts = 0., 0
    for result_path in paths:
        folder = result_path.parent
        artifact = load_artifact(folder)
        with np.load(folder / "validation.npz", allow_pickle=False) as cached:
            positions = np.array([0, len(cached["ids"])//2, len(cached["ids"])-1])
            recomputed = predict_artifact(artifact, x[cached["ids"][positions]])
            expected = cached["scores"][positions]
            difference = float(np.abs(recomputed-expected).max())
            assert np.allclose(recomputed, expected, atol=3e-6), (folder, difference)
            assert np.array_equal(cached["y"], y[cached["ids"]])
            largest = max(largest, difference)
        if "model" in artifact:
            del artifact["model"]
        del artifact
        artifacts += 1
    report = {"status": "passed", "splits_checked": checked_splits, "artifacts_reloaded": artifacts,
              "max_reload_difference": largest, "raw_validation_clips_regenerated": raw_checks,
              "checks": ["source/data hashes", "labels match manifest", "subject/view/repetition isolation",
                         "held-out outliers cannot change training scaler", "raw visibility preserved",
                         "FP/FN and F1 vs sklearn", "threshold search optimum", "finite extreme-logit BCE",
                         "all-zero-label gradient", "independent encoder initialization", "eval deterministic dropout",
                         "raw-to-cache feature agreement", "saved validation scores reproduced"]}
    save_json(RUN / ("verification_all.json" if args.all_artifacts else "verification_initial.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
