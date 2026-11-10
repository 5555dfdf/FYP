"""Train compact full-data research models using validation-derived fixed choices.

These models see all five people. Outer-test performance belongs to the nested
models, never to these full-data fits. This policy is written before test reveal.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from core import HERE, RUN, configure_torch, digest, fit_scaler, read_data, save_json, source_identity, transform
from train import fit_neural


def main():
    configure_torch()
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    assert source_identity() == json.loads((RUN / "identity.json").read_text())
    x, _, y, observations, metadata = read_data()
    release = RUN / "deployment"
    decisions = {}
    for exercise in protocol["exercises"]:
        ranking = []
        for name, config in protocol["candidates"].items():
            if config["family"] != "transformer":
                continue
            files = list((RUN / f"fits/ex{exercise}").glob(f"test_*/val_*/{name}/seed_{protocol['seeds'][0]}/result.json"))
            assert len(files) == 20
            results = [json.loads(p.read_text(encoding="utf-8")) for p in files]
            ranking.append((float(np.mean([r["validation"]["macro_ap"] for r in results])), name, files, results))
        ranking.sort(key=lambda r: (-r[0], r[1]))
        ap, name, files, results = ranking[0]
        epochs = max(1, int(np.median([r["best_epoch"] for r in results])))
        thresholds, precision = [], []
        for p in files:
            with np.load(p.parent / "validation.npz", allow_pickle=False) as cached:
                thresholds.append(cached["thresholds"])
                precision.append(cached["precision_thresholds"])
        decisions[str(exercise)] = {"candidate": name, "fixed_epochs": epochs, "validation_macro_ap": ap,
                                    "thresholds": np.median(thresholds, axis=0).tolist(),
                                    "precision_thresholds": np.median(precision, axis=0).tolist(),
                                    "selection_evidence": "all 20 inner validation results at screening seed, no test scores used"}
    policy = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "decisions": decisions,
              "status": "full_data_research_models_not_independently_evaluated",
              "limitations": "Trained on all five volunteers; outer test results do not measure these weights. Validation-derived thresholds are provisional after full-data refit; no clinical reliability or calibration established."}
    if (release / "selection.json").exists():
        assert json.loads((release / "selection.json").read_text(encoding="utf-8"))["decisions"] == decisions
    else:
        save_json(release / "selection.json", policy)
    for exercise in protocol["exercises"]:
        decision = decisions[str(exercise)]
        config = protocol["candidates"][decision["candidate"]]
        ids = np.asarray([i for i,r in enumerate(observations) if r["exercise_id"] == exercise])
        mean, std = fit_scaler(x, ids, config)
        xx, yy = transform(x[ids], config, mean, std), y[ids]
        members = []
        for seed in protocol["seeds"]:
            target = release / f"ex{exercise}/seed_{seed}"
            members.append(os.path.relpath(target, release / f"ex{exercise}"))
            if (target / "model.pt").exists():
                continue
            print(f"Full-data refit ex{exercise} {decision['candidate']} seed={seed} epochs={decision['fixed_epochs']}", flush=True)
            # Training-data diagnostic only. Fixed epoch count has already been
            # selected from earlier held-out validation; no early stopping here.
            model, diagnostics = fit_neural(xx, yy, xx, yy, config, protocol["training"], seed,
                                            fixed_epochs=decision["fixed_epochs"])
            target.mkdir(parents=True, exist_ok=True)
            artifact = {"identity": source_identity(), "config": config,
                        "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "mean": mean.tolist(), "std": std.tolist(), "seed": seed, "exercise": exercise,
                        "thresholds": decision["thresholds"], "precision_thresholds": decision["precision_thresholds"],
                        "split": {"train": sorted({r["person"] for r in observations}), "validation": [], "test": []},
                        "best_epoch": decision["fixed_epochs"], "status": policy["status"]}
            torch.save(artifact, target / "model.pt")
            # Rename fields so full-data diagnostics cannot be mistaken for validation.
            for row in diagnostics["history"]:
                row["training_diagnostic_bce"] = row.pop("validation_bce")
                row["training_diagnostic_macro_ap"] = row.pop("validation_macro_ap")
            save_json(target / "training.json", {"selection": decision, "diagnostics": diagnostics,
                      "note": "All curves here use training data only; fixed epochs, no validation/early stopping"})
            del model
            torch.cuda.empty_cache()
        save_json(release / f"ex{exercise}/bundle.json", {"exercise": exercise, "members": members,
                  "candidate": decision["candidate"], "status": policy["status"], "limitations": policy["limitations"]})
    print("Full-data research bundles saved; not an independent evaluation.", flush=True)


if __name__ == "__main__":
    main()
