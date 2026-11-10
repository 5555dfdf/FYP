"""Freeze, screen using validation only, confirm selections, then reveal outer tests."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import sklearn
import torch

from core import (HERE, RUN, configure_torch, digest, groups, metrics, read_data,
                  save_json, source_identity)
from train import fit_one, load_artifact, predict_artifact


def candidates():
    reference = {"family": "transformer", "features": "angles", "activation": "gelu", "pre_norm": True,
                 "dropout": .1, "weight_decay": .001, "standardize": True, "loss": "none", "lr": .0003}
    variants = {"tr_ref": {}, "tr_xyz": {"features": "xyz"}, "tr_sqrt": {"loss": "sqrt"},
                "tr_raw": {"loss": "raw"}, "tr_relu": {"activation": "relu"},
                "tr_postnorm": {"pre_norm": False}, "tr_noz": {"standardize": False},
                "tr_nodrop": {"dropout": 0.}, "tr_nodecay": {"weight_decay": 0.},
                "tr_visibility": {"features": "visibility"},
                "tr_lr1e3": {"lr": .001}, "tr_small": {"width": 32, "layers": 1}}
    result = {name: {**reference, **overrides} for name, overrides in variants.items()}
    result.update({"tcn_xyz": {**reference, "family": "tcn", "features": "xyz"},
                   "tcn_angles": {**reference, "family": "tcn"},
                   "angle_lr": {"family": "lr", "features": "angles"},
                   "angle_rf": {"family": "rf", "features": "angles"}})
    return result


def initialize():
    protocol_path = HERE / "protocol.json"
    if not protocol_path.exists():
        save_json(protocol_path, {"version": "rehab26_transformer_v2_20261109", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "training": {"max_epochs": 80, "batch_size": 32, "patience": 12, "min_delta": .0001},
            "candidates": candidates(), "exercises": [1, 7], "seeds": [20261109, 20261110, 20261111],
            "selection": "per-exercise/per-outer-person rank candidates by mean four-inner-validation macro-AP; tie mean validation F1, then lexicographic name; no outer predictions read until selections frozen",
            "screen": "all candidates, first seed, all 20 ordered test/validation splits per exercise",
            "confirm": "for each outer fold: its selected Transformer, selected TCN, and both angle baselines; 3 seeds for neural/RF, deterministic LR once; selected configurations fixed before extra seeds",
            "secondary": "F0.5 validation thresholds reported as precision-oriented operating point, not selected using tests",
            "test_ensemble": "four inner models within each seed; mean(p-threshold)>=0; mean probabilities for AP; no seed ensembling in main metrics",
            "scope": "exploratory; all five volunteers appeared in earlier inspected tests; simulated errors; known action and repetition boundaries",
            "tuning": "prespecified one-factor and lr/capacity candidates; optional second-round combinations require separate timestamped validation-only amendment"})
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    identity = source_identity()
    identity_path = RUN / "identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, "Code/data/protocol changed; use a new run version"
    else:
        save_json(identity_path, identity)
        save_json(RUN / "protocol.json", protocol)
        save_json(RUN / "environment.json", {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "executable": sys.executable, "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "conda_env": "fyp", "deterministic": True, "tf32": False, "amp": False})
        (RUN / "requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    return protocol, identity


def folder(ex, test, val, name, seed):
    return RUN / f"fits/ex{ex}/test_{test}/val_{val}/{name}/seed_{seed}"


def fit_jobs(protocol, identity, x, y, observations, jobs):
    for number, (ex, test, val, name, seed) in enumerate(jobs, 1):
        start = time.perf_counter()
        destination = folder(ex, test, val, name, seed)
        existed = (destination / "result.json").exists()
        if not existed:
            print(f"START [{number}/{len(jobs)}] ex{ex} test={test} val={val} {name} seed={seed}", flush=True)
        result = fit_one(x, y, observations, ex, test, val, protocol["candidates"][name],
                         protocol["training"], seed, destination, identity)
        if not existed:
            print(f"DONE AP={result['validation']['macro_ap']:.4f} F1={result['validation']['macro_f1']:.4f} "
                  f"epoch={result['best_epoch']} sec={time.perf_counter()-start:.1f}", flush=True)
        save_json(RUN / "progress.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "job": number,
                  "jobs_in_current_stage": len(jobs), "last_fit": str(destination.relative_to(RUN)),
                  "last_validation": result["validation"], "outer_test_revealed": (RUN / "test_summary.json").exists()})


def select(protocol, subjects):
    selection = {}
    for ex in protocol["exercises"]:
        selection[str(ex)] = {}
        for test in subjects:
            ranking = []
            for name, config in protocol["candidates"].items():
                results = [json.loads((folder(ex, test, val, name, protocol["seeds"][0]) / "result.json").read_text(encoding="utf-8")) for val in subjects if val != test]
                ranking.append({"name": name, "family": config["family"],
                                "validation_macro_ap": float(np.mean([r["validation"]["macro_ap"] for r in results])),
                                "validation_macro_f1": float(np.mean([r["validation"]["macro_f1"] for r in results])),
                                "validation_false_alert_rate": float(np.mean([r["validation"]["false_alert_rate"] for r in results]))})
            ranking.sort(key=lambda r: (-r["validation_macro_ap"], -r["validation_macro_f1"], r["name"]))
            selection[str(ex)][test] = {"transformer": next(r["name"] for r in ranking if r["family"] == "transformer"),
                                       "tcn": next(r["name"] for r in ranking if r["family"] == "tcn"),
                                       "ranking": ranking}
    output = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
              "rule": protocol["selection"], "test_predictions_available_at_selection": False}
    path = RUN / "selection.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        assert old["selection"] == selection
        return old
    save_json(path, output)
    return output


def evaluate(protocol, selection, x, noisy, y, observations, subjects):
    # Only this function accesses test features. No scores are exposed during screening.
    rows = []
    for ex in protocol["exercises"]:
        for test in subjects:
            choices = selection["selection"][str(ex)][test]
            methods = {"selected_transformer": choices["transformer"], "selected_tcn": choices["tcn"],
                       "angle_lr": "angle_lr", "angle_rf": "angle_rf", "reference_transformer": "tr_ref"}
            for method, name in methods.items():
                seeds = protocol["seeds"][:1] if method in ("angle_lr", "reference_transformer") else protocol["seeds"]
                for seed in seeds:
                    scores_all, noisy_all, thresholds_all, precision_all = [], [], [], []
                    single = []
                    for val in subjects:
                        if val == test:
                            continue
                        artifact = load_artifact(folder(ex, test, val, name, seed))
                        ids, _ = groups(observations, ex, test, val)
                        ids = ids["test"]
                        scores = predict_artifact(artifact, x[ids])
                        nscores = predict_artifact(artifact, noisy[ids])
                        scores_all.append(scores)
                        noisy_all.append(nscores)
                        thresholds_all.append(artifact["thresholds"])
                        precision_all.append(artifact["precision_thresholds"])
                        single.append({"validation_person": val, "test": metrics(y[ids], scores, artifact["thresholds"])})
                        if "model" in artifact:
                            del artifact["model"]
                        del artifact
                        torch.cuda.empty_cache()
                    stack = np.stack(scores_all)
                    scores, noisy_scores = stack.mean(0), np.stack(noisy_all).mean(0)
                    thresholds, pthresholds = np.mean(thresholds_all, 0), np.mean(precision_all, 0)
                    result = {"exercise": ex, "test_person": test, "method": method, "candidate": name, "seed": seed,
                              "test": metrics(y[ids], scores, thresholds),
                              "test_precision": metrics(y[ids], scores, pthresholds),
                              "test_fixed05": metrics(y[ids], scores),
                              "test_synthetic_noise": metrics(y[ids], noisy_scores, thresholds),
                              "single_inner_models": single}
                    target = RUN / f"outer_predictions/ex{ex}/{test}/{method}/seed_{seed}"
                    target.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target / "predictions.npz", ids=ids, y=y[ids], scores=scores,
                                        noisy_scores=noisy_scores, thresholds=thresholds,
                                        precision_thresholds=pthresholds, inner_scores=stack,
                                        inner_thresholds=np.asarray(thresholds_all))
                    save_json(target / "result.json", result)
                    rows.append(result)
                    print(f"TEST ex{ex} {test} {method} seed={seed} F1={result['test']['macro_f1']:.4f} FA={result['test']['false_alert_rate']:.3f}", flush=True)
    save_json(RUN / "test_summary.json", {"selection_sha256": digest(selection), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "screen", "confirm", "evaluate", "all"])
    args = parser.parse_args()
    configure_torch()
    protocol, identity = initialize()
    x, noisy, y, observations, _ = read_data()
    subjects = sorted({r["person"] for r in observations})
    if args.stage == "smoke":
        smoke_training = {**protocol["training"], "max_epochs": 8, "patience": 12}
        result = fit_one(x, y, observations, 1, subjects[0], subjects[1], protocol["candidates"]["tr_ref"], smoke_training,
                         protocol["seeds"][0], RUN / "smoke", identity)
        artifact = load_artifact(RUN / "smoke")
        cached = np.load(RUN / "smoke/validation.npz")
        recalculated = predict_artifact(artifact, x[cached["ids"]])
        assert np.allclose(cached["scores"], recalculated, atol=2e-6)
        save_json(RUN / "smoke_check.json", {"reloaded_max_difference": float(np.abs(cached["scores"]-recalculated).max()),
                  "best_epoch": result["best_epoch"], "parameters": result["parameters"], "seconds": result["elapsed_seconds"],
                  "status": "passed"})
        print(json.dumps(json.loads((RUN / "smoke_check.json").read_text()), indent=2))
        return
    if args.stage in ("screen", "all"):
        jobs = [(ex, test, val, name, protocol["seeds"][0]) for ex in protocol["exercises"]
                for test in subjects for val in subjects if val != test for name in protocol["candidates"]]
        fit_jobs(protocol, identity, x, y, observations, jobs)
    selection = select(protocol, subjects)
    if args.stage in ("confirm", "all"):
        jobs = []
        for ex in protocol["exercises"]:
            for test in subjects:
                chosen = selection["selection"][str(ex)][test]
                for val in subjects:
                    if val != test:
                        for name in (chosen["transformer"], chosen["tcn"], "angle_rf"):
                            for seed in protocol["seeds"][1:]:
                                jobs.append((ex, test, val, name, seed))
        fit_jobs(protocol, identity, x, y, observations, jobs)
    if args.stage in ("evaluate", "all"):
        evaluate(protocol, selection, x, noisy, y, observations, subjects)


if __name__ == "__main__":
    main()
