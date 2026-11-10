"""Repeat the v2 validation search on fused skeletons; preserve every v2 file."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHARED = ROOT / "experiments/rehab26_transformer_v2"
sys.path.insert(0, str(SHARED))
import core
import suite
import train

RUN = ROOT / "runs/rehab26_fusion_v3"
DATA = ROOT / "data/REHAB26-ViewAngles/processed_fusion_v3"
suite.RUN = RUN
_cache = None


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_data():
    metadata = load_json(DATA / "dataset.json")
    assert core.sha256(DATA / "samples.npz") == metadata["samples_sha256"]
    assert core.sha256(DATA / "observations.json") == metadata["observations_sha256"]
    assert core.sha256(HERE / "fusion.py") == metadata["fusion_sha256"]
    assert core.sha256(HERE / "prepare.py") == metadata["prepare_sha256"]
    observations = load_json(DATA / "observations.json")
    with np.load(DATA / "samples.npz", allow_pickle=False) as handle:
        arrays = {k: handle[k].copy() for k in handle.files}
    assert arrays["x"].shape == (1111, 96, 88)
    assert len({r["sample_id"] for r in observations}) == 1111
    return arrays, observations, metadata


def initialize():
    base = load_json(SHARED / "protocol.json")
    protocol_path = HERE / "protocol.json"
    if not protocol_path.exists():
        protocol = {**base, "version": "rehab26_fusion_v3_20261110", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": "three-view visibility-squared fused 3D skeleton; one sample per physical repetition",
            "scope": "exploratory, five previously inspected volunteers; approximate uncalibrated fusion; known action boundaries",
            "iteration_rule": "same v2 rule: combine one-factor improvements >=0.015 validation AP, per outer fold, structure/training/combined, remove duplicates",
            "controls": "tr_ref first seed, all 40 ordered outer/inner exercise splits, on equal-mean fusion and aligned camera2; fixed before test reveal",
            "comparison": "old single-view pooled and per-camera results plus old three-view probability averaging at physical-repetition level; no claim of equal camera budget",
            "test_access": "all main choices and control fits complete before any new outer-test predictions; no subsequent test-driven changes"}
        core.save_json(protocol_path, protocol)
    protocol = load_json(protocol_path)
    files = [HERE / name for name in ["fusion.py", "prepare.py", "run.py"]]
    files += [SHARED / name for name in ["core.py", "train.py", "suite.py"]]
    files += [ROOT / "experiments/rehab26_v1/pipeline.py", protocol_path, DATA / "dataset.json", DATA / "observations.json", DATA / "samples.npz"]
    identity = {str(p.relative_to(ROOT)): core.sha256(p) for p in files}
    if (RUN / "identity.json").exists():
        assert load_json(RUN / "identity.json") == identity, "Changed code/data/protocol; use new run version"
    else:
        core.save_json(RUN / "identity.json", identity)
        core.save_json(RUN / "protocol.json", protocol)
        core.save_json(RUN / "environment.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "torch": core.torch.__version__,
            "numpy": np.__version__, "gpu": core.torch.cuda.get_device_name(0), "conda_env": "fyp"})
        (RUN / "requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    return protocol, identity


def worker_init():
    global _cache
    core.configure_torch()
    _cache = read_data()


def worker_fit(job):
    ex, test, val, name, seed, config, training, identity, data_key = job
    arrays, observations, _ = _cache
    destination = suite.folder(ex, test, val, name, seed) if data_key == "x" else RUN / f"controls/{data_key}/ex{ex}/test_{test}/val_{val}/seed_{seed}"
    start = time.perf_counter()
    result = train.fit_one(arrays[data_key], arrays["y"].astype(np.float32), observations, ex, test, val, config, training, seed, destination, identity)
    return {"exercise": ex, "test": test, "validation_person": val, "candidate": name, "input": data_key,
            "seed": seed, "ap": result["validation"]["macro_ap"], "f1": result["validation"]["macro_f1"],
            "epoch": result["best_epoch"], "seconds": time.perf_counter() - start}


def fit_jobs(protocol, identity, jobs, stage, workers=2):
    pending = []
    for ex, test, val, name, seed, data_key in jobs:
        destination = suite.folder(ex, test, val, name, seed) if data_key == "x" else RUN / f"controls/{data_key}/ex{ex}/test_{test}/val_{val}/seed_{seed}"
        config = protocol["candidates"][name]
        if (destination / "result.json").exists():
            result = load_json(destination / "result.json")
            expected = {"identity": identity, "config": config, "training": protocol["training"], "seed": seed, "exercise": ex, "test": test, "validation": val}
            assert result["job_hash"] == core.digest(expected)
        else:
            pending.append((ex, test, val, name, seed, config, protocol["training"], identity, data_key))
    print(f"STAGE {stage}: {len(pending)} pending / {len(jobs)} planned; workers={workers}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = [pool.submit(worker_fit, job) for job in pending]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            print(f"FIT {stage} [{i}/{len(pending)}] {json.dumps(result)}", flush=True)
            core.save_json(RUN / "progress.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
                "completed_pending": i, "pending_total": len(pending), "already_completed": len(jobs) - len(pending), "last": result,
                "outer_test_revealed": (RUN / "test_summary.json").exists()})


def iteration_plan(base, selection, subjects):
    path = RUN / "iteration_plan.json"
    if path.exists():
        return load_json(path)
    assert not (RUN / "test_summary.json").exists()
    plan = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "rule": base["iteration_rule"], "candidates": {}, "allowed": {}}
    factors = {
        "structure": [("tr_relu", ["activation"]), ("tr_postnorm", ["pre_norm"]), ("tr_noz", ["standardize"]),
                      ("tr_visibility", ["features"]), ("tr_small", ["width", "layers"])],
        "training": [("tr_sqrt", ["loss"]), ("tr_raw", ["loss"]), ("tr_nodrop", ["dropout"]),
                     ("tr_nodecay", ["weight_decay"]), ("tr_lr1e3", ["lr"])]}
    for ex in base["exercises"]:
        plan["allowed"][str(ex)] = {}
        for test in subjects:
            ap = {r["name"]: r["validation_macro_ap"] for r in selection["selection"][str(ex)][test]["ranking"]}
            reference = base["candidates"]["tr_ref"]
            configs = {}
            for group, changes in factors.items():
                config = copy.deepcopy(reference)
                for name, keys in sorted(changes, key=lambda item: ap[item[0]]):
                    if ap[name] >= ap["tr_ref"] + .015:
                        for key in keys:
                            config[key] = base["candidates"][name].get(key, reference.get(key))
                configs[group] = config
            combined = copy.deepcopy(reference)
            for config in configs.values():
                combined.update({k: v for k, v in config.items() if v != reference.get(k)})
            configs["combined"] = combined
            seen = {core.digest(c) for c in base["candidates"].values()}
            allowed = []
            for group, config in configs.items():
                if core.digest(config) not in seen:
                    name = f"iter_ex{ex}_{test}_{group}"
                    allowed.append(name)
                    plan["candidates"][name] = config
                    seen.add(core.digest(config))
            plan["allowed"][str(ex)][test] = allowed
    core.save_json(path, plan)
    return plan


def final_selection(base, selection, plan, subjects):
    result = copy.deepcopy(selection)
    for ex in base["exercises"]:
        for test in subjects:
            item = result["selection"][str(ex)][test]
            for name in plan["allowed"][str(ex)][test]:
                rows = [load_json(suite.folder(ex, test, val, name, base["seeds"][0]) / "result.json") for val in subjects if val != test]
                item["ranking"].append({"name": name, "family": "transformer",
                    "validation_macro_ap": float(np.mean([r["validation"]["macro_ap"] for r in rows])),
                    "validation_macro_f1": float(np.mean([r["validation"]["macro_f1"] for r in rows])),
                    "validation_false_alert_rate": float(np.mean([r["validation"]["false_alert_rate"] for r in rows]))})
            item["ranking"].sort(key=lambda r: (-r["validation_macro_ap"], -r["validation_macro_f1"], r["name"]))
            item["transformer"] = next(r["name"] for r in item["ranking"] if r["family"] == "transformer")
    result.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(), amendment_sha256=core.digest(plan), test_predictions_available_at_selection=False)
    path = RUN / "selection_final.json"
    if path.exists():
        assert load_json(path)["selection"] == result["selection"]
        return load_json(path)
    core.save_json(path, result)
    return result


def evaluate_controls(protocol, arrays, observations, subjects):
    rows = []
    for key in ["x_equal", "x_cam2"]:
        for ex in protocol["exercises"]:
            for test in subjects:
                scores, thresholds = [], []
                for val in subjects:
                    if val == test:
                        continue
                    destination = RUN / f"controls/{key}/ex{ex}/test_{test}/val_{val}/seed_{protocol['seeds'][0]}"
                    artifact = train.load_artifact(destination)
                    ids = core.groups(observations, ex, test, val)[0]["test"]
                    scores.append(train.predict_artifact(artifact, arrays[key][ids]))
                    thresholds.append(artifact["thresholds"])
                    del artifact
                    core.torch.cuda.empty_cache()
                scores, thresholds = np.mean(scores, 0), np.mean(thresholds, 0)
                folder = RUN / f"control_predictions/{key}/ex{ex}/{test}"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "predictions.npz", ids=ids, y=arrays["y"][ids], scores=scores, thresholds=thresholds)
                rows.append({"input": key, "exercise": ex, "test_person": test, "test": core.metrics(arrays["y"][ids], scores, thresholds)})
    core.save_json(RUN / "control_summary.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "screen", "all"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    core.configure_torch()
    base, identity = initialize()
    arrays, observations, metadata = read_data()
    x, y = arrays["x"], arrays["y"].astype(np.float32)
    subjects = sorted({r["person"] for r in observations})
    if args.stage == "smoke":
        result = train.fit_one(x, y, observations, 1, subjects[0], subjects[1], base["candidates"]["tr_ref"],
            {**base["training"], "max_epochs": 8}, base["seeds"][0], RUN / "smoke", identity)
        artifact = train.load_artifact(RUN / "smoke")
        with np.load(RUN / "smoke/validation.npz") as cached:
            delta = float(np.abs(train.predict_artifact(artifact, x[cached["ids"]]) - cached["scores"]).max())
        assert delta < 2e-6
        core.save_json(RUN / "smoke_check.json", {"status": "passed", "reload_max_difference": delta, "parameters": result["parameters"]})
        print(result["validation"], flush=True)
        return
    jobs = [(ex, test, val, name, base["seeds"][0], "x") for ex in base["exercises"] for test in subjects for val in subjects if val != test for name in base["candidates"]]
    fit_jobs(base, identity, jobs, "screen", args.workers)
    selection = suite.select(base, subjects)
    if args.stage == "screen":
        return
    plan = iteration_plan(base, selection, subjects)
    protocol = copy.deepcopy(base)
    protocol["candidates"].update(plan["candidates"])
    jobs = [(ex, test, val, name, base["seeds"][0], "x") for ex in base["exercises"] for test in subjects for name in plan["allowed"][str(ex)][test] for val in subjects if val != test]
    fit_jobs(protocol, identity, jobs, "combinations", args.workers)
    selection = final_selection(base, selection, plan, subjects)
    core.save_json(RUN / "protocol.final.json", protocol)
    jobs = [(ex, test, val, name, seed, "x") for ex in base["exercises"] for test in subjects for val in subjects if val != test
            for name in [selection["selection"][str(ex)][test]["transformer"], selection["selection"][str(ex)][test]["tcn"], "angle_rf"] for seed in base["seeds"][1:]]
    fit_jobs(protocol, identity, jobs, "confirm", args.workers)
    jobs = [(ex, test, val, "tr_ref", base["seeds"][0], key) for key in ["x_equal", "x_cam2"] for ex in base["exercises"] for test in subjects for val in subjects if val != test]
    fit_jobs(protocol, identity, jobs, "controls", args.workers)
    core.save_json(RUN / "test_release_gate.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": core.sha256(RUN / "selection_final.json"), "fits_complete": len(list((RUN / "fits").rglob("result.json"))),
        "control_fits_complete": len(list((RUN / "controls").rglob("result.json")))})
    suite.evaluate(protocol, selection, x, arrays["x_noisy"], y, observations, subjects)
    evaluate_controls(protocol, arrays, observations, subjects)
    print("TRAINING AND OUTER EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
