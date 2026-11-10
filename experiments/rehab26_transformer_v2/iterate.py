"""Second validation-only iteration, frozen as an amendment before seeing tests.

Each outer fold derives its combinations ONLY from that fold's four validation
people. This script then confirms seeds and evaluates with the frozen selections.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np

from core import HERE, RUN, configure_torch, digest, read_data, save_json, sha256
from suite import evaluate, fit_jobs, folder, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "confirm", "evaluate", "all"])
    args = parser.parse_args()
    configure_torch()
    base, identity = initialize()
    x, noisy, y, observations, _ = read_data()
    subjects = sorted({r["person"] for r in observations})
    base_selection = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    amendment_path = RUN / "iteration_plan.json"
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        assert amendment["script_sha256"] == sha256(HERE / "iterate.py")
    else:
        assert not (RUN / "test_summary.json").exists(), "Do not derive a new tuning plan after test reveal"
        amendment = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(HERE / "iterate.py"),
                     "rule": "per outer fold, merge one-factor changes beating reference validation AP by >=0.015, separated into structure/training/full; remove duplicates; all selection within this outer fold", "candidates": {}, "allowed": {}}
        factor_groups = {
            "structure": [("tr_relu", ["activation"]), ("tr_postnorm", ["pre_norm"]),
                          ("tr_noz", ["standardize"]), ("tr_visibility", ["features"]),
                          ("tr_small", ["width", "layers"])],
            "training": [("tr_sqrt", ["loss"]), ("tr_raw", ["loss"]),
                         ("tr_nodrop", ["dropout"]), ("tr_nodecay", ["weight_decay"]),
                         ("tr_lr1e3", ["lr"])]}
        for ex in base["exercises"]:
            amendment["allowed"][str(ex)] = {}
            for test in subjects:
                ranking = base_selection["selection"][str(ex)][test]["ranking"]
                ap = {r["name"]: r["validation_macro_ap"] for r in ranking}
                reference = base["candidates"]["tr_ref"]
                group_configs = {}
                for group, factors in factor_groups.items():
                    config = copy.deepcopy(reference)
                    # Sorting makes the stronger alternative win if two factors
                    # modify the same key (e.g. sqrt and raw loss weights).
                    for name, keys in sorted(factors, key=lambda item: ap[item[0]]):
                        if ap[name] >= ap["tr_ref"]+.015:
                            for key in keys:
                                config[key] = base["candidates"][name].get(key, reference.get(key))
                    group_configs[group] = config
                combined = copy.deepcopy(reference)
                for config in group_configs.values():
                    combined.update({k:v for k,v in config.items() if v != reference.get(k)})
                group_configs["combined"] = combined
                seen = {digest(c) for c in base["candidates"].values()}
                names = []
                for group, config in group_configs.items():
                    if digest(config) in seen:
                        continue
                    name = f"iter_ex{ex}_{test}_{group}"
                    names.append(name)
                    amendment["candidates"][name] = config
                    seen.add(digest(config))
                amendment["allowed"][str(ex)][test] = names
        save_json(amendment_path, amendment)
    protocol = copy.deepcopy(base)
    protocol["candidates"].update(amendment["candidates"])
    if args.stage in ("tune", "all"):
        jobs = [(ex,test,val,name,base["seeds"][0]) for ex in base["exercises"] for test in subjects
                for name in amendment["allowed"][str(ex)][test] for val in subjects if val != test]
        fit_jobs(protocol, identity, x, y, observations, jobs)
    selection = copy.deepcopy(base_selection)
    for ex in base["exercises"]:
        for test in subjects:
            item = selection["selection"][str(ex)][test]
            for name in amendment["allowed"][str(ex)][test]:
                results = [json.loads((folder(ex,test,val,name,base["seeds"][0]) / "result.json").read_text(encoding="utf-8")) for val in subjects if val != test]
                item["ranking"].append({"name": name, "family": "transformer",
                    "validation_macro_ap": float(np.mean([r["validation"]["macro_ap"] for r in results])),
                    "validation_macro_f1": float(np.mean([r["validation"]["macro_f1"] for r in results])),
                    "validation_false_alert_rate": float(np.mean([r["validation"]["false_alert_rate"] for r in results]))})
            item["ranking"].sort(key=lambda r:(-r["validation_macro_ap"],-r["validation_macro_f1"],r["name"]))
            item["transformer"] = next(r["name"] for r in item["ranking"] if r["family"] == "transformer")
    selection.update({"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "amendment_sha256": digest(amendment),
                      "test_predictions_available_at_selection": False})
    selected_path = RUN / "selection_final.json"
    if selected_path.exists():
        assert json.loads(selected_path.read_text(encoding="utf-8"))["selection"] == selection["selection"]
        selection = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        save_json(selected_path, selection)
    save_json(RUN / "protocol.final.json", protocol)
    if args.stage in ("confirm", "all"):
        jobs=[]
        for ex in base["exercises"]:
            for test in subjects:
                chosen=selection["selection"][str(ex)][test]
                for val in subjects:
                    if val == test:
                        continue
                    for name in [chosen["transformer"], chosen["tcn"], "angle_rf"]:
                        for seed in base["seeds"][1:]:
                            jobs.append((ex,test,val,name,seed))
        fit_jobs(protocol, identity, x, y, observations, jobs)
    if args.stage in ("evaluate", "all"):
        evaluate(protocol, selection, x, noisy, y, observations, subjects)


if __name__ == "__main__":
    main()
