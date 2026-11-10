"""Finish a completed run: verification, reports, evaluated-fold bundles and refits."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from core import HERE, ROOT, RUN, configure_torch, metrics, read_data, save_json, sha256, transform
from train import load_artifact


def main():
    assert (RUN/"test_summary.json").exists(), "Wait for all selection, confirmation and outer evaluation"
    for script,extra in [("verify.py",["--all-artifacts"]),("analyze.py",[]),("release.py",[])]:
        subprocess.run([sys.executable,"-X","utf8",str(HERE/script),*extra],check=True)
    configure_torch()
    protocol=json.loads((RUN/"protocol.final.json").read_text(encoding="utf-8"))
    selection=json.loads((RUN/"selection_final.json").read_text(encoding="utf-8"))
    ensemble_report=[]
    # This seed ensemble is a predeclared extra report, not a test-selected seed.
    for ex in [1,7]:
        pooled={k:[] for k in ["y","scores","thresholds","precision_thresholds"]}
        per_person=[]
        for test,chosen in selection["selection"][str(ex)].items():
            members=[]
            scores=[];thresholds=[];precision=[]
            for seed in protocol["seeds"]:
                data=np.load(RUN/f"outer_predictions/ex{ex}/{test}/selected_transformer/seed_{seed}/predictions.npz")
                scores.append(data["scores"]);thresholds.append(data["thresholds"]);precision.append(data["precision_thresholds"])
                y=data["y"]
                for val in sorted(selection["selection"][str(ex)]):
                    if val!=test:
                        members.append(os.path.relpath(RUN/f"fits/ex{ex}/test_{test}/val_{val}/{chosen['transformer']}/seed_{seed}",RUN/f"evaluated_bundles/ex{ex}/{test}"))
            target=RUN/f"evaluated_bundles/ex{ex}/{test}"
            save_json(target/"bundle.json",{"exercise":ex,"members":members,
                "status":"subject_excluded_research_ensemble",
                "excluded_person":test,"candidate":chosen["transformer"],
                "limitations":"Twelve-model ensemble across four inner validation people and three seeds. All members exclude named outer person; simulated errors, no calibrated confidence or clinical validation."})
            score=np.mean(scores,0);threshold=np.mean(thresholds,0);pthreshold=np.mean(precision,0)
            per_person.append({"person":test,"test":metrics(y,score,threshold),"test_precision":metrics(y,score,pthreshold)})
            for key,value in [("y",y),("scores",score),("thresholds",np.broadcast_to(threshold,y.shape)),("precision_thresholds",np.broadcast_to(pthreshold,y.shape))]:
                pooled[key].append(value)
        pooled={key:np.concatenate(value) for key,value in pooled.items()}
        ensemble_report.append({"exercise":ex,"test":metrics(pooled["y"],pooled["scores"],pooled["thresholds"]),
            "test_precision":metrics(pooled["y"],pooled["scores"],pooled["precision_thresholds"]),"per_person":per_person})
    save_json(HERE/"seed_ensemble_results.json",ensemble_report)
    # Network-only timing, one specified fold and member, no choice by test score.
    x,_,_,observations,_=read_data()
    benchmark=[]
    for ex in [1,7]:
        test=sorted(selection["selection"][str(ex)])[0]
        val=sorted(p for p in selection["selection"][str(ex)] if p!=test)[0]
        name=selection["selection"][str(ex)][test]["transformer"]
        folder=RUN/f"fits/ex{ex}/test_{test}/val_{val}/{name}/seed_{protocol['seeds'][0]}"
        artifact=load_artifact(folder)
        row=next(i for i,r in enumerate(observations) if r["exercise_id"]==ex and r["person"]==test)
        tensor=torch.from_numpy(transform(x[row:row+1],artifact["config"],np.asarray(artifact["mean"],np.float32),np.asarray(artifact["std"],np.float32))).cuda()
        model=artifact["model"].eval()
        with torch.inference_mode():
            for _ in range(20):model(tensor)
            torch.cuda.synchronize()
            latencies=[]
            for _ in range(100):
                start=time.perf_counter();model(tensor);torch.cuda.synchronize()
                latencies.append((time.perf_counter()-start)*1000)
        benchmark.append({"exercise":ex,"candidate":name,"parameters":sum(p.numel() for p in model.parameters()),
            "batch":1,"shape":list(tensor.shape),"network_only_gpu_latency_median_ms":float(np.median(latencies)),
            "network_only_gpu_latency_p95_ms":float(np.quantile(latencies,.95)),
            "timing_scope":"one model forward only; excludes pose extraction, preprocessing, transfer, and ensemble repetitions", "warmup":20,"repeats":100})
        del model,artifact,tensor
        torch.cuda.empty_cache()
    save_json(HERE/"benchmark.json",benchmark)
    subprocess.run([sys.executable,"-X","utf8",str(HERE/"verify_bundles.py")],check=True)
    scripts={str(p.relative_to(ROOT)):sha256(p) for p in sorted(HERE.glob("*.py"))}
    save_json(RUN/"completion.json",{"completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "fit_count":len(list((RUN/"fits").rglob("result.json"))),"scripts_sha256":scripts,
        "verification":json.loads((RUN/"verification_all.json").read_text()),
        "benchmarks":benchmark,"status":"complete","deployment":"full-data research refits, no independent full-data score"})
    print(json.dumps({"completion":str(RUN/"completion.json"),"ensemble_results":ensemble_report,"benchmark":benchmark},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
