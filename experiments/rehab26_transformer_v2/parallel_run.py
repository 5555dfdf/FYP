"""Run independent fits in two local processes; no change to scientific protocol."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import datetime, timezone

from core import RUN, configure_torch, read_data, save_json

_cache = None


def worker_init():
    global _cache
    configure_torch()
    _cache = read_data()


def worker_fit(job):
    from suite import folder
    from train import fit_one
    ex, test, val, name, seed, config, training, identity = job
    x, _, y, observations, _ = _cache
    target = folder(ex,test,val,name,seed)
    start=time.perf_counter()
    result=fit_one(x,y,observations,ex,test,val,config,training,seed,target,identity)
    return {"exercise":ex,"test":test,"val":val,"candidate":name,"seed":seed,
            "ap":result["validation"]["macro_ap"],"f1":result["validation"]["macro_f1"],
            "best_epoch":result["best_epoch"],"seconds":time.perf_counter()-start}


def parallel_fit_jobs(protocol, identity, x, y, observations, jobs):
    from suite import folder
    pending=[]
    for ex,test,val,name,seed in jobs:
        if (folder(ex,test,val,name,seed)/"result.json").exists():
            # Check compatibility even for completed jobs, without CUDA reload.
            from core import digest
            result=json.loads((folder(ex,test,val,name,seed)/"result.json").read_text(encoding="utf-8"))
            expected={"identity":identity,"config":protocol["candidates"][name],"training":protocol["training"],
                      "seed":seed,"exercise":ex,"test":test,"validation":val}
            assert result["job_hash"]==digest(expected)
        else:
            pending.append((ex,test,val,name,seed,protocol["candidates"][name],protocol["training"],identity))
    print(f"Local process pool=2; {len(pending)} pending of {len(jobs)} fits",flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,initializer=worker_init) as pool:
        futures=[pool.submit(worker_fit,job) for job in pending]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            result=future.result()
            print(f"FIT [{i}/{len(pending)}] {json.dumps(result,ensure_ascii=False)}",flush=True)
            save_json(RUN/"progress_parallel.json",{"updated_at_utc":datetime.now(timezone.utc).isoformat(),
                      "completed_pending":i,"pending_total":len(pending),"already_completed":len(jobs)-len(pending),
                      "last_validation_result":result,"test_revealed":(RUN/"test_summary.json").exists()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["screen","iterate","all"])
    args=parser.parse_args()
    configure_torch()
    if args.stage in ("screen","all"):
        from suite import initialize,select
        protocol,identity=initialize()
        x,_,y,observations,_=read_data()
        subjects=sorted({r["person"] for r in observations})
        jobs=[(ex,test,val,name,protocol["seeds"][0]) for ex in protocol["exercises"]
              for test in subjects for val in subjects if val!=test for name in protocol["candidates"]]
        parallel_fit_jobs(protocol,identity,x,y,observations,jobs)
        select(protocol,subjects)
        del x,y,observations
    if args.stage in ("iterate","all"):
        import iterate
        iterate.fit_jobs=parallel_fit_jobs
        sys.argv=[sys.argv[0],"all"]
        iterate.main()


if __name__=="__main__":
    main()
