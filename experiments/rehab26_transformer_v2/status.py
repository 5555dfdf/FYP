"""Compact progress snapshot; reads validation summaries only until test stage."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from core import RUN


def main():
    files=list((RUN/"fits").rglob("result.json"))
    counts={}
    seconds=[]
    by_candidate={}
    for path in files:
        result=json.loads(path.read_text(encoding="utf-8"))
        ex=str(result["exercise"])
        counts[ex]=counts.get(ex,0)+1
        seconds.append(result["elapsed_seconds"])
        name=path.parent.parent.name
        by_candidate.setdefault(name,[]).append(result["validation"]["macro_ap"])
    progress=json.loads((RUN/"progress_parallel.json").read_text(encoding="utf-8")) if (RUN/"progress_parallel.json").exists() else {}
    phase="screen"
    if (RUN/"iteration_plan.json").exists():phase="iteration"
    if (RUN/"selection_final.json").exists():phase="confirm"
    if (RUN/"test_summary.json").exists():phase="evaluated"
    print(json.dumps({"time_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,
        "fits_complete":len(files),"fits_by_exercise":counts,
        "current_stage_completed":progress.get("completed_pending"),"current_stage_pending_total":progress.get("pending_total"),
        "last":progress.get("last_validation_result"),
        "selection_frozen":(RUN/"selection_final.json").exists(),
        "test_revealed":(RUN/"test_summary.json").exists()},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
