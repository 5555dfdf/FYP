"""Verify final raw-skeleton inference against frozen outer predictions and refits."""
from __future__ import annotations

import json

import numpy as np

from core import ROOT, RUN, configure_torch, read_data, save_json
from infer import predict_segment


def main():
    configure_torch()
    x,_,y,observations,_=read_data()
    selection=json.loads((RUN/"selection_final.json").read_text(encoding="utf-8"))
    checked=0
    max_difference=0.
    examples=[]
    for exercise in [1,7]:
        for person in sorted(selection["selection"][str(exercise)]):
            seeds=[]
            for seed in [20261109,20261110,20261111]:
                seeds.append(np.load(RUN/f"outer_predictions/ex{exercise}/{person}/selected_transformer/seed_{seed}/predictions.npz"))
            assert all(np.array_equal(seeds[0]["ids"],s["ids"]) for s in seeds)
            expected=np.mean([s["scores"] for s in seeds],axis=0)
            expected_threshold=np.mean([s["thresholds"] for s in seeds],axis=0)
            # Three observations are selected by index, independent of correctness.
            positions=[0,len(seeds[0]["ids"])//2,len(seeds[0]["ids"])-1]
            bundle=RUN/f"evaluated_bundles/ex{exercise}/{person}/bundle.json"
            for position in positions:
                row=observations[int(seeds[0]["ids"][position])]
                raw=np.load(ROOT/"data/REHAB26-ViewAngles/raw"/row["skeleton_file"],mmap_mode="r")
                result=predict_segment(bundle,raw[row["first_frame"]:row["last_frame"]+1])
                scores=np.asarray([r["score"] for r in result["per_error"]])
                thresholds=np.asarray([r["threshold"] for r in result["per_error"]])
                difference=float(np.max(np.abs(scores-expected[position])))
                assert np.allclose(scores,expected[position],atol=3e-6),(exercise,person,difference)
                assert np.allclose(thresholds,expected_threshold,atol=1e-12)
                assert result["members"]==12
                max_difference=max(max_difference,difference)
                checked+=1
                if position==0:
                    examples.append({"observation_id":row["observation_id"],"true_errors":row["error_labels"],"inference":result})
            for seed in seeds:seed.close()
    deployment=[]
    for exercise in [1,7]:
        bundle=RUN/f"deployment/ex{exercise}/bundle.json"
        row=next(r for r in observations if r["exercise_id"]==exercise)
        raw=np.load(ROOT/"data/REHAB26-ViewAngles/raw"/row["skeleton_file"],mmap_mode="r")
        result=predict_segment(bundle,raw[row["first_frame"]:row["last_frame"]+1])
        assert result["members"]==3 and all(np.isfinite(r["score"]) for r in result["per_error"])
        assert result["status"]=="full_data_research_models_not_independently_evaluated"
        deployment.append({"exercise":exercise,"members":3,"raw_inference":"passed","scope":"training-data smoke check, not independent score"})
    report={"status":"passed","evaluated_bundle_raw_clips":checked,"max_score_difference":max_difference,
            "deployment_checks":deployment,"scope":"raw preprocessing + saved weights + all ensemble members + thresholds"}
    save_json(RUN/"verification_bundles.json",report)
    save_json(RUN/"inference_examples.json",examples)
    save_json(ROOT/"experiments/rehab26_transformer_v2/verification_bundles.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
