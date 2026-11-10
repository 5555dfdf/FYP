"""Export compact homepage evidence from the existing, unchanged results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'results_summary.json'
    summary=json.loads(path.read_text(encoding='utf-8'))
    choices=[('abduction_best',1,'angle_rf','late_calibrated'),
             ('abduction_transformer',1,'selected_transformer','late_calibrated'),
             ('extension_best',7,'reference_transformer','x_equal'),
             ('extension_three_seeds',7,'selected_transformer','late_calibrated')]
    results=[]
    for label,exercise,method,variant in choices:
        row=next(r for r in summary['aggregate'] if (r['exercise'],r['method'],r['variant'])==(exercise,method,variant))
        results.append({'id':label,**row})
    output={'exported_at_utc':datetime.now(timezone.utc).isoformat(),
        'data_source':'https://doi.org/10.5281/zenodo.23018128',
        'dataset_paper':'https://arxiv.org/abs/2609.35726',
        'result_origin':'Experiments in this repository; not scores quoted from the dataset paper',
        'source_file':'experiments/rehab26_fusion_v3/results_summary.json',
        'source_sha256':hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
        'source_hash_normalization':'UTF-8 text with LF line endings, matching the Git blob',
        'metric_definition':'Per-seed pooled five-person outer-test macro-F1; seed sample standard deviation ddof=1',
        'rows':results}
    (HERE/'best_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Exported four homepage result records, including per-seed/per-person metrics')


if __name__=='__main__':main()
