"""Save reproducible literature-search evidence; no model training or source-data edits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'sources'
SOURCES.mkdir(parents=True, exist_ok=True)
QUERIES = {
    'review': '(rehabilitation AND ("movement quality" OR "exercise assessment" OR "action quality assessment") AND (review OR survey))',
    'datasets': '(KIMORE OR "UI-PRMD" OR IntelliRehabDS OR "Rehab-Pile" OR "RehabPile")',
    'compensation': '(rehabilitation AND ("compensatory movement" OR "compensation detection") AND (dataset OR learning))',
    'rehab_datasets': '("REHAB24-6" OR "Keraal" OR "MIRA" OR "Toronto Rehab Stroke Pose" OR "UCO Physical Rehabilitation")',
    'quality': '(rehabilitation AND skeleton AND ("contrastive" OR "transformer" OR "graph convolutional" OR "self-supervised"))',
}

def fetch(name, url, params=None):
    r = requests.get(url, params=params, timeout=60, headers={'User-Agent':'FYP-Literature-Review/1.0'})
    r.raise_for_status()
    suffix = '.json' if 'json' in r.headers.get('Content-Type','') else '.txt'
    (SOURCES / (name + suffix)).write_text(r.text, encoding='utf-8')
    return name, r

def main():
    jobs = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for name, query in QUERIES.items():
            jobs.append(pool.submit(fetch, 'epmc_'+name, 'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                {'query': f'({query}) AND FIRST_PDATE:[2000-01-01 TO 2026-11-06]', 'format':'json','resultType':'core','pageSize':60}))
        for name, query in {
            'survey':'skeleton rehabilitation exercise assessment survey review',
            'keraal':'Keraal dataset rehabilitation',
            'uiprmd':'dataset human motion rehabilitation exercises Vakanski',
            'rehab246':'REHAB24-6 dataset rehabilitation',
            'aquality':'action quality assessment survey learning',
        }.items():
            jobs.append(pool.submit(fetch,'crossref_'+name,'https://api.crossref.org/works',
                {'query':query,'rows':12,'filter':'until-pub-date:2026-11-06'}))
        jobs.append(pool.submit(fetch,'deeprehabpile_readme','https://raw.githubusercontent.com/MSD-IRIMAS/DeepRehabPile/main/README.md'))
        for job in as_completed(jobs):
            try:
                name, r = job.result()
                if name.startswith('epmc'):
                    d=r.json();items=d['resultList']['result']; print('\nSEARCH',name,'HITS',d['hitCount'])
                    for x in items:
                        print(json.dumps({k:x.get(k) for k in ['id','source','pmcid','doi','title','pubYear','isOpenAccess','firstPublicationDate']},ensure_ascii=False))
                elif name.startswith('crossref'):
                    print('\nSEARCH',name)
                    for x in r.json()['message']['items']:
                        print(json.dumps({'title':x.get('title'),'doi':x.get('DOI'),'date':x.get('published'),'url':x.get('URL')},ensure_ascii=False))
                else: print('\nREADME',r.text[:26000])
            except Exception as e: print('ERROR',repr(e))

if __name__ == '__main__': main()
