from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, sys, time
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sources'
OUT.mkdir(exist_ok=True)

def get(url, params=None):
    last=None
    for attempt in range(3):
        try:
            r=requests.get(url,params=params,timeout=(12,40),headers={'User-Agent':'FYP-Literature-Review/1.0'})
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last=e
            time.sleep(attempt+1)
    raise last

PMCS={
 'review2020':'PMC7189627', 'review2025':'PMC11902703',
 'uiprmd2018':'PMC5773117','deepframework2020':'PMC7032994',
 'vinet2020':'PMC7570706','ftfhgcn2025':'PMC12287435',
 'liverpool2025':'PMC12062305','llmfms2025':'PMC11896072',
 'mobiphysio2026':'PMC12992533','strokevideo2026':'PMC13195782',
 'irdsleakage2026':'PMC13416899', 'lowerlimbreview2024':'PMC11151759',
}
URLS={
 'deeprehabpile_readme':'https://raw.githubusercontent.com/MSD-IRIMAS/DeepRehabPile/main/README.md',
 'deeprehabpile_paper':'https://arxiv.org/html/2507.21018',
 'contrastive_karlov_paper':'https://arxiv.org/html/2403.02772',
 'rotation_repo':'https://raw.githubusercontent.com/Kelly510/RehabExerAssess/main/README.md',
 'uiprmd_official':'https://webpages.uidaho.edu/ui-prmd/',
 'keraal_official':'https://keraal.enstb.org/',
}
QUERIES={
 'compensation_focused':'rehabilitation AND ("compensatory movement" OR "compensation detection") AND (dataset OR learning)',
 'keraal_focused':'Keraal',
 'toronto_focused':'"Toronto Rehab Stroke Pose"',
 'rehab246_focused':'"REHAB24-6"',
 'mri_focused':'"multimodal rehabilitation" AND (dataset OR benchmark)',
}

def pmc(name, pid):
    r=get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pid}/fullTextXML')
    (OUT/f'{name}.xml').write_text(r.text,encoding='utf-8')
    soup=BeautifulSoup(r.content,'xml')
    (OUT/f'{name}_fulltext.txt').write_text(soup.get_text('\n',strip=True),encoding='utf-8')
    return name, 'FULLTEXT '+str(len(r.content))

def url(name, address):
    r=get(address)
    (OUT/f'{name}_web.txt').write_text(r.text,encoding='utf-8')
    if 'html' in r.headers.get('Content-Type',''):
        (OUT/f'{name}_fulltext.txt').write_text(BeautifulSoup(r.content,'html.parser').get_text('\n',strip=True),encoding='utf-8')
    return name,'WEB '+str(len(r.content))

def query(name, q):
    r=get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',{'query':f'({q}) AND FIRST_PDATE:[2000-01-01 TO 2026-11-06]','format':'json','resultType':'core','pageSize':30})
    (OUT/f'epmc_{name}.json').write_text(r.text,encoding='utf-8')
    items=r.json()['resultList']['result']
    return name, [{'title':x.get('title'),'doi':x.get('doi'),'pmcid':x.get('pmcid'),'year':x.get('pubYear'),'abstract':x.get('abstractText')} for x in items]

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(pmc,n,p) for n,p in PMCS.items()]+[pool.submit(url,n,u) for n,u in URLS.items()]+[pool.submit(query,n,q) for n,q in QUERIES.items()]
        for job in as_completed(jobs):
            try: print(json.dumps(job.result(),ensure_ascii=False),flush=True)
            except Exception as e: print('ERROR',str(e),flush=True)
