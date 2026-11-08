from concurrent.futures import ThreadPoolExecutor, as_completed
from fetch_sources import get, pmc, url, query, OUT
import json,sys
sys.stdout.reconfigure(encoding='utf-8')

def crossref(name,title):
    r=get('https://api.crossref.org/works',{'query.title':title,'rows':4,'filter':'until-pub-date:2026-11-06'})
    (OUT/f'crossref_{name}.json').write_text(r.text,encoding='utf-8')
    return name,[{'title':x.get('title'),'doi':x.get('DOI'),'link':x.get('link'),'date':x.get('published')} for x in r.json()['message']['items']]

def github(name, q):
    r=get('https://api.github.com/search/repositories',{'q':q,'per_page':4})
    (OUT/f'github_search_{name}.json').write_text(r.text,encoding='utf-8')
    return name,[{'repo':x['full_name'],'description':x.get('description'),'url':x['html_url'],'default_branch':x['default_branch']} for x in r.json().get('items',[])]

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(crossref,n,t) for n,t in {
            'review_skeleton2023':'Artificial intelligence for skeleton-based physical rehabilitation action evaluation: a systematic review',
            'finerehab':'FineRehab A Multi-modality and Multi-task Dataset for Rehabilitation Analysis',
            'keraal':'The Keraal Dataset for Robot Assisted Physical Rehabilitation',
            'keraaltransformer':'Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises',
            'rehab246':'REHAB24-6 A Multi-Modal Dataset of Physical Rehabilitation Exercises',
            'ec3d':'Exercise correction using a 3D human pose estimation',
        }.items()]
        jobs += [pool.submit(github,n,q) for n,q in {
            'finerehab':'FineRehab','keraal':'Keraal','lightpra':'LightPRA','egcn':'EGCN rehabilitation','rehab246':'REHAB24-6','contrastive':'rehabilitation supervised contrastive learning',
        }.items()]
        jobs += [pool.submit(url,'keraal_publications','https://keraal.enstb.org/publications.html'),
                 pool.submit(url,'deepframework2020_html','https://pmc.ncbi.nlm.nih.gov/articles/PMC7032994/'),
                 pool.submit(url,'review2020_html','https://pmc.ncbi.nlm.nih.gov/articles/PMC7189627/'),
                 pool.submit(pmc,'compensationwebcam2025','PMC12660197'),
                 pool.submit(query,'review_skeleton2023','EXT_ID:37182489 OR DOI:10.1016/j.compbiomed.2023.106835')]
        for job in as_completed(jobs):
            try:print(json.dumps(job.result(),ensure_ascii=False),flush=True)
            except Exception as e:print('ERROR',str(e),flush=True)
