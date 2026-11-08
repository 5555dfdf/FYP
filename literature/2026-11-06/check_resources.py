from concurrent.futures import ThreadPoolExecutor, as_completed
from fetch_sources import get, url, OUT
import base64,json,sys
sys.stdout.reconfigure(encoding='utf-8')

def readme(name,repo):
    r=get('https://api.github.com/repos/'+repo+'/readme')
    (OUT/(name+'_github.json')).write_text(r.text,encoding='utf-8')
    data=r.json()
    txt=base64.b64decode(data['content']).decode('utf-8')
    (OUT/(name+'_readme_web.txt')).write_text(txt,encoding='utf-8')
    return name,txt[:16000]

def metadata(name,address):
    r=get(address)
    (OUT/(name+'_metadata.json')).write_text(r.text,encoding='utf-8')
    d=r.json()
    if name=='rehab246':
        return name,{'title':d.get('metadata',{}).get('title'),'files':[(f.get('key'),f.get('size')) for f in d.get('files',[])],'license':d.get('metadata',{}).get('license'),'description':d.get('metadata',{}).get('description')}
    return name,{'status':d.get('status'),'bytes':len(r.content)}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(readme,n,r) for n,r in {'keraal':'nguyensmai/KeraalDataset','recon':'shamarkus/ReCon','egcn':'bruceyo/EGCN'}.items()]
        jobs += [pool.submit(metadata,'rehab246','https://zenodo.org/api/records/13305826'),
            pool.submit(metadata,'mobiphysio','https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/XSI0QN'),
            pool.submit(url,'kimore_official','https://vrai.dii.univpm.it/content/kimore-dataset'),
            pool.submit(url,'rehabpile_kimore','https://maxime-devanne.com/datasets/RehabPile/KIMORE_reg/'),
        ]
        for job in as_completed(jobs):
            try:print(json.dumps(job.result(),ensure_ascii=False),flush=True)
            except Exception as e:print('ERROR',str(e),flush=True)
