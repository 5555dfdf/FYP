from concurrent.futures import ThreadPoolExecutor, as_completed
from fetch_sources import get, url, query, OUT
from fetch_more import crossref
from pypdf import PdfReader
import json,sys
sys.stdout.reconfigure(encoding='utf-8')

def pdf(name,address):
    r=get(address)
    if not r.content.startswith(b'%PDF'): raise ValueError('Not PDF: '+address)
    target=OUT/(name+'.pdf');target.write_bytes(r.content)
    reader=PdfReader(target)
    (OUT/(name+'_fulltext.txt')).write_text('\n\n'.join(f'=== PAGE {i+1} ===\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages)),encoding='utf-8')
    return name, {'pages':len(reader.pages),'url':address}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs=[pool.submit(url,n,u) for n,u in {
          'keraal_readme':'https://raw.githubusercontent.com/nguyensmai/KeraalDataset/main/README.md',
          'lightpra_readme':'https://raw.githubusercontent.com/SaraS92/LightPRA/main/README.md',
          'recon_readme':'https://raw.githubusercontent.com/shamarkus/ReCon/master/README.md',
          'rehab246_readme':'https://raw.githubusercontent.com/abediai/rehab24-6-dataset-creation/main/README.md',
          'deeprehabpile_download':'https://raw.githubusercontent.com/MSD-IRIMAS/DeepRehabPile/main/datasets/get_datasets.sh',
          'finerehab_cvf':'https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Li_FineRehab_A_Multi-modality_and_Multi-task_Dataset_for_Rehabilitation_Analysis_CVPRW_2024_paper.html',
          'uiprmd_author':'https://www.aleksandarvakanski.com/datasets.html',
        }.items()]
        jobs+=[pool.submit(pdf,n,u) for n,u in {
          'keraal2024':'https://keraal.enstb.org/publication/Nguyen2024IJCNN.pdf',
          'finerehab2024':'https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Li_FineRehab_A_Multi-modality_and_Multi-task_Dataset_for_Rehabilitation_Analysis_CVPRW_2024_paper.pdf',
          'egcn2022':'https://www.ijcai.org/proceedings/2022/0511.pdf',
        }.items()]
        jobs += [pool.submit(crossref,n,t) for n,t in {
          'keraal2024':'A Medical Low-Back Pain Physical Rehabilitation Database for Human Body Movement Analysis',
          'ucdhe2023':'Fast and robust video-based exercise classification via body pose tracking and scalable multivariate time series classifiers',
        }.items()]
        for job in as_completed(jobs):
            try:print(json.dumps(job.result(),ensure_ascii=False),flush=True)
            except Exception as e:print('ERROR',str(e),flush=True)
