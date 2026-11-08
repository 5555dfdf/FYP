"""Build a curated, evidence-labelled bibliography; metadata is not a reading claim."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, sys
from fetch_sources import get

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sources'
ITEMS=[
('R01','综述','10.1016/j.compbiomed.2020.103687','摘要核对','计算方法总体框架；待取得全文','review2020'),
('R02','综述','10.1016/j.compbiomed.2023.106835','摘要核对','骨架康复评估的特征学习、评估与反馈综述','epmc_review_skeleton2023'),
('R03','综述','10.3390/s25051586','全文重点核对','核对纳入研究、数据类型、验证及泛化限制','review2025'),
('R04','综述','10.1177/20556683241259256','全文重点核对','扩展到下肢、视觉及可穿戴方案','lowerlimbreview2024'),
('D01','数据集','10.3390/data6050046','已有本地原文与全量数据审计','真实患者正确/错误；原标签无具体错误类型','../../../IRDS/analysis/IRDS_assessment_zh.md'),
('D02','数据集','10.1109/TNSRE.2019.2923060','本地PDF重点核对','治疗师EAAQ总分与分项；RGB需申请；数据未下载','KIMORE_local'),
('D03','数据集','10.3390/data3010002','全文重点核对','10名健康人、10类动作；正确/错误；不把代理连续分当临床评分','uiprmd2018'),
('D04','数据集','10.1109/IJCNN60899.2024.10650036','全文重点核对','KERAAL；错误类别、部位及时间标注；大量记录无标注，需核对计数差异','keraal2024'),
('D05','数据集','10.1109/CVPRW63382.2024.00324','全文及官方项目页重点核对','FineRehab；多维分项评分；官方页仍写will be released soon，未确认公开下载','finerehab2024'),
('D06','数据集','10.1038/s41597-025-05099-1','全文重点核对','UL-RED；速度与重复、跨传感器；未证实有临床质量标签','liverpool2025'),
('D07','数据集','10.1016/j.dib.2026.112635','全文重点核对','MobiPhysio；RGB与专家评分；人数叙述有待核对','mobiphysio2026'),
('D08','数据集','10.1371/journal.pone.0313707','全文重点核对','LLM-FMS；关键帧、分级和细粒度标注；不等于时序模型','llmfms2025'),
('D09','数据集','10.1007/978-3-031-75823-2_2','原始Zenodo元数据、文件清单及书目信息核对','REHAB24-6；10人、6动作、1072次重复，分段与正确性标签；正文未读','rehab246'),
('D10','方法/数据集','10.1007/s10618-022-00895-4','书目信息及基准全文中的数据描述核对','UCDHE相关：动作错误分类、二维骨架与时序分类器；原文摘要/全文待核对；2022在线/2023卷期','ucdhe2023'),
('M01','模型方法','10.1109/TNSRE.2020.2966249','摘要核对','深度自编码/GMM代理分数及监督预测；非独立临床评分','deepframework2020'),
('M02','模型方法','10.3390/s20185258','全文重点核对','VI-Net；RGB跨视角；QMAR为健康人模拟','vinet2020'),
('M03','模型方法','10.1109/TNSRE.2022.3150392','摘要核对','ST-GCN适配连续评分；作为图模型起点','gcn2022'),
('M04','模型方法','10.24963/ijcai.2022/511','全文重点核对','EGCN；位置/朝向融合；其KIMORE分类用人群代理标签，不能等同逐次正确性','egcn2022'),
('M05','模型方法','10.1109/TNSRE.2023.3282675','官方README及已有代码划分核对','旋转不变描述符；IRDS官方加载器未按人分组','rotation'),
('M06','模型方法','10.1016/j.compbiomed.2023.107420','摘要核对','D-STGCNT；图卷积、GRU、Transformer','dstgcnt2023'),
('M07','模型方法','10.1109/TNSRE.2023.3317411','摘要核对','对比学习、评分与关节注意力差异','contrastive2023'),
('M08','模型方法','10.1016/j.compbiomed.2024.108382','摘要及仓库说明核对','LightPRA；轻量TCN；README不能证明已复现','lightpra'),
('M09','模型方法','10.1109/TPAMI.2024.3378753','摘要核对','EGCN++；图模型集成的后续工作','egcnpp2024'),
('M10','模型方法','10.1109/TNSRE.2024.3400790','摘要核对','专家知识图卷积、门控池化、Transformer','expertgcn2024'),
('M11','模型方法','10.1007/s11517-024-03177-x','全文及作者仓库README重点核对','ReCon；多动作共享对比学习与IRDS到KIMORE迁移；2024在线/2025卷期','contrastive_karlov_paper'),
('M12','模型方法','10.1109/TNSRE.2024.3523906','摘要核对','位置与角度多视图、回归对比学习','hiercontrastive2024'),
('M13','模型方法','10.1016/j.compbiomed.2024.109578','摘要核对','轻量空间/时间图模型、动作分类后评分','optimized2025'),
('M14','模型方法','10.1038/s41598-025-12020-8','全文重点核对','FTF-HGCN；逐帧拓扑与多尺度时序；评分定义需独立核对','ftfhgcn2025'),
('M15','模型方法','10.1109/ICORR66766.2025.11063192','摘要核对','KERAAL Transformer错误分类；对反馈目标直接相关','keraaltransformer'),
('B01','基准/评价','10.1109/FG67764.2026.11556986','详细预印本及官方README重点核对','Rehab-Pile；39分类+21回归任务；按人划分；版本须区分','deeprehabpile_paper'),
('B02','基准/评价','10.3389/fmed.2026.1878438','全文重点核对','IRDS嵌套按人评估；简单特征与ST-SkelNet','irdsleakage2026'),
('B03','基准/评价','10.1016/j.ijmedinf.2026.106464','摘要核对','直接比较按样本与按人划分造成的偏差','impropersplit2026'),
('C01','应用/可行性','10.3389/fmed.2025.1645369','全文重点核对','真实卒中20人；跨人困难；数据未直接公开，不能作为现成训练源','compensationwebcam2025'),
]

def existing_metadata():
    found={}
    for p in OUT.glob('epmc_*.json'):
        for x in json.loads(p.read_text(encoding='utf-8')).get('resultList',{}).get('result',[]):
            if x.get('doi'):
                found[x['doi'].lower()]={'title':x.get('title'),'year':x.get('pubYear'),'first_publication_date':x.get('firstPublicationDate'),'authors':x.get('authorString'),'pmcid':x.get('pmcid'),'abstract':x.get('abstractText'),'metadata_source':p.name}
    for p in OUT.glob('crossref_*.json'):
        for x in json.loads(p.read_text(encoding='utf-8')).get('message',{}).get('items',[]):
            doi=x.get('DOI','').lower()
            if doi and doi not in found: found[doi]=from_crossref(x,p.name)
    return found

def from_crossref(x, source):
    return {'title':' '.join(x.get('title',[])),'year':str(x.get('published',{}).get('date-parts',[[None]])[0][0]),'authors':'; '.join(' '.join([a.get('given',''),a.get('family','')]).strip() for a in x.get('author',[])),'abstract':x.get('abstract'),'metadata_source':source,'container':' '.join(x.get('container-title',[]))}

def retrieve(doi):
    r=get('https://api.crossref.org/works/'+doi)
    name='doi_'+doi.replace('/','_').replace(':','_')+'.json'
    (OUT/name).write_text(r.text,encoding='utf-8')
    return doi.lower(),from_crossref(r.json()['message'],name)

def main():
    metadata=existing_metadata()
    # Exact-title searches provide authoritative identifiers for these two records.
    items=list(ITEMS)
    for i, row in enumerate(items):
        key={'D04':'keraal2024','D10':'ucdhe2023'}.get(row[0])
        if key:
            f=OUT/f'crossref_{key}.json'
            if f.exists():
                x=json.loads(f.read_text(encoding='utf-8'))['message']['items'][0]
                items[i]=row[:2]+(x['DOI'],)+row[3:]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(retrieve,row[2]):row[2] for row in items if row[2].lower() not in metadata}
        for job in as_completed(jobs):
            try:
                doi, value=job.result();metadata[doi]=value;print('METADATA',doi,value['title'],flush=True)
            except Exception as e:print('ERROR',jobs[job],str(e),flush=True)
    records=[]
    for pid,category,doi,evidence,note,stem in items:
        m={k:v for k,v in metadata.get(doi.lower(),{}).items() if k != 'abstract'}
        rec={'id':pid,'category':category,'doi':doi,'url':'https://doi.org/'+doi,'evidence_level':evidence,'note':note,'source_stem':stem,**m}
        records.append(rec)
    (ROOT/'paper_registry.json').write_text(json.dumps({'updated':'2026-11-06','scope':'康复动作评估；公开数据训练模型；尚未最终定题','records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print('RECORDS',len(records),'MISSING',[r['id'] for r in records if not r.get('title')])

if __name__=='__main__':main()
