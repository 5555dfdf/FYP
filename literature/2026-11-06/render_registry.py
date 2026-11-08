"""Render the curated registry as a human-readable Markdown comparison table."""
from pathlib import Path
from collections import Counter
import json,sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT=Path(__file__).resolve().parent
CODE={
 'D04':'https://github.com/nguyensmai/KeraalDataset',
 'D05':'https://bsu3dvlab.github.io/FineRehab/',
 'D08':'https://github.com/MarioTed/LLM-FMS',
 'D09':'https://zenodo.org/records/13305826',
 'M04':'https://github.com/bruceyo/EGCN',
 'M05':'https://github.com/Kelly510/RehabExerAssess',
 'M08':'https://github.com/SaraS92/LightPRA',
 'M09':'https://github.com/bruceyo/egcnplusplus',
 'M11':'https://github.com/shamarkus/ReCon',
 'B01':'https://github.com/MSD-IRIMAS/DeepRehabPile',
}
def clean(s): return str(s or '').replace('|',' / ').replace('\n',' ')

def main():
    doc=json.loads((ROOT/'paper_registry.json').read_text(encoding='utf-8'))
    rows=doc['records']
    assert len(rows)==33
    assert len({r['id'] for r in rows})==33
    assert len({r['doi'].lower() for r in rows})==33
    assert all(r.get('title') for r in rows)
    head='''# 康复动作模型训练：候选论文比较表

更新：2026-11-06。总计 33 篇，包含早期已经涉及的文献；不是 33 篇新论文，也不是全部已经全文精读。

**阅读等级是本轮证据状态。** “全文重点核对”表示已经查看与选题相关的方法、标签或实验段落，不代表复现；“摘要核对”不能用于确认实现和所有划分；“书目信息”仅证明文献存在。完整作者和来源索引保存在 `paper_registry.json`，原始来源缓存在本地 `sources/`，不随仓库发布。

年份采用可得的期刊年份/出版元数据。ReCon 为 2024 在线、2025 卷期；UCDHE 方法为 2022 在线、2023 卷期；Rehab-Pile 读的是详细的 2025 预印本及当前仓库，正式会议版本为 2026，二者不计作两篇。

| ID | 年份 | 类别 | 论文与 DOI | 本轮核对程度 | 与训练研究的关系/限制 | 代码或数据入口 |
|---|---|---|---|---|---|---|
'''
    lines=[head]
    for r in rows:
        year={'M11':'2024/2025','D10':'2022/2023','B01':'2025/2026'}.get(r['id'],r.get('year','待核对'))
        link=f"[资源]({CODE[r['id']]})" if r['id'] in CODE else '待定位/核对'
        lines.append(f"| {r['id']} | {year} | {clean(r['category'])} | [{clean(r['title'])}]({r['url']}) | {clean(r['evidence_level'])} | {clean(r['note'])} | {link} |\n")
    lines.append('''
## 这些入口不代表已经完成复现

- DeepRehabPile：已核对详细预印本、README、下载脚本。提供单模型单折训练入口及结果；当前未运行。原数据来源、筛选与标签必须保留。
- ReCon：已核对论文和作者 README。README 当前只简要说明 KIMORE/UI-PRMD，论文另有 IRDS 实验；需检查代码是否覆盖目标实验。
- EGCN：已核对论文和 README。模型特征融合可借鉴，但其 KIMORE 正常/异常代理标签不等同逐次临床正确性。
- Rotation Invariance：已核对官方 README，早期项目记录还检查过 IRDS 加载器的非按人划分；复用时需更换评价协议。
- LightPRA：已定位作者仓库，README 较简略；依赖、数据与训练入口待审查。
- EGCN++：EGCN 作者 README 提供此后续入口；本轮未独立审计内容。
- FineRehab：项目页仍称即将发布，不能把资源入口当作下载成功。

## 与此前八项 IRDS 后续文献调研的关系

本表优先扩展数据源和模型家族，没有把早期调研每一项都重复收录。早期的分割计数、公平性、2025 五模型比较等仍保存在原调研，不表示它们被否定。当前两表的篇数不能直接相加作为互不重复的总数。

另查得 2026 上肢 RGB 数据集 DOI `10.1016/j.dib.2026.112819` 并保存全文，作为补充线索；没有为凑数量计入 33 篇核心登记。更多同名/无关检索结果保留在搜索日志，不计入已筛选候选。
''')
    (ROOT/'paper_matrix_zh.md').write_text(''.join(lines),encoding='utf-8')
    print('rows',len(rows),'evidence',dict(Counter(r['evidence_level'] for r in rows)))

if __name__=='__main__':main()
