# 数据来源与复现说明

整理日期：2026-11-08。

## 仓库包含什么

本仓库发布项目原创的研究记录、比较表、审计脚本、派生汇总和骨架可视化。原始数据、论文 PDF、第三方全文提取、网页缓存和下载日志保留在本地，通过 `.gitignore` 排除。

`sample_manifest.jsonl` 是每个原始动作文件的派生检查记录，包含公开数据中的伪匿名 ID、动作与标签，不包含原始坐标序列。它没有人为补充真实姓名，也不是训练/测试划分。

论文和数据使用各自的许可证或申请条件。引用时请引用原始来源，不把本仓库的派生汇总当作原始数据发布。尚未为整个仓库声明统一许可证，第三方材料的许可不由本仓库决定。

## 1. 安装研究依赖

从仓库根目录执行：

```bash
python -m pip install -r requirements-research.txt
```

依赖文件列出兼容范围，尚不是锁定环境。文献工具和审计脚本尚未组成正式模型训练环境；不要据此推断某个深度学习模型已可运行。

## 2. IRDS 审计

原论文：Miron et al., *IntelliRehabDS (IRDS)—A Dataset of Physical Rehabilitation Movements*, Data, 2021。[论文 DOI](https://doi.org/10.3390/data6050046)、[原始数据 DOI](https://doi.org/10.5281/zenodo.4610859)、[原始读取/探索代码](https://github.com/alina-miron/intellirehabds)。数据论文标注 CC BY 4.0，使用时保留原作者引用。

从原始来源取得数据，解压为：

```text
IRDS/SkeletonData/
  Simplified/*.txt
  RawData/*.txt
```

两种格式对应同一批动作，不算成两倍样本。本地审计版本包含各 2,589 个动作文件；版本可能变化，新的统计应与提交结果对比并记录差异。

在仓库根目录执行：

```bash
python IRDS/analysis/audit_irds.py
```

脚本只读取源动作数据，在 `IRDS/analysis/` 写入或更新汇总 JSON、样本清单和图表。已有报告记录的是 2026-10-17 的运行；本次发布未重新运行全量审计。脚本使用相对其自身的路径，不依赖 `D:\_FYP` 或其他固定盘符。

已有输出：

- [audit_summary.json](../IRDS/analysis/audit_summary.json)
- [sample_manifest.jsonl](../IRDS/analysis/sample_manifest.jsonl)
- [visualized_samples.json](../IRDS/analysis/visualized_samples.json)
- [数据分布图](../IRDS/analysis/data_overview.png)
- [动作示例图](../IRDS/analysis/shoulder_examples.png)

结果边界和异常解释见 [IRDS 评估](../IRDS/analysis/IRDS_assessment_zh.md)。标称时间为帧数/30，不能未经时间字段核实就用于速度或平滑性结论；patient/control 分组是文件 ID 约定的工作性解释。

## 3. 离线重建论文表

已提交 `paper_registry.json`，可以在不访问网络的情况下重新生成论文表：

```bash
python literature/2026-11-06/render_registry.py
```

结构化登记保存书目、阅读状态和原创笔记；第三方全文/摘要内容保留在本地来源文件，不随登记上传。`metadata_source` 和 `source_stem` 指向本地证据名称，未承诺每个条目都已有全文。

## 4. 文献检索与来源重新获取

以下脚本是 2026-11-06 调研时使用的辅助工具，查询带当时日期上限。按需执行，不必为了阅读仓库全部重跑：

```bash
python literature/2026-11-06/collect_literature.py
python literature/2026-11-06/fetch_sources.py
python literature/2026-11-06/fetch_more.py
python literature/2026-11-06/fetch_priority.py
python literature/2026-11-06/check_resources.py
python literature/2026-11-06/build_registry.py
python literature/2026-11-06/render_registry.py
```

获取脚本在日期目录下的 `sources/` 保存响应，需联网并受外部网站的许可、权限、限流和内容变化影响。日志中的请求失败属于当时事实，不能认定为对应数据永久不可用。部分脚本仅打印错误而继续执行，退出成功不代表每篇都获取成功；请检查响应、文件格式及论文阅读等级。

`build_registry.py` 尝试从原始 API 响应或 DOI 查询建立登记，不会替你完成论文阅读。运行后应检查缺失书目、题名匹配和阅读状态，避免把自动获取元数据理解成自动全文核对。

[source_inventory.json](../literature/2026-11-06/source_inventory.json) 保存本次发布前本地来源文件的大小与 SHA-256，并列出论文/项目入口。文件哈希可识别本地副本，但本身不证明论文结论正确，也不保证未来下载字节一致。用以下命令可按当前本地文件重建该索引：

```bash
python tools/build_source_inventory.py
```

## 5. 候选数据入口

| 数据 | 原始入口 | 截至 2026-11-06 的状态 |
|---|---|---|
| KIMORE | [原论文](https://doi.org/10.1109/TNSRE.2019.2923060)、[项目](https://vrai.dii.univpm.it/content/kimore-dataset) | 已核对论文；动作未取得；RGB 原文要求请求与 EULA |
| KERAAL | [原论文](https://doi.org/10.1109/IJCNN60899.2024.10650036)、[作者仓库](https://github.com/nguyensmai/KeraalDataset) | 已核对论文/说明；动作与标注未全量解析 |
| REHAB24-6 | [原始 Zenodo](https://zenodo.org/records/13305826) | 已核对元数据、文件及标签说明；ZIP 未下载 |
| UI-PRMD | [原论文](https://doi.org/10.3390/data3010002)、[原始网站](https://webpages.uidaho.edu/ui-prmd/) | 已核对原论文；本地无动作数据 |
| FineRehab | [官方项目](https://bsu3dvlab.github.io/FineRehab/) | 页中仍写即将发布；不作确定训练依赖 |
| MobiPhysio | [Dataverse DOI](https://doi.org/10.7910/DVN/XSI0QN) | 全文来源已定位；实际文件访问核对未完成 |
| Rehab-Pile | [官方实现](https://github.com/MSD-IRIMAS/DeepRehabPile) | 统一整理版；原始数据版本、过滤和派生标签需单独核实 |

更完整的标签与样本限制见[数据集比较](../literature/2026-11-06/dataset_comparison_zh.md)。

## 6. 将来训练结果的记录约定

正式训练开始后，应记录原始版本、每个划分的人 ID、标签生成规则、训练/验证/测试预测、配置、随机种子、依赖版本和权重来源。按人划分应同时约束重复、视角、切片和预训练数据。模型成绩、文献报告和探索性几何指标分别标明来源。

当前没有模型训练命令、权重下载或模型性能表；这些应在实际训练完成后加入。
