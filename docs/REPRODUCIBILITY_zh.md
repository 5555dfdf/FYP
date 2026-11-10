# 环境、数据与实验复现

更新：2026-11-10。当前训练主线为REHAB26肩外展ex1与肩后伸ex7。仓库提供代码、正式协议、汇总成绩和核验记录；原始数据、缓存、权重与完整预测保留本地。仅查看结果、打开公开动作示意，不需要下载训练数据。

## 1. 当前环境

以下命令均从仓库根目录执行。本机PowerShell先加载Conda hook：

```powershell
& 'D:\conda\shell\condabin\conda-hook.ps1'
conda activate fyp
python -c "import sys, torch, sklearn, numpy, scipy; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

其他机器使用自己的Conda安装路径。训练使用RTX4050 Laptop；原始v1是CPU实验。当前训练环境的包版本见[v2环境快照](../experiments/rehab26_transformer_v2/environment.json)、[Conda导出](../experiments/rehab26_transformer_v2/environment.yml)与[依赖锁定记录](../experiments/rehab26_transformer_v2/requirements-lock.txt)。新机器需按平台安装兼容的PyTorch/CUDA依赖，不直接照搬导出文件的本机prefix。

`requirements-research.txt`用于文献和审计工具，不能替代训练依赖。演示生成使用NumPy/SciPy，架构绘图使用Pillow；浏览器核验另需Playwright和本机Chrome。

## 2. 原始数据与v1缓存

从[REHAB26-ViewAngles官方记录](https://doi.org/10.5281/zenodo.23018128)取得骨架ZIP与四份配套文件，按[审计说明](REHAB26_AUDIT_zh.md)放置：

```text
jointSequences-mediapipe.zip
data/REHAB26-ViewAngles/raw/
  Segmentation.csv
  Segmentation.txt
  Error_types.csv
  joint_names.txt
```

审计脚本校验并解压骨架ZIP，生成动作索引；之后v1预处理创建当前实验共同依赖的单视角缓存：

```powershell
python -X utf8 tools/audit_rehab26.py
python -X utf8 experiments/rehab26_v1/prepare.py
```

输出位于`data/REHAB26-ViewAngles/audit/`和`processed_v1/`。已存在完整`processed_v1`时，`prepare.py`会拒绝覆盖；已有核验缓存可直接沿用。当前两种动作合计1111次物理动作、3333个相机观察。无需重跑v1的50次基线训练即可准备v3输入。

## 3. 融合实验与完整比较的依赖

在上述缓存就绪后，v3从原始三路骨架重新构建融合样本。以下是需要重训时的执行顺序，会写入本地缓存和运行目录：

```powershell
python -X utf8 experiments/rehab26_fusion_v3/prepare.py
python -X utf8 experiments/rehab26_fusion_v3/verify_fusion.py
python -X utf8 experiments/rehab26_fusion_v3/run.py smoke
python -X utf8 experiments/rehab26_fusion_v3/run.py all --workers 2
python -X utf8 experiments/rehab26_fusion_v3/verify_models.py
```

融合缓存位于`processed_fusion_v3/`，包含1111个动作、96帧、88维缓存；参考网络使用其中69维。正式运行目录为`runs/rehab26_fusion_v3/`。已有完整缓存会保留；运行身份绑定代码、协议和数据哈希，改变处理方式时应另建版本，不能混用旧权重。

**完整比较表还依赖v2预测。** `analyze.py`读取`runs/rehab26_transformer_v2/`下的验证和测试预测，计算三视角概率融合与验证重选阈值；它不只读取v3结果。新机器若要重建整张比较表，应先依照[v2复现说明](../experiments/rehab26_transformer_v2/README.md)完成`parallel_run.py all`及`finalize.py`，再执行：

```powershell
python -X utf8 experiments/rehab26_fusion_v3/analyze.py
python -X utf8 experiments/rehab26_fusion_v3/visualize_fusion.py
python -X utf8 experiments/rehab26_fusion_v3/export_best_results.py
```

这些分析/导出脚本会重写派生报告或成绩证据，只有实际复现时才运行；阅读现有成绩不需要运行。现有冻结配置与结果见[v3实验说明](../experiments/rehab26_fusion_v3/README.md)和[最佳设定](../experiments/rehab26_fusion_v3/BEST_SETTINGS_zh.md)。当前v3模型继续导入v2共享实现和v1的`pipeline.py`，保留三个目录才能复现。

各轮推理入口见[v1](../experiments/rehab26_v1/README.md)、[v2](../experiments/rehab26_transformer_v2/README.md)与[v3](../experiments/rehab26_fusion_v3/README.md)。推理需要本地权重；仓库没有可直接下载的权重文件。全数据研究bundle与按人测试成绩分别解释。

## 4. 公开动作示意与架构图

公开演示可直接下载`demos/rehab_motion_svg/demo.html`，用Chrome/Edge离线打开。基于已提交的示意数据重建HTML与14个SVG不需要原始REHAB26或模型权重：

```powershell
python demos/rehab_motion_svg/build.py --public-only
python demos/rehab_motion_svg/check_player.py --public
python tools/render_transformer_design.py
```

交互检查会生成临时截图（Git忽略）及公开QA报告；架构脚本生成两组PNG/SVG。GIF另由`render_previews.py`生成。真实骨架回放还需本地数据和融合依赖，生成步骤见[演示说明](../demos/rehab_motion_svg/README.md)。公开示意不运行分类器；真实坐标及包含它们的完整页面保持本地。

## 5. 早期调研复现（历史材料）

以下为IRDS及2026-11-06文献选型记录，保留用于追溯，不是当前REHAB26训练的前置步骤。文献工具依赖可按需安装：

```powershell
python -m pip install -r requirements-research.txt
```

### IRDS审计（2026-10-17）

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

### 离线重建论文表

已提交 `paper_registry.json`，可以在不访问网络的情况下重新生成论文表：

```bash
python literature/2026-11-06/render_registry.py
```

结构化登记保存书目、阅读状态和原创笔记；第三方全文/摘要内容保留在本地来源文件，不随登记上传。`metadata_source` 和 `source_stem` 指向本地证据名称，未承诺每个条目都已有全文。

### 文献检索与来源重新获取

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

### 候选数据入口（截至2026-11-06）

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
