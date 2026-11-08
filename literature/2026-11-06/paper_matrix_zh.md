# 康复动作模型训练：候选论文比较表

更新：2026-11-06。总计 33 篇，包含早期已经涉及的文献；不是 33 篇新论文，也不是全部已经全文精读。

**阅读等级是本轮证据状态。** “全文重点核对”表示已经查看与选题相关的方法、标签或实验段落，不代表复现；“摘要核对”不能用于确认实现和所有划分；“书目信息”仅证明文献存在。完整作者和来源索引保存在 `paper_registry.json`，原始来源缓存在本地 `sources/`，不随仓库发布。

年份采用可得的期刊年份/出版元数据。ReCon 为 2024 在线、2025 卷期；UCDHE 方法为 2022 在线、2023 卷期；Rehab-Pile 读的是详细的 2025 预印本及当前仓库，正式会议版本为 2026，二者不计作两篇。

| ID | 年份 | 类别 | 论文与 DOI | 本轮核对程度 | 与训练研究的关系/限制 | 代码或数据入口 |
|---|---|---|---|---|---|---|
| R01 | 2020 | 综述 | [A review of computational approaches for evaluation of rehabilitation exercises.](https://doi.org/10.1016/j.compbiomed.2020.103687) | 摘要核对 | 计算方法总体框架；待取得全文 | 待定位/核对 |
| R02 | 2023 | 综述 | [Artificial Intelligence for skeleton-based physical rehabilitation action evaluation: A systematic review.](https://doi.org/10.1016/j.compbiomed.2023.106835) | 摘要核对 | 骨架康复评估的特征学习、评估与反馈综述 | 待定位/核对 |
| R03 | 2025 | 综述 | [Machine Learning-Based Computer Vision for Depth Camera-Based Physiotherapy Movement Assessment: A Systematic Review.](https://doi.org/10.3390/s25051586) | 全文重点核对 | 核对纳入研究、数据类型、验证及泛化限制 | 待定位/核对 |
| R04 | 2024 | 综述 | [Technological advances in lower-limb tele-rehabilitation: A review of literature.](https://doi.org/10.1177/20556683241259256) | 全文重点核对 | 扩展到下肢、视觉及可穿戴方案 | 待定位/核对 |
| D01 | 2021 | 数据集 | [IntelliRehabDS (IRDS)—A Dataset of Physical Rehabilitation Movements](https://doi.org/10.3390/data6050046) | 已有本地原文与全量数据审计 | 真实患者正确/错误；原标签无具体错误类型 | 待定位/核对 |
| D02 | 2019 | 数据集 | [The KIMORE Dataset: KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical REhabilitation.](https://doi.org/10.1109/TNSRE.2019.2923060) | 本地PDF重点核对 | 治疗师EAAQ总分与分项；RGB需申请；数据未下载 | 待定位/核对 |
| D03 | 2018 | 数据集 | [A Data Set of Human Body Movements for Physical Rehabilitation Exercises.](https://doi.org/10.3390/data3010002) | 全文重点核对 | 10名健康人、10类动作；正确/错误；不把代理连续分当临床评分 | 待定位/核对 |
| D04 | 2024 | 数据集 | [A Medical Low-Back Pain Physical Rehabilitation Database for Human Body Movement Analysis](https://doi.org/10.1109/ijcnn60899.2024.10650036) | 全文重点核对 | KERAAL；错误类别、部位及时间标注；大量记录无标注，需核对计数差异 | [资源](https://github.com/nguyensmai/KeraalDataset) |
| D05 | 2024 | 数据集 | [FineRehab: A Multi-modality and Multi-task Dataset for Rehabilitation Analysis](https://doi.org/10.1109/CVPRW63382.2024.00324) | 全文及官方项目页重点核对 | FineRehab；多维分项评分；官方页仍写will be released soon，未确认公开下载 | [资源](https://bsu3dvlab.github.io/FineRehab/) |
| D06 | 2025 | 数据集 | [A Human Motion Data Capture Study The University of Liverpool Rehabilitation Exercise Dataset.](https://doi.org/10.1038/s41597-025-05099-1) | 全文重点核对 | UL-RED；速度与重复、跨传感器；未证实有临床质量标签 | 待定位/核对 |
| D07 | 2026 | 数据集 | [MobiPhysio: A 2D video dataset of physiotherapy exercises for AI-driven assessment and monitoring.](https://doi.org/10.1016/j.dib.2026.112635) | 全文重点核对 | MobiPhysio；RGB与专家评分；人数叙述有待核对 | 待定位/核对 |
| D08 | 2025 | 数据集 | [LLM-FMS: A fine-grained dataset for functional movement screen action quality assessment.](https://doi.org/10.1371/journal.pone.0313707) | 全文重点核对 | LLM-FMS；关键帧、分级和细粒度标注；不等于时序模型 | [资源](https://github.com/MarioTed/LLM-FMS) |
| D09 | 2024 | 数据集 | [REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods](https://doi.org/10.1007/978-3-031-75823-2_2) | 原始Zenodo元数据、文件清单及书目信息核对 | REHAB24-6；10人、6动作、1072次重复，分段与正确性标签；正文未读 | [资源](https://zenodo.org/records/13305826) |
| D10 | 2022/2023 | 方法/数据集 | [Fast and robust video-based exercise classification via body pose tracking and scalable multivariate time series classifiers](https://doi.org/10.1007/s10618-022-00895-4) | 书目信息及基准全文中的数据描述核对 | UCDHE相关：动作错误分类、二维骨架与时序分类器；原文摘要/全文待核对；2022在线/2023卷期 | 待定位/核对 |
| M01 | 2020 | 模型方法 | [A Deep Learning Framework for Assessing Physical Rehabilitation Exercises.](https://doi.org/10.1109/TNSRE.2020.2966249) | 摘要核对 | 深度自编码/GMM代理分数及监督预测；非独立临床评分 | 待定位/核对 |
| M02 | 2020 | 模型方法 | [VI-Net-View-Invariant Quality of Human Movement Assessment.](https://doi.org/10.3390/s20185258) | 全文重点核对 | VI-Net；RGB跨视角；QMAR为健康人模拟 | 待定位/核对 |
| M03 | 2022 | 模型方法 | [Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises.](https://doi.org/10.1109/TNSRE.2022.3150392) | 摘要核对 | ST-GCN适配连续评分；作为图模型起点 | 待定位/核对 |
| M04 | 2022 | 模型方法 | [EGCN: An Ensemble-based Learning Framework for Exploring Effective Skeleton-based Rehabilitation Exercise Assessment](https://doi.org/10.24963/ijcai.2022/511) | 全文重点核对 | EGCN；位置/朝向融合；其KIMORE分类用人群代理标签，不能等同逐次正确性 | [资源](https://github.com/bruceyo/EGCN) |
| M05 | 2023 | 模型方法 | [A Skeleton-Based Rehabilitation Exercise Assessment System With Rotation Invariance](https://doi.org/10.1109/TNSRE.2023.3282675) | 官方README及已有代码划分核对 | 旋转不变描述符；IRDS官方加载器未按人分组 | [资源](https://github.com/Kelly510/RehabExerAssess) |
| M06 | 2023 | 模型方法 | [D-STGCNT: A Dense Spatio-Temporal Graph Conv-GRU Network based on transformer for assessment of patient physical rehabilitation.](https://doi.org/10.1016/j.compbiomed.2023.107420) | 摘要核对 | D-STGCNT；图卷积、GRU、Transformer | 待定位/核对 |
| M07 | 2023 | 模型方法 | [A Contrastive Learning Network for Performance Metric and Assessment of Physical Rehabilitation Exercises.](https://doi.org/10.1109/TNSRE.2023.3317411) | 摘要核对 | 对比学习、评分与关节注意力差异 | 待定位/核对 |
| M08 | 2024 | 模型方法 | [LightPRA: A Lightweight Temporal Convolutional Network for Automatic Physical Rehabilitation Exercise Assessment.](https://doi.org/10.1016/j.compbiomed.2024.108382) | 摘要及仓库说明核对 | LightPRA；轻量TCN；README不能证明已复现 | [资源](https://github.com/SaraS92/LightPRA) |
| M09 | 2024 | 模型方法 | [EGCN++: A New Fusion Strategy for Ensemble Learning in Skeleton-Based Rehabilitation Exercise Assessment.](https://doi.org/10.1109/TPAMI.2024.3378753) | 摘要核对 | EGCN++；图模型集成的后续工作 | [资源](https://github.com/bruceyo/egcnplusplus) |
| M10 | 2024 | 模型方法 | [An Expert-Knowledge-Based Graph Convolutional Network for Skeleton- Based Physical Rehabilitation Exercises Assessment.](https://doi.org/10.1109/TNSRE.2024.3400790) | 摘要核对 | 专家知识图卷积、门控池化、Transformer | 待定位/核对 |
| M11 | 2024/2025 | 模型方法 | [Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives.](https://doi.org/10.1007/s11517-024-03177-x) | 全文及作者仓库README重点核对 | ReCon；多动作共享对比学习与IRDS到KIMORE迁移；2024在线/2025卷期 | [资源](https://github.com/shamarkus/ReCon) |
| M12 | 2024 | 模型方法 | [Hierarchical Contrastive Representation for Accurate Evaluation of Rehabilitation Exercises via Multi-view Skeletal Representations.](https://doi.org/10.1109/TNSRE.2024.3523906) | 摘要核对 | 位置与角度多视图、回归对比学习 | 待定位/核对 |
| M13 | 2025 | 模型方法 | [Optimized assessment of physical rehabilitation exercises using spatiotemporal, sequential graph-convolutional networks.](https://doi.org/10.1016/j.compbiomed.2024.109578) | 摘要核对 | 轻量空间/时间图模型、动作分类后评分 | 待定位/核对 |
| M14 | 2025 | 模型方法 | [Frame topology fusion-based hierarchical graph convolution for automatic assessment of physical rehabilitation exercises.](https://doi.org/10.1038/s41598-025-12020-8) | 全文重点核对 | FTF-HGCN；逐帧拓扑与多尺度时序；评分定义需独立核对 | 待定位/核对 |
| M15 | 2025 | 模型方法 | [Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises.](https://doi.org/10.1109/ICORR66766.2025.11063192) | 摘要核对 | KERAAL Transformer错误分类；对反馈目标直接相关 | 待定位/核对 |
| B01 | 2025/2026 | 基准/评价 | [A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment Using Deep Learning](https://doi.org/10.1109/FG67764.2026.11556986) | 详细预印本及官方README重点核对 | Rehab-Pile；39分类+21回归任务；按人划分；版本须区分 | [资源](https://github.com/MSD-IRIMAS/DeepRehabPile) |
| B02 | 2026 | 基准/评价 | [Skeleton-based rehabilitation movement quality classification: a leakage-controlled benchmark on IntelliRehabDS.](https://doi.org/10.3389/fmed.2026.1878438) | 全文重点核对 | IRDS嵌套按人评估；简单特征与ST-SkelNet | 待定位/核对 |
| B03 | 2026 | 基准/评价 | [How improper dataset split hinders model generalizability: a systematic comparison in Human activity recognition and exercise evaluation tasks.](https://doi.org/10.1016/j.ijmedinf.2026.106464) | 摘要核对 | 直接比较按样本与按人划分造成的偏差 | 待定位/核对 |
| C01 | 2025 | 应用/可行性 | [Using deep learning to detect upper limb compensation in individuals post-stroke using consumer-grade webcams-A feasibility study.](https://doi.org/10.3389/fmed.2025.1645369) | 全文重点核对 | 真实卒中20人；跨人困难；数据未直接公开，不能作为现成训练源 | 待定位/核对 |

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
