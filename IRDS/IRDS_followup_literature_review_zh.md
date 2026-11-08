# IRDS 后续训练与评估研究调研

调研日期：2026-10-17。

调研问题：IntelliRehabDS（IRDS）发布后，是否有研究实际使用该数据集训练或评估模型？这些研究对我们的动作选择、错误检测和评价设计有什么帮助？

## 1. 主要结论

**有。IRDS 已被用于康复动作整体正确性分类、重复动作分割与计数、跨人泛化、公平性及统一基准评测。** 因此，“使用 IRDS 训练一个正确／错误分类器”本身已经有较多先例。

本次沿原始数据集论文的引用链检索，并通过 Semantic Scholar、Crossref、Europe PMC、论文全文和官方代码进行交叉核对。下表筛选出八项与课题相关、明确使用 IRDS 训练或评估模型的研究。其中六项进一步核对了全文或官方实现，两项目前确认到摘要层面。该清单是相关文献筛选结果，不代表穷尽全部后续论文。

与我们现阶段目标最相关的认识是：

1. **必须区分动作类别识别与动作正确性评估。** 两种任务的准确率不能混用。
2. **必须区分按动作样本划分与按受试者划分。** 同一人的重复动作进入训练和测试，会使对新患者的泛化能力难以得到可靠评价。
3. **正确／错误二分类不能直接验证具体纠错原因。** 模型判错、注意力热图和临床错误解释之间，需要额外证据连接。
4. **不同论文的高分不宜直接排名。** 数据筛选、任务定义、模型按动作单独训练还是跨动作统一训练、划分及指标汇总方式都可能不同。

## 2. 已确认使用 IRDS 的论文清单

| 编号 | 年份与发表渠道 | 论文 | IRDS 的实际用途 | 本次核对程度 |
|---|---|---|---|---|
| A | 2023，IEEE Transactions on Neural Systems and Rehabilitation Engineering（TNSRE） | [A Skeleton-Based Rehabilitation Exercise Assessment System With Rotation Invariance](https://doi.org/10.1109/TNSRE.2023.3282675) | 骨架动作正确性分类，旋转不变描述符及可视化解释 | 摘要＋官方代码 |
| B | 2023，Conference on Robots and Vision（CRV） | [Rehabilitation Exercise Repetition Segmentation and Counting Using Skeletal Body Joints](https://doi.org/10.1109/CRV60082.2023.00044) | 训练序列网络进行动作重复分割与计数 | 全文＋代码链接 |
| C | 2024，Medical & Biological Engineering & Computing | [Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives](https://doi.org/10.1007/s11517-024-03177-x) | ST-GCN 与监督对比学习；跨动作类型质量评估；IRDS 到 KIMORE 的迁移 | 全文 |
| D | 2024，Computers in Biology and Medicine | [Promoting fairness in activity recognition algorithms for patient’s monitoring and evaluation systems in healthcare](https://doi.org/10.1016/j.compbiomed.2024.108826) | 研究训练人群组成、身体特征和残障属性对动作识别的影响 | 摘要 |
| E | 2025，International Conference on Intelligent Computer Communication and Processing（ICCP） | [Comparative Analysis of Deep Learning Models for Human Movement Assessment in Physical Rehabilitation](https://doi.org/10.1109/ICCP68926.2025.11427109) | 比较 BiLSTM、LSTM-CNN、FCN、ResNet、InceptionTime 的正确性分类及划分方式 | 全文 |
| F | 2026，IEEE International Conference on Automatic Face and Gesture Recognition（FG）；2025 年已有预印本 | [A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment Using Deep Learning](https://doi.org/10.1109/FG67764.2026.11556986) | 将 IRDS 纳入 Rehab-Pile，构建按动作二分类任务，跨人比较九种深度学习架构 | 全文＋官方代码 |
| G | 2026，International Journal of Medical Informatics | [How improper dataset split hinders model generalizability: a systematic comparison in Human activity recognition and exercise evaluation tasks](https://doi.org/10.1016/j.ijmedinf.2026.106464) | 在 IRDS 和 NTU RGB+D 120 上训练共 12 种模型，比较按样本和按人划分 | 摘要 |
| H | 2026，Frontiers in Medicine | [Skeleton-based rehabilitation movement quality classification: a leakage-controlled benchmark on IntelliRehabDS](https://doi.org/10.3389/fmed.2026.1878438) | 统一按人划分和嵌套交叉验证，比较传统特征模型与 ST-SkelNet | 全文 |

“摘要＋官方代码”表示本次未完整核对该论文全文；“全文＋代码链接”表示核对了正文及其提供的代码地址，不表示已经运行或全面审查实现。

## 3. 重点论文阅读与判断

### A. 旋转不变的骨架康复动作评估（2023）

**作者**：Kaili Zheng、Ji Wu、Jialin Zhang、Chenyi Guo。

**研究问题**：人体朝向相机的方向变化，会改变原始关节坐标，但动作质量判断不应仅因为坐标朝向改变而变化。此外，分类结果需要一定的可视化解释。

**方法**：提出基于人体骨架点积矩阵的旋转不变描述符，结合骨架评估模型；使用 Grad-CAM 展示关注位置，并提出 Overlap Ratio（OvR）评价可视化结果。

**实际使用数据的证据**：官方仓库 README 明确说明使用 IRDS、UI-PRMD 和作者自建的俯卧撑数据集；仓库包含 IRDS 数据加载器、训练命令及预训练模型入口。

**重要的划分发现**：本次检查的官方 IRDS 加载器按照排序后的文件序号，每四个文件取三个训练、一个测试，没有按照受试者 ID 分组。这套划分不能直接证明模型对未见患者的泛化能力。

**对我们的帮助**：可借鉴旋转处理、骨架建模和解释方法，也适合作为可复现模型候选。正式实验应另建按人划分。Grad-CAM 显示模型关注部位，不能自动等同于已经验证的临床错误原因。

资源：

- [论文 DOI](https://doi.org/10.1109/TNSRE.2023.3282675)
- [官方代码与预训练模型入口](https://github.com/Kelly510/RehabExerAssess)
- [IRDS 划分实现](https://github.com/Kelly510/RehabExerAssess/blob/main/mvn/datasets/IRDS.py#L58-L76)

### B. 基于骨架的动作重复分割与计数（2023）

**研究问题**：真实患者的动作速度、幅度和重复间隔不规则，如何从骨架序列中找到每次动作的边界并计数？

**方法**：比较以 LSTM 为核心的序列网络，包括逐时刻二元输出、密度图输出，以及直接预测重复次数的输出形式；输入可为骨架坐标、手工特征或二者组合。

**IRDS 结果**：正文报告密度图方法在 IRDS 上的总体重复分割 IoU 约为 **0.68**，平均帧边界误差 MAE-F 约为 **39 帧**。这些是分割指标，不能当成正确性分类成绩。

**数据组织限制**：论文将 IRDS 描述为 361 条包含多次重复的序列，平均每条约 5.69 次重复。这与我们本地按次切分的 2,589 个文件不同。复现前必须查明这些长序列如何取得或组织，不能把单次动作文件直接当作同样的计数任务。正文提及五折交叉验证，本次未确认其 IRDS 实验是否严格按人划分。

**对我们的帮助**：如果课题以后扩展到完整训练过程的自动分割和计数，可作为方法入口；现阶段整体正确性任务不必立即加入这一模块。

资源：

- [论文 DOI](https://doi.org/10.1109/CRV60082.2023.00044)
- [可访问全文](https://arxiv.org/html/2304.09735)
- [代码及标注资源](https://github.com/abedicodes/repetition-segmentation)

### C. 使用困难和较容易负样本的监督对比学习（2024）

**作者**：Mark Karlov、Ali Abedi、Shehroz S. Khan。

**研究问题**：每种康复动作的样本有限，为每种动作分别训练模型会减少可利用的数据，也增加模型数量。

**方法**：以 ST-GCN 为编码器训练统一的动作质量表示，通过以下样本关系建立监督对比学习：

- 同一种动作的正确执行构成正样本关系。
- 同一种动作的正确与错误执行构成困难负样本关系。
- 不同动作之间的样本构成较容易的负样本关系。

作者还把 IRDS 上预训练的编码器迁移至 KIMORE，替换输出部分以适配连续评分任务。

**报告结果**：可访问全文 Table 1 中，所提方法在 IRDS 九类动作上的平均准确率为 **99.20%**，加入旋转不变描述符的变体为 **99.02%**。

**解释边界**：这项比较沿用了 Zheng 等研究的 3∶1 训练／验证划分。不能将其直接表述为“对未见患者有 99.20% 准确率”。论文还报告其他交叉验证实验，各表的实验设置需要分别核对，不能混为一个成绩。

**对我们的帮助**：适合研究先使用多个动作学习骨架表示、再聚焦肩前屈的策略。但是否优于仅用肩前屈训练，需要在相同的按人划分下验证。IRDS 上的二分类预训练也不意味着 IRDS 本身具有连续质量评分标签。

资源：

- [期刊论文 DOI](https://doi.org/10.1007/s11517-024-03177-x)
- [本次核对的可访问全文及表格](https://arxiv.org/html/2403.02772)

### D. 动作识别中的公平性与人群组成（2024）

**作者**：Ciro Mennella、Massimo Esposito、Giuseppe De Pietro、Umberto Maniscalco。

**使用 IRDS 的方式**：摘要明确说明在 IRDS 上比较处理多通道时间序列的 tCNN、混合 LSTM、Transformer 等模型，并有意改变训练数据中的人群组成，研究身体特征和残障属性对算法表现的影响。

**摘要报告**：CNN-LSTM 的最高准确率达到 **88%**。摘要还报告考虑残障属性异质性的训练分布，在相关评估中达到 51%，而未考虑时平均为 33%。这些数字对应不同实验条件，不能合并解释，也不能直接当作我们的整体正确性分类基线。

**对我们的帮助**：可用于说明为什么训练集需要覆盖不同功能状态，以及为什么仅有总准确率不足以证明系统适用于不同患者。

**本次边界**：核对到摘要，尚未完整核对具体任务输出、分组定义和各成绩对应的划分方式。

资源：[论文 DOI](https://doi.org/10.1016/j.compbiomed.2024.108826)。

### E. 五种深度模型的康复动作正确性比较（2025）

**作者**：Camelia Lazar、Crina Grosan、Maria Iuliana Bocicor。

**方法与任务**：在 IRDS 和 UI-PRMD 上进行整次动作正确／错误分类，比较 BiLSTM、LSTM-CNN、FCN、ResNet、InceptionTime。

**评价设计**：

- 混合受试者的样本划分：80% 用于训练部分，20% 用于测试；训练部分再留出验证数据。IRDS 使用分层划分。
- 留一受试者测试（LOSO）：每次将一人的全部动作用于测试，其余人用于训练。

**IRDS 报告结果**：正文 Table II 中，混合受试者划分下 BiLSTM 的准确率为 **96.0%**；LOSO 下 LSTM-CNN 的准确率为 **98.0%**、F1 为 **0.974**。这些是论文报告值，本次没有运行代码复现，不应据此断定 LSTM-CNN 在其他协议下也一定能达到该水平。

**重要观察**：

- 正文明确指出公开版本有 **30 个个体**，而原始论文写 29 人，并称使用了全部数据。这与我们本地发现 30 个唯一 ID 相呼应，但没有解决 ID 217 的具体身份问题。
- 论文同时报告混合受试者与 LOSO，值得借鉴；其较高的 LOSO 成绩应结合完整预处理、模型选择、验证受试者划分及按人指标汇总方式进一步复核。
- 正文和比较表中存在需要核对的数字不一致。例如 Table IV 对 Zheng 与 Karlov 两项工作的准确率对应关系，与本次核对的 2024 年论文 Table 1 不一致。因此本报告不将该比较表直接当作可靠排名。

**对我们的帮助**：适合建立 BiLSTM、LSTM-CNN 和通用时序卷积模型的基线列表，并说明为何需同时观察准确率、错误召回率和 F1。

资源：

- [会议论文 DOI](https://doi.org/10.1109/ICCP68926.2025.11427109)
- [作者机构提供的论文版本](https://kclpure.kcl.ac.uk/ws/files/350856938/Gesture_Correctness_ICCP.pdf)

### F. Rehab-Pile／DeepRehabPile 统一基准（2025 预印本／2026 FG）

**作者**：Ali Ismail-Fawaz、Maxime Devanne、Stefano Berretti、Jonathan Weber、Germain Forestier。

**研究问题**：康复评估研究使用的数据集、预处理、划分和指标不统一，方法之间难以公平比较和复现。

**方法与产出**：将多个康复数据来源整理为 Rehab-Pile，对分类与回归任务进行统一评测，比较九种深度学习架构，公开实现、数据整理说明和结果。IRDS 被组织为九个按动作进行的正确／错误二分类任务。

**与我们相关的评价设计**：采用跨受试者划分，明确训练和测试不能出现同一人；条件允许时，测试集仅包含患者样本。

**对我们的帮助**：优先参考其按人划分、统一基线和复现框架。它可以帮助我们比较简单时序模型、卷积模型和骨架模型，避免只挑选一个复杂模型作为唯一方案。

**限制**：其整理后的 IRDS 各动作样本数与我们当前原始数据统计并不完全一致，需要先核实筛选、缺失值处理等规则。不能直接混用处理后版本与原始版本的结果。预印本早期题名和正式会议题名不同，引用时应使用对应版本的信息。

资源：

- [正式论文 DOI](https://doi.org/10.1109/FG67764.2026.11556986)
- [详细预印本](https://arxiv.org/html/2507.21018)
- [官方代码](https://github.com/MSD-IRIMAS/DeepRehabPile)
- [数据与结果项目页面](https://msd-irimas.github.io/pages/DeepRehabPile/)

### G. 数据划分如何影响泛化能力（2026）

**研究问题**：不按人划分的测试是否会高估康复动作模型表现？这种影响是否与模型或任务复杂度有关？

**使用 IRDS 的方式**：在 IRDS 和 NTU RGB+D 120 上训练共 12 种机器学习和深度学习模型，通过模拟比较 noncross-subject（NCS，受试者可重叠）和 cross-subject（CS，受试者不重叠）划分，并分析预测方差。

**摘要结论**：NCS 划分会高估表现，差异随任务和模型复杂度增加；受试者差异会影响模型稳定性。

**对我们的帮助**：是“为什么必须按人测试”的直接方法学依据，与我们观察到的受试者和错误分布混杂高度相关。

**本次边界**：核对到摘要，尚未核对全部模型清单及各实验数值，不在此提供未经确认的性能差值。

资源：[论文 DOI](https://doi.org/10.1016/j.ijmedinf.2026.106464)。

### H. IRDS 整体正确性分类的防泄漏基准（2026）

**研究问题**：在相同的按人划分、预处理和模型选择框架下，传统特征模型与骨架序列模型表现如何？

**数据与预处理**：排除 12 条特殊标签样本和 2 条不足 16 帧的序列，保留 2,575 条；包括根节点中心化、躯干尺度归一化、Savitzky–Golay 平滑和 64 帧重采样等。

**评价设计**：外层按人五折、内层按人三折交叉验证；传统模型的缩放和 SMOTE 仅在相应训练部分拟合；ST-SkelNet 使用类别加权损失。错误执行为阳性类别。

**正文 Table 3 的报告结果**：

| 模型 | 准确率 | 错误动作召回率 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 81.8% | 61.2% | 0.776 |
| Random Forest | 83.7% | 39.8% | 0.861 |
| Gradient Boosting | 85.1% | 51.1% | 0.823 |
| SVM（RBF） | 85.5% | 62.9% | 0.846 |
| MLP | 86.9% | 58.7% | 0.824 |
| ST-SkelNet | 84.6% | 61.2% | 0.855 |

以上为作者报告值，不是本次复现结果。正文另有汇总的外层预测及类别失衡指标表，使用时应保留对应的汇总定义。

**重要认识**：

- 总准确率达到 80% 多，并不意味着大多数错误都能被检出。
- ST-SkelNet 没有全面超过传统特征模型；不能先假定复杂骨架网络一定最好。
- 随机森林的特征重要性中，原始序列帧数排在首位，支持我们保留仅使用帧数／时长的简单基线。

**需要复核的限制**：文章仍将参与者写作 29 人，没有解释我们本地的 30 个唯一 ID；它还使用成人／青少年等描述，而原始论文的年龄表未提供青少年参与者证据。因此应独立核对人口学和具体数据清单。按人协议的文字描述也不代替代码复现。

资源：

- [论文 DOI](https://doi.org/10.3389/fmed.2026.1878438)
- [可访问全文及结果表](https://pmc.ncbi.nlm.nih.gov/articles/PMC13416899/)

## 4. 发现的一项明显可疑的数据集描述

论文：[Spatiotemporal skeleton based learning for automatic exercise quality assessment and real time posture correction](https://doi.org/10.1038/s41598-026-52118-1)，2026 年发表于 Scientific Reports，提出 Anatomical-Prior Sparse Attention（APSA），声称在 IRDS 和 KIMORE 上实验。

本次核对全文后发现：

| 文中描述 | 与原始 IRDS 的冲突 |
|---|---|
| Table 4 将 IRDS 动作列为深蹲、弓步、俯卧撑、平板支撑、波比跳 | 原始 IRDS 是肘屈伸、肩前屈、肩外展、双臂前举和侧点地等九类动作 |
| Figure 6 使用 IRDS 五级质量评分 | 原始数据主要是正确／错误标签，另有特殊标签 3，并非五级质量量表 |
| Table 3 报告膝内扣等具体错误类型检测 | 原始 IRDS 没有提供这些逐类错误标签 |

在额外动作、质量评分和错误类型真值的来源得到充分解释前，**不建议将其成绩作为本课题的可靠基线**。这里指出的是可核查的内容不一致，不据此推断作者意图。

资源：[可访问全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13357749/)。

## 5. 对我们现阶段四项成果的帮助

| 阶段成果 | 本次调研提供的帮助 | 仍需完成的工作 |
|---|---|---|
| 数据集比较表 | 确认 IRDS 已支持分类、计数及跨人研究，并有多种复现资源 | 独立核对 KIMORE、UI-PRMD 等数据的实际标签与获取情况 |
| 核心论文阅读记录 | 提供八项后续工作及其任务、方法、证据和限制 | 将摘要级确认的 D、G 两项推进到全文阅读；深入检查优先方法的实现 |
| 数据检查结果 | 外部论文 E 同样指出公开版本为 30 个个体；多篇研究强调划分风险 | 澄清 ID 217、左右约定和不同整理版本的样本筛选差异 |
| 初步课题范围 | 整体正确性已有较多工作；具体反馈验证和按人泛化更值得聚焦 | 根据补充错误标注的可行性确定最终任务 |

### 建议阅读顺序

1. **F：Rehab-Pile／DeepRehabPile**——了解跨人划分和统一基线，优先判断能否复用实验框架。
2. **H：IRDS 防泄漏分类基准**——理解准确率与错误检出能力的差别，设计传统特征基线和评价指标。
3. **A：旋转不变评估**——学习骨架表示、方向处理和可视化，但替换其按文件划分。
4. **C：监督对比学习**——研究多动作共同学习及向肩前屈迁移的可能性。
5. **G、D：划分与公平性**——为受试者分组、患者子集和体位分层评价提供依据。
6. **B：重复分割与计数**——只有在扩大到连续训练序列时再优先研究。

### 初步实验建议

- 以肩前屈一次重复的整体正确／错误判断作为可直接启动的任务。
- 先建立多数类、仅帧数／时长、几何特征模型及简单时序网络基线，再加入 ST-GCN 或对比学习。
- 所有模型使用同一份明确记录的受试者划分，训练内部完成归一化、增广、类别平衡与参数选择。
- 以错误检出为重点报告 recall、precision/F1、macro-F1、balanced accuracy 和混淆矩阵，并给出患者及体位分层结果。
- 对具体错误反馈建立单独的验证标签，不能把整体分类命中或注意力图直接当作原因正确的证据。
- 不以不同论文的最高准确率直接设定目标；先核对任务、样本版本、划分、阳性类别和统计方式。

## 6. 调研范围与未完成事项

本次完成的是文献检索、证据筛选及部分全文／实现核对，没有训练、运行或复现上述模型，也未全面审计其代码。未将仅引用 IRDS 的综述或其他数据集论文算作实际使用案例。

检索还出现了 Point Cloud Transformer、重采样、联邦学习、动作生成等方向的候选论文，但本次未将它们全部推进到任务、训练数据与评价协议的可靠核对，因此不在已确认的八项清单中展开，也不据其摘要数字作性能判断。

后续最有价值的核对是：取得优先论文的具体样本清单及划分代码，弄清 30 个唯一 ID、数据筛选和左右约定，随后在我们的固定按人协议下复现一至两个基线。

## 7. 原始数据集来源

- Miron et al. *IntelliRehabDS (IRDS)—A Dataset of Physical Rehabilitation Movements*. Data, 2021, 6(5), 46. [DOI](https://doi.org/10.3390/data6050046)。
- [原始数据存储库](https://doi.org/10.5281/zenodo.4610859)。
- [原始读取与探索代码](https://github.com/alina-miron/intellirehabds)。

本调研结果应与本地数据检查结果结合使用；源数据和论文自身的叙述不一致时，保留两者证据，并明确尚未解决的部分。
