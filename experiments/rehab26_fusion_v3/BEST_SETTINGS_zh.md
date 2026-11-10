# 当前最佳实验设定与来源

更新：2026-11-10。这里的成绩来自本仓库已完成实验，按当前测试macro-F1整理，不是引用论文中的模型成绩。肩外展、肩后伸分别训练与评价。

| 动作与范围 | 模型与输入 | macro-F1 | macro-AP | 误警 | 漏警 | 随机种子 |
|---|---|---:|---:|---:|---:|---|
| 肩外展：所有模型最高 | 角度RF，三视角概率平均，验证重选阈值 | **0.641 ± 0.005** | 0.669 | 54.3% | 15.7% | 3个 |
| 肩外展：Transformer最高 | 选定Transformer，三视角概率平均，验证重选阈值 | 0.570 ± 0.018 | 0.635 | 44.1% | 21.3% | 3个 |
| 肩后伸：当前最高 | 参考Transformer，三视角骨架等权融合 | **0.734** | 0.784 | 19.1% | 9.7% | 1个 |
| 肩后伸：已完成三种子最高 | 选定Transformer，三视角概率平均，验证重选阈值 | 0.690 ± 0.029 | 0.756 | 20.9% | 14.4% | 3个 |

±为训练种子样本标准差，不能当作人群置信区间。误警：正确动作被报告任一错误；漏警：有错误动作没有报告任何错误。当前最高分是查看已有测试后的描述性排名；没有据此继续修改本轮模型。肩后伸0.734仍需额外种子确认，肩外展RF虽然F1最高，误警仍达54.3%。

## 数据与评价来源

- **原始数据**：[REHAB26-ViewAngles，版本DOI 10.5281/zenodo.23018128](https://doi.org/10.5281/zenodo.23018128)。本轮使用5位志愿者、肩外展556次和肩后伸555次动作，每次有三个相机观察，保留主要与附加错误的多标签。
- **数据集方法论文**：Jánošová, Lang, Budikova, Sedmidubsky, *Impact of Patient Orientation in Single- and Multi-View Camera Environments for AI-based Rehabilitation Monitoring*, [arXiv:2609.35726](https://arxiv.org/abs/2609.35726)。原文任务与本仓库的跨人五错误分类不同，不能将其分数直接拼入本表；差异见[全文核查](../../literature/2026-11-09_rehab26_methods/review_memo_zh.md)。
- **本仓库完整汇总**：[results_summary.json](results_summary.json)的`aggregate`，按`exercise`、`method`、`variant`定位。紧凑证据见[best_results.json](best_results.json)，包含每个种子的指标及原汇总SHA-256。
- **划分**：外层5人留1人测试，内层其余4人轮流验证，每次3人训练、1人验证；同一人的动作及三个视角不跨组。
- **汇总**：先对每个种子合并五名外层测试者的预测计算逐类F1并取均值，再跨种子平均。不是平均五个受试者的F1，也不是准确率。
- **测试使用**：本轮主模型选择和全部固定输入对照训练在新测试预测前完成。历史v2测试已参与探索，当前仍是五名已知志愿者上的方法比较。

<a id="shoulder-extension"></a>

## 肩后伸最高分：等权骨架融合 + 参考Transformer

记录定位：`exercise=7, method=reference_transformer, variant=x_equal`；种子`20261109`。

| 项目 | 实际设定 |
|---|---|
| 三视图输入 | MediaPipe单目三维骨架，每路33点 |
| 预处理 | 每帧髋中心平移；整次动作躯干尺度归一化；96帧动作进度重采样；每路固定髋方向yaw对齐 |
| 融合 | 三路关节坐标等权平均，融合后重定中心与尺度；不是标定三角测量 |
| 网络特征 | 每帧57维坐标 + 12维几何特征 = 69维 |
| 标准化 | 每个内层仅用训练集拟合均值、标准差 |
| 编码器 | 2层Transformer，64维，4头，FFN 128维，正弦位置编码，GELU，Pre-LN，末尾LayerNorm |
| 池化与分类头 | 时间均值/最大池化拼接；128→32→5，GELU与dropout |
| 损失 | 无类别加权的BCEWithLogitsLoss；推理使用5个独立sigmoid |
| 正则化 | dropout=0.1；AdamW weight_decay=0.001，仅矩阵参数 |
| 优化 | lr=0.0003，batch=32，梯度范数裁剪1.0 |
| 调度 | 前5 epoch warmup，随后余弦下降至1e-5 |
| 停止 | 最多80 epoch；验证macro-AP早停，patience=12，min_delta=1e-4 |
| 测试预测 | 每个外层平均4个内层模型的概率及验证阈值 |

来源：[冻结配置](protocol.final.json)中的`candidates.tr_ref`与`training`；[融合实现](fusion.py)的`merge_aligned(method="equal")`；[共享模型](../rehab26_transformer_v2/core.py)；[共享训练](../rehab26_transformer_v2/train.py)；[输入对照运行](run.py)。

<a id="shoulder-abduction"></a>

## 肩外展最高分：角度RF + 概率融合

记录定位：`exercise=1, method=angle_rf, variant=late_calibrated`；种子`20261109 / 20261110 / 20261111`。

1. 复用v2模型：每路MediaPipe骨架做原v2中心/尺度归一化与96帧重采样，提取12维几何特征。
2. 每个几何特征取均值、标准差、最小/最大值、10%/90%分位数、平均绝对帧间变化，得到84维统计特征。
3. 每种错误单独训练RF：160棵树、`max_depth=8`、`min_samples_leaf=3`、`class_weight="balanced"`、`random_state=seed+错误类别索引`。
4. 每个外层先集成4个内层模型，再按同次动作平均3个相机概率。
5. 在各旧内层验证集平均三视角概率后，用原v2的F1规则在0.05–0.95、步长0.05的网格选逐类阈值，最后平均4个内层阈值。没有用测试标签选阈值。

来源：[RF训练实现](../rehab26_transformer_v2/train.py)的`fit_one`、[统计特征与阈值实现](../rehab26_transformer_v2/core.py)、[概率融合分析](analyze.py)的`collect`。

## 已完成三种子的Transformer方案

肩后伸记录：`exercise=7, method=selected_transformer, variant=late_calibrated`。复用v2各外层按验证选择的配置，完整名称和参数见[v2最终选择](../rehab26_transformer_v2/selection_final.json)与[v2最终配置](../rehab26_transformer_v2/protocol.final.json)；三个种子分别计算后报告均值及标准差，没有将三个种子的概率再合并。

## 核验与方法来源

本轮984个模型检查点重载通过，验证预测最大差4.44e-16；训练集标准化参数重算差0；10次原始三视图到外层四模型集成的最大分数差1.19e-7。见[核验记录](verification_models.json)、[完成记录](completion.json)及[完整报告](RESULTS_zh.md)。模型权重和原始数据保留本地。

融合方法检查了[MediaPipe三角测量实现](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe)、[Pose2Sim](https://github.com/perfanalytics/pose2sim)、[FreeMoCap](https://github.com/freemocap/freemocap)和[多视图骨架融合脚本](https://github.com/Fdevmsy/Multiview_3D_Skeleton_Fusion)。本轮中心/尺度/yaw对齐与等权/visibility²融合为独立实现，未直接接入缺少K/R/t输入的三角测量代码；源码适配与版本见[核查记录](../../literature/2026-11-10_skeleton_fusion/REVIEW_zh.md)。
