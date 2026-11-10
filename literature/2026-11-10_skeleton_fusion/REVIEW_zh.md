# GitHub 骨架融合实现适配核查

日期：2026-11-10。目标：在缺少可靠相机内外参、逐帧同步未核实的 REHAB26 上，先做基础三视图骨架融合，再复用既有分类实验。此次先读代码、判断输入假设，没有安装整套运动捕捉软件。

| 仓库 | 实际检查 | 适配结论 |
|---|---|---|
| [Yuri-Vlasqz/3D_Human_Pose_MediaPipe](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe) | `modules/pose_triangulation.py`、`models/cameras_parameters.py`、README | 33 点 MediaPipe；先用 K[R\|t] 和二维点进行相机对 DLT 三角测量，再用两视图最小 visibility 或其平方融合各对的三维结果。**没有 K/R/t 不能直接使用三角测量部分**；可借鉴逐关节加权、平方加权和均值对照。 |
| [perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim) | `Pose2Sim/triangulation.py`、`common.py`、LICENSE | 对二维检测执行置信度加权三角测量，配合重投影误差筛相机；入口需要标定 TOML。可作为以后标定重建的参考，目前不直接接入。 |
| [freemocap/freemocap](https://github.com/freemocap/freemocap) | `triangulate_simple.py` | 接口要求去畸变的归一化二维点和相机外参 `[R\|t]`。不等于对 MediaPipe world landmarks 求平均；当前缺少输入条件。只核对该基础函数，不声称完整审计其系统。 |
| [Fdevmsy/Multiview_3D_Skeleton_Fusion](https://github.com/Fdevmsy/Multiview_3D_Skeleton_Fusion) | README、完整 `mmview-merge.py` | README 写 heatmap-weighted，但脚本主流程实际逐帧刚体对齐，再用相对上次融合结果的距离加权；写死 17 点和两个视图，另有首帧重复/输出多一帧、零分母风险。没有核实到许可文件。不直接复制。 |

## 实际选择

采用独立实现的 NumPy 基础融合，借鉴公开代码中的逐关节 visibility² 权重形式，而不是复制项目源码：

1. 依原有动作起止帧，读取同一次动作的三路 MediaPipe world landmarks。
2. 每路独立减髋中点、按整次动作躯干长度缩放，重采样到 96 帧；按相同动作进度配对，不声称通过时间戳恢复同步。
3. 用整次动作的髋部水平轴估计一次固定 yaw 旋转，各路转到相同身体朝向。保持 y 轴和动作内旋转变化，不用逐帧全身 Procrustes 拟合；假设相机竖直方向近似一致，没有估计 roll/pitch 或测得真实重力方向。
4. 每帧、每关节按 `visibility² / sum(visibility²)` 融合；全零权重显式回退等权平均。融合后重新定中心和尺度。
5. 由融合骨架重新计算原有 12 个几何特征，供既有 LR/RF/TCN/Transformer 使用。

这一步与上述 MediaPipe 仓库有明确区别：**我们融合的是三个已经预测出的单目三维骨架；仓库先从标定二维点三角测量，再融合相机对结果。** 相同的权重公式不意味着相同的几何精度。

增加同预处理的等权融合，以及仅使用 yaw 对齐后 cam2 的参考 Transformer 对照，区分加权、融合和坐标对齐的影响。visibility 是可见度，不是独立三维准确度真值；多个视图共同出错或动作进度不一致时，均值可能稀释错误。结果要通过实际跨人分类验证。

## 来源固定

实际读取的树提交：

- MediaPipe triangulation：`d491ff9a6a2fa238e7b00dc58dbddbc4ff21e08b`
- Pose2Sim：`c6db0ca3c620a7543c5f7960c95309dad87d4e6c`
- FreeMoCap：`8055477c927ee73886ef8f2eed4d39b745ef69c6`
- 17 点融合脚本：`c64cfa533127f2d677ba7ed645894b40d427db6b`

请求地址、实际时间、文件哈希和失败记录保存在 `search_manifest.json`、`detail_manifest.json`、`code_manifest.json`。Raw GitHub 域名连接失败后，通过官方 GitHub API 的固定 blob 取得代码；没有把搜索描述当作实现证据。另定位 ATHENA，但未深入源码，不纳入上述适配结论。

本地源码缓存保存在忽略的 `sources/`，实验实现见 `experiments/rehab26_fusion_v3/`。

适配后的训练与对照已完成，结果见[实验结论](../../experiments/rehab26_fusion_v3/FINDINGS_zh.md)：可见度平方加权没有一致优于等权；GitHub中的权重形式需要在本数据上验证，不能仅凭形式合理就默认更好。
