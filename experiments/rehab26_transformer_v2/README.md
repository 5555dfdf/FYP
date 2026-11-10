# REHAB26 Transformer v2：实际训练与复现

2026-11-09 启动。训练使用 `conda activate fyp`，原始数据与已核验 `processed_v1` 缓存沿用 v1。训练/验证/测试始终按人分组；外层 5 人留一测试，每个外层有 4 个内层验证者。两种动作分别训练、每动作五个多标签错误。

本轮授权包含依赖安装、实现代码、实际训练及自主验证集调参。参考设计见 `docs/TRANSFORMER_EXPERIMENT_DESIGN_zh.md`；精确运行清单在本目录 `protocol.json` 中运行前冻结，记录 UTC 时间。原始设计草案保留，不把本轮结果反写成事先已知的最优配置。

```powershell
& 'D:\conda\shell\condabin\conda-hook.ps1'
conda activate fyp
python -m pip install scikit-learn matplotlib
python -X utf8 experiments/rehab26_transformer_v2/suite.py smoke
python -X utf8 experiments/rehab26_transformer_v2/verify.py
python -X utf8 experiments/rehab26_transformer_v2/parallel_run.py all
python -X utf8 experiments/rehab26_transformer_v2/finalize.py
```

本轮实际入口为 `parallel_run.py all`：先完成单因素筛查，再执行 `iterate.py` 的验证驱动组合迭代、三种子确认及外层测试。两个本地进程只并行独立拟合，不改变科学协议。`suite.py all` 则是**不包含第二轮组合迭代**的基础流程，不能用它覆盖本轮最终测试输出。

screen 期间只计算内层验证，不生成外层测试分数；筛查后保存 `selection.json`，组合规则在 `iteration_plan.json` 中冻结，组合验证后保存 `selection_final.json`，之后再补种子。额外种子不能根据测试重新选配置。每次拟合保存训练曲线、权重、训练标准化统计、正类权重、验证预测和两套验证阈值。确认范围是每外层按验证选择的 Transformer、TCN，以及角度 LR/RF；LR 确定性只运行一次，不制造三种子方差。

主选择指标为四名内层验证者 macro-AP 的均值；平局取验证 macro-F1，再按候选名固定排序。主阈值按验证 F1 选，附加 F0.5 阈值只作为偏向 precision 的预设工作点。四模型集成使用 `mean(p - threshold)>=0`；用于 AP 的排序分数是均值原始 sigmoid。没有置信度校准或拒判承诺。

神经模型使用相同训练预算、数据与阈值规则，但 TCN 本身为 ReLU+GroupNorm，Transformer 为 GELU+LayerNorm，首轮比较的是整个架构方案。候选包括参考 Transformer、xyz、两种类别加权、ReLU、Post-LN、去标准化、去 dropout、去权重衰减、可见度、较高学习率、较小网络，以及 TCN xyz/angles、角度 LR/RF，共 16 种。固定一个种子筛查，共 640 次拟合；第二轮组合数量由预定规则和各外层内层验证证据决定，完整清单保存到 `iteration_plan.json`。按每外层选择配置后，仅入选神经方法和 RF 扩展至三种子，共额外 240 次拟合。若不同外层选择同一配置，其测试人仍隔离，不能合并成无分组训练。

`finalize.py` 会检查已保存权重、生成报告与图表、建立每个外层的 12 模型推理 bundle，并根据验证证据生成全五人重训的三种子研究权重。全数据研究权重不继承外层独立测试成绩，阈值也需要后续重新验证。`infer.py` 可接受单个模型目录或 bundle.json；原始输入为 MediaPipe world landmarks `[frames,33,4]`，起止帧采用零基、包含末帧。

所有结果保存在 `runs/rehab26_transformer_v2/`，数据和权重不会提交 Git。运行身份绑定核心模型、训练代码、协议和输入数据哈希；不兼容的代码/配置不能静默复用旧结果。训练任务可断点续跑。PyTorch 确定性模式开启，关闭 TF32、AMP、Flash/Memory-efficient attention 与 eval fastpath，使用 math SDPA；实际依赖冻结在运行目录。

当前五人都已经出现在较早探索实验中，本轮属于探索性研究。即便新一轮严格嵌套选模型，也不能恢复成完全未经查看的新测试人；不据此宣称真实患者或临床疗效。逐摄像头观察相关，不能按 3333 个独立个体做显著性推断。
