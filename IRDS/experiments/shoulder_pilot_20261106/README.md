# 肩前屈基线脚本草稿

2026-11-08 发布整理时，本目录已有 `run_pilot.py` 和本地虚拟环境，未发现 `results/`、运行报告、外折预测或模型文件。**当前作为已有代码草稿保留，不能当作实验已经成功运行的证据。** 本次仅检查 Python 语法，未执行训练，也未完成方法审查。

脚本设计：使用 IRDS Gesture 2/3 的 692 次动作，以错误为阳性；按文件受试者 ID 固定五折，比较多数类、帧数 Logistic Regression、几何特征 Logistic Regression/Random Forest 等。它还计划输出按人结果、bootstrap 区间与参考偏离提示。

此范围是早期试验候选，不是最终 FYP 选题。原始数据和 IRDS 的身份、左右语义、跟踪质量限制仍适用；参考偏离不是已验证的错误原因。源代码内的报告日期是预写模板，只有真实运行产物才可用于报告实验日期与成绩。

若后续决定验证该草稿，先取得 [IRDS 原始数据](https://doi.org/10.5281/zenodo.4610859)，按根目录[复现说明](../../../docs/REPRODUCIBILITY_zh.md)放置，然后从仓库根目录准备独立环境并运行：

```bash
python -m pip install -r IRDS/experiments/shoulder_pilot_20261106/requirements.txt
python IRDS/experiments/shoulder_pilot_20261106/run_pilot.py
```

依赖范围并非已锁定环境。执行前应审查划分、特征、训练与最终演示模型的区别；执行后核对每折受试者、指标、输出和运行环境，再决定是否收录结果。

`results/`、本地 `.venv/` 与序列化模型暂时排除提交；需要正式发布实验结果时，应显式选择经过核对的报告、配置与预测文件。
