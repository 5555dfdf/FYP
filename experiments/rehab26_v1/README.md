# REHAB26 第一轮实际训练实验（历史基线与共享预处理）

输入为公开数据中的三维骨架与整次动作错误标签。先研究肩外展（ex1）及肩后伸（ex7），按人划分，单视角输入，多标签预测。主要与附加错误均保留。视频、自动动作选择和自动分段不属于本轮实验。

实验配置在训练前写入 [protocol.json](protocol.json)。[RESULTS_zh.md](RESULTS_zh.md) 是全量实验结束后生成的实际结果，未生成前不代表实验完成。全部原始数据、缓存、模型与完整预测留在被 Git 忽略的 `data/` 和 `runs/`。

当前v3继续使用本目录的`pipeline.py`与`processed_v1`缓存，不能整目录移除。以下记录v1原始CPU运行环境；当前Conda环境与完整依赖顺序见[复现说明](../../docs/REPRODUCIBILITY_zh.md)。

## 环境

本机为 Windows、Python 3.13，首轮使用 CPU。独立环境名为 `.venv-rehab26`，未修改系统安装的机器学习包。

```powershell
python -m venv --system-site-packages .venv-rehab26
./.venv-rehab26/Scripts/python.exe -m pip --isolated install "scikit-learn>=1.6,<2"
./.venv-rehab26/Scripts/python.exe -m pip --isolated install torch --index-url https://download.pytorch.org/whl/cpu
```

上面命令用于建立环境；首次实际使用的精确包版本另存于 `requirements-lock.txt` 和运行目录的 `environment.json`。NumPy、SciPy、joblib、matplotlib 为本机已有依赖。没有 GPU 的机器也可运行本轮实验。

## 复现顺序

先按 [数据审计说明](../../docs/REHAB26_AUDIT_zh.md)准备原始文件，再运行：

```powershell
./.venv-rehab26/Scripts/python.exe experiments/rehab26_v1/prepare.py
./.venv-rehab26/Scripts/python.exe experiments/rehab26_v1/train.py
./.venv-rehab26/Scripts/python.exe experiments/rehab26_v1/analyze.py
./.venv-rehab26/Scripts/python.exe experiments/rehab26_v1/verify.py
```

`prepare.py` 不覆盖已有的完整 `processed_v1` 数据集；改预处理应另建版本。`train.py` 校验协议、代码和样本哈希，已完成的模型会跳过；若代码或数据变化，需换用新运行目录，不可混合结果。原始骨架从未改写。

`train.py --models angle_logistic --exercises 1 --folds 0` 可用于指定配置，但完整报告要求训练前声明的50组拟合全部结束。默认目录为 `runs/rehab26_v1_20261109_cpu/`。

## 运行目录

每种动作、每名测试者、每种模型均单独保存：

```text
runs/rehab26_v1_20261109_cpu/
  identity.json                 # 协议、训练代码、预处理、样本哈希
  protocol.json
  environment.json
  ex1|ex7/
    fold0_Honza|.../
      angle_logistic|angle_random_forest|tcn_.../
        model.joblib|model.pt   # 实际拟合出的分类器或时序模型权重
        result.json             # 训练记录、早停轮数、验证阈值、测试结果
        predictions.npz         # 验证/测试分数、真值与样本索引
  analysis/
    summary.json
    RESULTS_zh.md
    failure_cases.csv
    baseline_comparison.png
```

保存的是交叉验证各折的模型，没有把同样5人全部重新拟合成一个“已独立验证的最终模型”。测试成绩只对应各折训练/验证/测试规则。推理接口可以加载任一折，但部署模型选择与新人的适用性仍需额外验证。

## 原始骨架推理

例如，以下对 Honza 的一段肩外展执行其留人测试模型的推理，起止帧来自公开标注：

```powershell
./.venv-rehab26/Scripts/python.exe experiments/rehab26_v1/infer.py --artifact runs/rehab26_v1_20261109_cpu/ex1/fold0_Honza/angle_logistic/model.joblib --skeleton data/REHAB26-ViewAngles/raw/mediapipe/ex1/ex1_Honza_cam1_3d.npy --first 8 --last 158 --output runs/rehab26_v1_20261109_cpu/example_prediction.json
```

输出五种错误的分数、验证集选出的分类阈值、原始错误预测与可靠性状态。分数不视为校准概率；没有错误预测不等于确认动作医学上正确。验证集上没有合格可靠性规则时，接口会明确返回“不支持可靠性规则”，而不是保证警示可信。

当前没有逐帧错误定位、视频上传、RGB叠加或专家复核的矫正提示。`verify.py` 会检查全部50个模型重新加载后的预测与落盘记录一致，并从原始骨架重新处理片段进行推理核验。
