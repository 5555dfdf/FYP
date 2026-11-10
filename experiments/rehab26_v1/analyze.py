"""Summarize all predeclared held-out predictions without tuning any model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pipeline import ROOT, HERE, metrics, reliability_margin, save_json, sha256


MODEL_NAMES = {
    "angle_logistic": "角度统计＋逻辑回归",
    "angle_random_forest": "角度统计＋随机森林",
    "tcn_xyz": "TCN：坐标",
    "tcn_xyz_angles": "TCN：坐标＋角度",
    "tcn_xyz_angles_visibility": "TCN：坐标＋角度＋可见度",
}


def percentage(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/rehab26_v1_20261109_cpu")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    base = ROOT / "data/REHAB26-ViewAngles/processed_v1"
    metadata = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
    observations = json.loads((base / "observations.json").read_text(encoding="utf-8"))
    folds = json.loads((base / "folds.json").read_text(encoding="utf-8"))
    entries, failure_cases = [], []
    total_training_seconds = 0.0
    for ex in protocol["exercises"]:
        labels = metadata["labels"][str(ex)]
        expected = {i for i, row in enumerate(observations) if row["exercise_id"] == ex}
        for model in protocol["models"]:
            truth, predictions, stress_predictions, retained, ids, scores, margins = [], [], [], [], [], [], []
            fold_results = []
            for fold in folds:
                directory = run / f'ex{ex}/fold{fold["fold"]}_{fold["test"][0]}/{model}'
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                with np.load(directory / "predictions.npz", allow_pickle=False) as p:
                    test_ids, y = p["test_indices"], p["y_test"]
                    score, stress_score, threshold, keep = p["test_scores"], p["stress_scores"], p["thresholds"], p["retained"]
                assert all(observations[i]["person"] == fold["test"][0] for i in test_ids)
                pred = score >= threshold
                truth.append(y); predictions.append(pred); stress_predictions.append(stress_score >= threshold)
                retained.append(keep); ids.extend(test_ids.tolist()); scores.append(score)
                margin = reliability_margin(score, threshold)
                margins.append(margin)
                fold_results.append({"test_person": fold["test"][0], "validation_person": fold["validation"][0],
                                     "n_physical_repetitions": result["n_test_physical_repetitions"],
                                     "unconditional": result["test_unconditional"], "stress": result["test_synthetic_noise"],
                                     "validation": result["validation"], "thresholds": result["thresholds"],
                                     "policy_qualified": result["reliability_policy"]["qualified_on_validation"],
                                     "policy_cutoff": result["reliability_policy"]["cutoff"],
                                     "test_coverage": result["test_coverage"], "retained": result["test_retained"],
                                     "best_epoch": result["best_epoch"], "n_parameters": result["n_parameters"],
                                     "elapsed_seconds": result["elapsed_seconds"]})
                total_training_seconds += result["elapsed_seconds"]
            assert len(ids) == len(set(ids)) and set(ids) == expected
            y, pred, stress_pred = map(np.concatenate, [truth, predictions, stress_predictions])
            keep = np.concatenate(retained)
            score, margin = np.concatenate(scores), np.concatenate(margins)
            rows = [observations[i] for i in ids]
            entry = {"exercise": ex, "model": model, "name": MODEL_NAMES[model],
                     "n_physical_repetitions": len({r["sample_id"] for r in rows}),
                     "unconditional": metrics(y, pred, labels), "synthetic_noise": metrics(y, stress_pred, labels),
                     "mean_fold_macro_f1": float(np.mean([r["unconditional"]["macro_f1"] for r in fold_results])),
                     "sd_fold_macro_f1": float(np.std([r["unconditional"]["macro_f1"] for r in fold_results], ddof=1)),
                     "n_qualified_reliability_folds": sum(r["policy_qualified"] for r in fold_results),
                     "mixed_policy_coverage": float(keep.mean()), "mixed_policy_retained": metrics(y[keep], pred[keep], labels),
                     "folds": fold_results, "by_camera": [], "by_relative_view": [], "by_visibility": [],
                     "by_multilabel": []}
            for camera in (1, 2, 3):
                selected = np.asarray([r["camera"] == camera for r in rows])
                entry["by_camera"].append({"camera": camera, **metrics(y[selected], pred[selected], labels)})
            for name, low, high in [("frontal_abs_angle_0_20", 0, 20), ("oblique_abs_angle_20_70", 20, 70), ("profile_abs_angle_70_90", 70, 91)]:
                selected = np.asarray([low <= abs(r["view_angle_degrees"]) <= high if low == 0 else low < abs(r["view_angle_degrees"]) < high if high == 70 else low <= abs(r["view_angle_degrees"]) < high for r in rows])
                entry["by_relative_view"].append({"group": name, **metrics(y[selected], pred[selected], labels)})
            for name, low, high in [("below_0.8", 0, .8), ("0.8_to_0.95", .8, .95), ("at_least_0.95", .95, 1.01)]:
                selected = np.asarray([low <= r["right_arm_mean_visibility"] < high for r in rows])
                entry["by_visibility"].append({"group": name, **metrics(y[selected], pred[selected], labels)})
            for name, selected in [("zero_or_one_error", y.sum(axis=1) <= 1), ("multiple_errors", y.sum(axis=1) > 1)]:
                entry["by_multilabel"].append({"group": name, **metrics(y[selected], pred[selected], labels)})
            failed = np.flatnonzero(~(y == pred).all(axis=1))
            distinct = set()
            for j in sorted(failed, key=lambda i: -margin[i]):
                row = rows[j]
                if row["sample_id"] in distinct:
                    continue
                distinct.add(row["sample_id"])
                failure_cases.append({"exercise": ex, "model": model, "observation_id": row["observation_id"],
                                      "person": row["person"], "camera": row["camera"],
                                      "relative_view_degrees": row["view_angle_degrees"],
                                      "true_errors": ",".join([label for label, on in zip(labels, y[j]) if on]),
                                      "predicted_errors": ",".join([label for label, on in zip(labels, pred[j]) if on]),
                                      "threshold_distance": float(margin[j]), "retained_by_policy": bool(keep[j]),
                                      "right_arm_visibility": row["right_arm_mean_visibility"],
                                      "skeleton_file": row["skeleton_file"], "first_frame": row["first_frame"], "last_frame": row["last_frame"]})
                if len(distinct) == 10:
                    break
            entries.append(entry)
    output = run / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": protocol["protocol_version"], "protocol_sha256": sha256(run / "protocol.json"),
               "n_completed_models": len(entries) * 5, "total_fit_and_evaluation_seconds": total_training_seconds,
               "evaluation_unit": "single-camera observations, three correlated views per physical repetition",
               "entries": entries}
    save_json(output / "summary.json", summary)
    with (output / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failure_cases[0]))
        writer.writeheader(); writer.writerows(failure_cases)
    report = ["# REHAB26 首轮按人测试实验结果", "", "实验日期：2026-11-09。所有结果来自实际训练与未见参与者测试；单随机种子的探索性基线，不是临床验证。", "",
              "## 实验协议", "", "肩外展与肩后伸分别预测五种错误，目标为主要及附加错误的并集。正确动作的五个标签均为零。", "",
              "共 1111 次物理动作、3333 个单摄像头观察。每个观察独立输入模型；没有把多相机预测融合成更容易的测试任务。5轮轮流留一人测试，另1人验证，余下3人训练；同一人的所有动作与视角同组。", "",
              "输入为已知起止帧内的三维骨架，按髋中点平移、躯干尺度归一化并重采样为96帧，不旋转身体。类别阈值和TCN早停只依赖验证者。没有将参与者、身体朝向、正确性或帧区间元数据作为模型特征。", "",
              "具体结构、超参数、扰动、划分、阈值和可靠性规则见训练前固定的 `experiments/rehab26_v1/protocol.json`。完整模型、日志、预测与输入哈希保存在本地 `runs/rehab26_v1_20261109_cpu/`。", "",
              "## 全量测试结果", "", "宏平均F1针对五种具体错误计算。下表主F1汇总所有留出观察；均值±标准差则在5名测试者之间计算，二者权重不同。标准差不是置信区间。摄像头观察相关，不能按3333个独立受试者推断统计显著性。", "",
              "| 动作 | 模型 | 汇总宏F1 | 5人宏F1均值±SD | 标签集合全对 | 正确动作误警率 | 错误动作无警率 |", "|---|---|---:|---:|---:|---:|---:|"]
    for e in entries:
        m = e["unconditional"]
        report.append(f'| ex{e["exercise"]} | {e["name"]} | {m["macro_f1"]:.3f} | {e["mean_fold_macro_f1"]:.3f} ± {e["sd_fold_macro_f1"]:.3f} | {percentage(m["exact_match"])} | {percentage(m["false_alert_rate"])} | {percentage(m["missed_alert_rate"])} |')
    report += ["", "错误动作无警率指存在任一错误却没有发出任何错误预测，不等于具体错误全部识别正确；具体错误误报/漏报见逐类表。", "",
               "## 本轮能得出的结论", "",
               "模型与权重已经实际得到，但当前误警和跨人变化较大，不能把这些模型当作已经可靠的康复警示系统。下面的数值是全部预定配置的探索性比较，不能按测试结果挑一个模型后再把同一测试集当作独立最终验证。", ""]
    for ex in protocol["exercises"]:
        group = [e for e in entries if e["exercise"] == ex]
        lookup = {e["model"]: e for e in group}
        best = max(group, key=lambda e: e["unconditional"]["macro_f1"])
        base = lookup["tcn_xyz"]["unconditional"]["macro_f1"]
        angle_f1 = lookup["tcn_xyz_angles"]["unconditional"]["macro_f1"]
        vis_f1 = lookup["tcn_xyz_angles_visibility"]["unconditional"]["macro_f1"]
        reference_exact = best["unconditional"]["n_correct"] / best["unconditional"]["n_observations"]
        report.append(f'- ex{ex} 本轮汇总宏F1最高的配置为{best["name"]}（{best["unconditional"]["macro_f1"]:.3f}），其正确动作误警率仍为{percentage(best["unconditional"]["false_alert_rate"])}。TCN坐标→加角度→再加可见度为 {base:.3f} → {angle_f1:.3f} → {vis_f1:.3f}，不能认定附加特征有一致收益。始终不报错的参考策略宏F1为0、标签集合全对率为{percentage(reference_exact)}；本轮模型的集合全对率也说明，不能只展示识别成功案例。')
    report += ["", "可靠性规则的跨人失败尤其需要注意：例如，若某折在验证者上达到预设要求，测试者上的保留预测仍可能大量出错。逐折覆盖率与保留指标见后表，不通过提高拒绝比例掩盖原始成绩。人工扰动下某些弱模型F1持平或略升，也不构成鲁棒性已经得到证明。", "",
               "## 每名测试者", "",
               "| 动作 / 模型 | Honza | Mima | Petra | Terka | Vasil |", "|---|---:|---:|---:|---:|---:|"]
    for e in entries:
        report.append(f'| ex{e["exercise"]} / {e["name"]} | ' + " | ".join(f'{f["unconditional"]["macro_f1"]:.3f}' for f in e["folds"]) + " |")
    report += ["", "## 逐类错误结果", "", "同一动作可有多个标签，因此各错误支持数之和不等于错误动作数。支持数按摄像头观察计，同一次动作可出现三次。", "",
               "| 动作 | 模型 | 错误 | 精确率 | 召回率 | F1 | 支持数 | FP | FN |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for e in entries:
        for c in e["unconditional"]["per_error"]:
            report.append(f'| ex{e["exercise"]} | {e["name"]} | {c["label"]} | {c["precision"]:.3f} | {c["recall"]:.3f} | {c["f1"]:.3f} | {c["support"]} | {c["false_positives"]} | {c["false_negatives"]} |')
    report += ["", "## 视角与自然可见度", "", "按官方相对视角分为正面（绝对角度≤20°）、斜侧面（20°<绝对角度<70°）、侧面（≥70°）。以下均为全量测试宏F1。", "",
               "| 动作 / 模型 | cam1 | cam2 | cam3 | 正面 | 斜侧面 | 侧面 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for e in entries:
        report.append(f'| ex{e["exercise"]} / {e["name"]} | ' + " | ".join(f'{r["macro_f1"]:.3f}' for r in e["by_camera"] + e["by_relative_view"]) + " |")
    report += ["", "右肩、右肘、右腕的时间平均visibility分组如下。各组的人、视角和标签分布不同，不能把性能差异直接解释成可见度的因果作用。visibility也不是姿态误差真值。", "",
               "| 动作 / 模型 | <0.8（F1 / n） | 0.8–0.95（F1 / n） | ≥0.95（F1 / n） |", "|---|---:|---:|---:|"]
    for e in entries:
        cells = [f'{r["macro_f1"]:.3f} / {r["n_observations"]}' if r["n_observations"] else "无样本" for r in e["by_visibility"]]
        report.append(f'| ex{e["exercise"]} / {e["name"]} | ' + " | ".join(cells) + " |")
    report += ["", "## 固定人工扰动测试", "", "只在测试输入的96帧中第32至63帧，对右肘和右腕加入标准差0.15躯干长度的三维高斯噪声，并把相应visibility设为0.1；重新计算受影响角度。模型、标准化和分类阈值保持原样。这是固定的人为坐标扰动，不代表真实遮挡或重新运行姿态估计器后的效果。", "",
               "| 动作 / 模型 | 原始宏F1 | 扰动宏F1 | 变化 |", "|---|---:|---:|---:|"]
    for e in entries:
        clean, noise = e["unconditional"]["macro_f1"], e["synthetic_noise"]["macro_f1"]
        report.append(f'| ex{e["exercise"]} / {e["name"]} | {clean:.3f} | {noise:.3f} | {noise-clean:+.3f} |')
    report += ["", "## 警示可靠性与覆盖率", "", "首先保留上面的全部测试结果。另在验证者上选择分数距分类阈值的最小标准化距离，要求验证覆盖率≥50%且标签集合全对率≥80%，取覆盖最高的候选。分数距离是启发式，不是经过校准的可信概率。", "",
               "未找到合格候选的折标为“不支持”，回退到不拒绝的原始预测，不能宣传为可靠警示。以下逐折列出资格、覆盖率和保留预测的表现；这些要求只约束验证集，不能保证测试集达标。", "",
               "| 动作 / 模型 | 测试者 | 验证规则达标 | 测试覆盖率 | 保留宏F1 | 保留标签集合全对率 |", "|---|---|---|---:|---:|---:|"]
    for e in entries:
        for f in e["folds"]:
            m = f["retained"]
            report.append(f'| ex{e["exercise"]} / {e["name"]} | {f["test_person"]} | {"是" if f["policy_qualified"] else "否，回退原始预测"} | {percentage(f["test_coverage"])} | {m.get("macro_f1", 0):.3f} | {percentage(m.get("exact_match"))} |')
    report += ["", "## 失败案例与交付边界", "", "本地 `analysis/failure_cases.csv` 为每种动作/模型保留最多10个较高阈值距离但判断错误的动作，并保留骨架来源、帧区间、真值、预测、视角与可见度。每组按动作去重，适合后续人工检查；它不是随机错误率估计。", "",
               "已完成：样本制作、5轮划分、50组实际模型拟合、权重保存、验证阈值选择、未见参与者测试，以及预先指定的输入消融和人工扰动诊断。没有根据测试结果重新选择参数。", "",
               "尚未完成：多随机种子确认、真实视频对齐、RGB上传与自动分段、专家复核的提示映射、真实患者和临床验证。当前推理需提供一次完整动作的起止帧。三种TCN输入的差异只能作为初步消融观察，不能凭单个种子认定方法创新或稳定收益。", "",
               f'训练及逐折评估计时总和约 {total_training_seconds/60:.1f} 分钟（不含依赖安装、样本制作与报告生成），使用CPU。环境版本见本地 `environment.json`。', ""]
    report_text = "\n".join(report)
    (output / "RESULTS_zh.md").write_text(report_text, encoding="utf-8")
    (HERE / "RESULTS_zh.md").write_text(report_text, encoding="utf-8")
    # Compact, reviewable evidence committed alongside the report; full predictions remain local.
    compact = {**{k: v for k, v in summary.items() if k != "entries"}, "entries": [
        {"exercise": e["exercise"], "model": e["model"], "unconditional": e["unconditional"],
         "synthetic_noise": e["synthetic_noise"], "n_physical_repetitions": e["n_physical_repetitions"],
         "mean_fold_macro_f1": e["mean_fold_macro_f1"], "sd_fold_macro_f1": e["sd_fold_macro_f1"],
         "n_qualified_reliability_folds": e["n_qualified_reliability_folds"]} for e in entries]}
    save_json(HERE / "results_summary.json", compact)
    make_plot(entries, output)
    print(f"Summarized 50 completed fits: {HERE / 'RESULTS_zh.md'}", flush=True)
    for e in entries:
        print(f'ex{e["exercise"]} {e["model"]}: F1={e["unconditional"]["macro_f1"]:.3f}, noise={e["synthetic_noise"]["macro_f1"]:.3f}, qualified_folds={e["n_qualified_reliability_folds"]}/5')


def make_plot(entries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Angle + logistic", "Angle + RF", "TCN xyz", "TCN xyz+angle", "TCN xyz+angle+vis"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for axis, ex, title in zip(axes, [1, 7], ["Arm abduction (ex1)", "Shoulder extension (ex7)"]):
        group = [e for e in entries if e["exercise"] == ex]
        positions = np.arange(len(group))
        clean = [e["unconditional"]["macro_f1"] for e in group]
        noisy = [e["synthetic_noise"]["macro_f1"] for e in group]
        axis.barh(positions - .18, clean, height=.34, label="Original pose", color="#2563eb")
        axis.barh(positions + .18, noisy, height=.34, label="Synthetic right-arm noise", color="#f59e0b")
        axis.set_yticks(positions, names); axis.invert_yaxis(); axis.set_xlim(0, 1)
        axis.set_title(title); axis.set_xlabel("Macro F1 over 5 error labels")
        axis.grid(axis="x", alpha=.2); axis.set_axisbelow(True)
        for i, (a, b) in enumerate(zip(clean, noisy)):
            axis.text(a + .012, i - .18, f"{a:.3f}", va="center", fontsize=8)
            axis.text(b + .012, i + .18, f"{b:.3f}", va="center", fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("REHAB26: 5 held-out people, single-view repetition-level errors", fontsize=13)
    fig.text(.5, .015, "One seed. Three correlated camera observations per repetition. Synthetic noise is not real occlusion.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.savefig(output / "baseline_comparison.png", dpi=160)
    fig.savefig(HERE / "baseline_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
