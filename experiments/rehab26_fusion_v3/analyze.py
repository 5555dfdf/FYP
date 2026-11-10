"""Compare saved outer predictions at observation and physical-repetition levels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import ROOT, HERE, RUN, DATA, core, load_json

OLD = ROOT / "runs/rehab26_transformer_v2"
METHODS = ["angle_lr", "angle_rf", "selected_tcn", "reference_transformer", "selected_transformer"]
KEYS = ["macro_f1", "macro_ap", "false_alert_rate", "missed_alert_rate", "exact_match"]


def read_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in ("ids", "y", "scores", "thresholds")}


def join(rows):
    return {k: np.concatenate([np.broadcast_to(r[k], r["y"].shape) if k == "thresholds" else r[k] for r in rows])
            for k in ["ids", "y", "scores", "thresholds"]}


def aggregate(ex, method, variant, rows):
    values = [r["metrics"] for r in rows]
    return {"exercise": ex, "method": method, "variant": variant, "seeds": [r["seed"] for r in rows],
            "mean": {k: float(np.mean([v[k] for v in values])) for k in KEYS},
            "std": {k: float(np.std([v[k] for v in values], ddof=1)) if len(values) > 1 else 0. for k in KEYS}, "per_seed": rows}


def collect(base, observations, fused_lookup=None):
    """No refitting or test threshold selection; old thresholds are retained."""
    out = []
    old_selection = load_json(OLD / "selection_final.json") if fused_lookup is not None else None
    for ex in [1, 7]:
        for method in METHODS:
            seeds = sorted({int(p.name.removeprefix("seed_")) for p in (base / f"outer_predictions/ex{ex}").glob(f"*/{method}/seed_*")})
            variants = defaultdict(list)
            for seed in seeds:
                raw, late, calibrated, cameras = [], [], [], defaultdict(list)
                by_person = defaultdict(list)
                for folder in sorted((base / f"outer_predictions/ex{ex}").glob(f"*/{method}/seed_{seed}")):
                    data = read_prediction(folder / "predictions.npz")
                    raw.append(data)
                    by_person["pose" if fused_lookup is None else "single"].append({"person": folder.parent.parent.name,
                        "metrics": core.metrics(data["y"], data["scores"], data["thresholds"])})
                    if fused_lookup is None:
                        continue
                    grouped = defaultdict(list)
                    for n, i in enumerate(data["ids"]):
                        grouped[observations[int(i)]["sample_id"]].append(n)
                    ids, targets, scores = [], [], []
                    for sample_id, positions in sorted(grouped.items()):
                        assert len(positions) == 3
                        assert {observations[int(data["ids"][n])]["camera"] for n in positions} == {1, 2, 3}
                        assert np.all(data["y"][positions] == data["y"][positions[0]])
                        ids.append(fused_lookup[sample_id])
                        targets.append(data["y"][positions[0]])
                        scores.append(data["scores"][positions].mean(0))
                    row = {"ids": np.asarray(ids), "y": np.asarray(targets), "scores": np.asarray(scores), "thresholds": data["thresholds"]}
                    late.append(row)
                    by_person["late"].append({"person": folder.parent.parent.name, "metrics": core.metrics(row["y"], row["scores"], row["thresholds"])})
                    save = RUN / f"historical_late_predictions/ex{ex}/{folder.parent.parent.name}/{method}/seed_{seed}"
                    save.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(save / "predictions.npz", **row)
                    choices = old_selection["selection"][str(ex)][folder.parent.parent.name]
                    candidate = {"selected_transformer": choices["transformer"], "selected_tcn": choices["tcn"],
                                 "angle_lr": "angle_lr", "angle_rf": "angle_rf", "reference_transformer": "tr_ref"}[method]
                    calibrated_thresholds = []
                    for val_path in sorted((OLD / f"fits/ex{ex}/test_{folder.parent.parent.name}").glob(f"val_*/{candidate}/seed_{seed}/validation.npz")):
                        valid = read_prediction(val_path)
                        val_groups = defaultdict(list)
                        for n, original_id in enumerate(valid["ids"]):
                            val_groups[observations[int(original_id)]["sample_id"]].append(n)
                        targets, probabilities = [], []
                        for positions in val_groups.values():
                            assert len(positions) == 3
                            assert np.all(valid["y"][positions] == valid["y"][positions[0]])
                            targets.append(valid["y"][positions[0]])
                            probabilities.append(valid["scores"][positions].mean(0))
                        calibrated_thresholds.append(core.select_thresholds(np.asarray(targets), np.asarray(probabilities)))
                    assert len(calibrated_thresholds) == 4
                    cal_row = {**row, "thresholds": np.mean(calibrated_thresholds, 0)}
                    calibrated.append(cal_row)
                    by_person["late_calibrated"].append({"person": folder.parent.parent.name,
                        "metrics": core.metrics(cal_row["y"], cal_row["scores"], cal_row["thresholds"])})
                    save_cal = RUN / f"historical_late_calibrated_predictions/ex{ex}/{folder.parent.parent.name}/{method}/seed_{seed}"
                    save_cal.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(save_cal / "predictions.npz", **cal_row)
                    for camera in [1, 2, 3]:
                        mask = np.asarray([observations[int(i)]["camera"] == camera for i in data["ids"]])
                        cameras[camera].append({k: v[mask] if k != "thresholds" else v for k, v in data.items()})
                assert len(raw) == 5
                variant = "pose" if fused_lookup is None else "single"
                for key, parts in [(variant, raw)] + ([("late", late), ("late_calibrated", calibrated)] + [(f"camera{c}", cameras[c]) for c in [1, 2, 3]] if late else []):
                    all_data = join(parts)
                    assert len(set(all_data["ids"])) == len(all_data["ids"])
                    variants[key].append({"seed": seed, "metrics": core.metrics(all_data["y"], all_data["scores"], all_data["thresholds"]),
                                          "by_person": by_person[key]})
            out.extend(aggregate(ex, method, variant, rows) for variant, rows in variants.items())
    return out


def controls():
    output = []
    for key in ["x_equal", "x_cam2"]:
        for ex in [1, 7]:
            parts = [read_prediction(p) for p in sorted((RUN / f"control_predictions/{key}/ex{ex}").glob("*/predictions.npz"))]
            assert len(parts) == 5
            data = join(parts)
            output.append(aggregate(ex, "reference_transformer", key, [{"seed": 20261109,
                "metrics": core.metrics(data["y"], data["scores"], data["thresholds"])}]))
    return output


def validation_effects(protocol):
    rows = []
    for ex in [1, 7]:
        for name in protocol["candidates"]:
            if name.startswith("iter_"):
                continue
            deltas, f1s = [], []
            for path in sorted((RUN / f"fits/ex{ex}").glob(f"test_*/val_*/{name}/seed_20261109/result.json")):
                row = load_json(path)
                ref = load_json(path.parents[2] / "tr_ref/seed_20261109/result.json")
                deltas.append(row["validation"]["macro_ap"] - ref["validation"]["macro_ap"])
                f1s.append(row["validation"]["macro_f1"] - ref["validation"]["macro_f1"])
            assert len(deltas) == 20
            rows.append({"exercise": ex, "candidate": name, "n_splits": len(deltas), "mean_delta_ap": float(np.mean(deltas)),
                         "mean_delta_f1": float(np.mean(f1s)), "ap_improved_splits": int(np.sum(np.asarray(deltas) > 0))})
    return rows


def fmt(row, key="macro_f1"):
    if key in ["false_alert_rate", "missed_alert_rate"]:
        return f"{row['mean'][key] * 100:.1f}%"
    return f"{row['mean'][key]:.3f}" + (f" ± {row['std'][key]:.3f}" if len(row["seeds"]) > 1 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args()
    calibration_plan = RUN / "historical_late_calibration_plan.json"
    if not calibration_plan.exists():
        assert not (RUN / "test_release_gate.json").exists(), "Freeze this comparison before the new test release"
        core.save_json(calibration_plan, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "rule": "Report both retained old thresholds and thresholds selected on old inner-validation three-view mean probabilities; same v2 grid/F1 rule, four-inner threshold mean; no retraining, no test threshold tuning",
            "use_for_primary_paired_comparison": "late_calibrated"})
    observations = load_json(DATA / "observations.json")
    lookup = {r["sample_id"]: i for i, r in enumerate(observations)}
    old_observations = load_json(ROOT / "data/REHAB26-ViewAngles/processed_v1/observations.json")
    old = collect(OLD, old_observations, lookup)
    with np.load(DATA / "samples.npz", allow_pickle=False) as arrays:
        fused_y = arrays["y"]
    for path in (RUN / "historical_late_predictions").rglob("predictions.npz"):
        data = read_prediction(path)
        np.testing.assert_array_equal(data["y"], fused_y[data["ids"]])
    previous_summary = load_json(ROOT / "experiments/rehab26_transformer_v2/results_summary.json")
    for row in old:
        if row["variant"] == "single":
            previous = next(r for r in previous_summary["aggregate"] if r["exercise"] == row["exercise"] and r["method"] == row["method"])
            for key in KEYS:
                np.testing.assert_allclose(row["mean"][key], np.mean([r[key] for r in previous["pooled_per_seed"]]), atol=1e-12)
            if len(row["seeds"]) > 1:
                np.testing.assert_allclose(row["std"]["macro_f1"], previous["seed_f1_sd"], atol=1e-12)
    core.save_json(RUN / "historical_comparison.json", old)
    if args.historical_only:
        for row in old:
            if row["method"] == "selected_transformer" and row["variant"] in ["single", "late", "late_calibrated"]:
                print(row["exercise"], row["variant"], row["mean"])
        return
    assert (RUN / "test_release_gate.json").exists()
    protocol = load_json(RUN / "protocol.final.json")
    selection = load_json(RUN / "selection_final.json")
    new = collect(RUN, observations)
    control = controls()
    effects = validation_effects(protocol)
    all_rows = old + new + control
    indexed = {(r["exercise"], r["method"], r["variant"]): r for r in all_rows}
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, ex in zip(axes, [1, 7]):
        variants = ["single", "late_calibrated", "pose"]
        rows = [indexed[ex, "selected_transformer", v] for v in variants]
        ax.bar(np.arange(3), [r["mean"]["macro_f1"] for r in rows], yerr=[r["std"]["macro_f1"] for r in rows],
               color=["#8295aa", "#d59b4a", "#428c83"], capsize=5)
        for i, r in enumerate(rows):
            ax.text(i, r["mean"]["macro_f1"] + r["std"]["macro_f1"] + .018, f"{r['mean']['macro_f1']:.3f}", ha="center")
        ax.set(xticks=np.arange(3), xticklabels=["Old single view\n(pooled)", "Old 3-view\nprobability mean", "New 3-view\nskeleton fusion"],
               ylim=(0, .85), ylabel="Outer-test macro F1", title={1: "Shoulder abduction", 7: "Shoulder extension"}[ex])
    fig.suptitle("Selected Transformer: three training seeds; error bars show seed SD")
    fig.savefig(figures / "fusion_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, ex in zip(axes, [1, 7]):
        rows = [indexed[ex, "reference_transformer", v] for v in ["x_cam2", "x_equal", "pose"]]
        ax.bar(np.arange(3), [r["mean"]["macro_f1"] for r in rows], color=["#8295aa", "#d59b4a", "#428c83"])
        for i, r in enumerate(rows):
            ax.text(i, r["mean"]["macro_f1"] + .015, f"{r['mean']['macro_f1']:.3f}", ha="center")
        ax.set(xticks=np.arange(3), xticklabels=["Aligned camera 2", "Equal fusion", "Visibility-squared\nfusion"], ylim=(0, .85),
               ylabel="Outer-test macro F1", title=f"Exercise {ex}: reference Transformer, seed 20261109")
    fig.savefig(figures / "fusion_controls.png", dpi=180)
    plt.close(fig)
    # A representative curve chosen by middle validation AP, never by test result.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    representatives = []
    for ax, ex in zip(axes, [1, 7]):
        candidates = []
        for person, choices in selection["selection"][str(ex)].items():
            for p in (RUN / f"fits/ex{ex}/test_{person}").glob(f"val_*/{choices['transformer']}/seed_20261109/result.json"):
                candidates.append((load_json(p), p))
        row, path = sorted(candidates, key=lambda pair: pair[0]["validation"]["macro_ap"])[len(candidates)//2]
        history = row["history"]
        ax.plot([r["epoch"] for r in history], [r["validation_macro_ap"] for r in history], label="Validation macro AP")
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label=f"Training BCE ({row['config']['loss']})")
        ax.plot([r["epoch"] for r in history], [r["validation_bce"] for r in history], label="Validation BCE", color="#9174a5", alpha=.8)
        ax.axvline(row["best_epoch"], ls="--", color="gray", label="Saved epoch")
        ax.set(xlabel="Epoch", title=f"Exercise {ex}; held out {row['test']}, val {row['split']['validation'][0]}")
        ax.legend(fontsize=8)
        representatives.append(str(path.relative_to(ROOT)))
    fig.savefig(figures / "learning_curves.png", dpi=180)
    plt.close(fig)
    paired = []
    for ex in [1, 7]:
        previous = indexed[ex, "selected_transformer", "late_calibrated"]
        current = indexed[ex, "selected_transformer", "pose"]
        for person in sorted({r["person"] for r in observations}):
            old_values = [p["metrics"]["macro_f1"] for r in previous["per_seed"] for p in r["by_person"] if p["person"] == person]
            new_values = [p["metrics"]["macro_f1"] for r in current["per_seed"] for p in r["by_person"] if p["person"] == person]
            paired.append({"exercise": ex, "person": person, "late_f1": float(np.mean(old_values)), "pose_f1": float(np.mean(new_values)),
                           "delta_f1": float(np.mean(new_values)-np.mean(old_values))})
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "aggregate": all_rows,
               "validation_effects": effects, "paired_person_f1": paired, "representative_curves": representatives,
               "old_prediction_identity": core.sha256(ROOT / "experiments/rehab26_transformer_v2/results_summary.json"),
               "new_test_summary_sha256": core.sha256(RUN / "test_summary.json")}
    core.save_json(HERE / "results_summary.json", summary)
    labels = load_json(DATA / "dataset.json")["labels"]
    gate = load_json(RUN / "test_release_gate.json")
    lines = ["# REHAB26 三视图基础融合实验结果", "", "日期：2026-11-10。环境：conda fyp，RTX4050。", "",
        "本轮把同次动作的三路 MediaPipe 单目三维骨架融合为一个输入，重跑昨日的 LR、RF、TCN 和 Transformer。融合不是标定三维重建：每路髋中心/躯干尺度归一化、96帧动作进度配对、整次动作固定髋方向 yaw 对齐，再逐关节 visibility² 加权。没有新增精确时间同步。", "",
        "## Transformer 主比较", "", "macro-F1/AP 是汇总五个外层受试者后逐类计算，再跨三个训练种子取均值；±是种子样本标准差(ddof=1，与v2一致)，不能解释为人群置信区间。误警是正确动作被报任一错误，漏警是有错误动作没有报任何错误。", "",
        "| 动作 | 输入方案 | macro-F1 | macro-AP | 误警 | 漏警 |", "|---|---|---|---|---|---|"]
    names = {"single": "昨日单视角观察汇总", "late": "昨日三视角概率平均（沿用阈值）", "late_calibrated": "昨日三视角概率平均（验证重选阈值）", "pose": "本轮骨架加权融合"}
    for ex in [1, 7]:
        for variant in ["single", "late", "late_calibrated", "pose"]:
            r = indexed[ex, "selected_transformer", variant]
            lines.append(f"| {'肩外展' if ex == 1 else '肩后伸'} | {names[variant]} | {fmt(r)} | {fmt(r,'macro_ap')} | {fmt(r,'false_alert_rate')} | {fmt(r,'missed_alert_rate')} |")
    lines += ["", "昨日单视角汇总以3333个相机观察为单位；概率平均和本轮融合均以1111次物理动作为单位（肩外展556次，肩后伸555次）。概率平均直接复用昨日保存的三路预测：一版沿用原阈值，另一版先对每个内层验证集的三视角概率取平均，再沿用v2网格/F1规则选择阈值，最后平均四个内层阈值；没有重训或用测试集挑阈值。两版都报告，后续逐人配对和主图预先指定使用验证重选阈值版，规则冻结于新测试释放前。历史原始指标与v2报告逐项核对；概率融合的每次动作标签与本轮标签逐项核对。", "",
        "## 相同参考网络的输入对照", "", "三种输入均用相同参考 Transformer、首种子、40个内层划分和每次动作一个样本；对比加权、等权与同样预处理的单cam2。此表不等于上表各外层独立选出的最佳配置。", "",
        "| 动作 | 输入 | macro-F1 | macro-AP | 误警 | 漏警 |", "|---|---|---|---|---|---|"]
    for ex in [1, 7]:
        for variant, name in [("x_cam2", "对齐后单cam2"), ("x_equal", "三路等权"), ("pose", "三路visibility²")]:
            r = indexed[ex, "reference_transformer", variant]
            lines.append(f"| {ex} | {name} | {fmt(r)} | {fmt(r,'macro_ap')} | {fmt(r,'false_alert_rate')} | {fmt(r,'missed_alert_rate')} |")
    lines += ["", "## 融合输入下的模型对照", "", "| 动作 | 模型 | macro-F1 | macro-AP | 误警 | 漏警 |", "|---|---|---|---|---|---|"]
    for ex in [1, 7]:
        for method in METHODS:
            r = indexed[ex, method, "pose"]
            lines.append(f"| {ex} | {method} | {fmt(r)} | {fmt(r,'macro_ap')} | {fmt(r,'false_alert_rate')} | {fmt(r,'missed_alert_rate')} |")
    lines += ["", "## 每位测试者的变化", "", "下表先对种子取均值；参与者只有5位，个体差异比种子差异更值得关注。", "", "| 动作 | 人 | 昨日概率融合F1 | 本轮骨架融合F1 | 差值 |", "|---|---|---|---|---|"]
    for r in paired:
        lines.append(f"| {r['exercise']} | {r['person']} | {r['late_f1']:.3f} | {r['pose_f1']:.3f} | {r['delta_f1']:+.3f} |")
    lines += ["", "## 错误类别", "", "选定Transformer在本轮融合输入上的逐类F1/AP为三个种子均值；support按物理动作计。", "", "| 动作 | 类别 | support | F1 | AP |", "|---|---|---|---|---|"]
    for ex in [1, 7]:
        r = indexed[ex, "selected_transformer", "pose"]
        for j, label in enumerate(labels[str(ex)]):
            values = [v["metrics"]["per_error"][j] for v in r["per_seed"]]
            lines.append(f"| {ex} | {label} | {values[0]['support']} | {np.mean([v['f1'] for v in values]):.3f} | {np.mean([v['ap'] for v in values]):.3f} |")
    lines += ["", "## 损失、激活、归一化和正则化的验证结果", "", "保持昨日候选网格和选择规则。下表为每动作20个内层划分相对参考Transformer的平均验证AP差；属于验证证据，不根据外层测试继续调参。参考使用无类别加权的BCEWithLogitsLoss、GELU、Pre-LN、训练集标准化、dropout=0.1与AdamW(weight_decay=0.001，仅矩阵参数)。损失对照包括按训练标签计算的原始正负比权重，以及平方根后截断到[1,3]的权重。输出五个独立sigmoid概率；没有softmax互斥假设。", "", "| 动作 | 候选 | AP差 | AP提高划分数/20 |", "|---|---|---|---|"]
    for r in effects:
        if r["candidate"].startswith("tr_"):
            lines.append(f"| {r['exercise']} | {r['candidate']} | {r['mean_delta_ap']:+.3f} | {r['ap_improved_splits']} |")
    lines += ["", "候选含义：`tr_xyz`只用坐标；`tr_sqrt/raw`改变正类权重；`tr_relu`改变激活；`tr_postnorm`改为Post-LN；`tr_noz`关闭输入标准化；`tr_nodrop/nodecay`分别关闭dropout/weight decay；`tr_visibility`增加19维可见度；`tr_lr1e3`学习率改为0.001；`tr_small`改为单层32维。参考输入为57维关节坐标+12维几何特征。全部配置保存在`protocol.final.json`。"]
    lines += ["", "各外层最终配置：", "", "| 动作 | 测试者 | Transformer | TCN |", "|---|---|---|---|"]
    for ex in [1, 7]:
        for person, choices in selection["selection"][str(ex)].items():
            lines.append(f"| {ex} | {person} | {choices['transformer']} | {choices['tcn']} |")
    lines += ["", "## 运行与适用范围", "",
        f"共完成{gate['fits_complete']}次主实验拟合和{gate['control_fits_complete']}次输入对照拟合；另有一次小规模smoke。640次基础筛查、验证生成的组合、240次额外种子确认、80次输入对照均在新外层测试预测前完成。人物划分、阈值、标准化和早停继续复用v2实现。", "",
        "融合后每次动作仅一个训练样本；原先每次动作三个相机观察，因此同样epoch预算下的梯度更新次数不同。主表反映融合与训练方案的整体变化。参考输入对照更适合判断当前加权/等权融合是否有帮助。", "",
        "visibility是可见度，不保证三维精度；共同偏差、髋方向估计误差和近似进度配对都可能削弱融合。数据只有5位已用于探索的志愿者；没有临床患者泛化或三维真值结论。synthetic noise只在融合后施加右臂扰动，不作为相机遮挡实验。", "",
        "GitHub适配核查见[检索记录](../../literature/2026-11-10_skeleton_fusion/REVIEW_zh.md)。未直接复用需要K/R/t的三角测量代码；独立实现并验证了适用于现有输入的加权数学运算。", "",
        "![主比较](figures/fusion_comparison.png)", "", "![输入对照](figures/fusion_controls.png)", "", "![学习曲线](figures/learning_curves.png)", "",
        "结构化明细：`results_summary.json`；运行权重/预测/阈值：`runs/rehab26_fusion_v3/`。几何与模型重载结果分别保存在`fusion_verification.json`和`model_verification.json`。"]
    (HERE / "RESULTS_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ex in [1, 7]:
        for v in ["late", "late_calibrated", "pose"]:
            r = indexed[ex, "selected_transformer", v]
            print(ex, v, r["mean"])


if __name__ == "__main__":
    main()
