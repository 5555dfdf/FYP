"""Generate honest cross-subject results, validation ablations, figures and failures."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import HERE, ROOT, RUN, metrics, save_json, sha256

NAMES={"selected_transformer":"Selected Transformer", "selected_tcn":"Selected TCN", "angle_lr":"Angle LR", "angle_rf":"Angle RF", "reference_transformer":"Reference Transformer"}
ORDER=["angle_lr","angle_rf","selected_tcn","reference_transformer","selected_transformer"]


def cached(exercise,method,seed):
    rows=[]
    for folder in sorted((RUN/f"outer_predictions/ex{exercise}").glob(f"*/{method}/seed_{seed}")):
        with np.load(folder/"predictions.npz",allow_pickle=False) as data:
            rows.append({k:data[k].copy() for k in ["ids","y","scores","noisy_scores","thresholds","precision_thresholds"]})
    assert len(rows)==5
    return {"ids":np.concatenate([r["ids"] for r in rows]), "y":np.concatenate([r["y"] for r in rows]),
            "scores":np.concatenate([r["scores"] for r in rows]),
            "noisy_scores":np.concatenate([r["noisy_scores"] for r in rows]),
            "thresholds":np.concatenate([np.broadcast_to(r["thresholds"],r["y"].shape) for r in rows]),
            "precision_thresholds":np.concatenate([np.broadcast_to(r["precision_thresholds"],r["y"].shape) for r in rows])}


def mean_metrics(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in ["macro_f1","macro_ap","false_alert_rate","missed_alert_rate","exact_match"]}


def main():
    summary=json.loads((RUN/"test_summary.json").read_text(encoding="utf-8"))
    selection=json.loads((RUN/"selection_final.json").read_text(encoding="utf-8"))
    protocol=json.loads((RUN/"protocol.final.json").read_text(encoding="utf-8"))
    observations=json.loads((ROOT/"data/REHAB26-ViewAngles/processed_v1/observations.json").read_text(encoding="utf-8"))
    metadata=json.loads((ROOT/"data/REHAB26-ViewAngles/processed_v1/dataset.json").read_text(encoding="utf-8"))
    figures=HERE/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    aggregate,failures,validation=[],[],[]
    for exercise in [1,7]:
        for method in ORDER:
            seeds=sorted({r["seed"] for r in summary["rows"] if r["exercise"]==exercise and r["method"]==method})
            result=[]
            operating=[]
            noise=[]
            fixed=[]
            strata={}
            for seed in seeds:
                data=cached(exercise,method,seed)
                assert len(set(data["ids"]))==len(data["ids"])
                result.append(metrics(data["y"],data["scores"],data["thresholds"]))
                operating.append(metrics(data["y"],data["scores"],data["precision_thresholds"]))
                noise.append(metrics(data["y"],data["noisy_scores"],data["thresholds"]))
                fixed.append(metrics(data["y"],data["scores"]))
                if method=="selected_transformer":
                    for camera in [1,2,3]:
                        mask=np.array([observations[i]["camera"]==camera for i in data["ids"]])
                        strata.setdefault(str(camera),[]).append(metrics(data["y"][mask],data["scores"][mask],data["thresholds"][mask]))
                    if seed==seeds[0]:
                        predicted=data["scores"]>=data["thresholds"]
                        for local,i in enumerate(data["ids"]):
                            row=observations[int(i)]
                            if not np.array_equal(predicted[local],data["y"][local]):
                                failures.append({"exercise":exercise,"observation_id":row["observation_id"],"person":row["person"],
                                  "camera":row["camera"],"view_angle_degrees":row["view_angle_degrees"],
                                  "true_errors":"|".join(row["error_labels"]),
                                  "predicted_errors":"|".join(label for label,p in zip(metadata["labels"][str(exercise)],predicted[local]) if p),
                                  "false_alert_on_correct":int(not data["y"][local].any() and predicted[local].any()),
                                  "no_alert_on_error":int(data["y"][local].any() and not predicted[local].any()),
                                  "max_score_margin":float((data["scores"][local]-data["thresholds"][local]).max())})
            per_person={person:mean_metrics([r["test"] for r in summary["rows"] if r["exercise"]==exercise and r["method"]==method and r["test_person"]==person]) for person in sorted({r["person"] for r in observations})}
            aggregate.append({"exercise":exercise,"method":method,"seeds":seeds,"pooled_per_seed":result,
                "mean":mean_metrics(result),"seed_f1_sd":float(np.std([r["macro_f1"] for r in result],ddof=1)) if len(seeds)>1 else None,
                "precision_operating_point":mean_metrics(operating),"fixed05":mean_metrics(fixed),
                "synthetic_noise":mean_metrics(noise),"per_person":per_person,
                "camera":{k:mean_metrics(v) for k,v in strata.items()},
                "per_error_mean":[{key:float(np.mean([r["per_error"][j][key] for r in result])) for key in ["precision","recall","f1","ap","support","fp","fn"]} for j in range(5)]})
        for name,config in protocol["candidates"].items():
            if name.startswith("iter_"):
                continue
            files=list((RUN/f"fits/ex{exercise}").glob(f"test_*/val_*/{name}/seed_{protocol['seeds'][0]}/result.json"))
            assert len(files)==20
            values=[json.loads(p.read_text(encoding="utf-8")) for p in files]
            validation.append({"exercise":exercise,"candidate":name,"family":config["family"],
                "macro_ap":float(np.mean([r["validation"]["macro_ap"] for r in values])),
                "macro_f1":float(np.mean([r["validation"]["macro_f1"] for r in values])),
                "false_alert_rate":float(np.mean([r["validation"]["false_alert_rate"] for r in values])),
                "best_epoch_median":float(np.median([r["best_epoch"] for r in values])) if config["family"] in ("transformer","tcn") else None,
                "fit_seconds_sum":float(sum(r["elapsed_seconds"] for r in values))})
    save_json(HERE/"results_summary.json",{"generated_at_utc":datetime.now(timezone.utc).isoformat(),"aggregate":aggregate,"validation_ablation":validation,"selection":selection["selection"],"scope":"exploratory, correlated camera observations, simulated errors, five previously inspected volunteers"})
    with (HERE/"failures_seed0.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(failures[0]))
        writer.writeheader();writer.writerows(failures)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for ax,ex in zip(axes,[1,7]):
        selected=[next(r for r in aggregate if r["exercise"]==ex and r["method"]==m) for m in ORDER]
        loc=np.arange(len(selected))
        ax.bar(loc-.18,[r["mean"]["macro_f1"] for r in selected],.36,label="Macro F1",color="#3266c4")
        ax.bar(loc+.18,[r["mean"]["false_alert_rate"] for r in selected],.36,label="False alerts on correct",color="#d28a3c")
        ax.set_xticks(loc,[NAMES[r["method"]].replace(" ","\n",1) for r in selected],fontsize=9)
        ax.set_ylim(0,1);ax.set_title(f"Exercise {ex}: subject-disjoint, validation-selected")
        ax.legend(fontsize=8);ax.grid(axis="y",alpha=.15)
    fig.savefig(figures/"comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for ax,ex in zip(axes,[1,7]):
        val=[r for r in validation if r["exercise"]==ex and r["family"]=="transformer"]
        ax.barh([r["candidate"] for r in val],[r["macro_ap"] for r in val],color="#318c8b")
        ref=next(r["macro_ap"] for r in val if r["candidate"]=="tr_ref")
        ax.axvline(ref,color="#ac5b30",ls="--",label="Reference")
        ax.set_title(f"Exercise {ex}: validation only (20 splits)");ax.set_xlabel("Mean macro AP")
        ax.invert_yaxis();ax.legend()
    fig.savefig(figures/"validation_ablation.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for row,ex in enumerate([1,7]):
        person=sorted(selection["selection"][str(ex)])[0]
        name=selection["selection"][str(ex)][person]["transformer"]
        files=sorted((RUN/f"fits/ex{ex}/test_{person}").glob(f"val_*/{name}/seed_{protocol['seeds'][0]}/result.json"))
        result=json.loads(files[0].read_text(encoding="utf-8"));history=result["history"]
        ep=[r["epoch"] for r in history]
        axes[row,0].plot(ep,[r["train_loss"] for r in history],label="Training objective")
        axes[row,0].plot(ep,[r["validation_bce"] for r in history],label="Validation unweighted BCE")
        axes[row,1].plot(ep,[r["validation_macro_ap"] for r in history],label="Validation macro AP",color="#318c8b")
        for ax in axes[row]:
            ax.axvline(result["best_epoch"],ls="--",alpha=.5,color="black",label="Saved checkpoint")
            ax.set_xlabel("Epoch");ax.legend(fontsize=8);ax.grid(alpha=.15)
            ax.set_title(f"ex{ex}, {name}, val={result['validation']}" if isinstance(result.get('validation'),str) else f"ex{ex}, {name}, val={result['split']['validation'][0]}",fontsize=9)
    fig.savefig(figures/"learning_curves.png",dpi=180);plt.close(fig)
    lines=["# REHAB26 Transformer v2 实际实验结果", "", "2026-11-09 启动，结果由实际权重与预测生成。探索性研究：5 名模拟错误志愿者，已知动作与动作边界；尚无独立患者或临床疗效验证。", "", "## 协议与选择", "", "外层留一人测试，内层其余四人轮流验证、三人训练。16 个配置先做单种子筛查；第二轮只在对应内层验证改善时组合因素。先保存最终选择，再补种子，最后计算外层测试。相同人的全部视角和重复保持同组。", "", "表中 F1 是各种子汇总全部外层观察后的宏 F1 的均值；±SD 仅表示三种子变化。LR 和参考 Transformer 只有一个种子。所有方法均在每个外层使用四个内层模型集成；不能与 v1 单模型数字直接归因比较。", "", "## 主结果", "", "| 动作 | 方法 | 宏F1（种子均值±SD） | 宏AP | 正确动作误警率 | 错误动作无警率 | 标签集合全对 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        m=r["mean"];sd="" if r["seed_f1_sd"] is None else f" ± {r['seed_f1_sd']:.3f}"
        lines.append(f"| ex{r['exercise']} | {NAMES[r['method']]} | {m['macro_f1']:.3f}{sd} | {m['macro_ap']:.3f} | {m['false_alert_rate']:.1%} | {m['missed_alert_rate']:.1%} | {m['exact_match']:.1%} |")
    lines += ["", "![主结果](figures/comparison.png)", "", "## 验证选定的配置", "", "| 动作 | 外层测试者 | Transformer | TCN |", "|---|---|---|---|"]
    for ex in [1,7]:
        for person,chosen in selection["selection"][str(ex)].items():
            lines.append(f"| ex{ex} | {person} | {chosen['transformer']} | {chosen['tcn']} |")
    lines += ["", "完整配置见运行目录 protocol.final.json。不同外层允许选择不同配置，成绩对应整个嵌套选择流程，不是一个固定配置在所有人上的成绩。", "", "## 偏向精确率的预设工作点", "", "F0.5 阈值与 F1 阈值分别仅在内层验证者上选定。以下是同一模型改变工作点后的结果，未据测试成绩再改阈值。", "", "| 动作 | Transformer 宏F1 | 正确动作误警率 | 错误动作无警率 |", "|---|---:|---:|---:|"]
    for r in aggregate:
        if r["method"]=="selected_transformer":
            m=r["precision_operating_point"]
            lines.append(f"| ex{r['exercise']} | {m['macro_f1']:.3f} | {m['false_alert_rate']:.1%} | {m['missed_alert_rate']:.1%} |")
    lines += ["", "降低误警时要同时检查漏警，不能只提高阈值后宣称整体改善。分数未经校准，未实现经过验证的拒判模块。", "", "## 按受试者结果", "", "| 动作 | 受试者 | Transformer F1 | 误警率 | 无警率 |", "|---|---|---:|---:|---:|"]
    for r in aggregate:
        if r["method"]=="selected_transformer":
            for person,m in r["per_person"].items():
                lines.append(f"| ex{r['exercise']} | {person} | {m['macro_f1']:.3f} | {m['false_alert_rate']:.1%} | {m['missed_alert_rate']:.1%} |")
    lines += ["", "## 基础因素消融与训练诊断", "", "下图仅汇总验证指标，不用测试选择激活、损失、归一化或正则化。20 个分组结果相关，均值差不代表统计显著。组合实验另存迭代计划，不能把单因素收益简单相加。", "", "![验证消融](figures/validation_ablation.png)", "", "学习曲线固定选字母排序第一位外层及内层，未按成功程度挑选。训练目标损失在加权配置中与验证未加权 BCE 尺度不同，不能直接比较绝对大小。", "", "![学习曲线](figures/learning_curves.png)", "", "## 逐类错误", "", "| 动作 | 错误 | precision | recall | F1 | AP |", "|---|---|---:|---:|---:|---:|"]
    for r in aggregate:
        if r["method"]=="selected_transformer":
            for label,m in zip(metadata["labels"][str(r["exercise"])],r["per_error_mean"]):
                lines.append(f"| ex{r['exercise']} | {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['ap']:.3f} |")
    lines += ["", "## 证据边界与产物", "", "- 三种子结果不等于增加受试者人数；三相机也不是独立受试者。没有按相机观察构造独立样本置信区间。", "- 五名志愿者均已出现在之前的探索测试中，新一轮嵌套划分不能恢复完全未见的独立确认。", "- 三维骨架由姿态估计器产生，固定人工噪声诊断不等同于真实遮挡或摄像头鲁棒性。", "- 当前输入是完整动作片段，不能宣称已实现连续视频分割、逐帧定位、实时纠正或患者恢复效果。", "- 全量权重、预测、曲线、配置、输入/代码哈希保存在 runs/rehab26_transformer_v2；results_summary.json 为可追溯汇总。", "- failures_seed0.csv 导出第一种子全部标签集合不匹配观察，不只展示成功例。", "- 若生成全数据部署权重，其使用所有五人，外层测试成绩不能当作这些部署权重的独立成绩；阈值仍是待确认的研究工作点。", ""]
    (HERE/"RESULTS_zh.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"aggregate_rows":len(aggregate),"failures_seed0":len(failures),"report":str(HERE/"RESULTS_zh.md")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
