"""Exploratory IRDS shoulder-flexion baseline. Never modifies source data.

Run with the adjacent .venv/Scripts/python.exe. Incorrect execution is positive.
Fixed settings; no hyperparameter or threshold selection on evaluation data.
"""
from pathlib import Path
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results'
SEED = 20261106


def angle(a, b):
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(
        np.sum(a * b, axis=-1) / np.maximum(denominator, 1e-10), -1, 1)))


def extract(x, gesture):
    # Explicit mapping from the prior audit; these are raw Kinect joint names.
    sh, el, wr = (8, 9, 10) if gesture == 2 else (4, 5, 6)
    up = x[:, 20] - x[:, 0]
    elevation = angle(x[:, el] - x[:, sh], -up)
    elbow = angle(x[:, sh] - x[:, el], x[:, wr] - x[:, el])
    other_sh, other_el = (4, 5) if gesture == 2 else (8, 9)
    other = angle(x[:, other_el] - x[:, other_sh], -up)
    torso_change = angle(up, np.median(up[:max(1, len(x)//10)], axis=0))
    high = elevation >= np.percentile(elevation, 75)
    scale = max(float(np.median(np.linalg.norm(up, axis=1))), 1e-8)
    wrist_relative = (x[:, wr] - x[:, sh]) / scale
    head_clearance = (x[:, wr, 1] - x[:, 3, 1]) / scale
    f = {}
    for name, values in [('elevation', elevation), ('elbow', elbow),
                         ('other_elevation', other), ('torso_change', torso_change),
                         ('wrist_above_head', head_clearance)]:
        for q in [10, 50, 90, 100]:
            f[f'{name}_p{q}'] = float(np.percentile(values, q))
        f[f'{name}_std'] = float(np.std(values))
    f['elbow_at_peak'] = float(elbow[np.argmax(elevation)])
    f['elbow_high_median'] = float(np.median(elbow[high]))
    f['torso_high_median'] = float(np.median(torso_change[high]))
    f['elevation_range'] = float(np.percentile(elevation, 95)-np.percentile(elevation, 5))
    f['wrist_displacement_p90'] = float(np.percentile(
        np.linalg.norm(wrist_relative - wrist_relative[0], axis=1), 90))
    # No velocity/acceleration: timestamps are not yet verified. No ID/position inputs.
    return f


def metrics(y, pred, probability):
    both = len(np.unique(y)) == 2
    return dict(n=len(y), incorrect=int(np.sum(y)),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)) if both else None,
                incorrect_precision=float(precision_score(y, pred, zero_division=0)),
                incorrect_recall=float(recall_score(y, pred, zero_division=0)) if np.sum(y) else None,
                incorrect_f1=float(f1_score(y, pred, zero_division=0)) if np.sum(y) else None,
                macro_f1=float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)) if both else None,
                roc_auc=float(roc_auc_score(y, probability)) if both else None,
                average_precision=float(average_precision_score(y, probability)) if both else None,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    excluded = []
    for p in sorted((ROOT / 'SkeletonData' / 'Simplified').glob('*.txt')):
        subject, date, gesture, repetition, label, position = p.stem.split('_', 5)
        gesture, label, subject = int(gesture), int(label), int(subject)
        if gesture not in [2, 3]:
            continue
        if label not in [1, 2]:
            excluded.append(p.name)
            continue
        x = np.loadtxt(p, delimiter=',', ndmin=2)
        if x.shape[1] != 75 or not np.isfinite(x).all():
            raise ValueError(f'Invalid source: {p}')
        rows.append(dict(file=p.name, subject=subject, gesture=gesture,
                         original_label=label, y=int(label == 2), position=position,
                         group='patient_id' if 200 <= subject < 300 else 'control_id',
                         frames=len(x), **extract(x.reshape(-1, 25, 3), gesture)))
    df = pd.DataFrame(rows)
    geometry = list(df.columns[df.columns.get_loc('frames')+1:])
    assert len(df) == 692 and int(df.y.sum()) == 251
    assert np.isfinite(df[geometry + ['frames']].values).all()
    df.to_csv(OUT / 'features.csv', index=False)
    y, groups = df.y.to_numpy(), df.subject.to_numpy()
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                  .split(df[geometry], y, groups))
    fold_id = np.full(len(df), -1)
    split_log = []
    for fold, (train, test) in enumerate(splits, 1):
        assert not set(groups[train]) & set(groups[test])
        assert len(np.unique(y[train])) == 2
        fold_id[test] = fold
        split_log.append(dict(fold=fold, train_subjects=sorted(set(groups[train].tolist())),
                              test_subjects=sorted(set(groups[test].tolist())),
                              train_n=len(train), test_n=len(test),
                              test_incorrect=int(y[test].sum())))
    assert (fold_id > 0).all()
    # Fixed, deliberately small baselines. Geometry RF is the prespecified primary model.
    specs = {
        'majority': (lambda: DummyClassifier(strategy='most_frequent'), ['frames']),
        'frames_logistic': (lambda: make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', C=1, max_iter=2000, random_state=SEED)), ['frames']),
        'geometry_logistic': (lambda: make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', C=1, max_iter=2000, random_state=SEED)), geometry),
        'geometry_rf': (lambda: RandomForestClassifier(n_estimators=300, max_depth=6,
            min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
            random_state=SEED, n_jobs=-1), geometry),
        'geometry_frames_rf': (lambda: RandomForestClassifier(n_estimators=300, max_depth=6,
            min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
            random_state=SEED, n_jobs=-1), geometry + ['frames']),
    }
    predictions = df[['file', 'subject', 'gesture', 'original_label', 'y', 'position', 'group']].copy()
    predictions['fold'] = fold_id
    summary = dict(seed=SEED, task='Gesture 2/3, original label 2 is positive',
                   sklearn=sklearn.__version__, python=platform.python_version(),
                   n=len(df), subjects=int(df.subject.nunique()),
                   correct=int((y == 0).sum()), incorrect=int(y.sum()),
                   excluded_special_labels=excluded, geometry_features=geometry,
                   protocol='Fixed 5-fold StratifiedGroupKFold by file subject ID; threshold 0.5; no tuning',
                   primary_model='geometry_rf', models={}, splits=split_log)
    for name, (factory, columns) in specs.items():
        probability = np.full(len(df), np.nan)
        fold_results = []
        for fold, (train, test) in enumerate(splits, 1):
            model = factory()
            model.fit(df.iloc[train][columns], y[train])
            probability[test] = model.predict_proba(df.iloc[test][columns])[:, list(model.classes_).index(1)]
            fold_results.append(dict(fold=fold, **metrics(y[test],
                (probability[test] >= .5).astype(int), probability[test])))
        assert np.isfinite(probability).all()
        pred = (probability >= .5).astype(int)
        predictions[name + '_p_incorrect'] = probability
        predictions[name + '_pred'] = pred
        subgroups = {}
        for field in ['gesture', 'group', 'position']:
            for value in sorted(df[field].unique()):
                mask = df[field].to_numpy() == value
                subgroups[f'{field}={value}'] = metrics(y[mask], pred[mask], probability[mask])
        summary['models'][name] = dict(overall=metrics(y, pred, probability),
                                       folds=fold_results, subgroups=subgroups)
        final_model = factory()
        final_model.fit(df[columns], y)
        joblib.dump(dict(model=final_model, feature_columns=columns,
                         note='Fitted on all data for demo only; metrics come from held-out folds.'),
                    OUT / f'{name}.joblib')
        print(name, json.dumps(summary['models'][name]['overall']), flush=True)

    predictions.to_csv(OUT / 'heldout_predictions.csv', index=False)
    primary = 'geometry_rf'
    p = predictions[primary+'_p_incorrect'].to_numpy()
    pred = (p >= .5).astype(int)
    per_subject = []
    for subject in sorted(df.subject.unique()):
        mask = groups == subject
        row = dict(subject=int(subject), fold=int(fold_id[mask][0]),
                   **metrics(y[mask], pred[mask], p[mask]))
        row.pop('confusion_matrix')
        per_subject.append(row)
    pd.DataFrame(per_subject).to_csv(OUT / 'primary_per_subject.csv', index=False)
    # Cluster bootstrap, not an independent-action bootstrap. Descriptive uncertainty only.
    rng = np.random.default_rng(SEED)
    subjects = np.unique(groups)
    bootstrap = []
    for _ in range(1000):
        indices = np.concatenate([np.flatnonzero(groups == s)
                                  for s in rng.choice(subjects, len(subjects), replace=True)])
        yy, pp = y[indices], pred[indices]
        if len(np.unique(yy)) == 2:
            bootstrap.append([balanced_accuracy_score(yy, pp), recall_score(yy, pp),
                              f1_score(yy, pp, zero_division=0)])
    summary['primary_subject_bootstrap_95_percent'] = {
        k: np.percentile(np.array(bootstrap)[:, i], [2.5, 97.5]).tolist()
        for i, k in enumerate(['balanced_accuracy', 'incorrect_recall', 'incorrect_f1'])}
    # Reference deviations are exploratory, not error-reason ground truth or clinical advice.
    feedback = []
    for fold, (train, test) in enumerate(splits, 1):
        for index in test:
            row = df.iloc[index]
            ref = df.iloc[train]
            ref = ref[(ref.y == 0) & (ref.gesture == row.gesture) & (ref.position == row.position)]
            if len(ref) < 10:
                feedback.append(dict(file=row.file, fold=fold, reference_n=len(ref),
                                     flags='insufficient same-gesture/same-position correct reference'))
                continue
            limits = dict(elevation_lower=float(ref.elevation_p100.quantile(.05)),
                          elbow_lower=float(ref.elbow_high_median.quantile(.05)),
                          torso_upper=float(ref.torso_change_p90.quantile(.95)))
            flags = []
            if row.elevation_p100 < limits['elevation_lower']:
                flags.append('lower peak arm elevation than reference')
            if row.elbow_high_median < limits['elbow_lower']:
                flags.append('more elbow flexion during high arm elevation than reference')
            if row.torso_change_p90 > limits['torso_upper']:
                flags.append('more torso orientation change than reference')
            feedback.append(dict(file=row.file, fold=fold, reference_n=len(ref),
                                 predicted_incorrect=bool(pred[index]), original_label=int(row.original_label),
                                 elevation_peak=float(row.elevation_p100),
                                 elbow_high_median=float(row.elbow_high_median),
                                 torso_change_p90=float(row.torso_change_p90),
                                 **limits, flags='; '.join(flags) or 'no selected reference deviation'))
    pd.DataFrame(feedback).to_csv(OUT / 'exploratory_reference_deviations.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({name: data['overall'] for name, data in summary['models'].items()}).T.drop(
        columns='confusion_matrix').to_csv(OUT / 'metrics.csv')
    plot_results(summary, per_subject)
    write_report(summary)
    (OUT / 'environment.txt').write_text(
        f'Python {sys.version}\nnumpy {np.__version__}\npandas {pd.__version__}\n'
        f'scikit-learn {sklearn.__version__}\njoblib {joblib.__version__}\n', encoding='utf-8')


def plot_results(summary, per_subject):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    names = list(summary['models'])
    labels = ['Majority', 'Frames LR', 'Geometry LR', 'Geometry RF', 'Geometry + frames RF']
    xx = np.arange(len(names))
    for shift, metric, label, color in [(-.23, 'accuracy', 'Accuracy', '#8fa9bf'),
                                      (0, 'balanced_accuracy', 'Balanced accuracy', '#287f79'),
                                      (.23, 'incorrect_recall', 'Incorrect recall', '#c66a40')]:
        axes[0].bar(xx+shift, [summary['models'][n]['overall'][metric] for n in names],
                    width=.23, label=label, color=color)
    axes[0].set(xticks=xx, xticklabels=labels, ylim=(0, 1.04), title='Same held-out subjects for every model')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(fontsize=8)
    cm = np.array(summary['models']['geometry_rf']['overall']['confusion_matrix'])
    axes[1].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20,
                         color='white' if cm[i, j] > cm.max()/2 else '#222222')
    axes[1].set(xticks=[0, 1], xticklabels=['Correct', 'Incorrect'], yticks=[0, 1],
                yticklabels=['Correct', 'Incorrect'], xlabel='Predicted', ylabel='Original label',
                title='Primary: geometry random forest')
    pts = [r for r in per_subject if r['incorrect'] > 0]
    axes[2].bar(range(len(pts)), [r['incorrect_recall'] for r in pts], color='#c66a40')
    axes[2].set(xticks=range(len(pts)), xticklabels=[str(r['subject']) for r in pts],
                ylim=(0, 1.04), ylabel='Incorrect recall', xlabel='Subject ID (has incorrect examples)',
                title='Performance can vary by person')
    axes[2].tick_params(axis='x', rotation=60)
    fig.suptitle('IRDS shoulder flexion | exploratory 5-fold subject-grouped evaluation', fontsize=14)
    fig.savefig(OUT / 'pilot_results.png', dpi=170)
    plt.close(fig)


def write_report(s):
    lines = ['# IRDS 肩前屈粗基线实验', '', '运行日期：2026-11-06。探索实验，不是最终模型或临床验证。', '',
             f"数据：Gesture 2/3，共 {s['n']} 次动作、{s['subjects']} 个文件受试者 ID；正确 {s['correct']}、错误 {s['incorrect']}。排除 2 条特殊标签。", '',
             '## 实验设置', '',
             '- 错误为阳性。按受试者 ID 固定五折分组交叉验证，同一个人的两侧、体位、重复均在同一折。',
             '- 所有表中成绩均来自各样本所在测试折的预测，模型未用该人的样本训练。',
             '- 固定模型参数和 0.5 阈值；没有调参或依据测试成绩选择阈值。标准化只在训练折拟合。',
             '- 预先指定 geometry_rf 为主模型。30 个几何特征包括双侧抬臂、肘角、躯干相对起始朝向变化、手腕相对头部高度等。',
             '- Gesture 2 使用 Raw Right，Gesture 3 使用 Raw Left；此为既有审计对应关系，不代表已解决解剖左右命名。',
             '- 不输入受试者 ID、患者身份、体位、绝对坐标或 Gesture ID。含帧数模型单独列出。',
             '- 输入仍可能包含体位和身体能力线索。未处理跟踪质量，也未验证时间单位，未使用速度或加速度。', '',
             '## 各模型结果', '',
             '| 模型 | Accuracy | Balanced accuracy | 错误 Precision | 错误 Recall | 错误 F1 | Macro-F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, v in s['models'].items():
        m = v['overall']
        lines.append('| '+name+' | '+' | '.join(f"{m[k]:.1%}" for k in
            ['accuracy', 'balanced_accuracy', 'incorrect_precision', 'incorrect_recall', 'incorrect_f1', 'macro_f1'])+' |')
    lines += ['', '## 主模型的限制与分层结果', '']
    for key, m in s['models']['geometry_rf']['subgroups'].items():
        ba = '不可计算' if m['balanced_accuracy'] is None else f"{m['balanced_accuracy']:.1%}"
        recall = '无错误样本' if m['incorrect_recall'] is None else f"{m['incorrect_recall']:.1%}"
        lines.append(f"- {key}：n={m['n']}，错误={m['incorrect']}，accuracy={m['accuracy']:.1%}，balanced accuracy={ba}，错误 recall={recall}。")
    lines += ['', '受试者聚类 bootstrap 95% 区间（基于已得到的外折预测，不包含重新训练和不同划分的不确定性）：', '']
    for k, v in s['primary_subject_bootstrap_95_percent'].items():
        lines.append(f'- {k}: {v[0]:.1%}–{v[1]:.1%}')
    lines += ['', '## 动作偏离解释', '',
              'exploratory_reference_deviations.csv 使用各训练折中同 Gesture、同体位的正确动作建立参考（至少 10 条）。低于抬臂峰值/高抬臂阶段肘角第 5 百分位，或高于躯干朝向变化第 95 百分位时给出提示。', '',
              '这些是相对参考的偏离提示，独立于分类决策；不是已验证的错误原因，不代表模型靠这些原因判错。阈值是本次探索规则，不是原论文数值准则。没有实现摆臂代偿的可靠识别。', '',
              '## 文件与复现', '',
              '- summary.json：完整指标、每折受试者、分层指标及配置。',
              '- heldout_predictions.csv：每次动作的外折预测，供检查具体成功/失败样本。',
              '- primary_per_subject.csv：主模型按受试者结果。只有一种标签时不计算 balanced accuracy。',
              '- features.csv：可复查的输入特征。',
              '- *.joblib：在全部 692 条数据上重训的演示模型；这些文件本身没有独立测试集，成绩来自外折预测。',
              '- pilot_results.png：比较图、主模型混淆矩阵和按人错误召回。',
              '- environment.txt：运行依赖版本。', '',
              '复现：使用本实验目录 .venv/Scripts/python.exe 执行 run_pilot.py。', '',
              '不能据此声称临床纠错有效、适用于手机视频、或确定哪个模型最终最好。ID 217 身份未核实；当前按文件 ID 分组。没有因测试错误而筛除样本。']
    (HERE / 'REPORT_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
