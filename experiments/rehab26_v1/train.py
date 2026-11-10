"""Run the predeclared five-person, two-exercise baseline/ablation suite on CPU."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline import ROOT, HERE, angle_statistics, build_tcn, feature_indices, metrics, reliability_margin, save_json, select_reliability_policy, sha256, thresholds_from_validation


def predict_tcn(model, x, batch_size=128):
    model.eval()
    outputs = []
    with torch.no_grad():
        for first in range(0, len(x), batch_size):
            outputs.append(torch.sigmoid(model(torch.from_numpy(x[first:first + batch_size]))).numpy())
    return np.concatenate(outputs)


def fit_tcn(x_train, y_train, x_validation, y_validation, config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_tcn(x_train.shape[1], config)
    positive = y_train.sum(axis=0)
    weights = torch.tensor((len(y_train) - positive) / np.maximum(positive, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
                        batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    validation_x, validation_y = torch.from_numpy(x_validation), torch.from_numpy(y_validation.astype(np.float32))
    best_loss, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_y)
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            for first in range(0, len(validation_x), 128):
                count = len(validation_x[first:first + 128])
                val_total += float(criterion(model(validation_x[first:first + 128]), validation_y[first:first + 128])) * count
        val_loss = val_total / len(validation_y)
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, int(sum(p.numel() for p in model.parameters()))


def index_groups(observations, exercise, fold):
    groups = {part: np.asarray([i for i, row in enumerate(observations)
                               if row["exercise_id"] == exercise and row["person"] in fold[part]])
              for part in ["train", "validation", "test"]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(groups[a]) & set(groups[b])
        assert not ({observations[i]["sample_id"] for i in groups[a]} & {observations[i]["sample_id"] for i in groups[b]})
        assert not set(fold[a]) & set(fold[b])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/rehab26_v1_20261109_cpu")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--exercises", type=int, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+")
    args = parser.parse_args()
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    data_dir = ROOT / "data/REHAB26-ViewAngles/processed_v1"
    metadata = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    if metadata["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Protocol changed after sample preparation")
    if metadata["pipeline_sha256"] != sha256(HERE / "pipeline.py"):
        raise ValueError("Preprocessing/shared code changed after sample preparation; use a new data version")
    if metadata["samples_sha256"] != sha256(data_dir / "samples.npz"):
        raise ValueError("Cached samples changed")
    observations = json.loads((data_dir / "observations.json").read_text(encoding="utf-8"))
    folds = json.loads((data_dir / "folds.json").read_text(encoding="utf-8"))
    with np.load(data_dir / "samples.npz", allow_pickle=False) as data:
        x, x_noisy, y = data["x"], data["x_noisy"], data["y"].astype(bool)
    stats, stats_noisy = angle_statistics(x), angle_statistics(x_noisy)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": sha256(protocol_path), "samples_sha256": metadata["samples_sha256"],
                "pipeline_sha256": sha256(HERE / "pipeline.py"), "train_sha256": sha256(HERE / "train.py")}
    identity_path = run / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Run directory belongs to different code/data; choose a new directory")
    save_json(identity_path, identity)
    save_json(run / "protocol.json", protocol)
    save_json(run / "environment.json", {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
        "device": "cpu", "threads": protocol["torch_threads"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    torch.set_num_threads(protocol["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    threadpool_limits(limits=protocol["torch_threads"])
    models = args.models or protocol["models"]
    if set(models) - set(protocol["models"]):
        raise ValueError("Model not in predeclared protocol")
    for exercise in args.exercises or protocol["exercises"]:
        labels = metadata["labels"][str(exercise)]
        for fold in folds:
            if args.folds is not None and fold["fold"] not in args.folds:
                continue
            groups = index_groups(observations, exercise, fold)
            train, validation, test = [groups[k] for k in ["train", "validation", "test"]]
            seed = protocol["seed"] + exercise * 100 + fold["fold"]
            for variant in models:
                destination = run / f'ex{exercise}/fold{fold["fold"]}_{fold["test"][0]}/{variant}'
                if (destination / "result.json").exists():
                    print(f"Already completed: {destination.relative_to(run)}", flush=True)
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                history, best_epoch, n_parameters = [], None, None
                print(f'Start ex{exercise} fold{fold["fold"]} test={fold["test"][0]} {variant}', flush=True)
                if variant.startswith("angle_"):
                    estimators = []
                    for j in range(5):
                        if variant == "angle_logistic":
                            estimator = make_pipeline(StandardScaler(), LogisticRegression(**protocol["logistic"], random_state=seed + j))
                        else:
                            estimator = RandomForestClassifier(**protocol["random_forest"], random_state=seed + j)
                        estimator.fit(stats[train], y[train, j])
                        estimators.append(estimator)
                    def score(data):
                        return np.stack([estimator.predict_proba(data)[:, 1] for estimator in estimators], axis=1)
                    validation_scores, test_scores, stress_scores = score(stats[validation]), score(stats[test]), score(stats_noisy[test])
                    artifact = {"estimators": estimators, "variant": variant, "exercise": exercise, "labels": labels,
                                "fold": fold, "seed": seed, "identity": identity}
                else:
                    indices = feature_indices(variant)
                    selected = x[:, :, indices]
                    mean = selected[train].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
                    std = np.maximum(selected[train].std(axis=(0, 1), dtype=np.float64), 1e-4).astype(np.float32)
                    def standardize(data):
                        return np.ascontiguousarray(((data[:, :, indices] - mean) / std).transpose(0, 2, 1))
                    train_x, validation_x, test_x = standardize(x[train]), standardize(x[validation]), standardize(x[test])
                    model, history, best_epoch, n_parameters = fit_tcn(train_x, y[train], validation_x, y[validation], protocol["tcn"], seed)
                    validation_scores = predict_tcn(model, validation_x)
                    test_scores = predict_tcn(model, test_x)
                    stress_scores = predict_tcn(model, standardize(x_noisy[test]))
                    artifact = {"state_dict": model.state_dict(), "variant": variant, "exercise": exercise, "labels": labels,
                                "fold": fold, "seed": seed, "identity": identity, "tcn_config": protocol["tcn"],
                                "feature_indices": indices.tolist(), "mean": mean.tolist(), "std": std.tolist()}
                thresholds = thresholds_from_validation(y[validation], validation_scores)
                policy = select_reliability_policy(y[validation], validation_scores, thresholds, protocol["reliability"]["margin_grid"])
                artifact.update({"thresholds": thresholds.tolist(), "reliability_policy": policy})
                if variant.startswith("angle_"):
                    joblib.dump(artifact, destination / "model.joblib", compress=3)
                else:
                    torch.save(artifact, destination / "model.pt")
                predicted = test_scores >= thresholds
                margin = reliability_margin(test_scores, thresholds)
                keep = margin >= policy["cutoff"] if policy["qualified_on_validation"] else np.ones(len(test), dtype=bool)
                np.savez_compressed(destination / "predictions.npz", test_indices=test, validation_indices=validation,
                                    y_test=y[test], test_scores=test_scores, stress_scores=stress_scores,
                                    y_validation=y[validation], validation_scores=validation_scores,
                                    thresholds=thresholds, retained=keep)
                result = {"exercise": exercise, "model": variant, "fold": fold, "seed": seed,
                          "n_train_observations": len(train), "n_validation_observations": len(validation),
                          "n_test_observations": len(test), "n_test_physical_repetitions": len(test) // 3,
                          "thresholds": thresholds.tolist(), "best_epoch": best_epoch, "n_parameters": n_parameters,
                          "elapsed_seconds": time.perf_counter() - start,
                          "validation": metrics(y[validation], validation_scores >= thresholds, labels),
                          "test_unconditional": metrics(y[test], predicted, labels),
                          "test_synthetic_noise": metrics(y[test], stress_scores >= thresholds, labels),
                          "reliability_policy": policy, "test_coverage": float(keep.mean()),
                          "test_retained": metrics(y[test][keep], predicted[keep], labels),
                          "history": history}
                save_json(destination / "result.json", result)
                print(f'Done F1={result["test_unconditional"]["macro_f1"]:.3f}, exact={result["test_unconditional"]["exact_match"]:.3f}, '
                      f'noise_F1={result["test_synthetic_noise"]["macro_f1"]:.3f}, seconds={result["elapsed_seconds"]:.1f}', flush=True)
                if variant.startswith("tcn_"):
                    del model, train_x, validation_x, test_x, selected
    print(f"Requested suite complete: {run}", flush=True)


if __name__ == "__main__":
    main()
