"""Train one frozen candidate on three people; return validation-only selection evidence."""
from __future__ import annotations

import copy
import json
import math
import random
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from core import (angle_statistics, build_model, digest, fit_scaler, groups, metrics,
                  save_json, score_model, select_thresholds, transform)


def fit_neural(x, y, xv, yv, config, training, seed, fixed_epochs=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(config).cuda()
    positive = y.sum(0)
    assert (positive > 0).all()
    ratio = (len(y)-positive)/positive
    weight = {"none": np.ones(5), "sqrt": np.clip(np.sqrt(ratio), 1, 3), "raw": ratio}[config.get("loss", "none")]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device="cuda", dtype=torch.float32))
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.get("weight_decay", .001)},
                                   {"params": no_decay, "weight_decay": 0.}], lr=config.get("lr", .0003))
    tensors = [torch.from_numpy(a).cuda() for a in [x, y, xv, yv]]
    tx, ty, vx, vy = tensors
    max_epochs = training["max_epochs"]
    best_ap, best_epoch, state, stale = -1., 0, None, 0
    history = []
    rng = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(1, (fixed_epochs or max_epochs)+1):
        base_lr = config.get("lr", .0003)
        factor = epoch/5 if epoch <= 5 else .5*(1+math.cos(math.pi*(epoch-5)/(max_epochs-5)))
        lr = base_lr*factor if epoch <= 5 else 1e-5+(base_lr-1e-5)*factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = torch.randperm(len(tx), generator=rng, device="cuda")
        model.train()
        total = torch.zeros((), device="cuda")
        norms = []
        for batch in order.split(training["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tx[batch]), ty[batch])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            norms.append(norm.detach())
            optimizer.step()
            total += loss.detach()*len(batch)
        scores = score_model(model, vx)
        val = metrics(yv, scores)
        with torch.no_grad():
            probs = np.clip(scores, 1e-7, 1-1e-7)
            unweighted_bce = float(-(yv*np.log(probs)+(1-yv)*np.log1p(-probs)).mean())
        history.append({"epoch": epoch, "lr": lr, "train_loss": float(total/len(tx)),
                        "validation_bce": unweighted_bce, "validation_macro_ap": val["macro_ap"],
                        "gradient_norm_mean_before_clip": float(torch.stack(norms).mean())})
        if val["macro_ap"] > best_ap+training["min_delta"]:
            best_ap, best_epoch, stale = val["macro_ap"], epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch > 5:
            stale += 1
        if fixed_epochs is None and stale >= training["patience"]:
            break
    if fixed_epochs is None:
        model.load_state_dict(state)
    else:
        best_epoch = fixed_epochs
    torch.cuda.synchronize()
    diagnostic = {"history": history, "best_epoch": best_epoch, "positive_weight": weight.tolist(),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.perf_counter()-start}
    return model, diagnostic


def fit_one(x, y, observations, exercise, test, validation, config, training, seed, folder, identity):
    expected = {"identity": identity, "config": config, "training": training, "seed": seed,
                "exercise": exercise, "test": test, "validation": validation}
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["job_hash"] == digest(expected), f"Incompatible resume: {folder}"
        return result
    folder.mkdir(parents=True, exist_ok=True)
    ids, split = groups(observations, exercise, test, validation)
    train, valid = ids["train"], ids["validation"]
    start = time.perf_counter()
    if config["family"] in ("lr", "rf"):
        stats = angle_statistics(x)
        estimators = []
        with threadpool_limits(limits=3):
            for j in range(5):
                if config["family"] == "lr":
                    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed+j))
                else:
                    estimator = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=3, class_weight="balanced", n_jobs=3, random_state=seed+j)
                estimator.fit(stats[train], y[train, j])
                estimators.append(estimator)
        scores = np.stack([e.predict_proba(stats[valid])[:, 1] for e in estimators], 1)
        artifact = {**expected, "estimators": estimators, "split": split}
        diagnostic = {"elapsed_seconds": time.perf_counter()-start, "history": [], "best_epoch": None, "parameters": None}
        extension = "joblib"
    else:
        mean, std = fit_scaler(x, train, config)
        model, diagnostic = fit_neural(transform(x[train], config, mean, std), y[train],
                                       transform(x[valid], config, mean, std), y[valid], config, training, seed)
        scores = score_model(model, transform(x[valid], config, mean, std))
        artifact = {**expected, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "mean": mean.tolist(), "std": std.tolist(), "split": split}
        extension = "pt"
    thresholds = select_thresholds(y[valid], scores)
    conservative = select_thresholds(y[valid], scores, beta=.5)
    artifact.update({"thresholds": thresholds.tolist(), "precision_thresholds": conservative.tolist(),
                     "best_epoch": diagnostic["best_epoch"]})
    if extension == "pt":
        torch.save(artifact, folder / "model.pt")
        del model
        torch.cuda.empty_cache()
    else:
        joblib.dump(artifact, folder / "model.joblib", compress=3)
    np.savez_compressed(folder / "validation.npz", ids=valid, y=y[valid], scores=scores,
                        thresholds=thresholds, precision_thresholds=conservative)
    result = {**expected, "job_hash": digest(expected), "split": split,
              "n_train": len(train), "n_validation": len(valid),
              "validation": metrics(y[valid], scores, thresholds),
              "validation_precision": metrics(y[valid], scores, conservative),
              "validation_fixed05": metrics(y[valid], scores), **diagnostic}
    save_json(result_path, result)
    return result


def load_artifact(folder):
    folder = __import__("pathlib").Path(folder)
    if (folder / "model.pt").exists():
        artifact = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        model = build_model(artifact["config"])
        model.load_state_dict(artifact["state_dict"])
        artifact["model"] = model.cuda().eval()
    else:
        artifact = joblib.load(folder / "model.joblib")
    return artifact


def predict_artifact(artifact, x):
    if "estimators" in artifact:
        stats = angle_statistics(x)
        return np.stack([e.predict_proba(stats)[:, 1] for e in artifact["estimators"]], 1)
    return score_model(artifact["model"], transform(x, artifact["config"],
                       np.asarray(artifact["mean"], np.float32), np.asarray(artifact["std"], np.float32)))
