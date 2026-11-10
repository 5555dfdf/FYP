"""Geometry checks for fusion, plus sample isolation and raw-data replay."""
from __future__ import annotations

import json
import numpy as np

from fusion import ROOT, align_view, fuse_segments, merge_aligned
from pipeline import make_features, save_json


def geometry_checks():
    rng = np.random.default_rng(611)
    xyz = rng.normal(0, .1, (96, 33, 3)).astype(np.float32)
    xyz[:, 23] = [-.2, 0, 0]
    xyz[:, 24] = [.2, 0, 0]
    xyz[:, 11] = [-.2, -1, 0]
    xyz[:, 12] = [.2, -1, 0]
    # Visible forward inclination and changing shoulder/pelvis rotation.
    xyz[:, 11:13, 2] += .3
    xyz[:, 11, 2] += np.linspace(0, .1, 96)
    xyz[:, 12, 2] -= np.linspace(0, .1, 96)
    segments = []
    for theta, scale, shift in zip([-.8, .2, 1.1], [.6, 1.4, 2.], [[1, 2, 3], [-2, 1, 0], [2, -1, 1]]):
        c, s = np.cos(theta), np.sin(theta)
        view = xyz.copy()
        view[..., 0] = c * xyz[..., 0] - s * xyz[..., 2]
        view[..., 2] = s * xyz[..., 0] + c * xyz[..., 2]
        view = view * scale + np.asarray(shift)
        segments.append(np.concatenate([view, np.ones((96, 33, 1))], -1))
    result = fuse_segments(segments)
    ref, vis, _ = align_view(segments[0])
    np.testing.assert_allclose(result["x"], make_features(ref, vis), atol=2e-5)
    # Both tilt and the change of relative torso yaw survive alignment/fusion.
    assert result["xyz"][:, 11:13, 2].mean() > .2
    shoulder = result["xyz"][:, 12] - result["xyz"][:, 11]
    assert abs(shoulder[-1, 2] - shoulder[0, 2]) > .15
    np.testing.assert_allclose(result["x"], fuse_segments(segments[::-1])["x"], atol=2e-5)
    clean = np.stack([ref] * 3)
    noisy = clean.copy()
    noisy[0, :, 16, 0] += 3
    visibility = np.ones((3, 96, 33), dtype=np.float32)
    visibility[0, :, 16] = .01
    weighted = merge_aligned(noisy, visibility)[0]
    equal = merge_aligned(noisy, visibility, "equal")[0]
    assert np.abs(weighted[:, 16] - ref[:, 16]).mean() < np.abs(equal[:, 16] - ref[:, 16]).mean() / 100
    fallback = merge_aligned(clean, np.zeros_like(visibility))
    assert fallback[2]["fallback_joint_frames"] == 96 * 33
    assert np.isfinite(fallback[0]).all()
    return ["yaw/translation/scale invariance", "torso tilt and relative rotation retained", "camera permutation invariant",
            "low-visibility corrupted joint downweighted", "all-zero weights explicit finite fallback"]


def main():
    checks = geometry_checks()
    data = ROOT / "data/REHAB26-ViewAngles/processed_fusion_v3"
    if (data / "dataset.json").exists():
        import run
        arrays, observations, _ = run.read_data()
        for ex in [1, 7]:
            subjects = sorted({r["person"] for r in observations})
            for test in subjects:
                for val in subjects:
                    if val == test:
                        continue
                    groups, _ = run.core.groups(observations, ex, test, val)
                    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                        assert not {observations[i]["sample_id"] for i in groups[a]} & {observations[i]["sample_id"] for i in groups[b]}
        deltas = []
        for person in sorted({r["person"] for r in observations}):
            for ex in [1, 7]:
                i = next(i for i, r in enumerate(observations) if r["person"] == person and r["exercise_id"] == ex)
                row = observations[i]
                segments = [np.load(ROOT / "data/REHAB26-ViewAngles/raw" / v["skeleton_file"], mmap_mode="r")[row["first_frame"]:row["last_frame"] + 1] for v in row["source_views"]]
                fresh = fuse_segments(segments)["x"]
                delta = float(np.abs(fresh - arrays["x"][i]).max())
                assert delta < 2e-6
                deltas.append(delta)
        checks += ["40 subject-isolated splits", "10 raw repetition replays"]
        report = {"status": "passed", "checks": checks, "raw_replay_max_difference": max(deltas)}
        save_json(ROOT / "runs/rehab26_fusion_v3/fusion_verification.json", report)
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
