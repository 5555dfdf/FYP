"""Audit REHAB26-ViewAngles locally; no downloads, training or preprocessing.

Run from any directory: python tools/audit_rehab26.py
Raw files are retained. Outputs under data/ are intentionally ignored by Git.
Only NumPy and the Python standard library are needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np


ARCHIVE_MD5 = "303ba8e1c236ee52dac7c0b8ac23789f"
ARCHIVE_BYTES = 686028292
ANNOTATION_MD5 = {
    "Error_types.csv": "2f89da48c2c26b4868a59deaa4fb3ed9",
    "Segmentation.txt": "502475f709390d96b599da65fa45ffd3",
    "joint_names.txt": "2d6999cedf3775d3e9dded2e87b8d035",
    "Segmentation.csv": "3d1cf6b552f60ef03eb8f5db977c139f",
}
FILE_PATTERN = re.compile(r"mediapipe/ex(1|6|7)/(ex\1_([^/]+))_cam([123])_(2d|3d)\.npy")
REQUIRED_COLUMNS = {
    "video_id", "exercise_id", "exercise_subtype", "person",
    "repetition_number", "first_frame", "last_frame", "person_orientation",
    "correctness_score", "error_description", "additional_errors",
}
ARM_JOINTS = [0, 11, 12, 14, 16, 23, 24]
SQUAT_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def fingerprint(path: Path) -> dict:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def distribution(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"n": 0}
    return {"n": int(values.size), **dict(zip(
        ["min", "p05", "median", "p95", "max"],
        map(float, np.quantile(values, [0, .05, .5, .95, 1])),
    ))}


def extract_checked(archive: Path, raw: Path) -> list[tuple[str, tuple]]:
    entries = []
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for info in bundle.infolist():
            if info.is_dir():
                continue
            match = FILE_PATTERN.fullmatch(info.filename)
            if not match or info.filename in seen:
                raise ValueError(f"Unexpected or duplicate ZIP member: {info.filename}")
            seen.add(info.filename)
            destination = (raw / Path(*PurePosixPath(info.filename).parts)).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError(f"Unsafe ZIP path: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # Verify an existing extraction rather than silently replacing it.
                import zlib
                crc = 0
                with destination.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if destination.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError(f"Existing file differs from ZIP: {destination}")
            else:
                # Reading to EOF also asks zipfile to verify the member CRC.
                with bundle.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            entries.append((info.filename, match.groups()))
    return sorted(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "data/REHAB26-ViewAngles"
    raw, output = base / "raw", base / "audit"
    archive = root / "jointSequences-mediapipe.zip"
    sources = {name: fingerprint(raw / name) for name in (
        "Segmentation.csv", "Error_types.csv", "Segmentation.txt", "joint_names.txt",
    )}
    for name, expected_md5 in ANNOTATION_MD5.items():
        if sources[name]["md5"] != expected_md5:
            raise ValueError(f"Annotation MD5 differs from official record: {name}")
    print("Checking archive hashes...", flush=True)
    archive_hash = fingerprint(archive)
    if archive_hash["bytes"] != ARCHIVE_BYTES or archive_hash["md5"] != ARCHIVE_MD5:
        raise ValueError("Archive size/MD5 differs from the recorded official values")
    print("Extracting/verifying 90 skeleton files...", flush=True)
    entries = extract_checked(archive, raw)
    annotations = read_csv(raw / "Segmentation.csv")
    if not annotations or not REQUIRED_COLUMNS.issubset(annotations[0]):
        raise ValueError("Annotation schema differs from the documented fields")
    error_types = read_csv(raw / "Error_types.csv")
    allowed = defaultdict(set)
    for row in error_types:
        allowed[int(row["exercise_id"])].add(row["error_code"])
    issues, warnings, samples = [], [], []
    if len(entries) != 90:
        issues.append({"code": "unexpected_array_count", "actual": len(entries), "expected": 90})
    per_recording = defaultdict(list)
    ids = set()
    for line, row in enumerate(annotations, 2):
        ex, first, last, orientation, repetition, score = [int(row[k]) for k in (
            "exercise_id", "first_frame", "last_frame", "person_orientation", "repetition_number", "correctness_score",
        )]
        sample_id = f'{row["video_id"]}_o{orientation:02d}_r{repetition:03d}'
        primary = row["error_description"].strip()
        additional = [v.strip() for v in row["additional_errors"].split(",") if v.strip()]
        labels = sorted(set(([primary] if primary else []) + additional))
        record = {
            "sample_id": sample_id, "source_csv_line": line,
            "video_id": row["video_id"], "person": row["person"], "exercise_id": ex,
            "exercise_subtype": row["exercise_subtype"], "person_orientation": orientation,
            "repetition_number": repetition, "first_frame": first, "last_frame": last,
            "n_frames_inclusive": last - first + 1, "duration_seconds_at_30fps": (last - first + 1) / 30,
            "correctness_score": score, "primary_error": primary or None,
            "additional_errors": additional, "error_labels": labels,
            "views": [{"camera": cam, "view_angle_degrees": {1: -orientation, 2: 45 - orientation, 3: 90 - orientation}[cam],
                       "skeleton_files": {kind: f'mediapipe/ex{ex}/{row["video_id"]}_cam{cam}_{kind}.npy' for kind in ("2d", "3d")}}
                      for cam in (1, 2, 3)],
        }
        def flag(code, **details):
            issues.append({"code": code, "sample_id": sample_id, "csv_line": line, **details})
        if sample_id in ids:
            flag("duplicate_sample_id")
        ids.add(sample_id)
        if row["video_id"] != f'ex{ex}_{row["person"]}':
            flag("recording_person_exercise_mismatch")
        if first < 0 or last < first:
            flag("invalid_frame_interval")
        if score not in (1, 5) or (score == 1 and labels) or (score == 5 and not primary):
            flag("correctness_label_mismatch")
        if orientation not in {*range(0, 91, 10), 45}:
            flag("unexpected_orientation")
        if (ex in (1, 7) and row["exercise_subtype"] != "right") or (ex == 6 and row["exercise_subtype"]):
            flag("unexpected_exercise_subtype")
        for code in labels:
            if code not in allowed[ex]:
                flag("unknown_error_code", error_code=code)
        if primary in additional or len(additional) != len(set(additional)):
            flag("duplicate_error_label")
        samples.append(record)
        per_recording[row["video_id"]].append(record)

    for video_id, records in per_recording.items():
        previous = None
        for record in sorted(records, key=lambda r: r["first_frame"]):
            if previous and record["first_frame"] <= previous["last_frame"]:
                issues.append({"code": "overlapping_intervals", "sample_id": record["sample_id"], "previous_id": previous["sample_id"]})
            if previous is None or record["last_frame"] > previous["last_frame"]:
                previous = record
        grouped = defaultdict(list)
        for record in records:
            grouped[record["person_orientation"]].append(record["repetition_number"])
        for orientation, repetitions in grouped.items():
            if sorted(repetitions) != list(range(1, max(repetitions) + 1)):
                issues.append({"code": "repetition_number_gap_or_duplicate", "video_id": video_id, "orientation": orientation})

    inventory, view_checks = [], []
    available = {name for name, _ in entries}
    expected = {path for s in samples for v in s["views"] for path in v["skeleton_files"].values()}
    for name in sorted(expected - available):
        issues.append({"code": "missing_skeleton_file", "file": name})
    for name in sorted(available - expected):
        issues.append({"code": "unreferenced_skeleton_file", "file": name})
    visibility_pair_checks = []
    for index, (name, groups) in enumerate(entries, 1):
        ex_string, video_id, person, camera_string, kind = groups
        ex, camera = int(ex_string), int(camera_string)
        a = np.load(raw / name, mmap_mode="r", allow_pickle=False)
        if a.ndim != 3 or a.shape[1:] != (33, 4) or a.dtype.kind != "f":
            issues.append({"code": "unexpected_array_schema", "file": name, "shape": list(a.shape), "dtype": str(a.dtype)})
            continue
        coords, visibility = a[..., :3], a[..., 3]
        finite_coords = np.isfinite(coords).all(axis=-1)
        finite_vis = np.isfinite(visibility)
        valid_joint = finite_coords & finite_vis
        zero_xyz_frames = (coords == 0).all(axis=(1, 2))
        zero_placeholder_frames = zero_xyz_frames & (visibility == 0).all(axis=1)
        xyz = coords.reshape(-1, 3)
        finite_xyz = xyz[np.isfinite(xyz).all(axis=1)]
        finite_v = visibility[finite_vis]
        entry = {
            "file": name, "exercise_id": ex, "person": person, "camera": camera, "modality": kind,
            "shape": list(a.shape), "dtype": str(a.dtype), "n_frames": len(a),
            "nonfinite_coordinate_values": int((~np.isfinite(coords)).sum()),
            "nonfinite_visibility_values": int((~finite_vis).sum()),
            "frames_all_joints_invalid": int((~valid_joint.any(axis=1)).sum()),
            "frames_any_joint_invalid": int((~valid_joint.all(axis=1)).sum()),
            "all_zero_xyz_frames": int(zero_xyz_frames.sum()),
            "all_zero_xyz_and_visibility_frames": int(zero_placeholder_frames.sum()),
            "visibility_outside_0_1": int(((visibility < 0) | (visibility > 1)).sum()),
            "visibility": distribution(finite_v),
            "visibility_below_0_5_fraction": float(np.mean(finite_v < .5)) if finite_v.size else None,
            "xyz_min": finite_xyz.min(axis=0).tolist() if finite_xyz.size else None,
            "xyz_max": finite_xyz.max(axis=0).tolist() if finite_xyz.size else None,
            "hip_midpoint_norm": distribution(np.linalg.norm((a[:, 23, :3] + a[:, 24, :3]) / 2, axis=1)),
            "joint_visibility_mean": [float(v) if np.isfinite(v) else None for v in np.nanmean(visibility, axis=0)],
        }
        if kind == "2d":
            xy = a[..., :2]
            entry["xy_outside_0_1_joint_fraction"] = float(((xy < 0) | (xy > 1)).any(axis=-1).mean())
        inventory.append(entry)
        for record in per_recording[video_id]:
            first, last = record["first_frame"], record["last_frame"]
            in_bounds = 0 <= first <= last < len(a)
            check = {"sample_id": record["sample_id"], "person": person, "exercise_id": ex,
                     "camera": camera, "modality": kind, "file": name, "in_bounds": in_bounds}
            if not in_bounds:
                issues.append({"code": "out_of_bounds", "sample_id": record["sample_id"], "file": name, "n_frames": len(a)})
                check.update({"invalid_joint_fraction": None, "relevant_visibility_mean": None,
                              "relevant_visibility_below_0_5_fraction": None, "frames_any_joint_invalid": None,
                              "all_zero_xyz_frames": None, "all_zero_xyz_and_visibility_frames": None})
            else:
                segment_valid = valid_joint[first:last + 1]
                joints = SQUAT_JOINTS if ex == 6 else ARM_JOINTS
                vis = visibility[first:last + 1, joints]
                check.update({"invalid_joint_fraction": float((~segment_valid).mean()),
                              "relevant_visibility_mean": float(np.mean(vis[np.isfinite(vis)])) if np.isfinite(vis).any() else None,
                              "relevant_visibility_below_0_5_fraction": float((vis[np.isfinite(vis)] < .5).mean()) if np.isfinite(vis).any() else None,
                              "frames_any_joint_invalid": int((~segment_valid.all(axis=1)).sum()),
                              "all_zero_xyz_frames": int(zero_xyz_frames[first:last + 1].sum()),
                              "all_zero_xyz_and_visibility_frames": int(zero_placeholder_frames[first:last + 1].sum())})
                if check["all_zero_xyz_and_visibility_frames"]:
                    warnings.append({"code": "zero_placeholder_inside_repetition", "sample_id": record["sample_id"], "file": name})
            view_checks.append(check)
        if kind == "3d":
            other_name = name.replace("_3d.npy", "_2d.npy")
            if other_name in available:
                other = np.load(raw / other_name, mmap_mode="r", allow_pickle=False)
                same_shape = other.shape == a.shape
                same_visibility = same_shape and np.array_equal(other[..., 3], visibility, equal_nan=True)
                visibility_pair_checks.append({"recording": video_id, "camera": camera,
                                               "same_shape": same_shape, "identical_visibility": same_visibility})
                if not same_shape:
                    issues.append({"code": "2d_3d_frame_count_mismatch", "file": name})
                del other
        del a
        if index % 15 == 0:
            print(f"Audited {index}/{len(entries)} arrays", flush=True)

    grouped_files = defaultdict(list)
    for record in inventory:
        grouped_files[f'ex{record["exercise_id"]}_{record["person"]}'].append(record)
    sync_checks = []
    for video_id, records in sorted(grouped_files.items()):
        lengths = {f'cam{r["camera"]}_{r["modality"]}': r["n_frames"] for r in records}
        same_length = len(set(lengths.values())) == 1
        sync_checks.append({"video_id": video_id, "lengths": lengths, "equal_lengths": same_length})
        if not same_length:
            warnings.append({"code": "camera_total_length_difference", "video_id": video_id, "lengths": lengths,
                             "note": "Official description allows truncated sequence tails; equal frame indices are documented, RGB alignment not independently verified."})

    counts = []
    for ex in sorted(allowed):
        for person in sorted({r["person"] for r in samples}):
            subset = [r for r in samples if r["exercise_id"] == ex and r["person"] == person]
            counts.append({"exercise_id": ex, "person": person, "repetitions": len(subset),
                           "correct": sum(r["correctness_score"] == 1 for r in subset),
                           "incorrect": sum(r["correctness_score"] == 5 for r in subset),
                           "with_additional_errors": sum(bool(r["additional_errors"]) for r in subset)})
    label_counts = []
    for ex in sorted(allowed):
        for code in sorted(allowed[ex]):
            for person in sorted({r["person"] for r in samples}):
                subset = [r for r in samples if r["exercise_id"] == ex and r["person"] == person]
                label_counts.append({"exercise_id": ex, "error_code": code, "person": person,
                                     "primary_count": sum(r["primary_error"] == code for r in subset),
                                     "any_label_count": sum(code in r["error_labels"] for r in subset)})
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "numpy_version": np.__version__, "source_record": "https://zenodo.org/records/23018128",
        "archive": {"path": str(archive.relative_to(root)), **archive_hash, "matches_recorded_official_md5": True},
        "annotation_fingerprints": sources,
        "annotation_md5_verified": True,
        "conventions": {"frame_interval": "inclusive [first_frame, last_frame], as first/last frame names imply; visual boundary accuracy unverified",
                        "fps": 30, "coordinates": "x,y,z,visibility per official record. 3d documented as world landmarks in meters, hip-centered. 2d documented as pixel xy but numeric ranges resemble normalized xy; discrepancy unresolved, no conversion applied.",
                        "visibility": "channel 4 treated as MediaPipe visibility; 0.5 is a descriptive threshold only, not a training or rejection policy",
                        "invalid_joint_fraction": "nonfinite coordinate/visibility entries only; low visibility and all-zero placeholders are counted separately",
                        "counting_unit": "one CSV row = one physical repetition; camera views and 2d/3d modalities are linked observations",
                        "relevant_joints_arm": ARM_JOINTS, "relevant_joints_squat": SQUAT_JOINTS},
        "n_repetitions": len(samples), "n_unique_sample_ids": len(ids),
        "n_recordings": len(per_recording), "n_arrays": len(inventory),
        "n_repetition_camera_observations": len(samples) * 3,
        "n_repetition_camera_modality_checks": len(view_checks),
        "exercise_counts": dict(Counter(r["exercise_id"] for r in samples)),
        "person_counts": dict(Counter(r["person"] for r in samples)),
        "correctness_counts": dict(Counter(r["correctness_score"] for r in samples)),
        "n_with_additional_errors": sum(bool(r["additional_errors"]) for r in samples),
        "n_multilabel_repetitions": sum(len(r["error_labels"]) > 1 for r in samples),
        "duration_seconds": {str(ex): distribution([r["duration_seconds_at_30fps"] for r in samples if r["exercise_id"] == ex]) for ex in sorted(allowed)},
        "n_out_of_bounds_checks": sum(not r["in_bounds"] for r in view_checks),
        "n_checks_with_invalid_joints": sum(bool(r["frames_any_joint_invalid"]) for r in view_checks),
        "n_nonfinite_coordinate_values": sum(r["nonfinite_coordinate_values"] for r in inventory),
        "n_nonfinite_visibility_values": sum(r["nonfinite_visibility_values"] for r in inventory),
        "n_frames_all_joints_invalid_array_sum": sum(r["frames_all_joints_invalid"] for r in inventory),
        "n_all_zero_xyz_frames_array_sum": sum(r["all_zero_xyz_frames"] for r in inventory),
        "n_all_zero_xyz_and_visibility_frames_array_sum": sum(r["all_zero_xyz_and_visibility_frames"] for r in inventory),
        "n_repetition_checks_with_all_zero_xyz_frames": sum(bool(r["all_zero_xyz_frames"]) for r in view_checks),
        "n_repetition_checks_with_zero_placeholders": sum(bool(r["all_zero_xyz_and_visibility_frames"]) for r in view_checks),
        "n_visibility_outside_0_1": sum(r["visibility_outside_0_1"] for r in inventory),
        "issue_counts": dict(Counter(r["code"] for r in issues)), "issues": issues,
        "warning_counts": dict(Counter(r["code"] for r in warnings)), "warnings": warnings,
        "participant_exercise_counts": counts, "participant_error_counts": label_counts,
        "synchronized_length_checks": sync_checks, "modality_pair_checks": visibility_pair_checks,
        "visibility_by_exercise_camera_modality": [],
        "not_done": ["training samples or preprocessing", "model training or evaluation", "RGB/skeleton alignment", "visual label verification", "clinical validation"],
    }
    for ex in sorted(allowed):
        for camera in (1, 2, 3):
            for kind in ("2d", "3d"):
                subset = [r for r in view_checks if r["exercise_id"] == ex and r["camera"] == camera and r["modality"] == kind]
                summary["visibility_by_exercise_camera_modality"].append({
                    "exercise_id": ex, "camera": camera, "modality": kind,
                    "repetition_mean_visibility": distribution([r["relevant_visibility_mean"] for r in subset if r["relevant_visibility_mean"] is not None]),
                    "repetition_low_visibility_fraction": distribution([r["relevant_visibility_below_0_5_fraction"] for r in subset if r["relevant_visibility_below_0_5_fraction"] is not None]),
                })
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_json(output / "array_inventory.json", inventory)
    with (output / "repetition_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(output / "repetition_view_audit.csv", view_checks)
    write_csv(output / "participant_exercise_counts.csv", counts)
    write_csv(output / "participant_error_counts.csv", label_counts)
    print(json.dumps({k: summary[k] for k in [
        "n_repetitions", "n_arrays", "n_with_additional_errors", "n_multilabel_repetitions", "n_out_of_bounds_checks",
        "n_nonfinite_coordinate_values", "n_nonfinite_visibility_values", "issue_counts", "warning_counts",
    ]}, indent=2), flush=True)
    print(f"Outputs: {output}", flush=True)
    if issues:
        raise SystemExit(2)  # Outputs are retained for inspection; issues are never silently repaired.


if __name__ == "__main__":
    main()
