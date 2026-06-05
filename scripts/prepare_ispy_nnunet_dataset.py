#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


VALID_DATASETS = {"spy1", "spy2"}
VALID_SPLITS = {"train", "val", "internal_test"}
CASE_MAP_COLUMNS = [
    "case_id",
    "pid",
    "dataset",
    "split_final",
    "pCR",
    "conversion_mode",
    "fallback_search_used",
    "source_dce_path",
    "source_mask_path",
    "status",
    "converted_success",
]
CONVERTED_COLUMNS = [
    "case_id",
    "pid",
    "dataset",
    "split_final",
    "pCR",
    "source_dce_path",
    "source_mask_path",
    "output_image_paths",
    "output_label_path",
    "fallback_search_used",
    "status",
]
MISSING_COLUMNS = [
    "case_id",
    "pid",
    "dataset",
    "split_final",
    "missing_reason",
    "dce_candidates",
    "mask_candidates",
]
CANDIDATE_COLUMNS = [
    "case_id",
    "pid",
    "dataset",
    "split_final",
    "file_kind",
    "candidate_path",
    "candidate_root",
    "phase_index_in_name",
    "fallback_search_used",
    "selected",
    "selected_roles",
    "search_scope",
]
LABEL_CHECK_COLUMNS = [
    "case_id",
    "pid",
    "original_unique_values_sample_or_summary",
    "final_unique_values",
    "tumor_voxel_count",
    "status",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare nnU-Net v2 raw dataset files from the BreastDCEDL ISPY metadata CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Metadata CSV with pid, dataset, split_final, and timepoint columns.",
    )
    parser.add_argument("--spy1-root", type=Path, required=True, help="Root folder for ISPY1 min-crop data.")
    parser.add_argument("--spy2-root", type=Path, required=True, help="Root folder for ISPY2 min-crop data.")
    parser.add_argument("--dataset-id", type=int, default=112)
    parser.add_argument("--dataset-name", type=str, default="BreastTumorISPY")
    parser.add_argument(
        "--out-audit-dir",
        type=Path,
        default=Path("outputs/nnunet_ispy_audit"),
        help="Directory for audit CSVs and summary JSON.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=["copy", "symlink"],
        default="copy",
        help=(
            "How to materialize optional internal-test GT mask audit files. "
            "nnU-Net images and labels are always rewritten as standardized NIfTI files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate existing nnUNet_raw dataset and audit outputs if they already exist.",
    )
    return parser.parse_args()


def require_env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"Export {name} before running this script."
        )
    return Path(value)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")


def normalize_token(text: Any) -> str:
    value = str(text).upper().replace("_", "-")
    value = re.sub(r"\.NII\.GZ$", "", value)
    value = re.sub(r"[^A-Z0-9-]", "", value)
    return value


def phase_index_from_name(path: Path) -> Optional[int]:
    match = re.search(r"_dce_aqc_(\d+)\.nii\.gz$", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def bool_from_series_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def summarize_unique_values(values: np.ndarray, max_show: int = 10) -> str:
    unique_values = np.unique(values)
    unique_list = unique_values.tolist()
    if len(unique_list) <= max_show:
        return safe_json_dumps(unique_list)
    preview = unique_list[:max_show]
    return f"{safe_json_dumps(preview)} ... total_unique={len(unique_list)}"


@dataclass
class CandidateRecord:
    path: Path
    root_name: str
    file_kind: str
    normalized_name: str
    phase_index: Optional[int]


@dataclass
class ResolvedDCE:
    mode: str
    selected_phase_paths: Dict[str, Path]
    selected_4d_path: Optional[Path]
    candidates: List[CandidateRecord]
    fallback_search_used: bool


@dataclass
class ResolvedMask:
    selected_path: Optional[Path]
    candidates: List[CandidateRecord]
    fallback_search_used: bool


@dataclass
class FileIndex:
    direct_map: Dict[str, Dict[str, List[CandidateRecord]]]
    all_candidates: Dict[str, List[CandidateRecord]]


def score_candidate(
    candidate: CandidateRecord,
    pid: str,
    dataset: str,
    expected_root_name: str,
    desired_phase_idx: Optional[int] = None,
) -> Tuple[int, int, int, int, int, int, str]:
    norm_pid = normalize_token(pid)
    dataset_token = dataset.upper()
    norm_name = candidate.normalized_name
    startswith_pid = 1 if norm_name.startswith(norm_pid) else 0
    dataset_match = 1 if dataset_token in norm_name else 0
    expected_root_match = 1 if candidate.root_name == expected_root_name else 0
    file_kind_match = 1 if candidate.file_kind.upper() in norm_name else 0
    desired_phase_match = 1 if desired_phase_idx is not None and candidate.phase_index == desired_phase_idx else 0
    return (
        desired_phase_match,
        startswith_pid,
        dataset_match,
        file_kind_match,
        expected_root_match,
        -len(norm_name),
        candidate.path.name,
    )


def collect_pid_candidates(dir_path: Path, pid: str, root_name: str, file_kind: str) -> List[CandidateRecord]:
    norm_pid = normalize_token(pid)
    candidates: List[CandidateRecord] = []
    if not dir_path.exists():
        return candidates
    for path in sorted(dir_path.rglob("*.nii.gz")):
        normalized_name = normalize_token(path.name)
        if norm_pid in normalized_name:
            candidates.append(
                CandidateRecord(
                    path=path,
                    root_name=root_name,
                    file_kind=file_kind,
                    normalized_name=normalized_name,
                    phase_index=phase_index_from_name(path) if file_kind == "dce" else None,
                )
            )
    return candidates


def candidate_key_from_filename(path: Path) -> str:
    stem = path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    split_markers = ["_spy1", "_spy2", "_vis", "_mask", "_dce"]
    lower_stem = stem.lower()
    cut_positions = [lower_stem.find(marker) for marker in split_markers if lower_stem.find(marker) >= 0]
    if cut_positions:
        stem = stem[: min(cut_positions)]
    return normalize_token(stem)


def build_file_index(root_map: Dict[str, Path], file_kind: str) -> FileIndex:
    direct_map: Dict[str, Dict[str, List[CandidateRecord]]] = {}
    all_candidates: Dict[str, List[CandidateRecord]] = {}
    for root_name, root_path in root_map.items():
        file_dir = root_path / file_kind
        root_candidates: List[CandidateRecord] = []
        root_direct_map: Dict[str, List[CandidateRecord]] = {}
        if file_dir.exists():
            for path in sorted(file_dir.rglob("*.nii.gz")):
                candidate = CandidateRecord(
                    path=path,
                    root_name=root_name,
                    file_kind=file_kind,
                    normalized_name=normalize_token(path.name),
                    phase_index=phase_index_from_name(path) if file_kind == "dce" else None,
                )
                root_candidates.append(candidate)
                root_direct_map.setdefault(candidate_key_from_filename(path), []).append(candidate)
        direct_map[root_name] = root_direct_map
        all_candidates[root_name] = root_candidates
    return FileIndex(direct_map=direct_map, all_candidates=all_candidates)


def search_candidates_for_pid(
    pid: str,
    expected_root_name: str,
    file_index: FileIndex,
) -> Tuple[List[CandidateRecord], bool]:
    norm_pid = normalize_token(pid)
    expected_candidates = list(file_index.direct_map.get(expected_root_name, {}).get(norm_pid, []))
    if not expected_candidates:
        expected_candidates = [
            candidate
            for candidate in file_index.all_candidates.get(expected_root_name, [])
            if norm_pid in candidate.normalized_name
        ]
    if expected_candidates:
        return expected_candidates, False

    fallback_candidates: List[CandidateRecord] = []
    for root_name in file_index.all_candidates.keys():
        direct_hits = list(file_index.direct_map.get(root_name, {}).get(norm_pid, []))
        if direct_hits:
            fallback_candidates.extend(direct_hits)
        else:
            fallback_candidates.extend(
                [
                    candidate
                    for candidate in file_index.all_candidates.get(root_name, [])
                    if norm_pid in candidate.normalized_name
                ]
            )
    return fallback_candidates, True


def select_best_candidate(
    candidates: List[CandidateRecord],
    pid: str,
    dataset: str,
    expected_root_name: str,
    desired_phase_idx: Optional[int] = None,
) -> Optional[CandidateRecord]:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: score_candidate(
            candidate,
            pid=pid,
            dataset=dataset,
            expected_root_name=expected_root_name,
            desired_phase_idx=desired_phase_idx,
        ),
    )


def resolve_dce_sources(
    row: pd.Series,
    expected_root_name: str,
    dce_index: FileIndex,
) -> ResolvedDCE:
    pid = str(row["pid"])
    dataset = str(row["dataset"])
    desired_phase_map = {
        "0000": int(round(float(row["pre"]))),
        "0001": int(round(float(row["post_early"]))),
        "0002": int(round(float(row["post_late"]))),
    }

    candidates, fallback_search_used = search_candidates_for_pid(
        pid=pid,
        expected_root_name=expected_root_name,
        file_index=dce_index,
    )

    if not candidates:
        return ResolvedDCE(
            mode="missing",
            selected_phase_paths={},
            selected_4d_path=None,
            candidates=[],
            fallback_search_used=fallback_search_used,
        )

    selected_phase_paths: Dict[str, Path] = {}
    for channel_name, phase_idx in desired_phase_map.items():
        phase_matches = [candidate for candidate in candidates if candidate.phase_index == phase_idx]
        best_phase = select_best_candidate(
            phase_matches,
            pid=pid,
            dataset=dataset,
            expected_root_name=expected_root_name,
            desired_phase_idx=phase_idx,
        )
        if best_phase is not None:
            selected_phase_paths[channel_name] = best_phase.path

    if len(selected_phase_paths) == 3:
        return ResolvedDCE(
            mode="separate_3d",
            selected_phase_paths=selected_phase_paths,
            selected_4d_path=None,
            candidates=candidates,
            fallback_search_used=fallback_search_used,
        )

    best_overall = select_best_candidate(
        candidates,
        pid=pid,
        dataset=dataset,
        expected_root_name=expected_root_name,
    )
    if best_overall is None:
        return ResolvedDCE(
            mode="missing",
            selected_phase_paths={},
            selected_4d_path=None,
            candidates=candidates,
            fallback_search_used=fallback_search_used,
        )

    try:
        dce_img = nib.load(str(best_overall.path))
    except Exception:
        return ResolvedDCE(
            mode="missing",
            selected_phase_paths=selected_phase_paths,
            selected_4d_path=None,
            candidates=candidates,
            fallback_search_used=fallback_search_used,
        )

    if len(dce_img.shape) == 4:
        max_phase_idx = max(desired_phase_map.values())
        if max_phase_idx < dce_img.shape[3]:
            return ResolvedDCE(
                mode="single_4d",
                selected_phase_paths={},
                selected_4d_path=best_overall.path,
                candidates=candidates,
                fallback_search_used=fallback_search_used,
            )

    return ResolvedDCE(
        mode="missing",
        selected_phase_paths=selected_phase_paths,
        selected_4d_path=None,
        candidates=candidates,
        fallback_search_used=fallback_search_used,
    )


def resolve_mask_source(
    row: pd.Series,
    expected_root_name: str,
    mask_index: FileIndex,
) -> ResolvedMask:
    pid = str(row["pid"])
    dataset = str(row["dataset"])
    candidates, fallback_search_used = search_candidates_for_pid(
        pid=pid,
        expected_root_name=expected_root_name,
        file_index=mask_index,
    )
    best_candidate = select_best_candidate(
        candidates,
        pid=pid,
        dataset=dataset,
        expected_root_name=expected_root_name,
    )
    return ResolvedMask(
        selected_path=best_candidate.path if best_candidate is not None else None,
        candidates=candidates,
        fallback_search_used=fallback_search_used,
    )


def save_nifti_like(source_img: nib.spatialimages.SpatialImage, data: np.ndarray, output_path: Path) -> None:
    header = source_img.header.copy()
    header.set_data_dtype(data.dtype)
    out_img = nib.Nifti1Image(data, source_img.affine, header=header)
    nib.save(out_img, str(output_path))


def materialize_internal_test_mask(source_mask_path: Path, target_path: Path, copy_mode: str) -> None:
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    if copy_mode == "symlink":
        os.symlink(source_mask_path.resolve(), target_path)
    else:
        shutil.copy2(source_mask_path, target_path)


def build_dataset_json(
    dataset_name: str,
    num_training: int,
    num_test: int,
) -> Dict[str, Any]:
    return {
        "name": dataset_name,
        "dataset_name": dataset_name,
        "description": "BreastDCEDL ISPY1+ISPY2 no-Duke tumor segmentation dataset for nnU-Net v2",
        "channel_names": {
            "0": "pre",
            "1": "post_early",
            "2": "post_late",
        },
        "labels": {
            "background": 0,
            "tumor": 1,
        },
        "numTraining": int(num_training),
        "numTest": int(num_test),
        "file_ending": ".nii.gz",
    }


def main() -> None:
    args = parse_args()
    nnunet_raw = require_env_path("nnUNet_raw")

    require_columns(
        pd.read_csv(args.csv, nrows=5),
        ["pid", "dataset", "split_final", "pCR", "pre", "post_early", "post_late"],
    )
    df = pd.read_csv(args.csv)

    dataset_values = set(df["dataset"].astype(str).unique().tolist())
    unexpected_datasets = sorted(dataset_values - VALID_DATASETS)
    if unexpected_datasets:
        raise ValueError(f"Unexpected dataset values in CSV: {unexpected_datasets}")

    split_values = set(df["split_final"].astype(str).unique().tolist())
    unexpected_splits = sorted(split_values - VALID_SPLITS)
    if unexpected_splits:
        raise ValueError(f"Unexpected split_final values in CSV: {unexpected_splits}")

    dataset_folder_name = f"Dataset{int(args.dataset_id):03d}_{args.dataset_name}"
    dataset_dir = nnunet_raw / dataset_folder_name
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"
    audit_dir = args.out_audit_dir
    internal_test_gt_dir = audit_dir / "internal_test_gt_masks"

    if args.overwrite:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        if audit_dir.exists():
            shutil.rmtree(audit_dir)
    elif dataset_dir.exists():
        raise FileExistsError(
            f"Output dataset directory already exists: {dataset_dir.as_posix()}. "
            "Use --overwrite to recreate it."
        )

    ensure_dir(images_tr_dir)
    ensure_dir(labels_tr_dir)
    ensure_dir(images_ts_dir)
    ensure_dir(audit_dir)
    ensure_dir(internal_test_gt_dir)

    root_map = {
        "spy1": args.spy1_root,
        "spy2": args.spy2_root,
    }
    for dataset_name, root_path in root_map.items():
        if not root_path.exists():
            raise FileNotFoundError(f"{dataset_name} root does not exist: {root_path.as_posix()}")
        for subdir in ["dce", "mask"]:
            if not (root_path / subdir).exists():
                raise FileNotFoundError(f"Expected subdirectory missing: {(root_path / subdir).as_posix()}")

    dce_index = build_file_index(root_map, "dce")
    mask_index = build_file_index(root_map, "mask")

    mapping_rows: List[Dict[str, Any]] = []
    converted_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    label_check_rows: List[Dict[str, Any]] = []

    converted_train = 0
    converted_val = 0
    converted_internal_test = 0

    print(f"[{now_utc()}] rows_loaded={len(df)}")
    print(f"[{now_utc()}] expected_split_counts={df['split_final'].value_counts().sort_index().to_dict()}")
    print(f"[{now_utc()}] copy_mode_note=nnU-Net outputs are rewritten as standardized NIfTI files; copy-mode only affects optional internal-test GT audit files.")

    for row_idx, row in df.reset_index(drop=True).iterrows():
        pid = str(row["pid"])
        dataset = str(row["dataset"])
        split_final = str(row["split_final"])
        case_id = f"BreastISPY_{row_idx + 1:06d}"
        expected_root_name = dataset

        dce_resolution = resolve_dce_sources(row=row, expected_root_name=expected_root_name, dce_index=dce_index)
        mask_resolution = resolve_mask_source(row=row, expected_root_name=expected_root_name, mask_index=mask_index)
        fallback_search_used = bool(dce_resolution.fallback_search_used or mask_resolution.fallback_search_used)

        candidate_roles: Dict[str, List[str]] = {}
        for channel_name, source_path in dce_resolution.selected_phase_paths.items():
            candidate_roles.setdefault(source_path.as_posix(), []).append(channel_name)
        if dce_resolution.selected_4d_path is not None:
            candidate_roles.setdefault(dce_resolution.selected_4d_path.as_posix(), []).append("4d_source")
        if mask_resolution.selected_path is not None:
            candidate_roles.setdefault(mask_resolution.selected_path.as_posix(), []).append("mask_selected")

        for candidate in dce_resolution.candidates + mask_resolution.candidates:
            candidate_rows.append(
                {
                    "case_id": case_id,
                    "pid": pid,
                    "dataset": dataset,
                    "split_final": split_final,
                    "file_kind": candidate.file_kind,
                    "candidate_path": candidate.path.as_posix(),
                    "candidate_root": candidate.root_name,
                    "phase_index_in_name": candidate.phase_index,
                    "fallback_search_used": fallback_search_used,
                    "selected": candidate.path.as_posix() in candidate_roles,
                    "selected_roles": safe_json_dumps(candidate_roles.get(candidate.path.as_posix(), [])),
                    "search_scope": "both_roots" if fallback_search_used else "expected_root_only",
                }
            )

        dce_candidate_paths = [candidate.path.as_posix() for candidate in dce_resolution.candidates]
        mask_candidate_paths = [candidate.path.as_posix() for candidate in mask_resolution.candidates]

        missing_reason = ""
        if dce_resolution.mode == "missing":
            missing_reason = "dce_missing_or_unusable"
        elif split_final in {"train", "val"} and mask_resolution.selected_path is None:
            missing_reason = "mask_missing_for_training_case"

        mapping_row = {
            "case_id": case_id,
            "pid": pid,
            "dataset": dataset,
            "split_final": split_final,
            "pCR": row["pCR"],
            "conversion_mode": dce_resolution.mode,
            "fallback_search_used": fallback_search_used,
            "source_dce_path": "",
            "source_mask_path": mask_resolution.selected_path.as_posix() if mask_resolution.selected_path else "",
            "status": "",
            "converted_success": False,
        }

        if missing_reason:
            if missing_reason == "dce_missing_or_unusable" and dce_candidate_paths:
                print(
                    f"[{now_utc()}] Warning: pid={pid} dataset={dataset} had DCE candidates "
                    "but 3 nnU-Net channels could not be resolved from them."
                )
            if split_final == "internal_test" and mask_resolution.selected_path is None:
                print(
                    f"[{now_utc()}] Warning: pid={pid} dataset={dataset} internal_test case has no mask; "
                    "image conversion is skipped only if DCE is unusable."
                )
            mapping_row["status"] = missing_reason
            mapping_rows.append(mapping_row)
            missing_rows.append(
                {
                    "case_id": case_id,
                    "pid": pid,
                    "dataset": dataset,
                    "split_final": split_final,
                    "missing_reason": missing_reason,
                    "dce_candidates": safe_json_dumps(dce_candidate_paths),
                    "mask_candidates": safe_json_dumps(mask_candidate_paths),
                }
            )
            continue

        output_image_dir = images_ts_dir if split_final == "internal_test" else images_tr_dir
        output_image_paths = {
            "0000": output_image_dir / f"{case_id}_0000.nii.gz",
            "0001": output_image_dir / f"{case_id}_0001.nii.gz",
            "0002": output_image_dir / f"{case_id}_0002.nii.gz",
        }
        output_label_path = labels_tr_dir / f"{case_id}.nii.gz" if split_final in {"train", "val"} else None

        if dce_resolution.mode == "separate_3d":
            source_phase_paths = dce_resolution.selected_phase_paths
            for channel_name, source_path in source_phase_paths.items():
                source_img = nib.load(str(source_path))
                if len(source_img.shape) != 3:
                    raise ValueError(
                        f"Expected 3D DCE phase file for {pid}, but got shape {source_img.shape} at {source_path}"
                    )
                phase_data = np.asarray(source_img.get_fdata(dtype=np.float32), dtype=np.float32)
                save_nifti_like(source_img, phase_data, output_image_paths[channel_name])
            mapping_row["source_dce_path"] = safe_json_dumps(
                {channel_name: path.as_posix() for channel_name, path in sorted(source_phase_paths.items())}
            )
        elif dce_resolution.mode == "single_4d":
            source_4d_path = dce_resolution.selected_4d_path
            if source_4d_path is None:
                raise RuntimeError(f"4D mode selected without source path for pid={pid}")
            source_img = nib.load(str(source_4d_path))
            if len(source_img.shape) != 4:
                raise ValueError(
                    f"Expected 4D DCE file for {pid}, but got shape {source_img.shape} at {source_4d_path}"
                )
            four_d_data = np.asarray(source_img.get_fdata(dtype=np.float32), dtype=np.float32)
            phase_indices = {
                "0000": int(round(float(row["pre"]))),
                "0001": int(round(float(row["post_early"]))),
                "0002": int(round(float(row["post_late"]))),
            }
            for channel_name, phase_idx in phase_indices.items():
                if phase_idx >= four_d_data.shape[3]:
                    raise IndexError(
                        f"Requested phase index {phase_idx} exceeds DCE time dimension {four_d_data.shape[3]} "
                        f"for pid={pid} at {source_4d_path}"
                    )
                phase_data = np.asarray(four_d_data[..., phase_idx], dtype=np.float32)
                save_nifti_like(source_img, phase_data, output_image_paths[channel_name])
            mapping_row["source_dce_path"] = source_4d_path.as_posix()
        else:
            raise RuntimeError(f"Unexpected DCE resolution mode for pid={pid}: {dce_resolution.mode}")

        source_mask_path = mask_resolution.selected_path
        source_mask_summary = ""
        if source_mask_path is not None:
            mask_img = nib.load(str(source_mask_path))
            mask_data = np.asarray(mask_img.get_fdata(), dtype=np.float32)
            binary_mask = (mask_data > 0).astype(np.uint8)
            tumor_voxel_count = int(binary_mask.sum())
            final_unique_values = np.unique(binary_mask).tolist()
            label_status = "ok" if tumor_voxel_count > 0 else "warning_empty_tumor_mask"
            label_check_rows.append(
                {
                    "case_id": case_id,
                    "pid": pid,
                    "original_unique_values_sample_or_summary": summarize_unique_values(mask_data),
                    "final_unique_values": safe_json_dumps(final_unique_values),
                    "tumor_voxel_count": tumor_voxel_count,
                    "status": label_status,
                }
            )
            source_mask_summary = source_mask_path.as_posix()

            if output_label_path is not None:
                save_nifti_like(mask_img, binary_mask, output_label_path)
            else:
                audit_gt_path = internal_test_gt_dir / f"{case_id}.nii.gz"
                materialize_internal_test_mask(source_mask_path, audit_gt_path, args.copy_mode)

        if split_final == "train":
            converted_train += 1
            status = "converted_train"
        elif split_final == "val":
            converted_val += 1
            status = "converted_val"
        else:
            converted_internal_test += 1
            if source_mask_path is None:
                print(
                    f"[{now_utc()}] Warning: pid={pid} dataset={dataset} converted for internal_test "
                    "without an audit GT mask."
                )
            status = "converted_internal_test" if source_mask_path is not None else "converted_internal_test_missing_mask"

        mapping_row["source_mask_path"] = source_mask_summary
        mapping_row["status"] = status
        mapping_row["converted_success"] = True
        mapping_rows.append(mapping_row)

        converted_rows.append(
            {
                "case_id": case_id,
                "pid": pid,
                "dataset": dataset,
                "split_final": split_final,
                "pCR": row["pCR"],
                "source_dce_path": mapping_row["source_dce_path"],
                "source_mask_path": source_mask_summary,
                "output_image_paths": safe_json_dumps(
                    {channel_name: path.as_posix() for channel_name, path in sorted(output_image_paths.items())}
                ),
                "output_label_path": output_label_path.as_posix() if output_label_path is not None else "",
                "fallback_search_used": fallback_search_used,
                "status": status,
            }
        )

    num_training_cases = converted_train + converted_val
    num_test_cases = converted_internal_test

    dataset_json = build_dataset_json(
        dataset_name=args.dataset_name,
        num_training=num_training_cases,
        num_test=num_test_cases,
    )
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    pd.DataFrame(mapping_rows, columns=CASE_MAP_COLUMNS).to_csv(audit_dir / "case_id_mapping.csv", index=False)
    pd.DataFrame(converted_rows, columns=CONVERTED_COLUMNS).to_csv(audit_dir / "converted_cases.csv", index=False)
    pd.DataFrame(missing_rows, columns=MISSING_COLUMNS).to_csv(audit_dir / "missing_cases.csv", index=False)
    pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS).to_csv(audit_dir / "file_candidates.csv", index=False)
    pd.DataFrame(label_check_rows, columns=LABEL_CHECK_COLUMNS).to_csv(
        audit_dir / "label_value_check.csv",
        index=False,
    )

    summary = {
        "created_at_utc": now_utc(),
        "input_csv": args.csv.as_posix(),
        "rows_loaded": int(len(df)),
        "dataset_counts": {str(k): int(v) for k, v in df["dataset"].value_counts().sort_index().items()},
        "split_counts_expected": {str(k): int(v) for k, v in df["split_final"].value_counts().sort_index().items()},
        "converted_train_cases": int(converted_train),
        "converted_val_cases": int(converted_val),
        "converted_internal_test_cases": int(converted_internal_test),
        "missing_cases": int(len(missing_rows)),
        "dataset_dir": dataset_dir.as_posix(),
        "imagesTr_dir": images_tr_dir.as_posix(),
        "labelsTr_dir": labels_tr_dir.as_posix(),
        "imagesTs_dir": images_ts_dir.as_posix(),
        "audit_dir": audit_dir.as_posix(),
        "internal_test_gt_dir": internal_test_gt_dir.as_posix(),
        "dataset_json_path": (dataset_dir / "dataset.json").as_posix(),
        "copy_mode": args.copy_mode,
        "overwrite": bool(args.overwrite),
        "nnUNet_raw": nnunet_raw.as_posix(),
        "next_command": f"nnUNetv2_plan_and_preprocess -d {int(args.dataset_id)} --verify_dataset_integrity",
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{now_utc()}] converted_train_cases={converted_train}")
    print(f"[{now_utc()}] converted_val_cases={converted_val}")
    print(f"[{now_utc()}] converted_internal_test_cases={converted_internal_test}")
    print(f"[{now_utc()}] missing_cases={len(missing_rows)}")
    print(f"[{now_utc()}] output_nnunet_dataset_path={dataset_dir.as_posix()}")
    print(f"[{now_utc()}] next_command=nnUNetv2_plan_and_preprocess -d {int(args.dataset_id)} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
