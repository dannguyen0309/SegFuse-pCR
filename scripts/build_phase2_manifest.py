#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib


VALID_DATASETS = {"spy1", "spy2"}
VALID_SPLITS = {"train", "val", "internal_test"}
CLINICAL_COLUMNS = ["age", "HR", "HER2", "menopause"]
MANIFEST_COLUMNS = [
    "pid",
    "dataset",
    "split_final",
    "pCR",
    "dce_path",
    "pre_path",
    "early_path",
    "late_path",
    "mask_path",
    "gt_mask_path",
    "pred_mask_path",
    "mask_source",
    "pre",
    "post_early",
    "post_late",
    "age",
    "HR",
    "HER2",
    "menopause",
    "menopause_missing",
    "fallback_search_used",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Phase 2 manifest CSV for pCR classification.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--spy1-root", type=Path, required=True)
    parser.add_argument("--spy2-root", type=Path, required=True)
    parser.add_argument("--mask-source", choices=["gt", "pred_nnunet"], required=True)
    parser.add_argument("--pred-mask-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/phase2_manifest_audit"))
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in metadata CSV: {missing}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_token(text: Any) -> str:
    value = str(text).upper().replace("_", "-")
    value = re.sub(r"\.NII\.GZ$", "", value)
    value = re.sub(r"[^A-Z0-9-]", "", value)
    return value


@dataclass
class CandidateRecord:
    path: Path
    root_name: str
    file_kind: str
    normalized_name: str
    phase_index: Optional[int]


def candidate_key_from_filename(path: Path) -> str:
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    markers = ["_spy1", "_spy2", "_vis", "_mask", "_dce"]
    lower_stem = stem.lower()
    positions = [lower_stem.find(marker) for marker in markers if lower_stem.find(marker) >= 0]
    if positions:
        stem = stem[: min(positions)]
    return normalize_token(stem)


def phase_index_from_name(path: Path) -> Optional[int]:
    match = re.search(r"_dce_aqc_(\d+)\.nii\.gz$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def build_index(root_map: Dict[str, Path], file_kind: str) -> Tuple[Dict[str, Dict[str, List[CandidateRecord]]], Dict[str, List[CandidateRecord]]]:
    direct: Dict[str, Dict[str, List[CandidateRecord]]] = {}
    all_candidates: Dict[str, List[CandidateRecord]] = {}
    for root_name, root_path in root_map.items():
        direct[root_name] = {}
        all_candidates[root_name] = []
        scan_root = root_path / file_kind
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.nii.gz")):
            candidate = CandidateRecord(
                path=path,
                root_name=root_name,
                file_kind=file_kind,
                normalized_name=normalize_token(path.name),
                phase_index=phase_index_from_name(path) if file_kind == "dce" else None,
            )
            all_candidates[root_name].append(candidate)
            direct[root_name].setdefault(candidate_key_from_filename(path), []).append(candidate)
    return direct, all_candidates


def score_candidate(candidate: CandidateRecord, pid: str, dataset: str, expected_root_name: str, desired_phase_idx: Optional[int] = None) -> Tuple[int, int, int, int, int, int, str]:
    norm_pid = normalize_token(pid)
    norm_name = candidate.normalized_name
    return (
        1 if desired_phase_idx is not None and candidate.phase_index == desired_phase_idx else 0,
        1 if norm_name.startswith(norm_pid) else 0,
        1 if dataset.upper() in norm_name else 0,
        1 if candidate.root_name == expected_root_name else 0,
        1 if candidate.file_kind.upper() in norm_name else 0,
        -len(norm_name),
        candidate.path.name,
    )


def search_candidates(
    pid: str,
    expected_root_name: str,
    direct_index: Dict[str, Dict[str, List[CandidateRecord]]],
    all_candidates: Dict[str, List[CandidateRecord]],
) -> Tuple[List[CandidateRecord], bool]:
    norm_pid = normalize_token(pid)
    expected_hits = list(direct_index.get(expected_root_name, {}).get(norm_pid, []))
    if not expected_hits:
        expected_hits = [candidate for candidate in all_candidates.get(expected_root_name, []) if norm_pid in candidate.normalized_name]
    if expected_hits:
        return expected_hits, False

    fallback_hits: List[CandidateRecord] = []
    for root_name in all_candidates.keys():
        direct_hits = list(direct_index.get(root_name, {}).get(norm_pid, []))
        if direct_hits:
            fallback_hits.extend(direct_hits)
        else:
            fallback_hits.extend([candidate for candidate in all_candidates.get(root_name, []) if norm_pid in candidate.normalized_name])
    return fallback_hits, True


def select_best_candidate(
    candidates: List[CandidateRecord],
    pid: str,
    dataset: str,
    desired_phase_idx: Optional[int] = None,
) -> Optional[CandidateRecord]:
    if not candidates:
        return None
    exact_phase_matches = [candidate for candidate in candidates if desired_phase_idx is not None and candidate.phase_index == desired_phase_idx]
    candidate_pool = exact_phase_matches if exact_phase_matches else candidates
    return max(candidate_pool, key=lambda candidate: score_candidate(candidate, pid, dataset, dataset, desired_phase_idx=desired_phase_idx))


def load_case_map_if_available() -> Optional[pd.DataFrame]:
    candidate_paths = [
        Path("outputs/nnunet_ispy_audit/case_id_mapping.csv"),
        Path("outputs/phase2_manifest_pred_mask.csv"),
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"pid", "dataset", "split_final"}.issubset(df.columns):
            if "pred_mask_path" in df.columns or "mask_path" in df.columns:
                return df
        required = {"case_id", "pid", "dataset", "split_final"}
        if required.issubset(df.columns):
            return df
    return None


def predicted_mask_candidates_for_case(row: pd.Series, pred_mask_root: Path, case_map_df: Optional[pd.DataFrame]) -> List[CandidateRecord]:
    candidates: List[CandidateRecord] = []
    pid = str(row["pid"])
    dataset = str(row["dataset"])
    split_final = str(row["split_final"])
    norm_pid = normalize_token(pid)
    if pred_mask_root.exists():
        for path in sorted(pred_mask_root.rglob("*.nii.gz")):
            normalized_name = normalize_token(path.name)
            if norm_pid in normalized_name:
                candidates.append(
                    CandidateRecord(path=path, root_name="pred_nnunet", file_kind="mask", normalized_name=normalized_name, phase_index=None)
                )
    if candidates or case_map_df is None:
        return candidates
    hits = case_map_df[
        (case_map_df["pid"].astype(str) == pid)
        & (case_map_df["dataset"].astype(str) == dataset)
        & (case_map_df["split_final"].astype(str) == split_final)
    ]
    path_column = "pred_mask_path" if "pred_mask_path" in case_map_df.columns else "mask_path" if "mask_path" in case_map_df.columns else None
    for _, hit in hits.iterrows():
        if path_column is not None and pd.notna(hit.get(path_column, np.nan)):
            path = Path(str(hit[path_column]))
        else:
            case_id = str(hit["case_id"])
            path = pred_mask_root / f"{case_id}.nii.gz"
        if path.exists():
            candidates.append(
                CandidateRecord(path=path, root_name="pred_nnunet", file_kind="mask", normalized_name=normalize_token(path.name), phase_index=None)
            )
    return candidates


def mask_is_empty(path: Optional[Path]) -> bool:
    if path is None or not path.exists():
        return True
    try:
        mask = np.asarray(nib.load(path.as_posix()).get_fdata(dtype=np.float32), dtype=np.float32)
    except Exception:
        return True
    if mask.ndim != 3:
        return True
    return float((mask > 0).sum()) <= 0.0


def main() -> None:
    args = parse_args()
    if args.mask_source == "pred_nnunet" and args.pred_mask_root is None:
        raise ValueError("--pred-mask-root is required when --mask-source == pred_nnunet")

    df = pd.read_csv(args.csv)
    require_columns(df, ["pid", "dataset", "split_final", "pCR", "pre", "post_early", "post_late"])
    ensure_dir(args.audit_dir)
    ensure_dir(args.out.parent)

    dataset_values = set(df["dataset"].astype(str).unique().tolist())
    if dataset_values - VALID_DATASETS:
        raise ValueError(f"Unexpected dataset values in CSV: {sorted(dataset_values - VALID_DATASETS)}")
    split_values = set(df["split_final"].astype(str).unique().tolist())
    if split_values - VALID_SPLITS:
        raise ValueError(f"Unexpected split_final values in CSV: {sorted(split_values - VALID_SPLITS)}")

    root_map = {"spy1": args.spy1_root, "spy2": args.spy2_root}
    dce_direct, dce_all = build_index(root_map, "dce")
    gt_mask_direct, gt_mask_all = build_index(root_map, "mask")
    case_map_df = load_case_map_if_available() if args.mask_source == "pred_nnunet" else None

    manifest_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    split_audits: Dict[str, Dict[str, Any]] = {
        split: {
            "cases": 0,
            "manifest_rows_kept": 0,
            "pred_mask_path_present": 0,
            "missing_pred_masks": 0,
            "empty_masks": 0,
            "fallback_center_crops": 0,
            "class_distribution": {"0": 0, "1": 0},
        }
        for split in VALID_SPLITS
    }

    for _, row in df.iterrows():
        pid = str(row["pid"])
        dataset = str(row["dataset"])
        split_final = str(row["split_final"])
        split_audit = split_audits.setdefault(
            split_final,
            {
                "cases": 0,
                "manifest_rows_kept": 0,
                "pred_mask_path_present": 0,
                "missing_pred_masks": 0,
                "empty_masks": 0,
                "fallback_center_crops": 0,
                "class_distribution": {"0": 0, "1": 0},
            },
        )
        split_audit["cases"] += 1
        label_bucket = "1" if float(row["pCR"]) >= 0.5 else "0"
        split_audit["class_distribution"][label_bucket] += 1
        pre_idx = int(round(float(row["pre"])))
        early_idx = int(round(float(row["post_early"])))
        late_idx = int(round(float(row["post_late"])))

        dce_candidates, dce_fallback = search_candidates(pid, dataset, dce_direct, dce_all)
        selected_pre = select_best_candidate(dce_candidates, pid, dataset, desired_phase_idx=pre_idx)
        selected_early = select_best_candidate(dce_candidates, pid, dataset, desired_phase_idx=early_idx)
        selected_late = select_best_candidate(dce_candidates, pid, dataset, desired_phase_idx=late_idx)

        if args.mask_source == "gt":
            gt_mask_candidates, mask_fallback = search_candidates(pid, dataset, gt_mask_direct, gt_mask_all)
            pred_mask_candidates: List[CandidateRecord] = []
        else:
            pred_root = args.pred_mask_root
            assert pred_root is not None
            pred_mask_candidates = predicted_mask_candidates_for_case(row, pred_root, case_map_df)
            gt_mask_candidates, _ = search_candidates(pid, dataset, gt_mask_direct, gt_mask_all)
            mask_fallback = False
        selected_gt_mask = select_best_candidate(gt_mask_candidates, pid, dataset)
        selected_pred_mask = select_best_candidate(pred_mask_candidates, pid, dataset)
        selected_mask = selected_gt_mask if args.mask_source == "gt" else selected_pred_mask

        if args.mask_source == "pred_nnunet":
            if selected_pred_mask is not None:
                split_audit["pred_mask_path_present"] += 1
                if mask_is_empty(selected_pred_mask.path):
                    split_audit["empty_masks"] += 1
                    split_audit["fallback_center_crops"] += 1
            else:
                split_audit["missing_pred_masks"] += 1

        fallback_search_used = bool(dce_fallback or mask_fallback)

        selected_paths = set()
        for selected_candidate in [selected_pre, selected_early, selected_late, selected_mask, selected_gt_mask, selected_pred_mask]:
            if selected_candidate is not None:
                selected_paths.add(selected_candidate.path.as_posix())

        phase_candidates = dce_candidates
        for candidate in phase_candidates:
            candidate_rows.append(
                {
                    "pid": pid,
                    "dataset": dataset,
                    "split_final": split_final,
                    "file_kind": "dce",
                    "candidate_path": candidate.path.as_posix(),
                    "candidate_root": candidate.root_name,
                    "phase_index": candidate.phase_index,
                    "selected": candidate.path.as_posix() in selected_paths,
                    "fallback_search_used": fallback_search_used,
                }
            )

        if args.mask_source == "gt":
            mask_candidates = gt_mask_candidates
        else:
            mask_candidates = pred_mask_candidates

        for candidate in mask_candidates:
            candidate_rows.append(
                {
                    "pid": pid,
                    "dataset": dataset,
                    "split_final": split_final,
                    "file_kind": "mask",
                    "candidate_path": candidate.path.as_posix(),
                    "candidate_root": candidate.root_name,
                    "phase_index": "",
                    "selected": candidate.path.as_posix() in selected_paths,
                    "fallback_search_used": fallback_search_used,
                }
            )

        if selected_pre is None or selected_early is None or selected_late is None or selected_mask is None:
            missing_rows.append(
                {
                    "pid": pid,
                    "dataset": dataset,
                    "split_final": split_final,
                    "missing_reason": "missing_dce" if selected_pre is None or selected_early is None or selected_late is None else "missing_mask",
                    "dce_candidates": json.dumps([candidate.path.as_posix() for candidate in dce_candidates]),
                    "mask_candidates": json.dumps([candidate.path.as_posix() for candidate in mask_candidates]),
                }
            )
            continue

        split_audit["manifest_rows_kept"] += 1

        manifest_row = {column: row[column] if column in row.index else np.nan for column in CLINICAL_COLUMNS}
        manifest_row.update(
            {
                "pid": pid,
                "dataset": dataset,
                "split_final": split_final,
                "pCR": row["pCR"],
                "dce_path": selected_pre.path.as_posix(),
                "pre_path": selected_pre.path.as_posix(),
                "early_path": selected_early.path.as_posix(),
                "late_path": selected_late.path.as_posix(),
                "mask_path": selected_mask.path.as_posix(),
                "gt_mask_path": selected_gt_mask.path.as_posix() if selected_gt_mask is not None else np.nan,
                "pred_mask_path": selected_pred_mask.path.as_posix() if selected_pred_mask is not None else np.nan,
                "mask_source": args.mask_source,
                "pre": row["pre"],
                "post_early": row["post_early"],
                "post_late": row["post_late"],
                "menopause_missing": 1 if pd.isna(row.get("menopause", np.nan)) else 0,
                "fallback_search_used": bool(fallback_search_used),
            }
        )
        manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    missing_df = pd.DataFrame(
        missing_rows,
        columns=["pid", "dataset", "split_final", "missing_reason", "dce_candidates", "mask_candidates"],
    )
    candidate_df = pd.DataFrame(
        candidate_rows,
        columns=["pid", "dataset", "split_final", "file_kind", "candidate_path", "candidate_root", "phase_index", "selected", "fallback_search_used"],
    )

    manifest_df.to_csv(args.out, index=False)
    missing_df.to_csv(args.audit_dir / f"{args.out.stem}_missing_cases.csv", index=False)
    candidate_df.to_csv(args.audit_dir / f"{args.out.stem}_file_candidates.csv", index=False)

    summary = {
        "created_at_utc": now_utc(),
        "input_csv": args.csv.as_posix(),
        "output_manifest": args.out.as_posix(),
        "mask_source": args.mask_source,
        "rows_loaded": int(len(df)),
        "rows_kept": int(len(manifest_df)),
        "rows_missing": int(len(missing_df)),
        "split_counts": {str(k): int(v) for k, v in manifest_df["split_final"].value_counts().sort_index().items()} if len(manifest_df) else {},
        "dataset_counts": {str(k): int(v) for k, v in manifest_df["dataset"].value_counts().sort_index().items()} if len(manifest_df) else {},
        "audit_dir": args.audit_dir.as_posix(),
        "split_audits": split_audits,
    }
    (args.audit_dir / f"{args.out.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.audit_dir / f"{args.out.stem}_audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{now_utc()}] rows_loaded={len(df)} rows_kept={len(manifest_df)} rows_missing={len(missing_df)}")
    for split_name in sorted(split_audits.keys()):
        audit = split_audits[split_name]
        print(
            f"[{now_utc()}] split={split_name} cases={audit['cases']} kept={audit['manifest_rows_kept']} "
            f"pred_mask_path_present={audit['pred_mask_path_present']} missing_pred_masks={audit['missing_pred_masks']} "
            f"empty_masks={audit['empty_masks']} fallback_center_crops={audit['fallback_center_crops']} "
            f"class_distribution={audit['class_distribution']}"
        )
    if args.mask_source == "pred_nnunet" and (
        split_audits.get("train", {}).get("pred_mask_path_present", 0) == 0
        or split_audits.get("val", {}).get("pred_mask_path_present", 0) == 0
    ):
        print(
            f"[{now_utc()}] WARNING: predicted masks are not available for all non-test splits. "
            "This manifest is not a fully realistic training manifest for predicted-mask ROI training. "
            "Use GT-mask training for train/val and reserve predicted masks for internal_test evaluation, "
            "or generate out-of-fold predicted masks for every split."
        )
    print(f"[{now_utc()}] output_manifest={args.out.as_posix()}")


if __name__ == "__main__":
    main()
