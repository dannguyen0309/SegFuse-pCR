#!/usr/bin/env python3
"""
Evaluate nnU-Net predictions on the internal_test set.

Compares:
  predicted masks from nnU-Net
  vs
  ground-truth masks saved for internal_test

Example:
python scripts/evaluate_nnunet_internal_test.py \
  --pred-dir outputs/nnunet_predictions/Dataset112_BreastTumorISPY_internal_test \
  --gt-dir outputs/nnunet_ispy_audit/internal_test_gt_masks \
  --out-dir outputs/nnunet_predictions/Dataset112_BreastTumorISPY_internal_test_metrics
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net predicted masks against internal_test ground-truth masks."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory containing nnU-Net predicted .nii.gz masks.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Directory containing ground-truth internal_test .nii.gz masks.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for metrics CSV and summary JSON.",
    )
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help=(
            "If set, shape mismatches are reported but skipped. "
            "This script does not resample masks automatically."
        ),
    )
    return parser.parse_args()


def strip_nii_gz_name(path: Path) -> str:
    """Return filename without .nii.gz or .nii suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def collect_nifti_files(folder: Path) -> Dict[str, Path]:
    """
    Collect .nii and .nii.gz files and map case_id/stem -> path.

    If duplicate stems appear, the later one overwrites. In a clean nnU-Net output
    folder this should not happen.
    """
    files = sorted(list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii")))
    return {strip_nii_gz_name(p): p for p in files}


def load_binary_mask(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a NIfTI mask and return:
      binary mask as bool array
      voxel spacing as tuple

    Any value > 0 is treated as foreground/tumor.
    """
    img = nib.load(str(path))
    data = img.get_fdata()

    # Squeeze singleton dimensions if present.
    data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {data.shape} for {path}")

    mask = data > 0

    zooms = img.header.get_zooms()
    if len(zooms) >= 3:
        spacing = tuple(float(z) for z in zooms[:3])
    else:
        spacing = (1.0, 1.0, 1.0)

    return mask.astype(bool), spacing


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def compute_overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute voxel-wise binary segmentation metrics."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)
    tn = np.logical_and(~pred, ~gt).sum(dtype=np.float64)

    dice = safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "tp_voxels": float(tp),
        "fp_voxels": float(fp),
        "fn_voxels": float(fn),
        "tn_voxels": float(tn),
        "gt_tumor_voxels": float(gt.sum()),
        "pred_tumor_voxels": float(pred.sum()),
        "volume_diff_voxels": float(pred.sum() - gt.sum()),
        "absolute_volume_diff_voxels": float(abs(pred.sum() - gt.sum())),
    }


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """
    Compute binary surface voxels.

    Surface = mask minus eroded mask.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not available.")

    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)
    return np.logical_and(mask, ~eroded)


def compute_hd95(pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """
    Compute symmetric 95th percentile Hausdorff distance.

    If pred or gt is empty, returns NaN.
    """
    if not SCIPY_AVAILABLE:
        return float("nan")

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    pred_surface = surface_voxels(pred)
    gt_surface = surface_voxels(gt)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("nan")

    # distance_transform_edt calculates distance to the nearest zero.
    # To get distance to surface, invert the surface mask so surface voxels are zeros.
    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_gt_distances = dt_to_gt[pred_surface]
    gt_to_pred_distances = dt_to_pred[gt_surface]

    all_distances = np.concatenate([pred_to_gt_distances, gt_to_pred_distances])

    if all_distances.size == 0:
        return float("nan")

    return float(np.percentile(all_distances, 95))


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def nanmedian(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def summarize_metrics(df: pd.DataFrame) -> Dict[str, object]:
    metric_cols = [
        "dice",
        "iou",
        "precision",
        "recall_sensitivity",
        "specificity",
        "hd95",
        "gt_tumor_voxels",
        "pred_tumor_voxels",
        "absolute_volume_diff_voxels",
    ]

    summary: Dict[str, object] = {}

    for col in metric_cols:
        if col in df.columns:
            vals = df[col].astype(float).tolist()
            summary[f"mean_{col}"] = nanmean(vals)
            summary[f"median_{col}"] = nanmedian(vals)

    if "dice" in df.columns:
        dice = df["dice"].astype(float)
        summary["min_dice"] = float(np.nanmin(dice)) if not np.all(np.isnan(dice)) else float("nan")
        summary["max_dice"] = float(np.nanmax(dice)) if not np.all(np.isnan(dice)) else float("nan")
        summary["num_cases_dice_below_0_50"] = int((dice < 0.50).sum())
        summary["num_cases_dice_below_0_70"] = int((dice < 0.70).sum())
        summary["num_cases_dice_above_0_80"] = int((dice >= 0.80).sum())

    return summary


def write_json(path: Path, obj: Dict[str, object]) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, float) and math.isnan(o):
            return None
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def main() -> None:
    args = parse_args()

    pred_dir: Path = args.pred_dir
    gt_dir: Path = args.gt_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    pred_files = collect_nifti_files(pred_dir)
    gt_files = collect_nifti_files(gt_dir)

    pred_ids = set(pred_files.keys())
    gt_ids = set(gt_files.keys())

    matched_ids = sorted(pred_ids.intersection(gt_ids))
    missing_pred_ids = sorted(gt_ids - pred_ids)
    missing_gt_ids = sorted(pred_ids - gt_ids)

    rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []

    for case_id in matched_ids:
        pred_path = pred_files[case_id]
        gt_path = gt_files[case_id]

        try:
            pred_mask, pred_spacing = load_binary_mask(pred_path)
            gt_mask, gt_spacing = load_binary_mask(gt_path)

            if pred_mask.shape != gt_mask.shape:
                msg = (
                    f"Shape mismatch for {case_id}: "
                    f"pred {pred_mask.shape}, gt {gt_mask.shape}"
                )
                skipped_rows.append({
                    "case_id": case_id,
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                    "reason": msg,
                })
                if args.allow_shape_mismatch:
                    print(f"WARNING: {msg}. Skipping.")
                    continue
                raise ValueError(msg)

            # Use GT spacing as reference. If spacing differs, record warning but continue.
            spacing_warning = ""
            spacing = gt_spacing
            if tuple(round(x, 5) for x in pred_spacing) != tuple(round(x, 5) for x in gt_spacing):
                spacing_warning = f"pred_spacing={pred_spacing}, gt_spacing={gt_spacing}"

            metrics = compute_overlap_metrics(pred_mask, gt_mask)
            hd95 = compute_hd95(pred_mask, gt_mask, spacing)
            metrics["hd95"] = hd95

            row = {
                "case_id": case_id,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "shape": str(tuple(gt_mask.shape)),
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "spacing_warning": spacing_warning,
                "status": "ok",
            }
            row.update(metrics)
            rows.append(row)

        except Exception as exc:
            skipped_rows.append({
                "case_id": case_id,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "reason": repr(exc),
            })
            print(f"WARNING: failed case {case_id}: {exc}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "segmentation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    missing_predictions_df = pd.DataFrame([
        {
            "case_id": case_id,
            "gt_path": str(gt_files[case_id]),
            "reason": "ground_truth_exists_but_prediction_missing",
        }
        for case_id in missing_pred_ids
    ])
    missing_predictions_df.to_csv(out_dir / "missing_predictions.csv", index=False)

    missing_gt_df = pd.DataFrame([
        {
            "case_id": case_id,
            "pred_path": str(pred_files[case_id]),
            "reason": "prediction_exists_but_ground_truth_missing",
        }
        for case_id in missing_gt_ids
    ])
    missing_gt_df.to_csv(out_dir / "missing_ground_truth.csv", index=False)

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df.to_csv(out_dir / "skipped_cases.csv", index=False)

    metric_summary = summarize_metrics(metrics_df) if len(metrics_df) > 0 else {}

    summary: Dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "out_dir": str(out_dir),
        "num_pred_masks": len(pred_files),
        "num_gt_masks": len(gt_files),
        "num_matched_cases": len(matched_ids),
        "num_evaluated_cases": len(metrics_df),
        "num_missing_predictions": len(missing_pred_ids),
        "num_missing_ground_truth": len(missing_gt_ids),
        "num_skipped_cases": len(skipped_rows),
        "scipy_available": SCIPY_AVAILABLE,
        "metrics_csv": str(metrics_path),
    }
    summary.update(metric_summary)

    write_json(out_dir / "summary.json", summary)

    print("\n=== nnU-Net internal_test segmentation evaluation ===")
    print(f"Prediction dir: {pred_dir}")
    print(f"Ground-truth dir: {gt_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Pred masks: {len(pred_files)}")
    print(f"GT masks: {len(gt_files)}")
    print(f"Matched cases: {len(matched_ids)}")
    print(f"Evaluated cases: {len(metrics_df)}")
    print(f"Missing predictions: {len(missing_pred_ids)}")
    print(f"Missing ground truth: {len(missing_gt_ids)}")
    print(f"Skipped cases: {len(skipped_rows)}")
    print(f"SciPy available for HD95: {SCIPY_AVAILABLE}")

    if len(metrics_df) > 0:
        print(f"Mean Dice: {summary.get('mean_dice'):.4f}")
        print(f"Median Dice: {summary.get('median_dice'):.4f}")
        print(f"Mean IoU: {summary.get('mean_iou'):.4f}")
        print(f"Median IoU: {summary.get('median_iou'):.4f}")
        if not math.isnan(float(summary.get("mean_hd95", float("nan")))):
            print(f"Mean HD95: {summary.get('mean_hd95'):.4f}")
            print(f"Median HD95: {summary.get('median_hd95'):.4f}")

    print("\nSaved:")
    print(f"- {metrics_path}")
    print(f"- {out_dir / 'summary.json'}")
    print(f"- {out_dir / 'missing_predictions.csv'}")
    print(f"- {out_dir / 'missing_ground_truth.csv'}")
    print(f"- {out_dir / 'skipped_cases.csv'}")


if __name__ == "__main__":
    main()