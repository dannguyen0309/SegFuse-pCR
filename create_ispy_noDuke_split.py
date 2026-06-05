#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


OUTPUT_CSV_NAME = "BreastDCEDL_ISPY1_ISPY2_noDuke_80_10_10_split.csv"
OUTPUT_SUMMARY_NAME = "BreastDCEDL_ISPY1_ISPY2_noDuke_split_summary.json"
RANDOM_STATE = 42


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an ISPY1+ISPY2 no-Duke 80/10/10 metadata split for BreastDCEDL pCR prediction."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/BreastDCEDL_metadata_min_crop.csv"),
        help="Input metadata CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for the split CSV and summary JSON.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input CSV: {missing}")


def to_int_dict(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def split_distribution(df: pd.DataFrame, split_col: str, value_col: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "val", "internal_test"]:
        subset = df[df[split_col] == split_name]
        out[split_name] = to_int_dict(subset[value_col])
    return out


def safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    label_col: str,
    split_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_col],
        )
        return train_df.copy(), test_df.copy(), True
    except ValueError as exc:
        print(
            f"[{now_utc()}] Warning: stratified split failed for {split_name} ({exc}). "
            "Falling back to non-stratified split."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        return train_df.copy(), test_df.copy(), False


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    output_csv = outdir / OUTPUT_CSV_NAME
    output_summary = outdir / OUTPUT_SUMMARY_NAME

    df = pd.read_csv(args.csv)
    require_columns(df, ["pid", "dataset", "pCR"])

    original_columns = df.columns.tolist()
    original_shape = [int(df.shape[0]), int(df.shape[1])]
    original_dataset_distribution = to_int_dict(df["dataset"])
    original_pcr_distribution = to_int_dict(df["pCR"])

    print(f"[{now_utc()}] original_shape={tuple(original_shape)}")
    print(f"[{now_utc()}] original_dataset_distribution={original_dataset_distribution}")
    print(f"[{now_utc()}] original_pCR_distribution={original_pcr_distribution}")

    # The metadata CSV is the source of truth for cohort membership.
    filtered_df = df[df["dataset"].astype(str).isin(["spy1", "spy2"])].copy()
    if len(filtered_df) == 0:
        raise RuntimeError("No rows remain after filtering to dataset values ['spy1', 'spy2'].")
    filtered_shape_after_removing_duke = [int(filtered_df.shape[0]), int(filtered_df.shape[1])]

    train_df, temp_df, stratified_stage1_used = safe_train_test_split(
        filtered_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        label_col="pCR",
        split_name="stage1_train_vs_temp",
    )
    val_df, internal_test_df, stratified_stage2_used = safe_train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        label_col="pCR",
        split_name="stage2_val_vs_internal_test",
    )

    filtered_df["split_final"] = ""
    filtered_df.loc[train_df.index, "split_final"] = "train"
    filtered_df.loc[val_df.index, "split_final"] = "val"
    filtered_df.loc[internal_test_df.index, "split_final"] = "internal_test"

    if (filtered_df["split_final"] == "").any():
        missing_count = int((filtered_df["split_final"] == "").sum())
        raise RuntimeError(f"Split assignment failed for {missing_count} rows.")

    final_columns = filtered_df.columns.tolist()
    added_columns = [column for column in final_columns if column not in original_columns]
    if added_columns != ["split_final"]:
        raise RuntimeError(f"Expected only ['split_final'] to be added, but found {added_columns}")
    if final_columns[:-1] != original_columns:
        raise RuntimeError("Original column order changed before appending split_final.")

    final_dataset_values = sorted(filtered_df["dataset"].astype(str).unique().tolist())
    if final_dataset_values != ["spy1", "spy2"]:
        raise RuntimeError(f"Final CSV contains unexpected dataset values: {final_dataset_values}")

    final_split_values = sorted(filtered_df["split_final"].astype(str).unique().tolist())
    if final_split_values != ["internal_test", "train", "val"]:
        raise RuntimeError(f"Unexpected split_final values: {final_split_values}")

    split_counts = {
        "train": int((filtered_df["split_final"] == "train").sum()),
        "val": int((filtered_df["split_final"] == "val").sum()),
        "internal_test": int((filtered_df["split_final"] == "internal_test").sum()),
    }
    if sum(split_counts.values()) != len(filtered_df):
        raise RuntimeError("Split counts do not sum to the filtered row count.")

    final_dataset_distribution = to_int_dict(filtered_df["dataset"])
    split_pcr_distribution = split_distribution(filtered_df, "split_final", "pCR")
    split_dataset_distribution = split_distribution(filtered_df, "split_final", "dataset")

    filtered_df.to_csv(output_csv, index=False)

    summary = {
        "created_at_utc": now_utc(),
        "input_csv": args.csv.as_posix(),
        "output_csv": output_csv.as_posix(),
        "original_shape": original_shape,
        "original_dataset_distribution": original_dataset_distribution,
        "original_pcr_distribution": original_pcr_distribution,
        "filtered_shape_after_removing_duke": filtered_shape_after_removing_duke,
        "final_dataset_distribution": final_dataset_distribution,
        "split_counts": split_counts,
        "split_pcr_distribution": split_pcr_distribution,
        "split_dataset_distribution": split_dataset_distribution,
        "original_columns": original_columns,
        "final_columns": final_columns,
        "added_columns": added_columns,
        "stratified_stage1_used": bool(stratified_stage1_used),
        "stratified_stage2_used": bool(stratified_stage2_used),
        "random_state": RANDOM_STATE,
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{now_utc()}] filtered_shape_after_removing_duke={tuple(filtered_shape_after_removing_duke)}")
    print(f"[{now_utc()}] final_dataset_distribution={final_dataset_distribution}")
    print(f"[{now_utc()}] split_counts={split_counts}")
    print(f"[{now_utc()}] split_dataset_distribution={split_dataset_distribution}")
    print(f"[{now_utc()}] split_pCR_distribution={split_pcr_distribution}")
    print(
        f"[{now_utc()}] schema_check=preserved {len(original_columns)} original columns and added only {added_columns}"
    )
    print(f"[{now_utc()}] final_dataset_values={final_dataset_values}")
    print(f"[{now_utc()}] final_split_values={final_split_values}")
    print(f"[{now_utc()}] output_csv={output_csv.as_posix()}")
    print(f"[{now_utc()}] output_summary_json={output_summary.as_posix()}")


if __name__ == "__main__":
    main()
