#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nnU-Net v2 splits_final.json for the BreastTumorISPY dataset."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Metadata CSV with split_final assignments.")
    parser.add_argument("--case-map", type=Path, required=True, help="case_id_mapping.csv from dataset prep.")
    parser.add_argument("--dataset-id", type=int, default=112)
    parser.add_argument("--dataset-name", type=str, default="BreastTumorISPY")
    return parser.parse_args()


def require_env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"Export {name} before running this script."
        )
    return Path(value)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {label}: {missing}")


def bool_from_series_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    nnunet_preprocessed = require_env_path("nnUNet_preprocessed")

    csv_df = pd.read_csv(args.csv)
    case_map_df = pd.read_csv(args.case_map)
    require_columns(csv_df, ["pid", "dataset", "split_final"], "metadata CSV")
    require_columns(case_map_df, ["case_id", "pid", "dataset", "split_final", "converted_success"], "case map")

    dataset_folder_name = f"Dataset{int(args.dataset_id):03d}_{args.dataset_name}"
    preprocessed_dataset_dir = nnunet_preprocessed / dataset_folder_name
    if not preprocessed_dataset_dir.exists():
        raise SystemExit(
            "Run nnUNetv2_plan_and_preprocess first, then rerun this script. "
            f"Missing folder: {preprocessed_dataset_dir.as_posix()}"
        )

    merged_df = case_map_df.merge(
        csv_df[["pid", "dataset", "split_final"]].drop_duplicates(),
        on=["pid", "dataset", "split_final"],
        how="inner",
    )
    merged_df["converted_success"] = merged_df["converted_success"].map(bool_from_series_value)
    usable_df = merged_df[merged_df["converted_success"]].copy()

    train_case_ids = sorted(usable_df.loc[usable_df["split_final"] == "train", "case_id"].astype(str).tolist())
    val_case_ids = sorted(usable_df.loc[usable_df["split_final"] == "val", "case_id"].astype(str).tolist())
    internal_test_case_ids = sorted(
        usable_df.loc[usable_df["split_final"] == "internal_test", "case_id"].astype(str).tolist()
    )

    overlap = sorted(set(train_case_ids) & set(val_case_ids))
    if overlap:
        raise RuntimeError(f"Train/val overlap detected in case IDs: {overlap[:10]}")
    if set(internal_test_case_ids) & (set(train_case_ids) | set(val_case_ids)):
        raise RuntimeError("Internal-test case IDs leaked into train/val split lists.")

    splits = [{"train": train_case_ids, "val": val_case_ids}]
    output_path = preprocessed_dataset_dir / "splits_final.json"
    output_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"train case count: {len(train_case_ids)}")
    print(f"val case count: {len(val_case_ids)}")
    print(f"total: {len(train_case_ids) + len(val_case_ids)}")
    print(f"splits_final.json: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
