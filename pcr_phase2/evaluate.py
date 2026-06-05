from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pcr_phase2.clinical import ClinicalPreprocessor
from pcr_phase2.dataset import BreastDCEPhase2Dataset
from pcr_phase2.metrics import compute_binary_metrics, compute_confusion_matrix, sigmoid
from pcr_phase2.model import Phase2MaskGuidedTransformerClassifier
from pcr_phase2.train import move_batch_to_device, run_inference
from pcr_phase2.utils import configure_logging, load_checkpoint, save_json, safe_mkdir


def _validate_checkpoint_schema(config: Dict[str, Any], clinical_dim: int) -> None:
    clinical_feature_version = int(config.get("clinical_feature_version", 0))
    if clinical_feature_version != 2:
        raise RuntimeError(
            "Checkpoint uses an incompatible clinical feature schema. "
            f"Expected clinical_feature_version=2, found {clinical_feature_version}. "
            "Retrain with the current five-feature clinical setup."
        )

    checkpoint_clinical_dim = config.get("clinical_dim")
    if checkpoint_clinical_dim is None:
        raise RuntimeError("Checkpoint is missing clinical_dim metadata and cannot be validated against the current schema.")
    if int(checkpoint_clinical_dim) != int(clinical_dim):
        raise RuntimeError(
            "Checkpoint clinical_dim does not match the current model input size. "
            f"Expected {clinical_dim}, found {checkpoint_clinical_dim}. "
            "Retrain or select a checkpoint built with the same clinical feature set."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 pCR classifier on a manifest split.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="internal_test")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = safe_mkdir(args.output_dir)
    logger = configure_logging(output_dir, log_name="evaluate.log")

    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    config: Dict[str, Any] = checkpoint["config"]
    best_threshold = float(checkpoint.get("best_threshold", 0.5))
    preprocessor_state = checkpoint.get("clinical_preprocessor_state")
    if preprocessor_state is None:
        raise RuntimeError("Checkpoint is missing clinical_preprocessor_state.")
    clinical_preprocessor = ClinicalPreprocessor.from_state_dict(preprocessor_state)

    use_image = bool(config.get("use_image", not config.get("clinical_only", False)))
    use_clinical = bool(config.get("use_clinical", not config.get("image_only", False)))
    clinical_dim = len(config["clinical_num_cols"]) + len(config["clinical_cat_cols"])
    _validate_checkpoint_schema(config, clinical_dim)

    dataset = BreastDCEPhase2Dataset(
        manifest_csv=args.manifest,
        split=args.split,
        clinical_num_cols=config["clinical_num_cols"],
        clinical_cat_cols=config["clinical_cat_cols"],
        clinical_preprocessor=clinical_preprocessor,
        roi_size=config["roi_size"],
        mask_mode=config.get("mask_mode", "gt"),
        roi_crop_enable=config.get("roi_crop_enable", True),
        roi_margin=config.get("roi_margin", 8),
        min_component_size=config.get("min_component_size", 16),
        keep_largest_component=config.get("keep_largest_component", True),
        normalize_mode=config.get("normalize_mode", "zscore"),
        target_col=config.get("target_col", "pCR"),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No rows found for split={args.split} in manifest={args.manifest}")

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    device = torch.device(args.device)
    model = Phase2MaskGuidedTransformerClassifier(
        clinical_dim=clinical_dim,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        dim_feedforward=config.get("dim_feedforward", 512),
        encoder_type=config.get("encoder_type", "official_medicalnet_resnet18"),
        encoder_base_channels=config.get("encoder_base_channels", 32),
        encoder_out_channels=config.get("encoder_out_channels", 128),
        shared_encoder=config.get("shared_encoder", True),
        use_image=use_image,
        use_clinical=use_clinical,
        fusion_type=config.get("fusion_type", "attention"),
        use_mask_channel=config.get("use_mask_channel", False),
        medicalnet_pretrained_path=config.get("medicalnet_pretrained_path", "third_party/pretrained/resnet_18_23dataset.pth"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    logits, labels, pids, datasets, splits = run_inference(model, loader, device, amp=config.get("amp", False))
    probs = sigmoid(logits)
    metrics = compute_binary_metrics(labels, probs, threshold=best_threshold)
    confusion = compute_confusion_matrix(labels, (probs >= best_threshold).astype(int))
    metrics["best_threshold"] = float(best_threshold)
    metrics["selected_threshold"] = float(best_threshold)

    predictions_df = pd.DataFrame(
        {
            "pid": pids,
            "y_true": labels.astype(float),
            "y_prob": probs.astype(float),
            "y_pred": (probs >= best_threshold).astype(int),
            "dataset": datasets,
            "split_final": splits,
        }
    )
    predictions_df.to_csv(Path(output_dir) / "predictions.csv", index=False)
    pd.DataFrame([confusion]).to_csv(Path(output_dir) / "confusion_matrix.csv", index=False)
    save_json(Path(output_dir) / "metrics.json", metrics)
    save_json(Path(output_dir) / "test_metrics.json", metrics)
    logger.info("evaluation_complete split=%s auroc=%s auprc=%s", args.split, metrics["auroc"], metrics["auprc"])


if __name__ == "__main__":
    main()
