from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcr_phase2.clinical import ClinicalPreprocessor
from pcr_phase2.dataset import BreastDCEPhase2Dataset
from pcr_phase2.metrics import compute_binary_metrics, find_best_threshold, sigmoid
from pcr_phase2.model import Phase2MaskGuidedTransformerClassifier
from pcr_phase2.utils import configure_logging, load_checkpoint, save_checkpoint, save_json, safe_mkdir, set_seed


class CombinedBCEFocalLoss(nn.Module):
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        logits = logits.float().view_as(targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_factor = targets * self.focal_alpha + (1.0 - targets) * (1.0 - self.focal_alpha)
        focal_loss = alpha_factor * torch.pow((1.0 - p_t).clamp_min(0.0), self.focal_gamma) * bce_loss
        combined = self.bce_weight * bce_loss + self.focal_weight * focal_loss
        return combined.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 mask-guided transformer pCR classifier.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--clinical-num-cols", nargs="*", default=["age", "menopause_missing"])
    parser.add_argument("--clinical-cat-cols", nargs="*", default=["HR", "HER2", "menopause"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--roi-size", type=int, nargs=3, default=[96, 160, 160])
    parser.add_argument("--roi-margin", type=int, default=8)
    parser.add_argument("--min-component-size", type=int, default=16)
    parser.add_argument("--mask-mode", choices=["gt", "pred", "none"], default="pred")
    parser.add_argument("--roi-crop-enable", dest="roi_crop_enable", action="store_true", default=True)
    parser.add_argument("--disable-roi-crop", dest="roi_crop_enable", action="store_false")
    parser.add_argument("--keep-largest-component", dest="keep_largest_component", action="store_true", default=True)
    parser.add_argument("--disable-keep-largest-component", dest="keep_largest_component", action="store_false")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--clinical-only", action="store_true")
    parser.add_argument("--fusion-type", choices=["attention", "concat"], default="attention")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["official_medicalnet_resnet18", "official_medicalnet_resnet34", "official_medicalnet_resnet50"],
        default="official_medicalnet_resnet18",
    )
    parser.add_argument("--encoder-base-channels", type=int, default=32)
    parser.add_argument("--encoder-out-channels", type=int, default=128)
    parser.add_argument("--shared-encoder", action="store_true", default=True)
    parser.add_argument("--disable-shared-encoder", dest="shared_encoder", action="store_false")
    parser.add_argument("--use-mask-channel", action="store_true")
    parser.add_argument("--medicalnet-pretrained-path", type=str, default=None)
    parser.add_argument("--enable-augmentation", action="store_true", default=False)
    parser.add_argument("--augmentation-strength", choices=["light", "medium", "strong"], default="light")
    parser.add_argument("--freeze-image-encoder", action="store_true", default=False)
    parser.add_argument("--debug-forward-only", action="store_true")
    parser.add_argument("--normalize-mode", choices=["zscore", "percentile"], default="zscore")
    parser.add_argument("--target-col", type=str, default="pCR")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", choices=["plateau", "cosine", "none"], default="plateau")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--monitor-metric", type=str, default="auroc")
    parser.add_argument("--threshold-objective", type=str, default="balanced_accuracy")
    parser.add_argument("--loss-bce-weight", type=float, default=0.5)
    parser.add_argument("--loss-focal-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume training from a checkpoint path.")
    return parser.parse_args()


def apply_encoder_defaults(args: argparse.Namespace) -> None:
    base_defaults = {
        "batch_size": 2,
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.2,
        "amp": False,
    }
    for field, default_value in base_defaults.items():
        if getattr(args, field) is None:
            setattr(args, field, default_value)


def compute_pos_weight(labels: np.ndarray) -> float:
    labels = labels.astype(np.float32)
    pos = float((labels > 0.5).sum())
    neg = float((labels <= 0.5).sum())
    if pos < 1:
        return 1.0
    return max(neg / pos, 1e-6)


def build_weighted_sampler(labels: np.ndarray) -> Optional[WeightedRandomSampler]:
    labels = labels.astype(np.float32)
    positives = float((labels > 0.5).sum())
    negatives = float((labels <= 0.5).sum())
    if positives < 1 or negatives < 1:
        return None
    positive_weight = 0.5 / positives
    negative_weight = 0.5 / negatives
    sample_weights = np.where(labels > 0.5, positive_weight, negative_weight).astype(np.float64)
    return WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_pids: List[str] = []
    all_datasets: List[str] = []
    all_splits: List[str] = []
    use_amp = amp and device.type == "cuda"
    autocast_device = device.type

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            outputs = model(batch["image"], batch["mask"], batch["clinical"])
        all_logits.append(outputs["logits"].detach().cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())
        all_pids.extend(list(batch["pid"]))
        all_datasets.extend(list(batch["dataset"]))
        all_splits.extend(list(batch["split_final"]))

    logits = np.concatenate(all_logits) if all_logits else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.float32)
    return logits, labels, all_pids, all_datasets, all_splits


def write_train_log(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_datasets(args: argparse.Namespace, clinical_preprocessor: ClinicalPreprocessor) -> Tuple[BreastDCEPhase2Dataset, BreastDCEPhase2Dataset]:
    train_ds = BreastDCEPhase2Dataset(
        manifest_csv=args.manifest,
        split="train",
        clinical_num_cols=args.clinical_num_cols,
        clinical_cat_cols=args.clinical_cat_cols,
        clinical_preprocessor=clinical_preprocessor,
        roi_size=args.roi_size,
        mask_mode=args.mask_mode,
        roi_crop_enable=args.roi_crop_enable,
        roi_margin=args.roi_margin,
        min_component_size=args.min_component_size,
        keep_largest_component=args.keep_largest_component,
        normalize_mode=args.normalize_mode,
        target_col=args.target_col,
        enable_augmentation=args.enable_augmentation,
        augmentation_strength=args.augmentation_strength,
    )
    val_ds = BreastDCEPhase2Dataset(
        manifest_csv=args.manifest,
        split="val",
        clinical_num_cols=args.clinical_num_cols,
        clinical_cat_cols=args.clinical_cat_cols,
        clinical_preprocessor=clinical_preprocessor,
        roi_size=args.roi_size,
        mask_mode=args.mask_mode,
        roi_crop_enable=args.roi_crop_enable,
        roi_margin=args.roi_margin,
        min_component_size=args.min_component_size,
        keep_largest_component=args.keep_largest_component,
        normalize_mode=args.normalize_mode,
        target_col=args.target_col,
        enable_augmentation=False,
        augmentation_strength=args.augmentation_strength,
    )
    return train_ds, val_ds


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return value


def _count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        parameter_count = int(parameter.numel())
        total_parameters += parameter_count
        if parameter.requires_grad:
            trainable_parameters += parameter_count
    return trainable_parameters, total_parameters


def _validate_resume_compatibility(args: argparse.Namespace, checkpoint_config: Dict[str, Any], clinical_dim: int) -> None:
    expected_fields = [
        "clinical_num_cols",
        "clinical_cat_cols",
        "clinical_dim",
        "clinical_feature_version",
        "roi_size",
        "roi_margin",
        "min_component_size",
        "d_model",
        "n_heads",
        "num_layers",
        "dropout",
        "dim_feedforward",
        "encoder_type",
        "encoder_base_channels",
        "encoder_out_channels",
        "fusion_type",
        "use_mask_channel",
        "mask_mode",
        "roi_crop_enable",
        "keep_largest_component",
        "image_only",
        "clinical_only",
        "enable_augmentation",
        "augmentation_strength",
    ]
    mismatches: List[str] = []
    for field in expected_fields:
        if field not in checkpoint_config:
            continue
        if field == "clinical_dim":
            current_value = clinical_dim
        elif field == "clinical_feature_version":
            current_value = 2
        else:
            current_value = _normalize_for_compare(getattr(args, field))
        checkpoint_value = _normalize_for_compare(checkpoint_config.get(field))
        if current_value != checkpoint_value:
            mismatches.append(f"{field}: current={current_value} checkpoint={checkpoint_value}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise RuntimeError(
            "resume checkpoint is incompatible with the current command-line configuration. "
            "Use the same clinical columns, encoder settings, and ROI/model sizes as the checkpoint, "
            f"or start a new run. Mismatches: {mismatch_text}"
        )


def main() -> None:
    args = parse_args()
    apply_encoder_defaults(args)
    if args.image_only and args.clinical_only:
        raise RuntimeError("--image-only and --clinical-only cannot both be set.")
    args.use_image = not args.clinical_only
    args.use_clinical = not args.image_only

    output_dir = safe_mkdir(args.output_dir)
    checkpoints_dir = safe_mkdir(Path(output_dir) / "checkpoints")
    logger = configure_logging(output_dir)
    set_seed(args.seed)

    manifest_df = pd.read_csv(args.manifest)
    train_df = manifest_df[manifest_df["split_final"].astype(str) == "train"].copy()
    val_df = manifest_df[manifest_df["split_final"].astype(str) == "val"].copy()
    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("Manifest must contain non-empty train and val splits.")

    clinical_preprocessor = ClinicalPreprocessor(args.clinical_num_cols, args.clinical_cat_cols)
    clinical_preprocessor.fit(train_df)
    clinical_preprocessor_path = Path(output_dir) / "clinical_preprocessor.json"
    clinical_preprocessor.save_json(clinical_preprocessor_path.as_posix())
    clinical_dim = len(args.clinical_num_cols) + len(args.clinical_cat_cols)

    train_ds, val_ds = build_datasets(args, clinical_preprocessor)
    train_labels = np.asarray([float(row.get(args.target_col, 0.0)) for row in train_ds.rows], dtype=np.float32)
    sampler = build_weighted_sampler(train_labels) if args.use_weighted_sampler else None
    pin_memory = torch.device(args.device).type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    device = torch.device(args.device)
    model = Phase2MaskGuidedTransformerClassifier(
        clinical_dim=clinical_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        encoder_type=args.encoder_type,
        encoder_base_channels=args.encoder_base_channels,
        encoder_out_channels=args.encoder_out_channels,
        shared_encoder=args.shared_encoder,
        use_image=args.use_image,
        use_clinical=args.use_clinical,
        fusion_type=args.fusion_type,
        use_mask_channel=args.use_mask_channel,
        medicalnet_pretrained_path=args.medicalnet_pretrained_path,
    ).to(device)

    model.set_image_encoder_frozen(args.freeze_image_encoder)
    trainable_parameters, total_parameters = _count_trainable_parameters(model)
    trainable_parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameter_list:
        raise RuntimeError("No trainable parameters remain after applying the requested freeze configuration.")

    train_augmentation_summary = train_ds.get_augmentation_summary()

    logger.info("train_audit=%s", train_ds.audit_summary)
    logger.info("val_audit=%s", val_ds.audit_summary)
    logger.info(
        "augmentation_enabled=%s augmentation_strength=%s",
        bool(args.enable_augmentation),
        args.augmentation_strength,
    )
    logger.info(
        "enabled_augmentations=%s",
        "random horizontal flip, small rotation, very light elastic deformation, MRI bias field / intensity inhomogeneity"
        if args.enable_augmentation
        else "none",
    )
    logger.info(
        "augmentation_probabilities=flip=%.2f rotation=%.2f elastic=%.2f bias_field=%.2f rotation_range=+/-%.1fdeg elastic_prob=%.2f synchronized_phase_augmentation=%s",
        float(train_augmentation_summary["flip_prob"]),
        float(train_augmentation_summary["rotation_prob"]),
        float(train_augmentation_summary["elastic_prob"]),
        float(train_augmentation_summary["bias_field_prob"]),
        float(train_augmentation_summary["rotation_degrees"]),
        float(train_augmentation_summary["elastic_prob"]),
        bool(train_augmentation_summary["synchronized_phase_augmentation"]),
    )
    logger.info(
        "image_encoder_frozen=%s trainable_parameters=%d total_parameters=%d",
        bool(args.freeze_image_encoder),
        trainable_parameters,
        total_parameters,
    )
    save_json(
        Path(output_dir) / "data_audit.json",
        {
            "train": train_ds.audit_summary,
            "val": val_ds.audit_summary,
            "manifest_rows": int(len(manifest_df)),
            "train_rows": int(len(train_ds.rows)),
            "val_rows": int(len(val_ds.rows)),
        },
    )

    resume_checkpoint_path: Path | None = None
    if args.resume_from is not None:
        resume_checkpoint_path = Path(args.resume_from)
    else:
        candidate_checkpoint = checkpoints_dir / "last.pt"
        if candidate_checkpoint.exists():
            resume_checkpoint_path = candidate_checkpoint

    resume_optimizer_state = True
    resume_scheduler_state = True
    resume_scaler_state = True

    pos_weight_value = compute_pos_weight(train_labels)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = CombinedBCEFocalLoss(
        pos_weight=pos_weight,
        bce_weight=args.loss_bce_weight,
        focal_weight=args.loss_focal_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
    optimizer = torch.optim.AdamW(trainable_parameter_list, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"
    best_score = -np.inf
    best_threshold = 0.5
    best_metrics: Dict[str, Any] = {}
    no_improve = 0
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if resume_checkpoint_path is not None and resume_checkpoint_path.exists():
        resume_checkpoint = load_checkpoint(resume_checkpoint_path, map_location=device)
        _validate_resume_compatibility(args, resume_checkpoint.get("config", {}), clinical_dim)
        model.load_state_dict(resume_checkpoint["model_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_score = float(resume_checkpoint.get("best_score", best_score))
        best_threshold = float(resume_checkpoint.get("best_threshold", best_threshold))
        best_metrics = dict(resume_checkpoint.get("best_metrics", resume_checkpoint.get("val_metrics", {})))
        no_improve = int(resume_checkpoint.get("no_improve", 0))
        history = list(resume_checkpoint.get("history", []))
        checkpoint_config = resume_checkpoint.get("config", {}) if isinstance(resume_checkpoint.get("config", {}), dict) else {}
        checkpoint_frozen = bool(checkpoint_config.get("freeze_image_encoder", False))
        if checkpoint_frozen != bool(args.freeze_image_encoder):
            resume_optimizer_state = False
            resume_scheduler_state = False
            resume_scaler_state = False
            logger.info(
                "resume_freeze_state_changed checkpoint_freeze_image_encoder=%s current_freeze_image_encoder=%s; reinitializing optimizer for staged fine-tuning",
                checkpoint_frozen,
                bool(args.freeze_image_encoder),
            )
        if resume_optimizer_state and resume_checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if scheduler is not None and resume_scheduler_state and resume_checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if resume_scaler_state and resume_checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        logger.info("resuming_from=%s start_epoch=%d", resume_checkpoint_path.as_posix(), start_epoch)

    config_payload = vars(args).copy()
    config_payload["use_image"] = bool(args.use_image)
    config_payload["use_clinical"] = bool(args.use_clinical)
    config_payload["clinical_dim"] = clinical_dim
    config_payload["clinical_feature_version"] = 2
    config_payload["clinical_preprocessor_json"] = clinical_preprocessor_path.as_posix()
    config_payload["enable_augmentation"] = bool(args.enable_augmentation)
    config_payload["augmentation_strength"] = str(args.augmentation_strength)
    config_payload["freeze_image_encoder"] = bool(args.freeze_image_encoder)
    config_payload["augmentation_summary"] = train_augmentation_summary
    config_payload["train_audit"] = train_ds.audit_summary
    config_payload["val_audit"] = val_ds.audit_summary
    save_json(Path(output_dir) / "config.json", config_payload)
    save_json(Path(output_dir) / "medicalnet_load_report.json", getattr(model, "medicalnet_load_report", {"enabled": False, "used": False}))

    if args.debug_forward_only:
        debug_batch = next(iter(train_loader))
        debug_batch = move_batch_to_device(debug_batch, device)
        print(f"image shape: {tuple(debug_batch['image'].shape)}")
        print(f"mask shape: {tuple(debug_batch['mask'].shape)}")
        print(f"clinical shape: {tuple(debug_batch['clinical'].shape)}")
        debug_outputs = model.debug_forward(debug_batch["image"], debug_batch["mask"], debug_batch["clinical"])
        for index, feature_map in enumerate(debug_outputs["phase_feature_maps"]):
            print(f"phase {index} feature map shape: {tuple(feature_map.shape)}")
        for index, phase_token in enumerate(debug_outputs["phase_tokens"]):
            print(f"phase {index} token shape: {tuple(phase_token.shape)}")
        print(f"final token tensor shape: {tuple(debug_outputs['final_tokens'].shape)}")
        print(f"logits shape: {tuple(debug_outputs['logits'].shape)}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        model.set_image_encoder_frozen(args.freeze_image_encoder)
        epoch_losses: List[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch["image"], batch["mask"], batch["clinical"])
                loss = criterion(outputs["logits"], batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu().item()))

        val_logits, val_labels, _, _, _ = run_inference(model, val_loader, device, amp=args.amp)
        val_probs = sigmoid(val_logits)
        epoch_threshold = find_best_threshold(val_labels, val_probs, objective=args.threshold_objective)
        val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=epoch_threshold)
        if len(val_logits) > 0:
            val_loss = float(
                criterion(
                    torch.from_numpy(val_logits).to(device=device, dtype=torch.float32),
                    torch.from_numpy(val_labels).to(device=device, dtype=torch.float32),
                )
                .detach()
                .cpu()
                .item()
            )
        else:
            val_loss = float("nan")
        val_metrics["loss"] = val_loss

        monitor_value = float(val_metrics.get(args.monitor_metric, float("nan")))
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(monitor_value if not np.isnan(monitor_value) else -np.inf)
            else:
                scheduler.step()

        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "val_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "best_threshold": epoch_threshold,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)

        logger.info(
            "epoch=%d train_loss=%.4f val_auroc=%s val_auprc=%s val_f1=%s threshold=%.4f",
            epoch,
            history_row["train_loss"],
            f"{val_metrics['auroc']:.4f}" if not np.isnan(val_metrics["auroc"]) else "nan",
            f"{val_metrics['auprc']:.4f}" if not np.isnan(val_metrics["auprc"]) else "nan",
            f"{val_metrics['f1']:.4f}" if not np.isnan(val_metrics["f1"]) else "nan",
            epoch_threshold,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "config": config_payload,
            "val_metrics": val_metrics,
            "best_threshold": epoch_threshold,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "no_improve": no_improve,
            "history": history,
            "clinical_preprocessor_state": clinical_preprocessor.state_dict(),
        }
        save_checkpoint(checkpoints_dir / "last.pt", checkpoint_payload)

        if not np.isnan(monitor_value) and monitor_value > best_score:
            best_score = monitor_value
            best_threshold = epoch_threshold
            best_metrics = dict(val_metrics)
            best_metrics["best_threshold"] = float(best_threshold)
            no_improve = 0
            checkpoint_payload["best_score"] = best_score
            checkpoint_payload["best_threshold"] = best_threshold
            checkpoint_payload["best_metrics"] = best_metrics
            save_checkpoint(checkpoints_dir / "best.pt", checkpoint_payload)
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            logger.info("Early stopping triggered at epoch=%d", epoch)
            break

    best_metrics = dict(best_metrics)
    best_metrics["best_threshold"] = float(best_threshold)
    save_json(Path(output_dir) / "val_metrics.json", best_metrics)
    write_train_log(Path(output_dir) / "train_log.csv", history)
    save_json(
        Path(output_dir) / "training_summary.json",
        {
            "best_score": float(best_score),
            "best_threshold": float(best_threshold),
            "monitor_metric": args.monitor_metric,
            "threshold_objective": args.threshold_objective,
            "train_cases": int(len(train_ds.rows)),
            "val_cases": int(len(val_ds.rows)),
            "enable_augmentation": bool(args.enable_augmentation),
            "augmentation_strength": str(args.augmentation_strength),
            "augmentation_summary": train_augmentation_summary,
            "freeze_image_encoder": bool(args.freeze_image_encoder),
        },
    )
    logger.info("training_complete best_%s=%s best_threshold=%.4f", args.monitor_metric, best_score, best_threshold)


if __name__ == "__main__":
    main()

