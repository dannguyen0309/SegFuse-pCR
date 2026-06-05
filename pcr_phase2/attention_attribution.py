#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ATTENTION_HEATMAP_CMAP = "viridis"
MODALITY_COLORS = {"DCE phases": "#1F77B4", "Clinical": "#FF7F0E"}
ATTENTION_TEXT_COLOR_THRESHOLD = 0.30
ATTENTION_TEXT_HIGH_COLOR = "#0B1F4D"
ATTENTION_TEXT_LOW_COLOR = "white"

from pcr_phase2.clinical import ClinicalPreprocessor
from pcr_phase2.dataset import BreastDCEPhase2Dataset
from pcr_phase2.metrics import compute_binary_metrics, compute_confusion_matrix, sigmoid
from pcr_phase2.model import Phase2MaskGuidedTransformerClassifier
from pcr_phase2.train import move_batch_to_device
from pcr_phase2.utils import configure_logging, load_checkpoint, save_json, safe_mkdir


@dataclass
class AttentionCaptureState:
    original_forwards: List[Tuple[nn.Module, Any]]
    captured: List[Optional[torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 transformer attention attribution on a manifest split.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="internal_test")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--case-id", type=str, default=None, help="Optional case id to evaluate one case only.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-pred-mask", action="store_true", help="Use predicted-mask ROI paths instead of GT mask paths.")
    parser.add_argument("--layer-index", type=int, default=-1, help="Focus layer for summaries and heatmaps. Defaults to the last layer.")
    return parser.parse_args()


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, *bases: Path) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    candidates = [path]
    for base in bases:
        candidates.append(base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _validate_checkpoint_schema(config: Dict[str, Any], clinical_dim: int) -> None:
    clinical_feature_version = int(config.get("clinical_feature_version", 0))
    if clinical_feature_version != 2:
        raise RuntimeError(
            "Checkpoint uses an incompatible clinical feature schema. "
            f"Expected clinical_feature_version=2, found {clinical_feature_version}."
        )

    checkpoint_clinical_dim = config.get("clinical_dim")
    if checkpoint_clinical_dim is None:
        raise RuntimeError("Checkpoint is missing clinical_dim metadata and cannot be validated against the current schema.")
    if int(checkpoint_clinical_dim) != int(clinical_dim):
        raise RuntimeError(
            "Checkpoint clinical_dim does not match the current model input size. "
            f"Expected {clinical_dim}, found {checkpoint_clinical_dim}."
        )


def _load_clinical_preprocessor(checkpoint: Dict[str, Any], checkpoint_path: Path) -> ClinicalPreprocessor:
    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    preprocessor_state = checkpoint.get("clinical_preprocessor_state")
    if isinstance(preprocessor_state, dict) and preprocessor_state:
        return ClinicalPreprocessor.from_state_dict(preprocessor_state)

    preprocessor_json = config.get("clinical_preprocessor_json")
    if preprocessor_json:
        resolved = _resolve_path(preprocessor_json, checkpoint_path.parent, _resolve_project_root())
        if resolved.exists():
            return ClinicalPreprocessor.load_json(resolved.as_posix())

    raise RuntimeError("Unable to recover the clinical preprocessor from the checkpoint.")


def _extract_model_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return {str(name).replace("module.", "").replace("model.", "").replace("net.", ""): tensor for name, tensor in value.items() if torch.is_tensor(tensor)}
    tensor_items = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
    if tensor_items:
        return tensor_items
    raise RuntimeError("Checkpoint does not contain a recognizable model state dict.")


def _build_model(checkpoint: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> Tuple[Phase2MaskGuidedTransformerClassifier, ClinicalPreprocessor, Dict[str, Any]]:
    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    clinical_preprocessor = _load_clinical_preprocessor(checkpoint, checkpoint_path)
    clinical_dim = int(config.get("clinical_dim", len(clinical_preprocessor.feature_columns)))
    _validate_checkpoint_schema(config, clinical_dim)

    pretrained_path = config.get("medicalnet_pretrained_path")
    resolved_pretrained_path = None
    if pretrained_path:
        resolved_pretrained_path = _resolve_path(pretrained_path, checkpoint_path.parent, _resolve_project_root())

    model = Phase2MaskGuidedTransformerClassifier(
        clinical_dim=clinical_dim,
        d_model=int(config.get("d_model", 256)),
        n_heads=int(config.get("n_heads", 4)),
        num_layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.2)),
        dim_feedforward=int(config.get("dim_feedforward", 512)),
        encoder_type=str(config.get("encoder_type", "official_medicalnet_resnet34")),
        encoder_base_channels=int(config.get("encoder_base_channels", 32)),
        encoder_out_channels=int(config.get("encoder_out_channels", 128)),
        shared_encoder=bool(config.get("shared_encoder", True)),
        use_image=bool(config.get("use_image", True)),
        use_clinical=bool(config.get("use_clinical", True)),
        fusion_type=str(config.get("fusion_type", "attention")),
        use_mask_channel=bool(config.get("use_mask_channel", False)),
        medicalnet_pretrained_path=resolved_pretrained_path.as_posix() if resolved_pretrained_path is not None else None,
    ).to(device)

    model_state = _extract_model_state(checkpoint)
    load_result = model.load_state_dict(model_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint loading failed unexpectedly. "
            f"missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}"
        )
    model.eval()
    return model, clinical_preprocessor, config


def _build_dataset(
    manifest: Path,
    split: str,
    clinical_preprocessor: ClinicalPreprocessor,
    use_pred_mask: bool,
    config: Dict[str, Any],
) -> BreastDCEPhase2Dataset:
    roi_size = config.get("roi_size", [96, 160, 160])
    mask_mode = "pred" if use_pred_mask else str(config.get("mask_mode", "gt"))
    return BreastDCEPhase2Dataset(
        manifest_csv=manifest.as_posix(),
        split=split,
        clinical_num_cols=list(config.get("clinical_num_cols", clinical_preprocessor.numeric_cols)),
        clinical_cat_cols=list(config.get("clinical_cat_cols", clinical_preprocessor.categorical_cols)),
        clinical_preprocessor=clinical_preprocessor,
        roi_size=roi_size,
        mask_mode=mask_mode,
        roi_crop_enable=bool(config.get("roi_crop_enable", True)),
        roi_margin=int(config.get("roi_margin", 8)),
        min_component_size=int(config.get("min_component_size", 16)),
        keep_largest_component=bool(config.get("keep_largest_component", True)),
        normalize_mode=str(config.get("normalize_mode", "zscore")),
        target_col=str(config.get("target_col", "pCR")),
        strict_mask=False,
        enable_augmentation=False,
        augmentation_strength=str(config.get("augmentation_strength", "light")),
    )


def _find_case_sample(dataset: BreastDCEPhase2Dataset, case_id: str) -> Dict[str, Any]:
    for index, row in enumerate(dataset.rows):
        pid = str(row.get("pid"))
        dataset_name = str(row.get("dataset"))
        if case_id in {pid, dataset_name, f"{pid}_{dataset_name}"}:
            return dataset[index]
    raise FileNotFoundError(f"case_id={case_id} was not found after dataset filtering and ROI resolution.")


def _parse_layer_index(requested: int, num_layers: int) -> int:
    if num_layers < 1:
        raise ValueError("The transformer must contain at least one layer.")
    if requested < 0:
        return num_layers - 1
    if requested >= num_layers:
        raise ValueError(f"Requested layer index {requested} is out of range for num_layers={num_layers}.")
    return requested


@contextlib.contextmanager
def _capture_attention_weights(model: nn.Module) -> Iterable[AttentionCaptureState]:
    transformer = getattr(model, "transformer", None)
    if transformer is None or not hasattr(transformer, "layers"):
        raise RuntimeError("The loaded model does not expose a transformer encoder.")

    original_forwards: List[Tuple[nn.Module, Any]] = []
    captured: List[Optional[torch.Tensor]] = [None for _ in transformer.layers]

    def _make_patched_forward(layer_index: int, original_forward: Any):
        def patched_forward(*args: Any, **kwargs: Any):
            patched_kwargs = dict(kwargs)
            patched_kwargs["need_weights"] = True
            patched_kwargs["average_attn_weights"] = False
            try:
                output, attention_weights = original_forward(*args, **patched_kwargs)
            except TypeError as exc:
                if "average_attn_weights" not in str(exc):
                    raise
                patched_kwargs.pop("average_attn_weights", None)
                output, attention_weights = original_forward(*args, **patched_kwargs)
            captured[layer_index] = attention_weights.detach()
            return output, attention_weights

        return patched_forward

    for layer_index, encoder_layer in enumerate(transformer.layers):
        self_attn = encoder_layer.self_attn
        original_forward = self_attn.forward
        original_forwards.append((self_attn, original_forward))
        self_attn.forward = _make_patched_forward(layer_index, original_forward)

    try:
        yield AttentionCaptureState(original_forwards=original_forwards, captured=captured)
    finally:
        for attention_module, original_forward in original_forwards:
            attention_module.forward = original_forward


def _token_labels(model: Phase2MaskGuidedTransformerClassifier) -> List[str]:
    labels: List[str] = []
    if getattr(model, "use_image", False):
        labels.extend(["phase_pre", "phase_early", "phase_late"])
    if getattr(model, "use_clinical", False):
        labels.append("Clinical")
    return labels


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _stack_attention_layers(captured: List[Optional[torch.Tensor]]) -> np.ndarray:
    tensors: List[np.ndarray] = []
    for attention in captured:
        if attention is None:
            raise RuntimeError("Attention was not captured for one or more transformer layers.")
        tensors.append(_tensor_to_numpy(attention))
    stacked = np.stack(tensors, axis=0)
    if stacked.ndim != 5:
        raise AssertionError(
            f"Expected stacked attention tensor with shape [layers, batch, heads, tokens, tokens], got {stacked.shape}"
        )
    return stacked


def _mean_over_heads(attention_layers: np.ndarray) -> np.ndarray:
    if attention_layers.ndim != 5:
        raise AssertionError(
            f"Expected attention tensor with shape [layers, batch, heads, tokens, tokens], got {attention_layers.shape}"
        )
    return attention_layers.mean(axis=2)


def _token_summary(mean_attention: np.ndarray, token_labels: Sequence[str]) -> Dict[str, Any]:
    if mean_attention.ndim != 3:
        raise AssertionError(
            f"Expected mean attention tensor with shape [layers, tokens, tokens], got {mean_attention.shape}"
        )
    focus_attention = mean_attention[-1]
    incoming = focus_attention.mean(axis=0)
    outgoing = focus_attention.mean(axis=1)
    focus_index = int(np.argmax(incoming))
    return {
        "focus_layer_index": int(mean_attention.shape[0] - 1),
        "focus_layer_token_incoming": {label: float(incoming[index]) for index, label in enumerate(token_labels)},
        "focus_layer_token_outgoing": {label: float(outgoing[index]) for index, label in enumerate(token_labels)},
        "focus_layer_top_token": str(token_labels[focus_index]),
        "focus_layer_top_token_score": float(incoming[focus_index]),
        "focus_layer_mean_attention": focus_attention.tolist(),
    }


def _render_attention_heatmap(mean_attention: np.ndarray, token_labels: Sequence[str], output_path: Path, title: str) -> None:
    if mean_attention.ndim != 3:
        raise AssertionError(f"Expected mean attention tensor with shape [layers, tokens, tokens], got {mean_attention.shape}")

    num_layers = mean_attention.shape[0]
    num_columns = min(2, num_layers)
    num_rows = int(np.ceil(num_layers / num_columns))
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(6.5 * num_columns, 5.5 * num_rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(num_rows, num_columns)
    vmax = float(np.max(mean_attention)) if mean_attention.size else 1.0
    vmax = vmax if vmax > 0.0 else 1.0
    display_labels = [
        "Pre" if label == "phase_pre" else "Early" if label == "phase_early" else "Late" if label == "phase_late" else label
        for label in token_labels
    ]

    for layer_index in range(num_rows * num_columns):
        row = layer_index // num_columns
        col = layer_index % num_columns
        axis = axes_array[row, col]
        if layer_index >= num_layers:
            axis.axis("off")
            continue
        layer_attention = mean_attention[layer_index]
        image = axis.imshow(layer_attention, cmap=ATTENTION_HEATMAP_CMAP, vmin=0.0, vmax=vmax)
        axis.set_title("")
        axis.set_xticks(range(len(token_labels)))
        axis.set_yticks(range(len(token_labels)))
        axis.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=13)
        axis.set_yticklabels(display_labels, fontsize=13)
        for i in range(layer_attention.shape[0]):
            for j in range(layer_attention.shape[1]):
                text_color = ATTENTION_TEXT_HIGH_COLOR if float(layer_attention[i, j]) > ATTENTION_TEXT_COLOR_THRESHOLD else ATTENTION_TEXT_LOW_COLOR
                axis.text(j, i, f"{layer_attention[i, j]:.2f}", ha="center", va="center", fontsize=11, color=text_color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(title, fontsize=18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _build_modality_contribution_table(case_rows: pd.DataFrame) -> pd.DataFrame:
    if case_rows.empty:
        return pd.DataFrame(columns=["modality", "mean_attention", "std_attention", "mean_share_pct"])

    table = case_rows.copy()
    phase_columns = ["focus_in_phase_pre", "focus_in_phase_early", "focus_in_phase_late"]
    table["phase_attention"] = table[phase_columns].sum(axis=1)
    clinical_column = "focus_in_clinical" if "focus_in_clinical" in table.columns else "focus_in_Clinical"
    if clinical_column not in table.columns:
        raise KeyError("Missing clinical attention column in attention summary table.")
    table["clinical_attention"] = table[clinical_column]
    table["phase_share_pct"] = 100.0 * table["phase_attention"]
    table["clinical_share_pct"] = 100.0 * table["clinical_attention"]

    summary_rows = []
    for modality_name, column_name in (("DCE phases", "phase_attention"), ("Clinical", "clinical_attention")):
        series = table[column_name].astype(float)
        summary_rows.append(
            {
                "modality": modality_name,
                "mean_attention": float(series.mean()),
                "std_attention": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "mean_share_pct": float((100.0 * series).mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def _render_modality_contribution_chart(summary_table: pd.DataFrame, output_path: Path, title: str) -> None:
    if summary_table.empty:
        raise ValueError("Cannot render a modality contribution chart from an empty summary table.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    modality_order = ["DCE phases", "Clinical"]

    summary_table = summary_table.set_index("modality").reindex(modality_order).reset_index()
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#F7F7F5")
    bars = axis.bar(
        summary_table["modality"],
        summary_table["mean_attention"],
        yerr=summary_table["std_attention"],
        capsize=8,
        color=[MODALITY_COLORS[modality] for modality in summary_table["modality"]],
        alpha=0.92,
        edgecolor="#1E1E1E",
        linewidth=0.8,
    )

    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean incoming attention", fontsize=13)
    axis.set_title(title, fontsize=18)
    axis.tick_params(axis="x", labelsize=13)
    axis.tick_params(axis="y", labelsize=12)
    axis.tick_params(axis="both", colors="#1E1E1E")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#1E1E1E")
    axis.spines["bottom"].set_color("#1E1E1E")

    for bar, share_value, std_value in zip(bars, summary_table["mean_share_pct"], summary_table["std_attention"]):
        label_y = bar.get_height() + float(std_value) + 0.025
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{bar.get_height():.3f}\n({share_value:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#1E1E1E",
        )

    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _case_dir_name(pid: str, dataset_name: str) -> str:
    return f"{pid}_{dataset_name}" if dataset_name and dataset_name != "nan" else str(pid)


def main() -> None:
    args = parse_args()
    output_dir = safe_mkdir(args.output_dir)
    logger = configure_logging(output_dir, log_name="attention_attribution.log")

    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    checkpoint_path = Path(args.checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    best_threshold = float(checkpoint.get("best_threshold", 0.5))
    use_amp = bool(config.get("amp", False))

    device = torch.device(args.device)
    model, clinical_preprocessor, model_config = _build_model(checkpoint, checkpoint_path, device)
    dataset = _build_dataset(Path(args.manifest), args.split, clinical_preprocessor, args.use_pred_mask, model_config)
    if len(dataset) == 0:
        raise RuntimeError(f"No rows found for split={args.split} in manifest={args.manifest}")

    if args.case_id is not None:
        samples = [_find_case_sample(dataset, args.case_id)]
    else:
        samples = dataset

    loader = DataLoader(
        samples,
        batch_size=1 if args.case_id is not None else int(config.get("batch_size", 2)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    token_labels = _token_labels(model)
    focus_layer_index = _parse_layer_index(args.layer_index, len(model.transformer.layers))
    case_rows: List[Dict[str, Any]] = []
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_pids: List[str] = []
    all_datasets: List[str] = []
    all_splits: List[str] = []

    attention_root = safe_mkdir(output_dir / "attention_cases")
    autocast_device = device.type

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with _capture_attention_weights(model) as capture_state:
                with torch.autocast(device_type=autocast_device, enabled=use_amp and device.type == "cuda"):
                    outputs = model(batch["image"], batch["mask"], batch["clinical"])

            logits = outputs["logits"].detach().cpu().numpy()
            probabilities = sigmoid(logits)
            labels = batch["label"].detach().cpu().numpy()
            pids = list(batch["pid"])
            datasets = list(batch["dataset"])
            splits = list(batch["split_final"])

            attention_layers = _stack_attention_layers(capture_state.captured)
            mean_attention = _mean_over_heads(attention_layers)

            all_logits.append(logits)
            all_labels.append(labels)
            all_pids.extend(pids)
            all_datasets.extend(datasets)
            all_splits.extend(splits)

            for batch_index, pid in enumerate(pids):
                dataset_name = str(datasets[batch_index])
                split_name = str(splits[batch_index])
                case_name = _case_dir_name(str(pid), dataset_name)
                case_dir = safe_mkdir(attention_root / case_name)

                case_attention = attention_layers[:, batch_index]
                case_mean_attention = mean_attention[:, batch_index]
                np.save(case_dir / "attention_weights.npy", case_attention)
                _render_attention_heatmap(
                    case_mean_attention,
                    token_labels,
                    case_dir / "attention_heatmap.png",
                    title="Attention Attribution",
                )

                case_summary = {
                    "pid": str(pid),
                    "dataset": dataset_name,
                    "split_final": split_name,
                    "y_true": float(labels[batch_index]),
                    "y_prob": float(probabilities[batch_index]),
                    "y_pred": int(probabilities[batch_index] >= best_threshold),
                    "best_threshold": float(best_threshold),
                    "token_labels": list(token_labels),
                    "num_layers": int(case_attention.shape[0]),
                    "num_heads": int(case_attention.shape[1]),
                    "num_tokens": int(case_attention.shape[2]),
                    "focus_layer_index": int(focus_layer_index),
                }
                focus_attention = case_mean_attention[focus_layer_index]
                incoming = focus_attention.mean(axis=0)
                outgoing = focus_attention.mean(axis=1)
                case_summary["focus_layer_incoming"] = {label: float(incoming[index]) for index, label in enumerate(token_labels)}
                case_summary["focus_layer_outgoing"] = {label: float(outgoing[index]) for index, label in enumerate(token_labels)}
                case_summary["focus_layer_top_token"] = str(token_labels[int(np.argmax(incoming))])
                case_summary["focus_layer_top_token_score"] = float(np.max(incoming))
                case_summary["focus_layer_mean_attention"] = focus_attention.tolist()
                save_json(case_dir / "attention_summary.json", case_summary)

                row = {
                    "pid": str(pid),
                    "dataset": dataset_name,
                    "split_final": split_name,
                    "y_true": float(labels[batch_index]),
                    "y_prob": float(probabilities[batch_index]),
                    "y_pred": int(probabilities[batch_index] >= best_threshold),
                    "best_threshold": float(best_threshold),
                    "num_layers": int(case_attention.shape[0]),
                    "num_heads": int(case_attention.shape[1]),
                    "num_tokens": int(case_attention.shape[2]),
                    "focus_layer_index": int(focus_layer_index),
                    "focus_layer_top_token": str(token_labels[int(np.argmax(incoming))]),
                    "focus_layer_top_token_score": float(np.max(incoming)),
                }
                for label, value in case_summary["focus_layer_incoming"].items():
                    normalized_label = label.lower() if label == "Clinical" else label
                    row[f"focus_in_{normalized_label}"] = float(value)
                for label, value in case_summary["focus_layer_outgoing"].items():
                    normalized_label = label.lower() if label == "Clinical" else label
                    row[f"focus_out_{normalized_label}"] = float(value)
                case_rows.append(row)

    logits = np.concatenate(all_logits) if all_logits else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.float32)
    probs = sigmoid(logits)
    metrics = compute_binary_metrics(labels, probs, threshold=best_threshold)
    confusion = compute_confusion_matrix(labels, (probs >= best_threshold).astype(int))
    metrics["best_threshold"] = float(best_threshold)
    metrics["selected_threshold"] = float(best_threshold)

    predictions_df = pd.DataFrame(
        {
            "pid": all_pids,
            "y_true": labels.astype(float),
            "y_prob": probs.astype(float),
            "y_pred": (probs >= best_threshold).astype(int),
            "dataset": all_datasets,
            "split_final": all_splits,
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([confusion]).to_csv(output_dir / "confusion_matrix.csv", index=False)
    attention_summary_df = pd.DataFrame(case_rows)
    attention_summary_df.to_csv(output_dir / "attention_summary.csv", index=False)

    modality_summary_df = _build_modality_contribution_table(attention_summary_df)
    modality_summary_df.to_csv(output_dir / "modality_contribution_summary.csv", index=False)
    if not modality_summary_df.empty:
        _render_modality_contribution_chart(
            modality_summary_df,
            output_dir / "modality_contribution.png",
            title="Modality Contribution to Final Prediction",
        )

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "test_metrics.json", metrics)
    save_json(
        output_dir / "attention_config.json",
        {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "split": str(args.split),
            "case_id": args.case_id,
            "use_pred_mask": bool(args.use_pred_mask),
            "layer_index": int(args.layer_index),
            "token_labels": token_labels,
            "model_config": model_config,
        },
    )
    logger.info(
        "attention_evaluation_complete split=%s auroc=%s auprc=%s output_dir=%s",
        args.split,
        metrics["auroc"],
        metrics["auprc"],
        output_dir.as_posix(),
    )


if __name__ == "__main__":
    main()


"""
python /home/serverai/dannguyen/BreastCancerDetection-MRI/pcr_phase2/attention_attribution.py \
  --manifest /home/serverai/dannguyen/BreastCancerDetection-MRI/outputs/phase2_manifest_pred_mask.csv \
  --checkpoint /home/serverai/dannguyen/BreastCancerDetection-MRI/outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_bce_light/checkpoints/best.pt \
  --split internal_test \
  --output-dir /home/serverai/dannguyen/BreastCancerDetection-MRI/outputs_v2/phase2_attention_attribution
"""