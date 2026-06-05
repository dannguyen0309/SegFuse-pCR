from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcr_phase2.clinical import ClinicalPreprocessor
from pcr_phase2.dataset import BreastDCEPhase2Dataset
from pcr_phase2.model import Phase2MaskGuidedTransformerClassifier
from pcr_phase2.utils import load_checkpoint, safe_mkdir


@dataclass
class HookState:
    outputs: List[torch.Tensor]
    handles: List[Any]


@dataclass
class PhaseCamResult:
    phase_index: int
    raw_cam: np.ndarray
    saliency: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ visualizations for Phase 2 breast MRI models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-class", type=int, default=None, choices=[0, 1])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-pred-mask", action="store_true", help="Use predicted-mask ROI paths instead of GT mask paths.")
    parser.add_argument("--slice-axis", type=str, default="0", help="Slice axis as 0/1/2 or z/y/x.")
    parser.add_argument("--cam-layer", type=str, default="layer3", choices=["layer3", "layer4"])
    parser.add_argument(
        "--render-mode",
        type=str,
        default="overlay",
        choices=["raw", "overlay", "comparison"],
        help="Render raw CAM only, MRI overlay, or side-by-side comparison.",
    )
    parser.add_argument("--cam-percentile-low", type=float, default=1.0)
    parser.add_argument("--cam-percentile-high", type=float, default=99.0)
    parser.add_argument("--per-slice-normalize", action="store_true")
    parser.add_argument(
        "--slice-selection",
        type=str,
        default="cam-peak",
        choices=["mask-largest-area", "cam-peak", "cam-centroid"],
    )
    parser.add_argument("--num-slices", type=int, default=1, help="Number of neighboring slices to export around the selected slice.")
    parser.add_argument(
        "--export-mean-cam-summary",
        action="store_true",
        help="Also export mean-combined CAM summaries as extra artifacts.",
    )
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


def _normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        clean_key = str(key)
        while True:
            for prefix in ("module.", "model.", "net."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break
            else:
                break
        cleaned[clean_key] = value
    return cleaned


def _extract_model_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _normalize_state_dict(value)
    tensor_items = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
    if tensor_items:
        return _normalize_state_dict(tensor_items)
    raise RuntimeError("Checkpoint does not contain a recognizable model state dict.")


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


def _build_model(checkpoint: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> Phase2MaskGuidedTransformerClassifier:
    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    clinical_preprocessor = _load_clinical_preprocessor(checkpoint, checkpoint_path)
    clinical_dim = int(config.get("clinical_dim", len(clinical_preprocessor.feature_columns)))

    pretrained_path = config.get("medicalnet_pretrained_path")
    resolved_pretrained_path = None
    if pretrained_path:
        resolved_pretrained_path = _resolve_path(pretrained_path, checkpoint_path.parent, _resolve_project_root())

    model_kwargs = {
        "clinical_dim": clinical_dim,
        "d_model": int(config.get("d_model", 256)),
        "n_heads": int(config.get("n_heads", 4)),
        "num_layers": int(config.get("num_layers", 1)),
        "dropout": float(config.get("dropout", 0.2)),
        "dim_feedforward": int(config.get("dim_feedforward", 512)),
        "encoder_type": str(config.get("encoder_type", "official_medicalnet_resnet34")),
        "encoder_base_channels": int(config.get("encoder_base_channels", 32)),
        "encoder_out_channels": int(config.get("encoder_out_channels", 128)),
        "shared_encoder": bool(config.get("shared_encoder", True)),
        "use_image": bool(config.get("use_image", True)),
        "use_clinical": bool(config.get("use_clinical", True)),
        "fusion_type": str(config.get("fusion_type", "attention")),
        "use_mask_channel": bool(config.get("use_mask_channel", False)),
        "medicalnet_pretrained_path": resolved_pretrained_path.as_posix() if resolved_pretrained_path is not None else None,
    }

    model = Phase2MaskGuidedTransformerClassifier(**model_kwargs).to(device)
    model_state = _extract_model_state(checkpoint)
    load_result = model.load_state_dict(model_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint loading failed unexpectedly. "
            f"missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}"
        )
    model.eval()
    return model


def _load_manifest_row(manifest: Path, case_id: str) -> Tuple[pd.Series, str]:
    df = pd.read_csv(manifest)
    candidate_columns = ["case_id", "pid", "patient_id", "study_id", "id"]
    matches: List[pd.Series] = []
    for column in candidate_columns:
        if column in df.columns:
            subset = df[df[column].astype(str) == str(case_id)]
            if len(subset) > 0:
                matches = [row for _, row in subset.iterrows()]
                break
    if not matches:
        text_matches = df[df.astype(str).eq(str(case_id)).any(axis=1)]
        matches = [row for _, row in text_matches.iterrows()]
    if not matches:
        raise FileNotFoundError(f"Could not find case_id={case_id} in {manifest.as_posix()}")
    if len(matches) > 1:
        split_values = ", ".join(sorted({str(row.get('split_final')) for row in matches}))
        raise RuntimeError(f"case_id={case_id} matched multiple manifest rows across splits: {split_values}")
    row = matches[0]
    split = str(row.get("split_final", "test"))
    return row, split


def _build_dataset_for_case(
    manifest: Path,
    split: str,
    clinical_preprocessor: ClinicalPreprocessor,
    use_pred_mask: bool,
    config: Dict[str, Any],
) -> BreastDCEPhase2Dataset:
    roi_size = config.get("roi_size", [96, 160, 160])
    mask_mode = "pred" if use_pred_mask else "gt"
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


def _parse_slice_axis(value: str) -> int:
    normalized = str(value).strip().lower()
    mapping = {"0": 0, "1": 1, "2": 2, "z": 0, "y": 1, "x": 2}
    if normalized not in mapping:
        raise ValueError("--slice-axis must be one of 0, 1, 2, z, y, or x")
    return mapping[normalized]


def _resolve_target_layer(model: nn.Module, preferred_layer: str = "layer3") -> Tuple[str, nn.Module]:
    image_encoder = getattr(model, "shared_3d_encoder", None) or getattr(model, "shared_encoder", None)
    if image_encoder is None:
        raise RuntimeError("The loaded model does not expose an image encoder branch.")

    backbone = getattr(image_encoder, "backbone", image_encoder)
    if preferred_layer == "layer4":
        candidate_names = ["layer4", "head", "stage3", "layer3"]
    else:
        candidate_names = ["layer3", "stage3", "layer4", "head"]

    for candidate_name in candidate_names:
        candidate = getattr(backbone, candidate_name, None)
        if isinstance(candidate, nn.Module) and any(isinstance(child, nn.Conv3d) for child in candidate.modules()):
            return candidate_name, candidate

    if any(isinstance(child, nn.Conv3d) for child in backbone.modules()):
        return "backbone", backbone

    raise RuntimeError("Unable to locate a 3D convolutional target layer for Grad-CAM++.")


def _register_hooks(target_layer: nn.Module) -> HookState:
    state = HookState(outputs=[], handles=[])

    def forward_hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not torch.is_tensor(output):
            raise TypeError("Grad-CAM++ target layer must produce a tensor output.")
        output.retain_grad()
        state.outputs.append(output)

    state.handles.append(target_layer.register_forward_hook(forward_hook))
    return state


def _cleanup_hooks(hook_state: HookState) -> None:
    for handle in hook_state.handles:
        handle.remove()
    hook_state.handles.clear()


def _target_score_from_logit(logit: torch.Tensor, target_class: int) -> torch.Tensor:
    if target_class not in (0, 1):
        raise ValueError("target_class must be 0 or 1 for the binary pCR classifier.")
    return logit if int(target_class) == 1 else -logit


def _upsample_cam(cam: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if cam.ndim != 5:
        raise AssertionError(f"CAM must have shape [B, 1, D, H, W], got {tuple(cam.shape)}")
    if tuple(cam.shape[2:]) == tuple(size):
        return cam
    return F.interpolate(cam, size=tuple(int(v) for v in size), mode="trilinear", align_corners=False)


def _compute_gradcam_pp(activation: torch.Tensor, gradient: torch.Tensor, input_size: Sequence[int]) -> torch.Tensor:
    if activation.ndim != 5 or gradient.ndim != 5:
        raise AssertionError("Grad-CAM++ expects [B, C, D, H, W] activations and gradients.")

    # The weighting terms are computed over the 3D spatial dimensions so the encoder output stays volumetric.
    grad_sq = gradient.pow(2)
    grad_cube = gradient.pow(3)
    sum_activations = activation.sum(dim=(2, 3, 4), keepdim=True)

    # Grad-CAM++ uses higher-order gradient weighting to emphasize voxels that contribute most strongly
    # to the selected class score, rather than averaging gradients uniformly across the feature volume.
    alpha_denom = 2.0 * grad_sq + sum_activations * grad_cube
    alpha_denom = torch.where(alpha_denom.abs() > 1e-8, alpha_denom, torch.ones_like(alpha_denom))
    alphas = grad_sq / alpha_denom
    positive_gradients = F.relu(gradient)
    weights = (alphas * positive_gradients).sum(dim=(2, 3, 4), keepdim=True)

    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = _upsample_cam(cam, input_size)
    return cam


def _normalize_volume(volume: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)
    clipped = volume.copy()
    clipped[~finite] = 0.0
    if percentile_high <= percentile_low:
        raise ValueError("cam_percentile_high must be greater than cam_percentile_low")
    lo = float(np.percentile(clipped[finite], percentile_low))
    hi = float(np.percentile(clipped[finite], percentile_high))
    if hi <= lo + 1e-8:
        return np.zeros_like(clipped, dtype=np.float32)
    clipped = np.clip(clipped, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _normalize_slice(slice_2d: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(slice_2d)
    if not finite.any():
        return np.zeros_like(slice_2d, dtype=np.float32)
    clipped = slice_2d.copy()
    clipped[~finite] = 0.0
    if percentile_high <= percentile_low:
        raise ValueError("cam_percentile_high must be greater than cam_percentile_low")
    lo = float(np.percentile(clipped[finite], percentile_low))
    hi = float(np.percentile(clipped[finite], percentile_high))
    if hi <= lo + 1e-8:
        return np.zeros_like(clipped, dtype=np.float32)
    clipped = np.clip(clipped, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _select_slice_index(mask_volume: np.ndarray, axis: int) -> int:
    if mask_volume.ndim != 3:
        raise AssertionError(f"mask_volume must have shape [D, H, W], got {mask_volume.shape}")
    if float(mask_volume.sum()) <= 0.0:
        return int(mask_volume.shape[axis] // 2)
    if axis == 0:
        projection = mask_volume.sum(axis=(1, 2))
    elif axis == 1:
        projection = mask_volume.sum(axis=(0, 2))
    else:
        projection = mask_volume.sum(axis=(0, 1))
    return int(np.argmax(projection))


def _select_slice_index_from_cam(cam_volume: np.ndarray, axis: int, strategy: str) -> int:
    if cam_volume.ndim != 3:
        raise AssertionError(f"cam_volume must have shape [D, H, W], got {cam_volume.shape}")
    if axis == 0:
        projection = cam_volume.sum(axis=(1, 2))
    elif axis == 1:
        projection = cam_volume.sum(axis=(0, 2))
    else:
        projection = cam_volume.sum(axis=(0, 1))

    if strategy == "cam-centroid":
        total = float(projection.sum())
        if total <= 0.0:
            return int(cam_volume.shape[axis] // 2)
        indices = np.arange(projection.shape[0], dtype=np.float32)
        centroid = float(np.sum(indices * projection) / total)
        return int(np.clip(np.rint(centroid), 0, projection.shape[0] - 1))

    if float(projection.sum()) <= 0.0:
        return int(cam_volume.shape[axis] // 2)
    return int(np.argmax(projection))


def _extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if volume.ndim != 3:
        raise AssertionError(f"volume must have shape [D, H, W], got {volume.shape}")
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def _format_slice_label(axis: int, index: int) -> str:
    axis_name = {0: "z", 1: "y", 2: "x"}[axis]
    return f"{axis_name}{index:02d}"


def _build_slice_indices(center_index: int, axis_length: int, num_slices: int) -> List[int]:
    num_slices = max(1, int(num_slices))
    if num_slices == 1:
        return [int(np.clip(center_index, 0, axis_length - 1))]
    left = num_slices // 2
    right = num_slices - left - 1
    start = center_index - left
    end = center_index + right
    indices = [int(np.clip(index, 0, axis_length - 1)) for index in range(start, end + 1)]
    deduped: List[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    return deduped


def _render_raw_cam(cam_slice: np.ndarray, title: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=180)
    ax.imshow(cam_slice, cmap="jet", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.05)
    return fig


def _render_mri_only(image_slice: np.ndarray, title: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=180)
    ax.imshow(image_slice, cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.05)
    return fig


def _render_overlay(image_slice: np.ndarray, cam_slice: np.ndarray, title: Optional[str] = None, overlay_alpha: float = 0.40) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=180)
    ax.imshow(image_slice, cmap="gray", interpolation="nearest")
    ax.imshow(cam_slice, cmap="jet", alpha=overlay_alpha, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.05)
    return fig


def _render_comparison_figure(
    image_slice: np.ndarray,
    cam_slice: np.ndarray,
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=180)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(image_slice, cmap="gray", interpolation="nearest")
    axes[0].set_title("MRI only", fontsize=10)
    axes[1].imshow(cam_slice, cmap="jet", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Raw CAM", fontsize=10)
    axes[2].imshow(image_slice, cmap="gray", interpolation="nearest")
    axes[2].imshow(cam_slice, cmap="jet", alpha=0.40, interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[2].set_title("Overlay", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.as_posix(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _build_output_payload(
    checkpoint_path: Path,
    config: Dict[str, Any],
    case_id: str,
    split: str,
    target_class: int,
    predicted_class: int,
    predicted_probability: float,
    slice_axis: int,
    slice_index: int,
    mask_mode: str,
    hooked_layer_name: str,
    normalization_strategy: Dict[str, Any],
    slice_selection_strategy: str,
    render_mode: str,
    mask_channel_enabled: bool,
) -> Dict[str, Any]:
    return {
        "checkpoint": checkpoint_path.as_posix(),
        "case_id": case_id,
        "split_final": split,
        "target_class": int(target_class),
        "predicted_class": int(predicted_class),
        "predicted_probability": float(predicted_probability),
        "slice_axis": int(slice_axis),
        "slice_index": int(slice_index),
        "mask_mode": mask_mode,
        "hooked_layer_name": hooked_layer_name,
        "normalization_strategy": normalization_strategy,
        "slice_selection_strategy": slice_selection_strategy,
        "render_mode": render_mode,
        "mask_channel_enabled": bool(mask_channel_enabled),
        "encoder_type": str(config.get("encoder_type", "unknown")),
        "fusion_type": str(config.get("fusion_type", "unknown")),
        "use_image": bool(config.get("use_image", True)),
        "use_clinical": bool(config.get("use_clinical", True)),
    }


def _compute_phase_cams(
    hook_outputs: Sequence[torch.Tensor],
    input_size: Sequence[int],
    percentile_low: float,
    percentile_high: float,
) -> List[PhaseCamResult]:
    phase_results: List[PhaseCamResult] = []
    for phase_index, activation in enumerate(hook_outputs):
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError("Grad-CAM++ could not retrieve gradients for the hooked activation.")
        cam_volume = _compute_gradcam_pp(activation.detach(), gradient.detach(), input_size)
        raw_cam = cam_volume[0, 0].detach().cpu().numpy().astype(np.float32)
        saliency = _normalize_volume(raw_cam, percentile_low=percentile_low, percentile_high=percentile_high)
        phase_results.append(PhaseCamResult(phase_index=phase_index, raw_cam=raw_cam, saliency=saliency))
    return phase_results


def _stack_volume_results(results: Sequence[PhaseCamResult], attribute: str, reducer: str = "max") -> np.ndarray:
    arrays = [getattr(result, attribute) for result in results]
    stacked = np.stack(arrays, axis=0)
    if reducer == "mean":
        return stacked.mean(axis=0)
    return stacked.max(axis=0)


def _render_case_slice(render_mode: str, image_slice: np.ndarray, cam_slice: np.ndarray, title: str) -> plt.Figure:
    if render_mode == "raw":
        return _render_raw_cam(cam_slice=cam_slice, title=title)
    if render_mode == "comparison":
        return _render_comparison_figure(image_slice=image_slice, cam_slice=cam_slice, title=title)
    return _render_overlay(image_slice=image_slice, cam_slice=cam_slice, title=title)


def _save_cam_artifacts(
    output_dir: Path,
    case_id: str,
    image_volume: np.ndarray,
    phase_results: Sequence[PhaseCamResult],
    slice_axis: int,
    slice_indices: Sequence[int],
    render_mode: str,
    percentile_low: float,
    percentile_high: float,
    per_slice_normalize: bool,
    export_mean_cam_summary: bool,
) -> Path:
    case_name = str(case_id).replace("/", "_").replace("\\", "_")
    case_dir = safe_mkdir(output_dir / case_name)

    for result in phase_results:
        np.save(case_dir / f"phase{result.phase_index}_heatmap.npy", result.raw_cam)
        np.save(case_dir / f"phase{result.phase_index}_saliency.npy", result.saliency)

    summary_heatmap = _stack_volume_results(phase_results, "raw_cam", reducer="max")
    summary_saliency = _stack_volume_results(phase_results, "saliency", reducer="max")
    np.save(case_dir / "heatmap.npy", summary_heatmap)
    np.save(case_dir / "saliency.npy", summary_saliency)

    if export_mean_cam_summary:
        mean_heatmap = _stack_volume_results(phase_results, "raw_cam", reducer="mean")
        mean_saliency = _stack_volume_results(phase_results, "saliency", reducer="mean")
        np.save(case_dir / "mean_heatmap.npy", mean_heatmap)
        np.save(case_dir / "mean_saliency.npy", mean_saliency)

    for slice_index in slice_indices:
        slice_label = _format_slice_label(slice_axis, slice_index)
        summary_slice = _extract_slice(summary_saliency, slice_axis, slice_index)
        if per_slice_normalize:
            summary_slice = _normalize_slice(summary_slice, percentile_low=percentile_low, percentile_high=percentile_high)

        for result in phase_results:
            image_slice = _extract_slice(image_volume[result.phase_index], slice_axis, slice_index)
            cam_slice = _extract_slice(result.saliency, slice_axis, slice_index)
            if per_slice_normalize:
                cam_slice = _normalize_slice(cam_slice, percentile_low=percentile_low, percentile_high=percentile_high)
            fig = _render_case_slice(
                render_mode=render_mode,
                image_slice=image_slice,
                cam_slice=cam_slice,
                title=f"{case_id} | Phase {result.phase_index} | {slice_label}",
            )
            _save_figure(fig, case_dir / f"phase{result.phase_index}_slice{slice_index:02d}_{render_mode}.png")

        summary_fig = _render_case_slice(
            render_mode=render_mode,
            image_slice=_extract_slice(image_volume[0], slice_axis, slice_index),
            cam_slice=summary_slice,
            title=f"{case_id} | Summary | {slice_label}",
        )
        _save_figure(summary_fig, case_dir / f"summary_slice{slice_index:02d}_{render_mode}.png")

    return case_dir


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = _resolve_path(args.checkpoint, _resolve_project_root())
    manifest_path = _resolve_path(args.manifest, _resolve_project_root())
    output_root = safe_mkdir(args.output_dir)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    clinical_preprocessor = _load_clinical_preprocessor(checkpoint, checkpoint_path)
    model = _build_model(checkpoint, checkpoint_path, device)

    row, split = _load_manifest_row(manifest_path, args.case_id)
    dataset = _build_dataset_for_case(
        manifest=manifest_path,
        split=split,
        clinical_preprocessor=clinical_preprocessor,
        use_pred_mask=bool(args.use_pred_mask),
        config=config,
    )
    sample = _find_case_sample(dataset, args.case_id)

    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    clinical = sample["clinical"].unsqueeze(0).to(device) if bool(config.get("use_clinical", True)) else None

    hooked_layer_name, target_layer = _resolve_target_layer(model, preferred_layer=str(args.cam_layer))
    hooks = _register_hooks(target_layer)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            outputs = model(image, mask if bool(config.get("use_mask_channel", False)) else None, clinical)
            logits = outputs["logits"]
            predicted_probability = float(torch.sigmoid(logits).detach().cpu().numpy()[0])
            predicted_class = int(predicted_probability >= 0.5)
            target_class = predicted_class if args.target_class is None else int(args.target_class)
            score = _target_score_from_logit(logits[0], target_class)
            score.backward()

        if len(hooks.outputs) == 0:
            raise RuntimeError("The Grad-CAM++ hook did not capture any encoder activations.")

        image_volume = sample["image"].detach().cpu().numpy().astype(np.float32)
        mask_volume = sample["mask"].detach().cpu().numpy().astype(np.float32)[0]
        slice_axis = _parse_slice_axis(args.slice_axis)

        phase_results = _compute_phase_cams(
            hook_outputs=hooks.outputs,
            input_size=image.shape[2:],
            percentile_low=float(args.cam_percentile_low),
            percentile_high=float(args.cam_percentile_high),
        )

        if len(phase_results) != image_volume.shape[0]:
            raise RuntimeError(
                f"Expected one CAM per phase, but captured {len(phase_results)} CAMs for image shape {tuple(image_volume.shape)}."
            )

        summary_saliency = _stack_volume_results(phase_results, "saliency", reducer="max")
        if args.slice_selection == "mask-largest-area":
            slice_index = _select_slice_index(mask_volume, slice_axis)
        else:
            slice_index = _select_slice_index_from_cam(summary_saliency, slice_axis, args.slice_selection)

        slice_indices = _build_slice_indices(slice_index, mask_volume.shape[slice_axis], args.num_slices)
        slice_label = _format_slice_label(slice_axis, slice_index)

        case_dir = _save_cam_artifacts(
            output_dir=output_root,
            case_id=str(args.case_id),
            image_volume=image_volume,
            phase_results=phase_results,
            slice_axis=slice_axis,
            slice_indices=slice_indices,
            render_mode=str(args.render_mode),
            percentile_low=float(args.cam_percentile_low),
            percentile_high=float(args.cam_percentile_high),
            per_slice_normalize=bool(args.per_slice_normalize),
            export_mean_cam_summary=bool(args.export_mean_cam_summary),
        )

        payload = _build_output_payload(
            checkpoint_path=checkpoint_path,
            config=config,
            case_id=str(args.case_id),
            split=split,
            target_class=target_class,
            predicted_class=predicted_class,
            predicted_probability=predicted_probability,
            slice_axis=slice_axis,
            slice_index=slice_index,
            mask_mode="pred" if bool(args.use_pred_mask) else "gt",
            hooked_layer_name=hooked_layer_name,
            normalization_strategy={
                "volume_percentile_low": float(args.cam_percentile_low),
                "volume_percentile_high": float(args.cam_percentile_high),
                "per_slice_normalize": bool(args.per_slice_normalize),
            },
            slice_selection_strategy=str(args.slice_selection),
            render_mode=str(args.render_mode),
            mask_channel_enabled=bool(config.get("use_mask_channel", False)),
        )
        (case_dir / "gradcam_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        _cleanup_hooks(hooks)


if __name__ == "__main__":
    main()

# Example usage:
# ./venv/bin/python -m pcr_phase2.gradcam_pp \
#   --manifest outputs/phase2_manifest_gt_mask.csv \
#   --checkpoint outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_balacc_auprc_bs4/checkpoints/best.pt \
#   --case-id ISPY2_XXXX \
#   --output-dir outputs_v2/gradcam_attention \
#   --device cuda
#
# ./venv/bin/python -m pcr_phase2.gradcam_pp \
#   --manifest outputs/phase2_manifest_pred_mask.csv \
#   --checkpoint outputs_v2/phase2_mednet34_gt_roi_clinical_attention_margin16_bs4_balacc_auprc_bs4/checkpoints/best.pt \
#   --case-id ISPY2_XXXX \
#   --output-dir outputs_v2/gradcam_pred_mask \
#   --use-pred-mask \
#   --target-class 1 \
#   --slice-axis z

"""
GT mask:
pid,y_true,y_prob,y_pred,dataset,split_final
ISPY1_1005,0.0,0.0989465855266303,0,spy1,internal_test
ISPY1_1029,0.0,0.5635581154468929,1,spy1,internal_test
ISPY1_1058,1.0,0.6530162847847695,1,spy1,internal_test
ISPY1_1088,1.0,0.13139597557765298,0,spy1,internal_test

Pred Mask:
pid,y_true,y_prob,y_pred,dataset,split_final
ISPY2-749097,0.0,0.13660839002621936,0,spy2,internal_test
ISPY2-413961,0.0,0.7647159477410339,1,spy2,internal_test
ISPY2-766967,1.0,0.3126847318221801,0,spy2,internal_test
ISPY2-378401,1.0,0.4722118905374165,1,spy2,internal_test

"""