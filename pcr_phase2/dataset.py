from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scipy import ndimage as ndi

from pcr_phase2.clinical import ClinicalPreprocessor
from pcr_phase2.roi import center_crop_3d, extract_roi_from_mask, pad_or_resize_3d
from pcr_phase2.utils import load_nifti


class BreastDCEPhase2Dataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        clinical_num_cols: List[str],
        clinical_cat_cols: List[str],
        clinical_preprocessor: ClinicalPreprocessor,
        roi_size: Sequence[int] = (96, 160, 160),
        mask_mode: str = "gt",
        roi_crop_enable: bool = True,
        roi_margin: int = 8,
        min_component_size: int = 16,
        keep_largest_component: bool = True,
        normalize_mode: str = "zscore",
        target_col: str = "pCR",
        strict_mask: bool = False,
        enable_augmentation: bool = False,
        augmentation_strength: str = "light",
    ) -> None:
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split_final"].astype(str) == str(split)].reset_index(drop=True).copy()
        self.split = split
        self.clinical_num_cols = list(clinical_num_cols)
        self.clinical_cat_cols = list(clinical_cat_cols)
        self.clinical_preprocessor = clinical_preprocessor
        self.roi_size = tuple(int(v) for v in roi_size)
        self.mask_mode = mask_mode
        self.roi_crop_enable = bool(roi_crop_enable)
        self.roi_margin = int(roi_margin)
        self.min_component_size = int(min_component_size)
        self.keep_largest_component = bool(keep_largest_component)
        self.normalize_mode = normalize_mode
        self.target_col = target_col
        self.strict_mask = bool(strict_mask)
        self.enable_augmentation = bool(enable_augmentation)
        self.augmentation_strength = str(augmentation_strength)
        self.augmentation_profile = self._build_augmentation_profile(self.augmentation_strength)
        self.rows: List[Dict[str, Any]] = []
        self.audit_summary = self._audit_and_filter_rows()
        self._label_columns = [self.target_col]

    def _path_from_row(self, row: Dict[str, Any], key: str) -> Path | None:
        value = row.get(key)
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return Path(text)

    def _is_missing(self, path: Path | None) -> bool:
        return path is None or not path.exists()

    def _resolve_phase_path(self, row: Dict[str, Any], phase_name: str, phase_index_key: str) -> Path:
        explicit = self._path_from_row(row, f"{phase_name}_path")
        if explicit is not None and explicit.exists():
            return explicit

        dce_path = self._path_from_row(row, "dce_path")
        if dce_path is None:
            raise FileNotFoundError(f"Missing dce_path for pid={row.get('pid')}")

        phase_index = int(round(float(row.get(phase_index_key, 0))))
        pid = str(row.get("pid"))
        dataset = str(row.get("dataset"))
        sibling = dce_path.parent / f"{pid}_{dataset}_vis1_dce_aqc_{phase_index}.nii.gz"
        if sibling.exists():
            return sibling
        if dce_path.exists():
            return dce_path
        raise FileNotFoundError(f"Could not resolve {phase_name}_path for pid={pid} dataset={dataset}")

    def _resolve_mask_path(self, row: Dict[str, Any]) -> Path | None:
        if self.mask_mode == "none":
            return None
        if self.mask_mode == "pred":
            preferred = self._path_from_row(row, "pred_mask_path")
            fallback = self._path_from_row(row, "mask_path")
            return preferred if preferred is not None else fallback
        preferred = self._path_from_row(row, "gt_mask_path")
        fallback = self._path_from_row(row, "mask_path")
        return preferred if preferred is not None else fallback

    def __len__(self) -> int:
        return len(self.rows)

    def _normalize_phase(self, phase: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase, dtype=np.float32)
        if self.normalize_mode == "percentile":
            lo = np.percentile(phase, 1.0)
            hi = np.percentile(phase, 99.0)
            if hi <= lo:
                hi = lo + 1e-3
            phase = np.clip(phase, lo, hi)
            phase = (phase - lo) / (hi - lo + 1e-6)
            return phase.astype(np.float32)

        mean = float(phase.mean())
        std = float(phase.std())
        if std < 1e-6:
            std = 1.0
        return ((phase - mean) / std).astype(np.float32)

    @staticmethod
    def _build_augmentation_profile(augmentation_strength: str) -> Dict[str, Any]:
        profiles: Dict[str, Dict[str, Any]] = {
            "light": {
                "flip_prob": 0.5,
                "rotation_prob": 0.85,
                "rotation_degrees": 5.0,
                "elastic_prob": 0.08,
                "elastic_alpha": 0.8,
                "elastic_sigma": 10.0,
                "bias_field_prob": 1.0,
                "bias_field_magnitude": 0.08,
                "bias_field_grid": 4,
                "bias_field_smoothing": 12.0,
            },
            "medium": {
                "flip_prob": 0.5,
                "rotation_prob": 0.90,
                "rotation_degrees": 7.0,
                "elastic_prob": 0.10,
                "elastic_alpha": 1.0,
                "elastic_sigma": 11.0,
                "bias_field_prob": 1.0,
                "bias_field_magnitude": 0.12,
                "bias_field_grid": 4,
                "bias_field_smoothing": 14.0,
            },
            "strong": {
                "flip_prob": 0.5,
                "rotation_prob": 0.95,
                "rotation_degrees": 10.0,
                "elastic_prob": 0.12,
                "elastic_alpha": 1.2,
                "elastic_sigma": 12.0,
                "bias_field_prob": 1.0,
                "bias_field_magnitude": 0.16,
                "bias_field_grid": 5,
                "bias_field_smoothing": 16.0,
            },
        }
        if augmentation_strength not in profiles:
            raise ValueError("augmentation_strength must be one of: light, medium, strong")
        return profiles[augmentation_strength]

    def _load_phase_volume(self, row: Dict[str, Any], phase_name: str, phase_index_key: str) -> np.ndarray:
        phase_path = self._resolve_phase_path(row, phase_name, phase_index_key)
        data = np.asarray(load_nifti(phase_path).get_fdata(dtype=np.float32), dtype=np.float32)
        if data.ndim == 3:
            return data
        if data.ndim == 4:
            phase_index = int(round(float(row.get(phase_index_key, 0))))
            phase_index = max(0, min(phase_index, data.shape[-1] - 1))
            return np.asarray(data[..., phase_index], dtype=np.float32)
        raise ValueError(f"Unsupported DCE dimensionality for {phase_path.as_posix()}: {data.shape}")

    def _load_phase_stack(self, row: Dict[str, Any]) -> np.ndarray:
        phases = [
            self._load_phase_volume(row, "pre", "pre"),
            self._load_phase_volume(row, "early", "post_early"),
            self._load_phase_volume(row, "late", "post_late"),
        ]
        return np.stack(phases, axis=0).astype(np.float32)

    @staticmethod
    def _rotation_matrix_xyz(angles_radians: Tuple[float, float, float]) -> np.ndarray:
        angle_x, angle_y, angle_z = angles_radians
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)

        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]],
            dtype=np.float32,
        )
        rot_y = np.array(
            [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
            dtype=np.float32,
        )
        rot_z = np.array(
            [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return rot_z @ rot_y @ rot_x

    def _apply_small_rotation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        angles_degrees: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if ndi is None:
            raise RuntimeError("scipy is required for MRI augmentation but is not available.")

        spatial_shape = np.asarray(image.shape[1:], dtype=np.float32)
        center = (spatial_shape - 1.0) / 2.0
        rotation = self._rotation_matrix_xyz(tuple(np.deg2rad(angles_degrees)))
        matrix = np.linalg.inv(rotation)
        offset = center - matrix @ center

        transformed_channels = [
            ndi.affine_transform(
                image[channel_index],
                matrix=matrix,
                offset=offset,
                order=1,
                mode="nearest",
                cval=0.0,
                prefilter=False,
            )
            for channel_index in range(image.shape[0])
        ]
        transformed_mask = ndi.affine_transform(
            mask,
            matrix=matrix,
            offset=offset,
            order=0,
            mode="nearest",
            cval=0.0,
            prefilter=False,
        )
        return np.stack(transformed_channels, axis=0).astype(np.float32), transformed_mask.astype(np.float32)

    def _apply_very_light_elastic_deformation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if ndi is None:
            raise RuntimeError("scipy is required for MRI augmentation but is not available.")

        profile = self.augmentation_profile
        alpha = float(profile["elastic_alpha"])
        sigma = float(profile["elastic_sigma"])
        spatial_shape = image.shape[1:]

        random_state = np.random.RandomState(np.random.randint(0, 2**31 - 1))
        dz = ndi.gaussian_filter((random_state.rand(*spatial_shape) * 2.0 - 1.0), sigma, mode="reflect") * alpha
        dy = ndi.gaussian_filter((random_state.rand(*spatial_shape) * 2.0 - 1.0), sigma, mode="reflect") * alpha
        dx = ndi.gaussian_filter((random_state.rand(*spatial_shape) * 2.0 - 1.0), sigma, mode="reflect") * alpha

        z, y, x = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]),
            np.arange(spatial_shape[2]),
            indexing="ij",
        )
        indices = (z + dz, y + dy, x + dx)

        elastic_channels = [
            ndi.map_coordinates(
                image[channel_index],
                indices,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for channel_index in range(image.shape[0])
        ]
        elastic_mask = ndi.map_coordinates(mask, indices, order=0, mode="nearest", prefilter=False)
        return np.stack(elastic_channels, axis=0).astype(np.float32), elastic_mask.astype(np.float32)

    def _apply_bias_field_augmentation(self, image: np.ndarray) -> np.ndarray:
        if ndi is None:
            raise RuntimeError("scipy is required for MRI augmentation but is not available.")

        profile = self.augmentation_profile
        magnitude = float(profile["bias_field_magnitude"])
        grid_size = int(profile["bias_field_grid"])
        smoothing = float(profile["bias_field_smoothing"])
        spatial_shape = tuple(int(v) for v in image.shape[1:])

        control_shape = (max(2, grid_size), max(2, grid_size), max(2, grid_size))
        control_points = np.random.normal(0.0, 1.0, size=control_shape).astype(np.float32)
        zoom_factors = tuple(spatial_shape[index] / float(control_shape[index]) for index in range(3))
        bias_field = ndi.zoom(control_points, zoom=zoom_factors, order=1)
        if bias_field.shape != spatial_shape:
            pad_width = [(0, max(0, spatial_shape[index] - bias_field.shape[index])) for index in range(3)]
            bias_field = np.pad(bias_field, pad_width, mode="edge")
            bias_field = bias_field[: spatial_shape[0], : spatial_shape[1], : spatial_shape[2]]
        bias_field = ndi.gaussian_filter(bias_field, sigma=smoothing, mode="nearest")
        bias_field = bias_field - float(bias_field.mean())
        std = float(bias_field.std())
        if std < 1e-6:
            std = 1.0
        bias_field = bias_field / std
        multiplicative_field = np.exp(np.clip(magnitude * bias_field, -0.35, 0.35)).astype(np.float32)
        return (image * multiplicative_field[None, ...]).astype(np.float32)

    def _apply_synchronized_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_augmentation or self.split != "train":
            return image.astype(np.float32), mask.astype(np.float32)

        profile = self.augmentation_profile

        if np.random.random() < float(profile["flip_prob"]):
            image = np.flip(image, axis=3)
            mask = np.flip(mask, axis=2)

        if np.random.random() < float(profile["rotation_prob"]):
            angles = tuple(
                float(np.random.uniform(-float(profile["rotation_degrees"]), float(profile["rotation_degrees"])))
                for _ in range(3)
            )
            image, mask = self._apply_small_rotation(image, mask, angles)

        if np.random.random() < float(profile["elastic_prob"]):
            image, mask = self._apply_very_light_elastic_deformation(image, mask)

        if np.random.random() < float(profile["bias_field_prob"]):
            image = self._apply_bias_field_augmentation(image)

        mask = (mask > 0.5).astype(np.float32)
        return image.astype(np.float32), mask.astype(np.float32)

    def _normalize_phase_stack(self, image: np.ndarray) -> np.ndarray:
        phases = [self._normalize_phase(image[channel_index]) for channel_index in range(image.shape[0])]
        return np.stack(phases, axis=0).astype(np.float32)

    def get_augmentation_summary(self) -> Dict[str, Any]:
        summary = dict(self.augmentation_profile)
        summary.update(
            {
                "enabled": bool(self.enable_augmentation and self.split == "train"),
                "synchronized_phase_augmentation": True,
                "train_split_only": True,
            }
        )
        return summary

    def _load_mask(self, row: Dict[str, Any]) -> np.ndarray:
        if self.mask_mode == "none":
            image_shape = self._load_phase_volume(row, "pre", "pre").shape
            return np.zeros(image_shape, dtype=np.float32)

        mask_path = self._resolve_mask_path(row)
        if mask_path is None or not mask_path.exists():
            if self.strict_mask:
                raise FileNotFoundError(f"Missing mask for pid={row.get('pid')} in split={self.split}")
            image_shape = self._load_phase_volume(row, "pre", "pre").shape
            return np.zeros(image_shape, dtype=np.float32)

        mask = np.asarray(load_nifti(mask_path).get_fdata(dtype=np.float32), dtype=np.float32)
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D for pid={row.get('pid')}, got shape {mask.shape}")
        return (mask > 0).astype(np.float32)

    def _audit_and_filter_rows(self) -> Dict[str, Any]:
        kept_rows: List[Dict[str, Any]] = []
        missing_mri_files = 0
        missing_mask_files = 0
        empty_predicted_masks = 0
        fallback_crops = 0
        labels: List[float] = []

        for _, raw_row in self.df.iterrows():
            row = raw_row.to_dict()
            try:
                phase_paths = [
                    self._resolve_phase_path(row, "pre", "pre"),
                    self._resolve_phase_path(row, "early", "post_early"),
                    self._resolve_phase_path(row, "late", "post_late"),
                ]
            except FileNotFoundError:
                missing_mri_files += 1
                continue

            if any(self._is_missing(path) for path in phase_paths):
                missing_mri_files += 1
                continue

            mask_path = self._resolve_mask_path(row)
            if self.mask_mode != "none" and self._is_missing(mask_path):
                missing_mask_files += 1
                continue

            kept_rows.append(row)
            labels.append(float(row.get(self.target_col, 0.0)))

            if self.mask_mode == "pred" and self.roi_crop_enable and mask_path is not None and mask_path.exists():
                if self._mask_is_empty(row):
                    empty_predicted_masks += 1
                    fallback_crops += 1

        self.rows = kept_rows
        labels_array = np.asarray(labels, dtype=np.float32) if labels else np.zeros((0,), dtype=np.float32)
        label_counts = {
            "0": int((labels_array < 0.5).sum()),
            "1": int((labels_array >= 0.5).sum()),
        }
        if self.mask_mode != "pred" or not self.roi_crop_enable:
            empty_predicted_masks = 0
            fallback_crops = 0

        return {
            "manifest_rows": int(len(self.df)),
            "loaded_cases": int(len(self.rows)),
            "missing_mri_files": int(missing_mri_files),
            "missing_mask_files": int(missing_mask_files),
            "empty_predicted_masks": int(empty_predicted_masks),
            "fallback_crops": int(fallback_crops),
            "class_distribution": label_counts,
            "split": self.split,
            "mask_mode": self.mask_mode,
            "roi_crop_enable": bool(self.roi_crop_enable),
        }

    def _mask_is_empty(self, row: Dict[str, Any]) -> bool:
        mask_path = self._resolve_mask_path(row)
        if mask_path is None or not mask_path.exists():
            return True
        mask = np.asarray(load_nifti(mask_path).get_fdata(dtype=np.float32), dtype=np.float32)
        if mask.ndim != 3:
            return True
        return float((mask > 0).sum()) <= 0.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image = self._load_phase_stack(row)
        mask = self._load_mask(row)

        roi_stats: Dict[str, Any]
        if self.mask_mode == "none":
            if self.roi_crop_enable:
                image, mask = center_crop_3d(image, mask, self.roi_size)
                image, mask = pad_or_resize_3d(image, mask, self.roi_size)
                roi_stats = {"empty_mask": True, "fallback_used": False, "roi_mode": "center_no_mask"}
            else:
                image, mask = pad_or_resize_3d(image, mask, self.roi_size)
                roi_stats = {"empty_mask": True, "fallback_used": False, "roi_mode": "whole_volume"}
        elif self.roi_crop_enable:
            image, mask, roi_stats = extract_roi_from_mask(
                image,
                mask,
                self.roi_size,
                margin=self.roi_margin,
                min_component_size=self.min_component_size,
                keep_largest_component=self.keep_largest_component,
                fallback="center",
            )
        else:
            image, mask = pad_or_resize_3d(image, mask, self.roi_size)
            roi_stats = {"empty_mask": False, "fallback_used": False, "roi_mode": "whole_volume"}

        image, mask = self._apply_synchronized_augmentation(image, mask)
        image = self._normalize_phase_stack(image)

        clinical = self.clinical_preprocessor.transform_row(pd.Series(row))

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "clinical": torch.from_numpy(clinical.astype(np.float32)),
            "label": torch.tensor(float(row.get(self.target_col, 0.0)), dtype=torch.float32),
            "pid": str(row.get("pid")),
            "dataset": str(row.get("dataset")),
            "split_final": str(row.get("split_final")),
            "empty_mask": torch.tensor(bool(roi_stats.get("empty_mask", False)), dtype=torch.bool),
            "fallback_used": torch.tensor(bool(roi_stats.get("fallback_used", False)), dtype=torch.bool),
            "roi_mode": str(roi_stats.get("roi_mode", "unknown")),
        }

