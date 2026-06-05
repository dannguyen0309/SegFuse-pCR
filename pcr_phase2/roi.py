from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency path
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

try:  # pragma: no cover - optional dependency path
    from skimage.measure import label as sk_label
except Exception:  # pragma: no cover
    sk_label = None


def _label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    binary = np.asarray(mask > 0, dtype=bool)
    if binary.size == 0:
        return np.zeros_like(binary, dtype=np.int32), 0
    if ndi is not None:
        labeled, num = ndi.label(binary)
        return labeled.astype(np.int32), int(num)
    if sk_label is not None:
        labeled = sk_label(binary, connectivity=1)
        return np.asarray(labeled, dtype=np.int32), int(labeled.max())

    labeled = np.zeros_like(binary, dtype=np.int32)
    component = 0
    depth, height, width = binary.shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if not binary[z, y, x] or labeled[z, y, x] != 0:
                    continue
                component += 1
                stack = [(z, y, x)]
                labeled[z, y, x] = component
                while stack:
                    cz, cy, cx = stack.pop()
                    for nz, ny, nx in (
                        (cz - 1, cy, cx),
                        (cz + 1, cy, cx),
                        (cz, cy - 1, cx),
                        (cz, cy + 1, cx),
                        (cz, cy, cx - 1),
                        (cz, cy, cx + 1),
                    ):
                        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                            if binary[nz, ny, nx] and labeled[nz, ny, nx] == 0:
                                labeled[nz, ny, nx] = component
                                stack.append((nz, ny, nx))
    return labeled, component


def get_mask_bbox(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return (int(z0), int(z1)), (int(y0), int(y1)), (int(x0), int(x1))


def postprocess_mask(mask: np.ndarray, min_component_size: int = 16, keep_largest_component: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
    binary = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    labeled, num_components = _label_connected_components(binary)
    if num_components == 0:
        return binary.astype(np.float32), {"num_components": 0, "largest_component_size": 0, "kept_component_size": 0}

    component_sizes = np.bincount(labeled.ravel(), minlength=num_components + 1)
    component_sizes[0] = 0
    largest_component = int(component_sizes.argmax()) if component_sizes.size > 1 else 0
    largest_size = int(component_sizes[largest_component]) if largest_component > 0 else 0

    if keep_largest_component and largest_component > 0:
        keep_components = {largest_component}
    else:
        keep_components = {int(index) for index, size in enumerate(component_sizes) if index > 0 and size >= int(min_component_size)}
        if not keep_components and largest_component > 0:
            keep_components = {largest_component}

    processed = np.isin(labeled, list(keep_components)).astype(np.float32)
    kept_size = int(processed.sum())
    return processed, {
        "num_components": int(num_components),
        "largest_component_size": largest_size,
        "kept_component_size": kept_size,
    }


def crop_to_bbox_with_margin(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    margin: int | Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(margin, int):
        mz = my = mx = int(margin)
    else:
        mz, my, mx = [int(v) for v in margin]
    (z0, z1), (y0, y1), (x0, x1) = bbox
    z0 = max(0, z0 - mz)
    y0 = max(0, y0 - my)
    x0 = max(0, x0 - mx)
    z1 = min(mask.shape[0], z1 + mz)
    y1 = min(mask.shape[1], y1 + my)
    x1 = min(mask.shape[2], x1 + mx)
    return image[:, z0:z1, y0:y1, x0:x1], mask[z0:z1, y0:y1, x0:x1]


def extract_roi_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    roi_size: Sequence[int],
    margin: int | Sequence[int] = 8,
    min_component_size: int = 16,
    keep_largest_component: bool = True,
    fallback: str = "center",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | bool | str]]:
    processed_mask, mask_stats = postprocess_mask(
        mask,
        min_component_size=min_component_size,
        keep_largest_component=keep_largest_component,
    )
    bbox = get_mask_bbox(processed_mask)
    fallback_used = False
    empty_mask = bbox is None

    if bbox is None:
        fallback_used = True
        if fallback == "whole":
            cropped_image, cropped_mask = image.astype(np.float32), processed_mask.astype(np.float32)
        else:
            cropped_image, cropped_mask = center_crop_3d(image, processed_mask, roi_size)
    else:
        cropped_image, cropped_mask = crop_to_bbox_with_margin(image, processed_mask, bbox, margin)

    cropped_image, cropped_mask = pad_or_resize_3d(cropped_image, cropped_mask, roi_size)
    roi_stats: Dict[str, int | bool | str] = {
        **mask_stats,
        "empty_mask": bool(empty_mask),
        "fallback_used": bool(fallback_used),
        "roi_mode": "center" if fallback_used and fallback != "whole" else ("whole" if fallback_used else "bbox"),
    }
    return cropped_image.astype(np.float32), cropped_mask.astype(np.float32), roi_stats


def center_crop_3d(image: np.ndarray, mask: np.ndarray, roi_size: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    target_d, target_h, target_w = [int(v) for v in roi_size]
    _, depth, height, width = image.shape

    def bounds(size: int, target: int) -> Tuple[int, int]:
        if size <= target:
            return 0, size
        start = (size - target) // 2
        return start, start + target

    z0, z1 = bounds(depth, target_d)
    y0, y1 = bounds(height, target_h)
    x0, x1 = bounds(width, target_w)
    return image[:, z0:z1, y0:y1, x0:x1], mask[z0:z1, y0:y1, x0:x1]


def _resize_numpy_volume(volume: np.ndarray, out_size: Sequence[int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(volume.astype(np.float32))[None, None]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=tuple(out_size), mode="nearest")
    else:
        resized = F.interpolate(tensor, size=tuple(out_size), mode="trilinear", align_corners=False)
    return resized[0, 0].cpu().numpy().astype(np.float32)


def pad_or_resize_3d(image: np.ndarray, mask: np.ndarray, roi_size: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    target_d, target_h, target_w = [int(v) for v in roi_size]
    channels, depth, height, width = image.shape

    if (depth, height, width) == (target_d, target_h, target_w):
        return image.astype(np.float32), mask.astype(np.float32)

    if depth <= target_d and height <= target_h and width <= target_w:
        out_image = np.zeros((channels, target_d, target_h, target_w), dtype=np.float32)
        out_mask = np.zeros((target_d, target_h, target_w), dtype=np.float32)
        z0 = (target_d - depth) // 2
        y0 = (target_h - height) // 2
        x0 = (target_w - width) // 2
        out_image[:, z0:z0 + depth, y0:y0 + height, x0:x0 + width] = image
        out_mask[z0:z0 + depth, y0:y0 + height, x0:x0 + width] = mask
        return out_image, out_mask

    resized_channels = [_resize_numpy_volume(image[c], roi_size, mode="trilinear") for c in range(channels)]
    resized_mask = _resize_numpy_volume(mask, roi_size, mode="nearest")
    resized_mask = (resized_mask > 0.5).astype(np.float32)
    return np.stack(resized_channels, axis=0).astype(np.float32), resized_mask


def roi_pool_feature_map(
    feature_map: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "masked_mean_max_concat",
) -> torch.Tensor:
    if feature_map.ndim != 5 or mask.ndim != 5:
        raise ValueError("feature_map and mask must be 5D tensors [B, C, D, H, W] and [B, 1, D, H, W]")

    if mask.shape[2:] != feature_map.shape[2:]:
        mask = F.interpolate(mask.float(), size=feature_map.shape[2:], mode="nearest")
    mask = (mask > 0.5).float()
    eps = 1e-6

    if mode == "masked_mean":
        masked = feature_map * mask
        denom = mask.sum(dim=(2, 3, 4)).clamp_min(eps)
        return masked.sum(dim=(2, 3, 4)) / denom

    if mode == "masked_max":
        neg_fill = torch.finfo(feature_map.dtype).min
        masked = feature_map.masked_fill(mask == 0, neg_fill)
        pooled = masked.amax(dim=(2, 3, 4))
        empty = mask.sum(dim=(2, 3, 4)) <= 0
        global_max = feature_map.amax(dim=(2, 3, 4))
        pooled = torch.where(empty.expand_as(pooled), global_max, pooled)
        return pooled

    if mode == "masked_mean_max_concat":
        return torch.cat(
            [
                roi_pool_feature_map(feature_map, mask, mode="masked_mean"),
                roi_pool_feature_map(feature_map, mask, mode="masked_max"),
            ],
            dim=1,
        )

    raise ValueError(f"Unsupported ROI pooling mode: {mode}")

