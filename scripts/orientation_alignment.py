#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_from_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check affine orientation for raw MRI, GT mask, and CAM; reorient them to the same canonical space; "
            "resample GT/CAM to the raw MRI space; and render one synchronized view axis."
        )
    )
    parser.add_argument("--raw", type=Path, required=True, help="Raw MRI NIfTI, usually case_0000.nii.gz")
    parser.add_argument("--gt", type=Path, required=True, help="GT mask NIfTI")
    parser.add_argument(
        "--cam",
        type=Path,
        required=True,
        help="CAM volume as .nii/.nii.gz or .npy; use a volume for exact alignment",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to save aligned panels")
    parser.add_argument(
        "--view-axis",
        type=str,
        default="sagittal",
        choices=["axial", "coronal", "sagittal"],
        help="Single axis to display after all volumes are aligned",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Optional slice index in the aligned reference space; defaults to the GT midpoint",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    return parser.parse_args()


def load_nifti(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return nib.load(path.as_posix())


def load_cam(cam_path: Path, reference_affine: np.ndarray | None = None) -> nib.Nifti1Image:
    if not cam_path.exists():
        raise FileNotFoundError(f"CAM path not found: {cam_path}")

    if cam_path.suffix.lower() == ".npy":
        cam_array = np.load(cam_path.as_posix())
        if cam_array.ndim == 4 and cam_array.shape[0] == 1:
            cam_array = cam_array[0]
        if cam_array.ndim != 3:
            raise ValueError(f"CAM .npy must be 3D or (1, D, H, W), got shape {cam_array.shape}")
        affine = reference_affine if reference_affine is not None else np.eye(4)
        return nib.Nifti1Image(cam_array.astype(np.float32), affine=affine)

    suffixes = "".join(cam_path.suffixes).lower()
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return nib.load(cam_path.as_posix())

    raise ValueError(f"Unsupported CAM format: {cam_path}")


def describe_image(label: str, img: nib.Nifti1Image) -> None:
    print(f"{label}: shape={img.shape}, axcodes={''.join(aff2axcodes(img.affine))}")
    print(f"  affine=\n{np.array2string(img.affine, precision=3, suppress_small=True)}")


def canonicalize(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)


def align_to_reference(moving: nib.Nifti1Image, reference: nib.Nifti1Image, order: int) -> nib.Nifti1Image:
    return resample_from_to(moving, reference, order=order)


def minmax(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def slice_axis(view_axis: str) -> int:
    # In canonical RAS space, the standard anatomical planes map to voxel axes as:
    # sagittal -> x-axis (0), coronal -> y-axis (1), axial -> z-axis (2).
    mapping = {"sagittal": 0, "coronal": 1, "axial": 2}
    if view_axis not in mapping:
        raise ValueError(f"Unknown view_axis={view_axis!r}")
    return mapping[view_axis]


def pick_slice_index(mask: np.ndarray, axis: int, explicit_index: Optional[int]) -> int:
    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= mask.shape[axis]:
            raise IndexError(f"slice-index={explicit_index} is out of bounds for axis {axis} with size {mask.shape[axis]}")
        return explicit_index

    nonzero = np.argwhere(mask > 0)
    if nonzero.size == 0:
        return mask.shape[axis] // 2
    return int(np.median(nonzero[:, axis]))


def extract_view(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = np.stack([minmax(image)] * 3, axis=-1)
    overlay = base.copy()
    mask_bool = mask > 0
    overlay[mask_bool, 0] = (1.0 - alpha) * overlay[mask_bool, 0] + alpha * 1.0
    overlay[mask_bool, 1] = (1.0 - alpha) * overlay[mask_bool, 1]
    overlay[mask_bool, 2] = (1.0 - alpha) * overlay[mask_bool, 2]
    return np.clip(overlay, 0.0, 1.0)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([minmax(image)] * 3, axis=-1)
    heat = minmax(heatmap)
    cmap = plt.get_cmap("magma")
    heat_rgb = cmap(heat)[..., :3]
    blended = (1.0 - alpha) * base + alpha * heat_rgb
    return np.clip(blended, 0.0, 1.0)


def render_aligned_panels(
    raw: np.ndarray,
    gt: np.ndarray,
    cam: np.ndarray,
    case_name: str,
    view_axis: str,
    slice_index: int,
    output_path: Path,
    show: bool,
) -> None:
    axis = slice_axis(view_axis)
    raw_slice = extract_view(raw, axis, slice_index)
    gt_slice = extract_view(gt, axis, slice_index)
    cam_slice = extract_view(cam, axis, slice_index)

    panels = [
        (minmax(raw_slice), "Raw MRI"),
        (overlay_mask(raw_slice, gt_slice), "Raw MRI + GT mask"),
        (overlay_heatmap(raw_slice, cam_slice), "Raw MRI + Grad-CAM++"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis_plot, (panel, title) in zip(axes, panels):
        axis_plot.imshow(panel, origin="lower")
        axis_plot.set_title(title, fontsize=18)
        axis_plot.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    raw_img = load_nifti(args.raw)
    gt_img = load_nifti(args.gt)
    cam_img = load_cam(args.cam, reference_affine=raw_img.affine)

    print("Original orientation:")
    describe_image("RAW", raw_img)
    describe_image("GT", gt_img)
    describe_image("CAM", cam_img)

    raw_axcodes = "".join(aff2axcodes(raw_img.affine))
    gt_axcodes = "".join(aff2axcodes(gt_img.affine))
    print(f"\nAffine check: RAW={raw_axcodes}, GT={gt_axcodes}")
    print("Using canonical RAS plane mapping: sagittal=x, coronal=y, axial=z")
    print("CAM .npy is assigned the RAW affine before canonicalization so it stays in the same physical space.")

    raw_can = canonicalize(raw_img)
    gt_can = canonicalize(gt_img)
    cam_can = canonicalize(cam_img)

    print("\nCanonical orientation:")
    describe_image("RAW_CAN", raw_can)
    describe_image("GT_CAN", gt_can)
    describe_image("CAM_CAN", cam_can)

    gt_aligned = align_to_reference(gt_can, raw_can, order=0)
    cam_aligned = align_to_reference(cam_can, raw_can, order=1)

    raw = raw_can.get_fdata().astype(np.float32)
    gt = gt_aligned.get_fdata().astype(np.float32)
    cam = cam_aligned.get_fdata().astype(np.float32)

    if raw.shape != gt.shape:
        raise ValueError(f"GT shape {gt.shape} does not match RAW shape {raw.shape} after resampling")
    if raw.shape != cam.shape:
        raise ValueError(f"CAM shape {cam.shape} does not match RAW shape {raw.shape} after resampling")

    axis = slice_axis(args.view_axis)
    slice_index = pick_slice_index(gt, axis, args.slice_index)
    case_name = args.raw.name.replace("_0000.nii.gz", ".nii.gz")

    output_path = args.output_dir / f"aligned_{args.view_axis}_slice_{slice_index:04d}.png"
    render_aligned_panels(raw, gt, cam, case_name, args.view_axis, slice_index, output_path, args.show)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


"""
./venv/bin/python scripts/orientation_alignment_demo.py \
  --raw /home/serverai/dannguyen/BreastCancerDetection-MRI/temp/ispy_case_000003/BreastISPY_000003_0000.nii.gz \
  --gt /home/serverai/dannguyen/BreastCancerDetection-MRI/nnUNet_raw/Dataset112_BreastTumorISPY/labelsTr/BreastISPY_000003.nii.gz \
  --cam /home/serverai/dannguyen/BreastCancerDetection-MRI/temp/outputs/ispy_case_000003/ispy_case_000003_cam.npy \
  --output-dir /home/serverai/dannguyen/BreastCancerDetection-MRI/temp/orientation_alignment/ispy_case_000003 \
  --view-axis sagittal
"""