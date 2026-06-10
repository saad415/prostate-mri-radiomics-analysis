from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_mask(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata() > 0, dtype=bool)


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    size_sum = mask_a.sum() + mask_b.sum()
    if size_sum == 0:
        return 1.0
    return float((2.0 * intersection) / size_sum)


def volume_voxels(mask: np.ndarray) -> int:
    return int(mask.sum())


def relative_volume_difference(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    volume_a = volume_voxels(mask_a)
    volume_b = volume_voxels(mask_b)
    denominator = (volume_a + volume_b) / 2.0
    if denominator == 0:
        return 0.0
    return float(abs(volume_a - volume_b) / denominator)


def hausdorff_distance_voxels(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if not mask_a.any() or not mask_b.any():
        return float("nan")

    surface_a = mask_a ^ ndimage.binary_erosion(mask_a)
    surface_b = mask_b ^ ndimage.binary_erosion(mask_b)

    distance_to_b = ndimage.distance_transform_edt(~surface_b)
    distance_to_a = ndimage.distance_transform_edt(~surface_a)

    return float(max(distance_to_b[surface_a].max(), distance_to_a[surface_b].max()))
