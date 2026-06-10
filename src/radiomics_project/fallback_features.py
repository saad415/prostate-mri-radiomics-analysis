from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats


DIRECTIONS_3D = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
]


def load_image_and_mask(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    image = np.asarray(image_nii.get_fdata(), dtype=np.float32)
    mask = np.asarray(mask_nii.get_fdata() > 0, dtype=bool)
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    return image, mask, spacing


def quantize(values: np.ndarray, bins: int = 16) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.uint8)

    low, high = np.percentile(values, [1, 99])
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=np.uint8)

    clipped = np.clip(values, low, high)
    scaled = (clipped - low) / (high - low)
    return np.minimum((scaled * bins).astype(np.uint8), bins - 1)


def first_order_features(values: np.ndarray) -> dict[str, float]:
    percentiles = {f"original_firstorder_percentile_{p:02d}": float(np.percentile(values, p)) for p in range(5, 100, 5)}
    histogram, _ = np.histogram(values, bins=32)
    histogram = histogram / max(histogram.sum(), 1)
    histogram_features = {f"original_firstorder_histogram_bin_{index:02d}": float(value) for index, value in enumerate(histogram)}

    return {
        "original_firstorder_mean": float(values.mean()),
        "original_firstorder_median": float(np.median(values)),
        "original_firstorder_minimum": float(values.min()),
        "original_firstorder_maximum": float(values.max()),
        "original_firstorder_range": float(values.max() - values.min()),
        "original_firstorder_variance": float(values.var()),
        "original_firstorder_standard_deviation": float(values.std()),
        "original_firstorder_skewness": float(stats.skew(values)),
        "original_firstorder_kurtosis": float(stats.kurtosis(values)),
        "original_firstorder_energy": float(np.sum(values**2)),
        "original_firstorder_entropy": float(stats.entropy(histogram + 1e-12)),
        **percentiles,
        **histogram_features,
    }


def shape_features(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    voxel_volume = float(np.prod(spacing))
    volume_voxels = int(mask.sum())
    labeled, component_count = ndimage.label(mask)
    object_slices = ndimage.find_objects(labeled)
    coords = np.argwhere(mask)

    if coords.size == 0:
        return {}

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents_voxels = maxs - mins + 1
    extents_mm = extents_voxels * np.asarray(spacing)
    bbox_volume = float(np.prod(extents_mm))
    filled = ndimage.binary_fill_holes(mask)
    surface = mask ^ ndimage.binary_erosion(mask)
    surface_voxels = int(surface.sum())

    return {
        "original_shape_volume_voxels": float(volume_voxels),
        "original_shape_volume_mm3": float(volume_voxels * voxel_volume),
        "original_shape_surface_voxels": float(surface_voxels),
        "original_shape_component_count": float(component_count),
        "original_shape_bbox_x_mm": float(extents_mm[0]),
        "original_shape_bbox_y_mm": float(extents_mm[1]),
        "original_shape_bbox_z_mm": float(extents_mm[2]),
        "original_shape_bbox_volume_mm3": bbox_volume,
        "original_shape_extent": float((volume_voxels * voxel_volume) / bbox_volume) if bbox_volume else 0.0,
        "original_shape_solidity": float(volume_voxels / max(filled.sum(), 1)),
        "original_shape_centroid_x": float(coords[:, 0].mean()),
        "original_shape_centroid_y": float(coords[:, 1].mean()),
        "original_shape_centroid_z": float(coords[:, 2].mean()),
        "original_shape_largest_component_voxels": float(max(np.bincount(labeled.ravel())[1:], default=0)),
        "original_shape_slice_count": float(len(object_slices)),
    }


def glcm_matrix(volume: np.ndarray, mask: np.ndarray, direction: tuple[int, int, int], bins: int = 16) -> np.ndarray:
    dx, dy, dz = direction
    source_slices = (
        slice(max(0, -dx), volume.shape[0] - max(0, dx)),
        slice(max(0, -dy), volume.shape[1] - max(0, dy)),
        slice(max(0, -dz), volume.shape[2] - max(0, dz)),
    )
    target_slices = (
        slice(max(0, dx), volume.shape[0] - max(0, -dx)),
        slice(max(0, dy), volume.shape[1] - max(0, -dy)),
        slice(max(0, dz), volume.shape[2] - max(0, -dz)),
    )

    valid = mask[source_slices] & mask[target_slices]
    if not valid.any():
        return np.zeros((bins, bins), dtype=np.float64)

    source = volume[source_slices][valid]
    target = volume[target_slices][valid]
    matrix = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(matrix, (source, target), 1)
    matrix += matrix.T
    return matrix / max(matrix.sum(), 1)


def glcm_properties(matrix: np.ndarray) -> dict[str, float]:
    bins = matrix.shape[0]
    i, j = np.indices(matrix.shape)
    contrast = np.sum(matrix * (i - j) ** 2)
    dissimilarity = np.sum(matrix * np.abs(i - j))
    homogeneity = np.sum(matrix / (1.0 + (i - j) ** 2))
    asm = np.sum(matrix**2)
    energy = np.sqrt(asm)
    entropy = -np.sum(matrix * np.log2(matrix + 1e-12))

    row_mean = np.sum(i * matrix)
    col_mean = np.sum(j * matrix)
    row_std = np.sqrt(np.sum(((i - row_mean) ** 2) * matrix))
    col_std = np.sqrt(np.sum(((j - col_mean) ** 2) * matrix))
    correlation = np.sum(((i - row_mean) * (j - col_mean) * matrix) / max(row_std * col_std, 1e-12))

    return {
        "contrast": float(contrast),
        "dissimilarity": float(dissimilarity),
        "homogeneity": float(homogeneity),
        "asm": float(asm),
        "energy": float(energy),
        "entropy": float(entropy),
        "correlation": float(correlation),
    }


def texture_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    quantized_values = quantize(image[mask], bins=16)
    quantized = np.zeros_like(image, dtype=np.uint8)
    quantized[mask] = quantized_values

    features: dict[str, float] = {}
    for index, direction in enumerate(DIRECTIONS_3D):
        matrix = glcm_matrix(quantized, mask, direction, bins=16)
        for name, value in glcm_properties(matrix).items():
            features[f"original_glcm_direction_{index:02d}_{name}"] = value
    return features


def extract_fallback_features(image_path: Path, mask_path: Path) -> dict[str, float]:
    image, mask, spacing = load_image_and_mask(image_path, mask_path)
    if not mask.any():
        raise ValueError(f"Mask is empty: {mask_path}")

    values = image[mask].astype(np.float64)
    return {
        **shape_features(mask, spacing),
        **first_order_features(values),
        **texture_features(image, mask),
    }
