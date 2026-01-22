"""
Helper Utilities for PFT_GEM

This module provides common utility functions used throughout the
PFT_GEM package for image processing, mask creation, and validation.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from scipy import ndimage


def create_spherical_tumor_mask(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius: float,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Create a binary spherical tumor mask.

    Parameters
    ----------
    shape : tuple
        Grid dimensions (nx, ny, nz)
    center : tuple
        Tumor center coordinates in voxels
    radius : float
        Tumor radius in mm
    voxel_size : tuple
        Voxel dimensions in mm

    Returns
    -------
    mask : np.ndarray
        Binary tumor mask (uint8)
    """
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute distance from center in mm
    distance = np.sqrt(
        ((X - center[0]) * voxel_size[0])**2 +
        ((Y - center[1]) * voxel_size[1])**2 +
        ((Z - center[2]) * voxel_size[2])**2
    )

    mask = (distance <= radius).astype(np.uint8)
    return mask


def create_ellipsoidal_tumor_mask(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    semi_axes: Tuple[float, float, float],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_angles: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Create a binary ellipsoidal tumor mask.

    Parameters
    ----------
    shape : tuple
        Grid dimensions (nx, ny, nz)
    center : tuple
        Tumor center coordinates in voxels
    semi_axes : tuple
        Semi-axes lengths (a, b, c) in mm
    voxel_size : tuple
        Voxel dimensions in mm
    rotation_angles : tuple, optional
        Euler rotation angles (rx, ry, rz) in radians

    Returns
    -------
    mask : np.ndarray
        Binary tumor mask (uint8)
    """
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Coordinates relative to center (in mm)
    dx = (X - center[0]) * voxel_size[0]
    dy = (Y - center[1]) * voxel_size[1]
    dz = (Z - center[2]) * voxel_size[2]

    # Apply rotation if specified
    if rotation_angles is not None:
        rx, ry, rz = rotation_angles

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx  # Combined rotation

        # Apply rotation
        coords = np.stack([dx.ravel(), dy.ravel(), dz.ravel()])
        rotated = R @ coords
        dx = rotated[0].reshape(shape)
        dy = rotated[1].reshape(shape)
        dz = rotated[2].reshape(shape)

    # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
    a, b, c = semi_axes
    ellipsoid = (dx/a)**2 + (dy/b)**2 + (dz/c)**2

    mask = (ellipsoid <= 1.0).astype(np.uint8)
    return mask


def compute_distance_transform(
    mask: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    signed: bool = False
) -> np.ndarray:
    """
    Compute distance transform from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    voxel_size : tuple
        Voxel dimensions in mm
    signed : bool
        If True, compute signed distance (negative inside mask)

    Returns
    -------
    distance : np.ndarray
        Distance transform in mm
    """
    # Distance from mask boundary (outside)
    dist_outside = ndimage.distance_transform_edt(~mask.astype(bool), sampling=voxel_size)

    if signed:
        # Distance from boundary (inside)
        dist_inside = ndimage.distance_transform_edt(mask.astype(bool), sampling=voxel_size)
        distance = dist_outside - dist_inside
    else:
        distance = dist_outside

    return distance


def resample_volume(
    volume: np.ndarray,
    current_voxel_size: Tuple[float, float, float],
    target_voxel_size: Tuple[float, float, float],
    order: int = 1
) -> np.ndarray:
    """
    Resample a volume to a different voxel size.

    Parameters
    ----------
    volume : np.ndarray
        Input volume
    current_voxel_size : tuple
        Current voxel dimensions in mm
    target_voxel_size : tuple
        Target voxel dimensions in mm
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns
    -------
    resampled : np.ndarray
        Resampled volume
    """
    # Compute zoom factors
    zoom_factors = tuple(
        current_voxel_size[i] / target_voxel_size[i]
        for i in range(3)
    )

    # Resample
    resampled = ndimage.zoom(volume, zoom_factors, order=order)

    return resampled


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    percentile_range: Tuple[float, float] = (1, 99),
    target_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normalize image intensities.

    Parameters
    ----------
    image : np.ndarray
        Input image
    method : str
        Normalization method: 'minmax', 'zscore', or 'percentile'
    percentile_range : tuple
        Percentiles for clipping (used with 'percentile' method)
    target_range : tuple
        Target intensity range

    Returns
    -------
    normalized : np.ndarray
        Normalized image
    """
    if method == "minmax":
        vmin, vmax = image.min(), image.max()
    elif method == "zscore":
        mean, std = image.mean(), image.std()
        normalized = (image - mean) / (std + 1e-10)
        return normalized
    elif method == "percentile":
        vmin = np.percentile(image, percentile_range[0])
        vmax = np.percentile(image, percentile_range[1])
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Scale to target range
    normalized = (image - vmin) / (vmax - vmin + 1e-10)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    normalized = np.clip(normalized, target_range[0], target_range[1])

    return normalized


def validate_inputs(
    displacement: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    expected_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """
    Validate input arrays for common errors.

    Parameters
    ----------
    displacement : np.ndarray, optional
        Displacement field to validate
    image : np.ndarray, optional
        Image to validate
    mask : np.ndarray, optional
        Mask to validate
    expected_shape : tuple, optional
        Expected spatial shape

    Raises
    ------
    ValueError
        If validation fails
    """
    if displacement is not None:
        if displacement.ndim != 4:
            raise ValueError(
                f"Displacement field must be 4D, got {displacement.ndim}D"
            )
        if displacement.shape[-1] != 3:
            raise ValueError(
                f"Displacement field must have 3 components, got {displacement.shape[-1]}"
            )
        if expected_shape is None:
            expected_shape = displacement.shape[:3]

    if image is not None:
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D, got {image.ndim}D")
        if expected_shape is not None and image.shape != expected_shape:
            raise ValueError(
                f"Image shape {image.shape} does not match expected {expected_shape}"
            )

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3D, got {mask.ndim}D")
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match expected {expected_shape}"
            )


def compute_overlap_metrics(
    mask1: np.ndarray,
    mask2: np.ndarray
) -> dict:
    """
    Compute overlap metrics between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask

    Returns
    -------
    metrics : dict
        Dictionary with Dice, Jaccard, sensitivity, specificity
    """
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)

    dice = 2 * intersection / (np.sum(m1) + np.sum(m2) + 1e-10)
    jaccard = intersection / (union + 1e-10)

    sensitivity = intersection / (np.sum(m2) + 1e-10)  # True positive rate
    specificity = np.sum(~m1 & ~m2) / (np.sum(~m2) + 1e-10)  # True negative rate

    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "volume_m1": int(np.sum(m1)),
        "volume_m2": int(np.sum(m2)),
        "intersection": int(intersection),
        "union": int(union),
    }


def find_tumor_center(
    mask: np.ndarray,
    method: str = "centroid"
) -> Tuple[float, float, float]:
    """
    Find the center of a tumor mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary tumor mask
    method : str
        Method: 'centroid' or 'weighted'

    Returns
    -------
    center : tuple
        (x, y, z) coordinates of tumor center
    """
    coords = np.array(np.where(mask > 0))

    if coords.size == 0:
        # Return center of volume if mask is empty
        return tuple(s // 2 for s in mask.shape)

    if method == "centroid":
        center = coords.mean(axis=1)
    elif method == "weighted":
        # Weight by mask values if not binary
        weights = mask[mask > 0]
        center = np.average(coords, axis=1, weights=weights)
    else:
        raise ValueError(f"Unknown method: {method}")

    return tuple(center)


def estimate_tumor_radius(mask: np.ndarray) -> float:
    """
    Estimate effective radius of tumor from mask volume.

    Parameters
    ----------
    mask : np.ndarray
        Binary tumor mask

    Returns
    -------
    radius : float
        Effective radius in voxels
    """
    volume = np.sum(mask > 0)
    # V = 4/3 * pi * r^3, so r = (3V / 4pi)^(1/3)
    radius = (3 * volume / (4 * np.pi)) ** (1/3)
    return float(radius)


def create_boundary_mask(
    mask: np.ndarray,
    thickness: int = 1,
    mode: str = "inner"
) -> np.ndarray:
    """
    Create a boundary mask from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    thickness : int
        Boundary thickness in voxels
    mode : str
        'inner', 'outer', or 'both'

    Returns
    -------
    boundary : np.ndarray
        Binary boundary mask
    """
    struct = ndimage.generate_binary_structure(3, 1)

    if mode in ["inner", "both"]:
        eroded = ndimage.binary_erosion(mask, struct, iterations=thickness)
        inner_boundary = mask.astype(bool) & ~eroded
    else:
        inner_boundary = np.zeros_like(mask, dtype=bool)

    if mode in ["outer", "both"]:
        dilated = ndimage.binary_dilation(mask, struct, iterations=thickness)
        outer_boundary = dilated & ~mask.astype(bool)
    else:
        outer_boundary = np.zeros_like(mask, dtype=bool)

    if mode == "inner":
        return inner_boundary.astype(np.uint8)
    elif mode == "outer":
        return outer_boundary.astype(np.uint8)
    else:
        return (inner_boundary | outer_boundary).astype(np.uint8)
