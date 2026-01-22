"""
NIfTI File Handler for Displacement Fields

This module provides utilities for loading and saving neuroimaging data
in NIfTI format, with special handling for displacement vector fields.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
import json

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class NiftiHandler:
    """
    Handler for NIfTI file operations.

    Provides methods for loading and saving brain images and
    displacement fields in NIfTI format, preserving spatial
    orientation and affine transformations.

    Parameters
    ----------
    reference_path : str or Path, optional
        Path to a reference NIfTI file to use for spatial information

    Examples
    --------
    >>> handler = NiftiHandler('reference.nii.gz')
    >>> data = handler.load('brain.nii.gz')
    >>> handler.save(processed_data, 'output.nii.gz')
    """

    def __init__(self, reference_path: Optional[Union[str, Path]] = None):
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required for NIfTI file operations. "
                "Install with: pip install nibabel"
            )

        self.reference_path = Path(reference_path) if reference_path else None
        self._reference_img: Optional[nib.Nifti1Image] = None
        self._affine: Optional[np.ndarray] = None
        self._header: Optional[nib.Nifti1Header] = None

        if self.reference_path is not None:
            self._load_reference()

    def _load_reference(self) -> None:
        """Load reference image for spatial information."""
        if self.reference_path is None or not self.reference_path.exists():
            return

        self._reference_img = nib.load(str(self.reference_path))
        self._affine = self._reference_img.affine.copy()
        self._header = self._reference_img.header.copy()

    @property
    def affine(self) -> Optional[np.ndarray]:
        """Return the affine transformation matrix."""
        return self._affine

    @property
    def voxel_size(self) -> Optional[Tuple[float, float, float]]:
        """Return voxel dimensions in mm."""
        if self._header is None:
            return None
        return tuple(self._header.get_zooms()[:3])

    @property
    def shape(self) -> Optional[Tuple[int, int, int]]:
        """Return image dimensions."""
        if self._reference_img is None:
            return None
        return self._reference_img.shape[:3]

    def load(
        self,
        path: Union[str, Path],
        dtype: Optional[np.dtype] = None
    ) -> np.ndarray:
        """
        Load a NIfTI file and return the data array.

        Parameters
        ----------
        path : str or Path
            Path to NIfTI file
        dtype : np.dtype, optional
            Data type for output array

        Returns
        -------
        data : np.ndarray
            Image data array
        """
        img = nib.load(str(path))
        data = img.get_fdata()

        if dtype is not None:
            data = data.astype(dtype)

        # Update reference if not set
        if self._affine is None:
            self._affine = img.affine.copy()
            self._header = img.header.copy()

        return data

    def load_with_metadata(
        self,
        path: Union[str, Path]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load NIfTI file with full metadata.

        Parameters
        ----------
        path : str or Path
            Path to NIfTI file

        Returns
        -------
        data : np.ndarray
            Image data
        metadata : dict
            Dictionary containing affine, header info, etc.
        """
        img = nib.load(str(path))
        data = img.get_fdata()

        metadata = {
            "affine": img.affine.copy(),
            "shape": img.shape,
            "voxel_size": tuple(img.header.get_zooms()[:3]),
            "dtype": str(img.get_data_dtype()),
            "sform_code": int(img.header['sform_code']),
            "qform_code": int(img.header['qform_code']),
        }

        return data, metadata

    def save(
        self,
        data: np.ndarray,
        path: Union[str, Path],
        affine: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None
    ) -> None:
        """
        Save data to a NIfTI file.

        Parameters
        ----------
        data : np.ndarray
            Data array to save
        path : str or Path
            Output path
        affine : np.ndarray, optional
            4x4 affine matrix. Uses reference if not provided.
        dtype : np.dtype, optional
            Data type for saved file
        """
        if affine is None:
            affine = self._affine if self._affine is not None else np.eye(4)

        if dtype is not None:
            data = data.astype(dtype)

        img = nib.Nifti1Image(data, affine)

        # Set some sensible defaults
        img.header.set_xyzt_units('mm', 'sec')

        nib.save(img, str(path))

    def get_origin(self) -> Tuple[float, float, float]:
        """Get the origin coordinates from the affine matrix."""
        if self._affine is None:
            return (0.0, 0.0, 0.0)
        return tuple(self._affine[:3, 3])

    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel coordinates to world coordinates.

        Parameters
        ----------
        voxel_coords : np.ndarray
            Nx3 array of voxel coordinates

        Returns
        -------
        world_coords : np.ndarray
            Nx3 array of world coordinates in mm
        """
        if self._affine is None:
            raise ValueError("No affine matrix available")

        if voxel_coords.ndim == 1:
            voxel_coords = voxel_coords.reshape(1, -1)

        # Add homogeneous coordinate
        ones = np.ones((voxel_coords.shape[0], 1))
        voxel_hom = np.hstack([voxel_coords, ones])

        world_hom = voxel_hom @ self._affine.T
        return world_hom[:, :3]

    def world_to_voxel(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to voxel coordinates.

        Parameters
        ----------
        world_coords : np.ndarray
            Nx3 array of world coordinates in mm

        Returns
        -------
        voxel_coords : np.ndarray
            Nx3 array of voxel coordinates
        """
        if self._affine is None:
            raise ValueError("No affine matrix available")

        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)

        # Add homogeneous coordinate
        ones = np.ones((world_coords.shape[0], 1))
        world_hom = np.hstack([world_coords, ones])

        affine_inv = np.linalg.inv(self._affine)
        voxel_hom = world_hom @ affine_inv.T
        return voxel_hom[:, :3]


def save_displacement_field(
    displacement: np.ndarray,
    path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    split_components: bool = False
) -> None:
    """
    Save a displacement field to NIfTI format.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3)
    path : str or Path
        Output path
    affine : np.ndarray, optional
        4x4 affine matrix
    voxel_size : tuple
        Voxel dimensions in mm
    split_components : bool
        If True, save x, y, z components as separate files
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for saving NIfTI files")

    if affine is None:
        affine = np.diag([*voxel_size, 1.0])

    path = Path(path)

    if split_components:
        # Save each component separately
        base = path.stem.replace('.nii', '')
        suffix = '.nii.gz' if path.suffix == '.gz' else '.nii'

        for i, comp in enumerate(['x', 'y', 'z']):
            comp_path = path.parent / f"{base}_{comp}{suffix}"
            img = nib.Nifti1Image(displacement[..., i], affine)
            nib.save(img, str(comp_path))
    else:
        # Save as 4D volume
        img = nib.Nifti1Image(displacement, affine)
        img.header.set_xyzt_units('mm', 'sec')
        nib.save(img, str(path))


def load_displacement_field(
    path: Union[str, Path],
    components: Optional[Tuple[str, str, str]] = None
) -> np.ndarray:
    """
    Load a displacement field from NIfTI format.

    Parameters
    ----------
    path : str or Path
        Path to displacement field (4D) or base path for components
    components : tuple, optional
        Paths to x, y, z component files if stored separately

    Returns
    -------
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3)
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for loading NIfTI files")

    if components is not None:
        # Load from separate component files
        dx = nib.load(str(components[0])).get_fdata()
        dy = nib.load(str(components[1])).get_fdata()
        dz = nib.load(str(components[2])).get_fdata()
        return np.stack([dx, dy, dz], axis=-1)
    else:
        # Load from single 4D file
        img = nib.load(str(path))
        data = img.get_fdata()

        if data.ndim == 3:
            raise ValueError(
                "Expected 4D displacement field. "
                "Use 'components' parameter for separate files."
            )

        return data


def create_identity_warp(
    shape: Tuple[int, int, int],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Create an identity warp field (zero displacement).

    Parameters
    ----------
    shape : tuple
        Grid dimensions
    voxel_size : tuple
        Voxel dimensions in mm
    output_path : str or Path, optional
        If provided, save to this path

    Returns
    -------
    warp : np.ndarray
        4D identity warp field
    """
    warp = np.zeros((*shape, 3), dtype=np.float32)

    if output_path is not None:
        save_displacement_field(warp, output_path, voxel_size=voxel_size)

    return warp


def combine_warps(
    warp1: np.ndarray,
    warp2: np.ndarray,
    method: str = "compose"
) -> np.ndarray:
    """
    Combine two warp fields.

    Parameters
    ----------
    warp1 : np.ndarray
        First warp field
    warp2 : np.ndarray
        Second warp field
    method : str
        'compose' for composition, 'add' for simple addition

    Returns
    -------
    combined : np.ndarray
        Combined warp field
    """
    if method == "add":
        return warp1 + warp2
    elif method == "compose":
        from ..core.displacement import DisplacementField
        df1 = DisplacementField(warp1)
        df2 = DisplacementField(warp2)
        return df1.compose(df2).data
    else:
        raise ValueError(f"Unknown method: {method}")
