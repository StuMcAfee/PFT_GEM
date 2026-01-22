"""
SUIT Template Loader

This module provides utilities for loading and working with the SUIT
(Spatially Unbiased Infratentorial Template) for cerebellum and brainstem
analysis. SUIT provides high-resolution templates and atlases specifically
designed for the posterior fossa region.

References:
-----------
Diedrichsen, J. (2006). A spatially unbiased atlas template of the human
cerebellum. NeuroImage, 33(1), 127-138.

Diedrichsen, J., et al. (2009). A probabilistic MR atlas of the human
cerebellum. NeuroImage, 46(1), 39-46.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import json

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


@dataclass
class TemplateData:
    """Container for SUIT template data.

    Attributes:
        template: T1-weighted template image
        mask: Brain/cerebellum mask
        atlas: Anatomical parcellation atlas
        labels: Dictionary mapping label IDs to region names
        affine: 4x4 affine transformation matrix
        voxel_size: Voxel dimensions in mm
        metadata: Additional metadata dictionary
    """
    template: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    atlas: Optional[np.ndarray] = None
    labels: Dict[int, str] = field(default_factory=dict)
    affine: Optional[np.ndarray] = None
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SUITTemplateLoader:
    """
    Loader for SUIT (Spatially Unbiased Infratentorial Template) data.

    The SUIT template provides standardized reference data for the
    cerebellum and brainstem, which is essential for posterior fossa
    tumor modeling.

    Parameters
    ----------
    suit_dir : str or Path
        Path to SUIT template directory
    resolution : str
        Template resolution: '1mm' or '2mm'

    Examples
    --------
    >>> loader = SUITTemplateLoader('/path/to/suit')
    >>> template_data = loader.load_template()
    >>> mask = loader.get_cerebellum_mask()
    """

    # SUIT atlas region labels
    SUIT_LABELS = {
        0: "Background",
        1: "Left_I_IV",
        2: "Right_I_IV",
        3: "Left_V",
        4: "Right_V",
        5: "Left_VI",
        6: "Vermis_VI",
        7: "Right_VI",
        8: "Left_Crus_I",
        9: "Vermis_Crus_I",
        10: "Right_Crus_I",
        11: "Left_Crus_II",
        12: "Vermis_Crus_II",
        13: "Right_Crus_II",
        14: "Left_VIIb",
        15: "Vermis_VIIb",
        16: "Right_VIIb",
        17: "Left_VIIIa",
        18: "Vermis_VIIIa",
        19: "Right_VIIIa",
        20: "Left_VIIIb",
        21: "Vermis_VIIIb",
        22: "Right_VIIIb",
        23: "Left_IX",
        24: "Vermis_IX",
        25: "Right_IX",
        26: "Left_X",
        27: "Vermis_X",
        28: "Right_X",
        29: "Left_Dentate",
        30: "Right_Dentate",
        31: "Left_Interposed",
        32: "Right_Interposed",
        33: "Left_Fastigial",
        34: "Right_Fastigial",
    }

    # Expected file names in SUIT directory
    TEMPLATE_FILES = {
        "template_1mm": "SUIT_T1_1mm.nii",
        "template_2mm": "SUIT_T1_2mm.nii",
        "mask_1mm": "SUIT_mask_1mm.nii",
        "mask_2mm": "SUIT_mask_2mm.nii",
        "atlas_1mm": "SUIT_atlas_1mm.nii",
        "atlas_2mm": "SUIT_atlas_2mm.nii",
    }

    def __init__(
        self,
        suit_dir: Optional[Union[str, Path]] = None,
        resolution: str = "1mm"
    ):
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required for template loading. "
                "Install with: pip install nibabel"
            )

        self.suit_dir = Path(suit_dir) if suit_dir else None
        self.resolution = resolution
        self._template_data: Optional[TemplateData] = None

    def _get_file_path(self, file_key: str) -> Optional[Path]:
        """Get path to a SUIT file if it exists."""
        if self.suit_dir is None:
            return None

        key = f"{file_key}_{self.resolution}"
        if key in self.TEMPLATE_FILES:
            path = self.suit_dir / self.TEMPLATE_FILES[key]
            if path.exists():
                return path

        # Try with .gz extension
        key_gz = key + ".gz"
        path_gz = self.suit_dir / (self.TEMPLATE_FILES.get(key, "") + ".gz")
        if path_gz.exists():
            return path_gz

        return None

    def is_available(self) -> bool:
        """Check if SUIT template data is available."""
        if self.suit_dir is None:
            return False
        return self.suit_dir.exists() and any(self.suit_dir.iterdir())

    def load_template(self) -> TemplateData:
        """
        Load all available SUIT template data.

        Returns
        -------
        template_data : TemplateData
            Container with loaded template, mask, atlas, etc.
        """
        if self._template_data is not None:
            return self._template_data

        data = TemplateData(labels=self.SUIT_LABELS.copy())

        # Try to load template
        template_path = self._get_file_path("template")
        if template_path is not None:
            img = nib.load(str(template_path))
            data.template = img.get_fdata()
            data.affine = img.affine.copy()
            data.voxel_size = tuple(img.header.get_zooms()[:3])

        # Try to load mask
        mask_path = self._get_file_path("mask")
        if mask_path is not None:
            data.mask = nib.load(str(mask_path)).get_fdata()

        # Try to load atlas
        atlas_path = self._get_file_path("atlas")
        if atlas_path is not None:
            data.atlas = nib.load(str(atlas_path)).get_fdata().astype(np.int32)

        # Add metadata
        data.metadata = {
            "source": "SUIT",
            "resolution": self.resolution,
            "reference": "Diedrichsen (2006) NeuroImage",
        }

        self._template_data = data
        return data

    def get_cerebellum_mask(self) -> Optional[np.ndarray]:
        """
        Get binary mask of the cerebellum.

        Returns
        -------
        mask : np.ndarray
            Binary cerebellum mask
        """
        data = self.load_template()
        return data.mask

    def get_region_mask(self, region_ids: List[int]) -> Optional[np.ndarray]:
        """
        Get mask for specific atlas regions.

        Parameters
        ----------
        region_ids : list
            List of region IDs from SUIT_LABELS

        Returns
        -------
        mask : np.ndarray
            Binary mask for specified regions
        """
        data = self.load_template()
        if data.atlas is None:
            return None

        mask = np.zeros(data.atlas.shape, dtype=np.uint8)
        for rid in region_ids:
            mask[data.atlas == rid] = 1

        return mask

    def get_left_hemisphere_mask(self) -> Optional[np.ndarray]:
        """Get mask for left cerebellar hemisphere."""
        left_regions = [1, 3, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31, 33]
        return self.get_region_mask(left_regions)

    def get_right_hemisphere_mask(self) -> Optional[np.ndarray]:
        """Get mask for right cerebellar hemisphere."""
        right_regions = [2, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30, 32, 34]
        return self.get_region_mask(right_regions)

    def get_vermis_mask(self) -> Optional[np.ndarray]:
        """Get mask for cerebellar vermis."""
        vermis_regions = [6, 9, 12, 15, 18, 21, 24, 27]
        return self.get_region_mask(vermis_regions)

    def get_deep_nuclei_mask(self) -> Optional[np.ndarray]:
        """Get mask for deep cerebellar nuclei."""
        nuclei_regions = [29, 30, 31, 32, 33, 34]
        return self.get_region_mask(nuclei_regions)

    def get_region_centroids(self) -> Dict[int, Tuple[float, float, float]]:
        """
        Compute centroid coordinates for each atlas region.

        Returns
        -------
        centroids : dict
            Dictionary mapping region IDs to (x, y, z) centroids in mm
        """
        data = self.load_template()
        if data.atlas is None or data.affine is None:
            return {}

        centroids = {}
        for rid in np.unique(data.atlas):
            if rid == 0:  # Skip background
                continue

            mask = data.atlas == rid
            coords = np.array(np.where(mask)).T

            if len(coords) == 0:
                continue

            # Compute centroid in voxel space
            centroid_vox = coords.mean(axis=0)

            # Convert to world coordinates
            centroid_world = data.affine @ np.append(centroid_vox, 1)
            centroids[int(rid)] = tuple(centroid_world[:3])

        return centroids

    def get_shape(self) -> Optional[Tuple[int, int, int]]:
        """Get template spatial dimensions."""
        data = self.load_template()
        if data.template is not None:
            return data.template.shape
        elif data.mask is not None:
            return data.mask.shape
        return None

    @staticmethod
    def create_synthetic_template(
        shape: Tuple[int, int, int] = (128, 128, 64),
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> TemplateData:
        """
        Create a synthetic template for testing when SUIT data is unavailable.

        Parameters
        ----------
        shape : tuple
            Template dimensions
        voxel_size : tuple
            Voxel size in mm

        Returns
        -------
        template_data : TemplateData
            Synthetic template data
        """
        # Create coordinate grids
        x = np.arange(shape[0]) - shape[0] // 2
        y = np.arange(shape[1]) - shape[1] // 2
        z = np.arange(shape[2]) - shape[2] // 2
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Create ellipsoidal "cerebellum" shape
        # Semi-axes for cerebellum-like shape
        a, b, c = shape[0] * 0.35, shape[1] * 0.4, shape[2] * 0.3

        ellipsoid = (X/a)**2 + (Y/b)**2 + ((Z + shape[2]*0.2)/c)**2

        # Create mask
        mask = (ellipsoid < 1.0).astype(np.uint8)

        # Create synthetic T1 template (higher intensity inside)
        template = np.zeros(shape, dtype=np.float32)
        template[mask > 0] = 0.8 + 0.2 * np.random.rand(np.sum(mask))

        # Add some structure
        template = template * (1 - 0.3 * ellipsoid)
        template = np.clip(template, 0, 1)

        # Create simple atlas (left/right/vermis)
        atlas = np.zeros(shape, dtype=np.int32)
        atlas[(mask > 0) & (X < -shape[0]*0.05)] = 1  # Left
        atlas[(mask > 0) & (X > shape[0]*0.05)] = 2   # Right
        atlas[(mask > 0) & (np.abs(X) <= shape[0]*0.05)] = 3  # Vermis

        # Create affine matrix
        affine = np.diag([*voxel_size, 1.0])
        affine[:3, 3] = [-shape[i] * voxel_size[i] / 2 for i in range(3)]

        return TemplateData(
            template=template,
            mask=mask,
            atlas=atlas,
            labels={0: "Background", 1: "Left", 2: "Right", 3: "Vermis"},
            affine=affine,
            voxel_size=voxel_size,
            metadata={
                "source": "synthetic",
                "resolution": f"{voxel_size[0]}mm",
            }
        )


def download_suit_template(
    output_dir: Union[str, Path],
    version: str = "3.4"
) -> Path:
    """
    Provide instructions for downloading SUIT template.

    Note: Due to licensing, SUIT templates cannot be automatically
    downloaded. This function provides instructions for manual download.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save template files
    version : str
        SUIT version to download

    Returns
    -------
    output_dir : Path
        Path where templates should be saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions = f"""
    SUIT Template Download Instructions
    ====================================

    The SUIT (Spatially Unbiased Infratentorial Template) can be downloaded from:
    http://www.diedrichsenlab.org/imaging/suit.htm

    Version: {version}

    Required files:
    - SUIT_T1_1mm.nii (or .nii.gz)
    - SUIT_mask_1mm.nii (or .nii.gz)
    - SUIT_atlas_1mm.nii (or .nii.gz)

    Please download and place these files in:
    {output_dir}

    For SPM users, SUIT is also available as an SPM toolbox from:
    http://www.diedrichsenlab.org/imaging/suit_download.htm

    Citation:
    ---------
    Diedrichsen, J. (2006). A spatially unbiased atlas template of the
    human cerebellum. NeuroImage, 33(1), 127-138.
    """

    # Save instructions
    instructions_path = output_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print(instructions)
    return output_dir
