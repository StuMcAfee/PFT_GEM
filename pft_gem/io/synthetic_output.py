"""
Synthetic MRI Output Generation

This module provides utilities for generating synthetic MRI data by applying
tumor-induced displacement fields to template images (e.g., SUIT). The output
includes displaced template images and displacement field transforms that can
be used with standard neuroimaging tools.

The displacement field can be saved in formats compatible with:
- FSL (warp fields)
- ANTs (displacement fields)
- ITK-based tools
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import json

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


@dataclass
class SyntheticMRIOutput:
    """Container for synthetic MRI output data.

    Attributes:
        displaced_image: The warped template image showing tumor effect
        original_image: The original template image
        displacement_field: 4D displacement vectors (nx, ny, nz, 3) in mm
        tumor_mask: Binary mask of the tumor region
        affine: 4x4 affine transformation matrix
        voxel_size: Voxel dimensions in mm
        metadata: Additional metadata dictionary
    """
    displaced_image: np.ndarray
    original_image: np.ndarray
    displacement_field: np.ndarray
    tumor_mask: np.ndarray
    affine: np.ndarray
    voxel_size: Tuple[float, float, float]
    metadata: Dict[str, Any]


def generate_displaced_template(
    template: np.ndarray,
    displacement: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
    tumor_mask: Optional[np.ndarray] = None,
    tumor_intensity: float = 0.6
) -> np.ndarray:
    """
    Apply displacement field to warp a template image.

    This function creates a synthetic MRI showing the effect of tumor-induced
    tissue displacement by warping the input template according to the
    displacement field.

    Parameters
    ----------
    template : np.ndarray
        3D template image to warp
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3) in mm
    voxel_size : tuple
        Voxel dimensions in mm
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)
    tumor_mask : np.ndarray, optional
        Binary tumor mask to add tumor appearance
    tumor_intensity : float
        Intensity value for tumor region (0-1)

    Returns
    -------
    displaced : np.ndarray
        Warped template image
    """
    from scipy import ndimage

    if template.shape != displacement.shape[:3]:
        raise ValueError(
            f"Template shape {template.shape} does not match "
            f"displacement shape {displacement.shape[:3]}"
        )

    # Create sampling coordinates
    coords = np.indices(template.shape).astype(np.float64)

    # Add displacement (convert from mm to voxels)
    for i in range(3):
        coords[i] += displacement[..., i] / voxel_size[i]

    # Warp image using map_coordinates
    displaced = ndimage.map_coordinates(
        template, coords, order=order, mode='constant', cval=0
    )

    # Add tumor appearance if mask provided
    if tumor_mask is not None:
        # Create tumor with slightly different intensity and texture
        tumor_region = tumor_mask > 0
        # Use the tumor_intensity as base with some variation
        noise = np.random.normal(0, 0.05, displaced.shape)
        displaced[tumor_region] = tumor_intensity + noise[tumor_region]
        displaced = np.clip(displaced, 0, 1)

    return displaced.astype(np.float32)


def save_displacement_as_warp(
    displacement: np.ndarray,
    output_path: Union[str, Path],
    affine: np.ndarray,
    format: str = "nifti",
    header_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save displacement field as a warp/transform file.

    The output format is compatible with standard neuroimaging tools.
    For FSL compatibility, the displacement is saved as a 4D NIfTI file.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3) in mm
    output_path : str or Path
        Output file path
    affine : np.ndarray
        4x4 affine transformation matrix
    format : str
        Output format: 'nifti' (default), 'fsl', or 'ants'
    header_info : dict, optional
        Additional header information to include
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for saving NIfTI files")

    output_path = Path(output_path)

    if format in ("nifti", "fsl"):
        # Standard NIfTI 4D warp field
        # FSL convention: displacement in mm
        img = nib.Nifti1Image(displacement.astype(np.float32), affine)
        img.header.set_xyzt_units('mm', 'sec')
        img.header['intent_code'] = 1006  # NIFTI_INTENT_DISPVECT
        img.header['intent_name'] = b'DISPVECT'
        nib.save(img, str(output_path))

    elif format == "ants":
        # ANTs convention: also saves as 4D, but may need different intent
        img = nib.Nifti1Image(displacement.astype(np.float32), affine)
        img.header.set_xyzt_units('mm', 'sec')
        img.header['intent_code'] = 1006
        nib.save(img, str(output_path))

    else:
        raise ValueError(f"Unknown format: {format}")


def save_displacement_as_coordinate_map(
    displacement: np.ndarray,
    output_path: Union[str, Path],
    affine: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Save displacement as an absolute coordinate map.

    Instead of displacement vectors, this saves the absolute coordinates
    that each voxel maps to. This format is used by some tools for
    direct coordinate lookup.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3) in mm
    output_path : str or Path
        Output file path
    affine : np.ndarray
        4x4 affine transformation matrix
    voxel_size : tuple
        Voxel dimensions in mm
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for saving NIfTI files")

    shape = displacement.shape[:3]

    # Create coordinate grids in world space
    coords = np.zeros_like(displacement)
    for i in range(3):
        idx = np.arange(shape[i])
        if i == 0:
            coords[..., 0] = idx[:, np.newaxis, np.newaxis] * voxel_size[0]
        elif i == 1:
            coords[..., 1] = idx[np.newaxis, :, np.newaxis] * voxel_size[1]
        else:
            coords[..., 2] = idx[np.newaxis, np.newaxis, :] * voxel_size[2]

    # Add origin from affine
    coords[..., 0] += affine[0, 3]
    coords[..., 1] += affine[1, 3]
    coords[..., 2] += affine[2, 3]

    # Add displacement to get final coordinates
    coord_map = coords + displacement

    img = nib.Nifti1Image(coord_map.astype(np.float32), affine)
    img.header.set_xyzt_units('mm')
    img.header['intent_code'] = 1007  # NIFTI_INTENT_POINTSET
    nib.save(img, str(output_path))


def apply_warp_to_image(
    image_path: Union[str, Path],
    warp_path: Union[str, Path],
    output_path: Union[str, Path],
    order: int = 1
) -> np.ndarray:
    """
    Apply a saved warp field to transform an image.

    This function loads a warp field (displacement) and applies it to
    transform an input image, saving the result.

    Parameters
    ----------
    image_path : str or Path
        Path to input image
    warp_path : str or Path
        Path to warp field (4D displacement NIfTI)
    output_path : str or Path
        Path for output warped image
    order : int
        Interpolation order

    Returns
    -------
    warped : np.ndarray
        Warped image data
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for NIfTI operations")

    from scipy import ndimage

    # Load image and warp
    img = nib.load(str(image_path))
    warp_img = nib.load(str(warp_path))

    image_data = img.get_fdata()
    displacement = warp_img.get_fdata()

    # Get voxel size
    voxel_size = img.header.get_zooms()[:3]

    # Create sampling coordinates
    coords = np.indices(image_data.shape).astype(np.float64)

    # Add displacement (convert from mm to voxels)
    for i in range(3):
        coords[i] += displacement[..., i] / voxel_size[i]

    # Warp
    warped = ndimage.map_coordinates(
        image_data, coords, order=order, mode='constant', cval=0
    )

    # Save
    warped_img = nib.Nifti1Image(warped.astype(np.float32), img.affine)
    nib.save(warped_img, str(output_path))

    return warped


def generate_synthetic_mri_output(
    template_data,  # TemplateData from template_loader
    tumor_center: Tuple[float, float, float],
    tumor_radius: float,
    tumor_expansion: float = 5.0,
    decay_exponent: float = 2.0,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "synthetic"
) -> SyntheticMRIOutput:
    """
    Generate complete synthetic MRI output with tumor displacement.

    This is a convenience function that:
    1. Creates a geometric expansion model for the tumor
    2. Computes the displacement field
    3. Applies the displacement to warp the template
    4. Optionally saves all outputs to files

    Parameters
    ----------
    template_data : TemplateData
        Loaded SUIT template data
    tumor_center : tuple
        (x, y, z) tumor center in voxel coordinates
    tumor_radius : float
        Tumor radius in mm
    tumor_expansion : float
        Amount of expansion in mm
    decay_exponent : float
        Displacement decay exponent (1-3)
    output_dir : str or Path, optional
        Directory to save output files
    prefix : str
        Prefix for output filenames

    Returns
    -------
    output : SyntheticMRIOutput
        Container with all generated data
    """
    from ..core.geometric_model import GeometricExpansionModel, TumorParameters, ModelParameters
    from ..utils.helpers import create_spherical_tumor_mask

    # Get template properties
    template = template_data.template
    mask = template_data.mask
    affine = template_data.affine
    voxel_size = template_data.voxel_size

    if template is None:
        raise ValueError("Template data must contain a template image")

    grid_shape = template.shape

    # Create tumor parameters
    tumor_params = TumorParameters(
        center=tumor_center,
        radius=tumor_radius,
        shape="spherical"
    )

    # Create model parameters
    model_params = ModelParameters(
        decay_exponent=decay_exponent,
        max_displacement=tumor_expansion * 1.5,
        smoothing_sigma=1.0
    )

    # Create and run model
    model = GeometricExpansionModel(
        tumor_params=tumor_params,
        model_params=model_params,
        grid_shape=grid_shape,
        voxel_size=voxel_size
    )

    # Add mask if available (resample if needed)
    if mask is not None:
        if mask.shape != grid_shape:
            from scipy.ndimage import zoom
            scale = [g / m for g, m in zip(grid_shape, mask.shape)]
            mask_resampled = zoom(mask, scale, order=0)
            model.set_boundary_mask(mask_resampled)
        else:
            model.set_boundary_mask(mask)

    # Compute displacement
    displacement = model.compute_displacement_field(tumor_expansion=tumor_expansion)

    # Create tumor mask
    tumor_mask = create_spherical_tumor_mask(
        grid_shape, tumor_center, tumor_radius
    )

    # Generate displaced template
    displaced = generate_displaced_template(
        template,
        displacement,
        voxel_size=voxel_size,
        tumor_mask=tumor_mask,
        tumor_intensity=0.65
    )

    # Prepare metadata
    metadata = {
        "tumor_center": list(tumor_center),
        "tumor_radius": tumor_radius,
        "tumor_expansion": tumor_expansion,
        "decay_exponent": decay_exponent,
        "grid_shape": list(grid_shape),
        "voxel_size": list(voxel_size),
        "source": "PFT_GEM synthetic output"
    }

    # Create output object
    output = SyntheticMRIOutput(
        displaced_image=displaced,
        original_image=template,
        displacement_field=displacement,
        tumor_mask=tumor_mask,
        affine=affine if affine is not None else np.eye(4),
        voxel_size=voxel_size,
        metadata=metadata
    )

    # Save outputs if directory specified
    if output_dir is not None:
        save_synthetic_output(output, output_dir, prefix)

    return output


def save_synthetic_output(
    output: SyntheticMRIOutput,
    output_dir: Union[str, Path],
    prefix: str = "synthetic"
) -> Dict[str, Path]:
    """
    Save all synthetic MRI output files.

    Saves:
    - Displaced template image (NIfTI)
    - Original template (NIfTI)
    - Displacement field as warp (NIfTI 4D)
    - Tumor mask (NIfTI)
    - Metadata (JSON)

    Parameters
    ----------
    output : SyntheticMRIOutput
        Generated synthetic output
    output_dir : str or Path
        Output directory
    prefix : str
        Prefix for filenames

    Returns
    -------
    paths : dict
        Dictionary mapping output type to file path
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for saving NIfTI files")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save displaced image
    displaced_path = output_dir / f"{prefix}_displaced.nii.gz"
    img = nib.Nifti1Image(output.displaced_image, output.affine)
    img.header.set_xyzt_units('mm')
    nib.save(img, str(displaced_path))
    paths['displaced'] = displaced_path

    # Save original template
    original_path = output_dir / f"{prefix}_original.nii.gz"
    img = nib.Nifti1Image(output.original_image.astype(np.float32), output.affine)
    img.header.set_xyzt_units('mm')
    nib.save(img, str(original_path))
    paths['original'] = original_path

    # Save displacement field as warp
    warp_path = output_dir / f"{prefix}_warp.nii.gz"
    save_displacement_as_warp(
        output.displacement_field,
        warp_path,
        output.affine,
        format="nifti"
    )
    paths['warp'] = warp_path

    # Save inverse warp (for applying in reverse direction)
    from ..core.displacement import DisplacementField, FieldMetadata
    field = DisplacementField(
        output.displacement_field,
        metadata=FieldMetadata(
            shape=output.displacement_field.shape[:3],
            voxel_size=output.voxel_size
        )
    )
    inverse_field = field.invert(iterations=20)
    inverse_warp_path = output_dir / f"{prefix}_warp_inverse.nii.gz"
    save_displacement_as_warp(
        inverse_field.data,
        inverse_warp_path,
        output.affine,
        format="nifti"
    )
    paths['warp_inverse'] = inverse_warp_path

    # Save tumor mask
    mask_path = output_dir / f"{prefix}_tumor_mask.nii.gz"
    img = nib.Nifti1Image(output.tumor_mask.astype(np.uint8), output.affine)
    nib.save(img, str(mask_path))
    paths['tumor_mask'] = mask_path

    # Save metadata
    metadata_path = output_dir / f"{prefix}_metadata.json"

    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    metadata_json = convert_to_json_serializable(output.metadata)

    with open(metadata_path, 'w') as f:
        json.dump(metadata_json, f, indent=2)
    paths['metadata'] = metadata_path

    return paths


def load_synthetic_output(
    output_dir: Union[str, Path],
    prefix: str = "synthetic"
) -> SyntheticMRIOutput:
    """
    Load previously saved synthetic MRI output.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing saved outputs
    prefix : str
        Prefix used when saving

    Returns
    -------
    output : SyntheticMRIOutput
        Loaded synthetic output
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for loading NIfTI files")

    output_dir = Path(output_dir)

    # Load displaced image
    displaced_img = nib.load(str(output_dir / f"{prefix}_displaced.nii.gz"))
    displaced = displaced_img.get_fdata()
    affine = displaced_img.affine
    voxel_size = tuple(displaced_img.header.get_zooms()[:3])

    # Load original
    original = nib.load(str(output_dir / f"{prefix}_original.nii.gz")).get_fdata()

    # Load warp
    displacement = nib.load(str(output_dir / f"{prefix}_warp.nii.gz")).get_fdata()

    # Load tumor mask
    tumor_mask = nib.load(str(output_dir / f"{prefix}_tumor_mask.nii.gz")).get_fdata()

    # Load metadata
    with open(output_dir / f"{prefix}_metadata.json", 'r') as f:
        metadata = json.load(f)

    return SyntheticMRIOutput(
        displaced_image=displaced,
        original_image=original,
        displacement_field=displacement,
        tumor_mask=tumor_mask,
        affine=affine,
        voxel_size=voxel_size,
        metadata=metadata
    )
