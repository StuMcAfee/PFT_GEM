"""
I/O modules for NIfTI file handling and SUIT template loading.
"""

from .nifti_handler import NiftiHandler, save_displacement_field, load_displacement_field
from .template_loader import SUITTemplateLoader, TemplateData
from .synthetic_output import (
    SyntheticMRIOutput,
    generate_displaced_template,
    save_displacement_as_warp,
    save_displacement_as_coordinate_map,
    apply_warp_to_image,
    generate_synthetic_mri_output,
    save_synthetic_output,
    load_synthetic_output,
)

__all__ = [
    "NiftiHandler",
    "save_displacement_field",
    "load_displacement_field",
    "SUITTemplateLoader",
    "TemplateData",
    # Synthetic MRI output
    "SyntheticMRIOutput",
    "generate_displaced_template",
    "save_displacement_as_warp",
    "save_displacement_as_coordinate_map",
    "apply_warp_to_image",
    "generate_synthetic_mri_output",
    "save_synthetic_output",
    "load_synthetic_output",
]
