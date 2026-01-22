# Example Data

This directory contains example data for testing and demonstrating PFT_GEM functionality.

## Synthetic Data Generation

PFT_GEM includes utilities for generating synthetic data:

```python
from pft_gem.core.constraints import create_synthetic_dti
from pft_gem.io.template_loader import SUITTemplateLoader
from pft_gem.utils.helpers import create_spherical_tumor_mask

# Create synthetic DTI data
dti_data = create_synthetic_dti(
    shape=(128, 128, 64),
    tumor_center=(64, 64, 32),
    tumor_radius=15.0
)

# Create synthetic template
template = SUITTemplateLoader.create_synthetic_template(
    shape=(128, 128, 64)
)

# Create tumor mask
tumor_mask = create_spherical_tumor_mask(
    shape=(128, 128, 64),
    center=(64, 64, 32),
    radius=15.0
)
```

## Expected Data Format

PFT_GEM works with:
- **NIfTI files** (.nii, .nii.gz) for brain images and masks
- **4D NIfTI files** for displacement fields (shape: nx, ny, nz, 3)
- Standard DTI metric maps (FA, MD, AD, RD)

## Sample Workflow

1. Load or generate brain template
2. Define tumor location and size
3. Optionally load DTI data for biophysical constraints
4. Run geometric expansion model
5. Visualize and export displacement field
