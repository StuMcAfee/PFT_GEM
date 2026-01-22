# SUIT Template Data

This directory should contain the SUIT (Spatially Unbiased Infratentorial Template)
reference data for cerebellum and brainstem analysis.

## Required Files

To use PFT_GEM with real anatomical data, please download the following files from
the official SUIT website and place them in this directory:

- `SUIT_T1_1mm.nii.gz` - T1-weighted template (1mm resolution)
- `SUIT_mask_1mm.nii.gz` - Binary cerebellum/brainstem mask
- `SUIT_atlas_1mm.nii.gz` - Anatomical parcellation atlas

For 2mm resolution versions, use:
- `SUIT_T1_2mm.nii.gz`
- `SUIT_mask_2mm.nii.gz`
- `SUIT_atlas_2mm.nii.gz`

## Download Instructions

1. Visit: http://www.diedrichsenlab.org/imaging/suit.htm
2. Download the SUIT toolbox or standalone templates
3. Extract the template files and place them in this directory

## Alternative: SPM Toolbox

If you use SPM, you can install SUIT as an SPM toolbox:
http://www.diedrichsenlab.org/imaging/suit_download.htm

The template files will be located in the `atlas` subdirectory of the toolbox.

## Using Synthetic Templates

If SUIT templates are not available, PFT_GEM can generate synthetic templates
for testing purposes:

```python
from pft_gem.io import SUITTemplateLoader

template = SUITTemplateLoader.create_synthetic_template(
    shape=(128, 128, 64),
    voxel_size=(1.0, 1.0, 1.0)
)
```

## Citation

When using SUIT templates, please cite:

Diedrichsen, J. (2006). A spatially unbiased atlas template of the human
cerebellum. NeuroImage, 33(1), 127-138.

Diedrichsen, J., Balsters, J. H., Flavell, J., Cussans, E., & Ramnani, N. (2009).
A probabilistic MR atlas of the human cerebellum. NeuroImage, 46(1), 39-46.
