# SUIT Template Data

This directory contains the SUIT (Spatially Unbiased Infratentorial Template)
reference data for cerebellum and brainstem analysis.

## Included Files

The following files are included from the official SUIT repositories:

### Template Files
- `tpl-SUIT_T1w.nii` - T1-weighted template image (from cerebellar_atlases)
- `SUIT_T1.nii` - T1-weighted template (from suit toolbox)
- `SUIT.nii` - SUIT template

### Mask Files
- `SUIT_mask.nii` - Binary cerebellum/brainstem mask
- `SUIT_pcerebellum.nii` - Probabilistic cerebellar mask

### Atlas/Parcellation Files
- `atl-Anatom_space-SUIT_dseg.nii` - Anatomical parcellation atlas (discrete labels)
- `atl-Anatom_space-SUIT_probseg.nii` - Probabilistic anatomical atlas
- `atl-Anatom.tsv` - Label lookup table (34 cerebellar regions)

### Tissue Probability Maps
- `gray_cereb.nii` - Cerebellar gray matter probability
- `white_cereb.nii` - Cerebellar white matter probability

## Atlas Regions

The anatomical atlas contains 34 labeled regions:

| Index | Region | Index | Region |
|-------|--------|-------|--------|
| 1 | Left_I_IV | 18 | Vermis_VIIIa |
| 2 | Right_I_IV | 19 | Right_VIIIa |
| 3 | Left_V | 20 | Left_VIIIb |
| 4 | Right_V | 21 | Vermis_VIIIb |
| 5 | Left_VI | 22 | Right_VIIIb |
| 6 | Vermis_VI | 23 | Left_IX |
| 7 | Right_VI | 24 | Vermis_IX |
| 8 | Left_CrusI | 25 | Right_IX |
| 9 | Vermis_CrusI | 26 | Left_X |
| 10 | Right_CrusI | 27 | Vermis_X |
| 11 | Left_CrusII | 28 | Right_X |
| 12 | Vermis_CrusII | 29 | Left_Dentate |
| 13 | Right_CrusII | 30 | Right_Dentate |
| 14 | Left_VIIb | 31 | Left_Interposed |
| 15 | Vermis_VIIb | 32 | Right_Interposed |
| 16 | Right_VIIb | 33 | Left_Fastigial |
| 17 | Left_VIIIa | 34 | Right_Fastigial |

## Usage

```python
from pft_gem.io import SUITTemplateLoader

# Load templates from this directory
loader = SUITTemplateLoader('data/suit_template')
template_data = loader.load_template()

# Access individual components
template = template_data.template    # T1 image
mask = template_data.mask            # Cerebellum mask
atlas = template_data.atlas          # Parcellation atlas
labels = template_data.labels        # Region name dictionary
```

## Sources

Files were obtained from:
- https://github.com/DiedrichsenLab/cerebellar_atlases
- https://github.com/jdiedrichsen/suit

## License

SUIT templates are distributed under Creative Commons Attribution-NonCommercial 3.0
Unported License (CC BY-NC 3.0).

## Citation

When using SUIT templates, please cite:

Diedrichsen, J. (2006). A spatially unbiased atlas template of the human
cerebellum. NeuroImage, 33(1), 127-138.

Diedrichsen, J., Balsters, J. H., Flavell, J., Cussans, E., & Ramnani, N. (2009).
A probabilistic MR atlas of the human cerebellum. NeuroImage, 46(1), 39-46.
