# PFT_GEM: Posterior Fossa Tumor - Geometric Expansion Model

A computationally efficient Python package for modeling tumor-induced displacement in brain tissue using geometric expansion methods. PFT_GEM provides a simplified alternative to finite element methods (FEM) while maintaining biophysically plausible displacement field estimates.

## Overview

PFT_GEM models how brain tumors, particularly those in the posterior fossa (cerebellum and brainstem), displace surrounding tissue as they grow. The geometric expansion approach offers significant computational advantages over FEM-based methods while incorporating biophysical constraints from diffusion MRI data.

### Key Features

- **Geometric Expansion Model**: Fast analytical displacement field computation
- **DTI Integration**: Biophysical constraints from diffusion tensor imaging (FA, MD)
- **SUIT Template Support**: Integration with Spatially Unbiased Infratentorial Template
- **NIfTI Compatibility**: Standard neuroimaging format support
- **Comprehensive Visualization**: Built-in tools for displacement field analysis
- **Jupyter Notebook**: Interactive tutorial and visualization workflows

## Theory

The geometric expansion model computes displacement based on radial expansion from the tumor boundary with tissue-specific modulation:

```
u(r) = u₀ × (R/r)^α × f(tissue) × g(boundaries)
```

Where:
- `u₀`: displacement at tumor boundary
- `R`: tumor radius
- `r`: distance from tumor center
- `α`: decay exponent (typically 1-3)
- `f(tissue)`: tissue stiffness modulation from DTI
- `g(boundaries)`: anatomical boundary constraints

### Biophysical Constraints from DTI

Diffusion MRI provides tissue microstructure information:

- **Fractional Anisotropy (FA)**: Higher FA indicates more organized tissue (typically stiffer)
- **Mean Diffusivity (MD)**: Lower MD indicates denser tissue (typically stiffer)

These are combined to create a stiffness map that modulates the displacement field, resulting in more realistic tissue behavior.

## Installation

### From Source

```bash
git clone https://github.com/StuMcAfee/PFT_GEM.git
cd PFT_GEM
pip install -e .
```

### Dependencies

Required:
- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7

Optional:
- nibabel >= 4.0 (NIfTI file support)
- matplotlib >= 3.5 (visualization)
- jupyter (interactive notebooks)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from pft_gem import GeometricExpansionModel
from pft_gem.core.geometric_model import TumorParameters, ModelParameters

# Define tumor parameters
tumor = TumorParameters(
    center=(64.0, 64.0, 32.0),  # mm
    radius=15.0,                 # mm
    shape="spherical"
)

# Define model parameters
params = ModelParameters(
    decay_exponent=2.0,
    max_displacement=15.0,
    use_dti_constraints=True
)

# Create model
model = GeometricExpansionModel(
    tumor_params=tumor,
    model_params=params,
    grid_shape=(128, 128, 64),
    voxel_size=(1.0, 1.0, 1.0)
)

# Compute displacement field
displacement = model.compute_displacement_field(tumor_expansion=5.0)

# Result: 4D array (nx, ny, nz, 3) with displacement vectors in mm
```

## Usage Examples

### Basic Displacement Field Computation

```python
from pft_gem import GeometricExpansionModel
from pft_gem.core.geometric_model import TumorParameters

# Simple spherical tumor
tumor = TumorParameters(center=(64, 64, 32), radius=15.0)
model = GeometricExpansionModel(tumor)

# Compute displacement for 5mm tumor expansion
displacement = model.compute_displacement_field(tumor_expansion=5.0)
```

### With DTI Constraints

```python
from pft_gem.core.constraints import BiophysicalConstraints, DTIData

# Load or create DTI data
dti = DTIData(fa=fa_map, md=md_map)

# Add constraints to model
model.set_dti_constraints(dti.fa, dti.md)

# Recompute with constraints
displacement = model.compute_displacement_field()
```

### Visualization

```python
from pft_gem.visualization import (
    plot_displacement_field,
    plot_vector_field,
    plot_jacobian
)

# Multi-view displacement magnitude
fig = plot_displacement_field(displacement, tumor_mask=tumor_mask)

# Vector field visualization
fig = plot_vector_field(displacement, slice_idx=32)

# Jacobian (volume change) visualization
jacobian = model.compute_jacobian_determinant()
fig = plot_jacobian(jacobian, slice_idx=32)
```

### Working with SUIT Templates

```python
from pft_gem.io import SUITTemplateLoader

# Load SUIT template (if available)
loader = SUITTemplateLoader('/path/to/suit')
template_data = loader.load_template()

# Or create synthetic template for testing
template_data = SUITTemplateLoader.create_synthetic_template(
    shape=(128, 128, 64)
)
```

### Saving Results

```python
from pft_gem.io import save_displacement_field

# Save as NIfTI
save_displacement_field(
    displacement,
    'displacement.nii.gz',
    voxel_size=(1.0, 1.0, 1.0)
)
```

## Project Structure

```
PFT_GEM/
├── pft_gem/                     # Main package
│   ├── __init__.py
│   ├── core/                    # Core modeling modules
│   │   ├── geometric_model.py   # Geometric expansion model
│   │   ├── displacement.py      # Displacement field operations
│   │   └── constraints.py       # DTI biophysical constraints
│   ├── io/                      # Input/output modules
│   │   ├── nifti_handler.py     # NIfTI file handling
│   │   └── template_loader.py   # SUIT template loading
│   ├── visualization/           # Visualization tools
│   │   └── plotting.py          # Plotting functions
│   └── utils/                   # Utility functions
│       └── helpers.py           # Helper utilities
├── data/                        # Reference data
│   ├── suit_template/           # SUIT template files
│   └── example/                 # Example data
├── notebooks/                   # Jupyter notebooks
│   └── pft_gem_tutorial.ipynb   # Interactive tutorial
├── tests/                       # Unit tests
├── examples/                    # Example scripts
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## API Reference

### Core Classes

#### `GeometricExpansionModel`
Main model class for computing displacement fields.

```python
model = GeometricExpansionModel(
    tumor_params,      # TumorParameters object
    model_params,      # ModelParameters object (optional)
    grid_shape,        # (nx, ny, nz) tuple
    voxel_size,        # (dx, dy, dz) in mm
    origin             # (x0, y0, z0) in mm
)

# Methods
displacement = model.compute_displacement_field(tumor_expansion=5.0)
strain = model.compute_strain_field()
jacobian = model.compute_jacobian_determinant()
```

#### `DisplacementField`
Class for displacement field operations.

```python
field = DisplacementField(data, metadata)

# Operations
magnitude = field.magnitude()
smoothed = field.smooth(sigma=1.0)
resampled = field.resample(new_shape)
warped_image = field.apply_to_image(image)
inverse = field.invert()
composed = field.compose(other_field)
```

#### `BiophysicalConstraints`
Class for DTI-derived tissue constraints.

```python
constraints = BiophysicalConstraints(dti_data)

stiffness = constraints.compute_stiffness_map()
compliance = constraints.compute_compliance_map()
modulated = constraints.modulate_displacement(displacement)
```

### Data Classes

#### `TumorParameters`
```python
tumor = TumorParameters(
    center=(x, y, z),        # mm
    radius=15.0,             # mm
    growth_rate=0.0,         # mm/day (optional)
    shape="spherical",       # or "ellipsoidal"
    semi_axes=(a, b, c)      # for ellipsoidal
)
```

#### `ModelParameters`
```python
params = ModelParameters(
    decay_exponent=2.0,      # 1-3 typical
    boundary_stiffness=0.8,  # 0-1
    csf_damping=0.1,         # 0-1
    max_displacement=15.0,   # mm
    tissue_modulation=True,
    use_dti_constraints=True,
    smoothing_sigma=1.0      # mm
)
```

## Comparison with PFT_FEM

| Feature | PFT_GEM | PFT_FEM |
|---------|---------|---------|
| Method | Geometric expansion | Finite Element |
| Computation time | Seconds | Minutes to hours |
| Accuracy | Good approximation | High fidelity |
| Memory usage | Low | High |
| Complex geometries | Limited | Excellent |
| Material models | Simplified | Full nonlinear |

PFT_GEM is recommended for:
- Rapid prototyping and exploration
- Large-scale studies requiring many simulations
- Initial displacement estimates
- Educational purposes

PFT_FEM is recommended for:
- High-fidelity surgical planning
- Complex material behavior
- Validation studies

## SUIT Template

The SUIT (Spatially Unbiased Infratentorial Template) provides standardized reference data for the cerebellum and brainstem:

1. Download from: http://www.diedrichsenlab.org/imaging/suit.htm
2. Place files in `data/suit_template/`
3. Required files:
   - `SUIT_T1_1mm.nii.gz`
   - `SUIT_mask_1mm.nii.gz`
   - `SUIT_atlas_1mm.nii.gz`

See `data/suit_template/README.md` for detailed instructions.

## Citation

If you use PFT_GEM in your research, please cite:

```bibtex
@software{pft_gem,
  title = {PFT_GEM: Geometric Expansion Model for Posterior Fossa Tumor Displacement},
  author = {McAfee, Stuart},
  year = {2024},
  url = {https://github.com/StuMcAfee/PFT_GEM}
}
```

For SUIT template:
```bibtex
@article{diedrichsen2006suit,
  title = {A spatially unbiased atlas template of the human cerebellum},
  author = {Diedrichsen, J{\"o}rn},
  journal = {NeuroImage},
  volume = {33},
  number = {1},
  pages = {127--138},
  year = {2006}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
