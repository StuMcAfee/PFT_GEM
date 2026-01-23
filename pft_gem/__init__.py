"""
PFT_GEM: Posterior Fossa Tumor - Geometric Expansion Model

A simplified approach to modeling tumor-induced displacement in brain tissue
using geometric expansion methods. This package provides computationally
efficient alternatives to finite element methods (FEM) while maintaining
biophysically plausible displacement field estimates.

Key Features:
- Geometric expansion modeling of tumor growth
- Integration with SUIT (Spatially Unbiased Infratentorial Template)
- Biophysical constraints from diffusion MRI data
- NIfTI format support for neuroimaging compatibility
- ANTsPy-compatible spatial transforms for image warping
"""

__version__ = "0.1.0"
__author__ = "PFT_GEM Development Team"

from .core.geometric_model import GeometricExpansionModel
from .core.displacement import DisplacementField
from .core.constraints import BiophysicalConstraints
from .core.tumor_growth_simulation import (
    TumorGrowthSimulator,
    TumorGrowthParams,
    TissueProperties,
    SimulationOutput,
    simulate_tumor_growth,
)

__all__ = [
    "GeometricExpansionModel",
    "DisplacementField",
    "BiophysicalConstraints",
    "TumorGrowthSimulator",
    "TumorGrowthParams",
    "TissueProperties",
    "SimulationOutput",
    "simulate_tumor_growth",
    "__version__",
]
