"""
Biophysical Constraints from Diffusion MRI

This module provides classes and functions for deriving biophysical
constraints from diffusion MRI data to modulate tumor displacement
fields. The constraints are based on tissue microstructure properties
inferred from DTI (Diffusion Tensor Imaging) metrics.

Theory:
-------
Diffusion MRI provides information about tissue microstructure that
correlates with mechanical properties:

1. Fractional Anisotropy (FA):
   - Higher FA indicates more organized/aligned tissue (e.g., white matter tracts)
   - These regions tend to be stiffer along fiber directions

2. Mean Diffusivity (MD):
   - Lower MD indicates denser tissue with more barriers to diffusion
   - Generally correlates with higher stiffness

3. Principal Diffusion Direction (PDD):
   - Indicates preferred direction of water diffusion
   - Tissues may be stiffer perpendicular to fibers
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from scipy import ndimage


class ConstraintType(Enum):
    """Types of biophysical constraints."""
    ISOTROPIC = "isotropic"
    ANISOTROPIC = "anisotropic"
    NONLINEAR = "nonlinear"


@dataclass
class DTIData:
    """Container for DTI-derived maps.

    Attributes:
        fa: Fractional anisotropy map (0-1)
        md: Mean diffusivity map (mm^2/s)
        ad: Axial diffusivity map (mm^2/s), optional
        rd: Radial diffusivity map (mm^2/s), optional
        v1: Principal eigenvector field (nx, ny, nz, 3), optional
        tensor: Full diffusion tensor (nx, ny, nz, 3, 3), optional
    """
    fa: np.ndarray
    md: np.ndarray
    ad: Optional[np.ndarray] = None
    rd: Optional[np.ndarray] = None
    v1: Optional[np.ndarray] = None
    tensor: Optional[np.ndarray] = None

    def __post_init__(self):
        # Validate shapes match
        if self.md.shape != self.fa.shape:
            raise ValueError("FA and MD maps must have same shape")
        if self.ad is not None and self.ad.shape != self.fa.shape:
            raise ValueError("AD map must have same shape as FA")
        if self.rd is not None and self.rd.shape != self.fa.shape:
            raise ValueError("RD map must have same shape as FA")
        if self.v1 is not None and self.v1.shape[:3] != self.fa.shape:
            raise ValueError("V1 field must have same spatial shape as FA")


@dataclass
class TissueParameters:
    """Mechanical tissue parameters.

    Attributes:
        young_modulus: Young's modulus in kPa (typical brain: 1-10 kPa)
        poisson_ratio: Poisson's ratio (0-0.5, brain ~0.45)
        stiffness_ratio: Anisotropic stiffness ratio (parallel/perpendicular)
    """
    young_modulus: float = 3.0  # kPa
    poisson_ratio: float = 0.45
    stiffness_ratio: float = 1.5


class BiophysicalConstraints:
    """
    Biophysical constraints derived from diffusion MRI data.

    This class computes tissue stiffness maps and anisotropic constraints
    from DTI data to modulate displacement fields in a physically
    plausible manner.

    Parameters
    ----------
    dti_data : DTIData
        DTI-derived maps (FA, MD, etc.)
    tissue_params : TissueParameters, optional
        Base tissue mechanical parameters
    constraint_type : ConstraintType
        Type of constraint model to use

    Examples
    --------
    >>> dti = DTIData(fa=fa_map, md=md_map)
    >>> constraints = BiophysicalConstraints(dti)
    >>> stiffness = constraints.compute_stiffness_map()
    """

    # Reference values for healthy adult brain
    REFERENCE_FA_WM = 0.45  # White matter
    REFERENCE_FA_GM = 0.15  # Gray matter
    REFERENCE_MD_WM = 0.7e-3  # mm^2/s
    REFERENCE_MD_GM = 0.8e-3  # mm^2/s

    # Stiffness conversion parameters
    FA_STIFFNESS_WEIGHT = 0.6
    MD_STIFFNESS_WEIGHT = 0.4

    def __init__(
        self,
        dti_data: DTIData,
        tissue_params: Optional[TissueParameters] = None,
        constraint_type: ConstraintType = ConstraintType.ISOTROPIC
    ):
        self.dti = dti_data
        self.params = tissue_params or TissueParameters()
        self.constraint_type = constraint_type

        # Cached computed maps
        self._stiffness_map: Optional[np.ndarray] = None
        self._anisotropy_tensor: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return spatial dimensions."""
        return self.dti.fa.shape

    def compute_stiffness_map(
        self,
        normalize: bool = True,
        smooth_sigma: float = 0.0
    ) -> np.ndarray:
        """
        Compute isotropic stiffness map from DTI metrics.

        The stiffness map is computed as a weighted combination of
        FA and MD contributions, where:
        - Higher FA → higher stiffness (more organized tissue)
        - Lower MD → higher stiffness (denser tissue)

        Parameters
        ----------
        normalize : bool
            Whether to normalize to [0, 1] range
        smooth_sigma : float
            Gaussian smoothing sigma in voxels (0 = no smoothing)

        Returns
        -------
        stiffness : np.ndarray
            3D stiffness map
        """
        # Normalize FA contribution (higher FA = stiffer)
        fa_contribution = np.clip(self.dti.fa / self.REFERENCE_FA_WM, 0, 2)

        # Normalize MD contribution (lower MD = stiffer)
        md_normalized = np.clip(self.dti.md / self.REFERENCE_MD_WM, 0.1, 3)
        md_contribution = 1.0 / md_normalized

        # Combine contributions
        stiffness = (
            self.FA_STIFFNESS_WEIGHT * fa_contribution +
            self.MD_STIFFNESS_WEIGHT * md_contribution
        )

        if smooth_sigma > 0:
            stiffness = ndimage.gaussian_filter(stiffness, sigma=smooth_sigma)

        if normalize:
            stiffness = (stiffness - stiffness.min()) / (
                stiffness.max() - stiffness.min() + 1e-10
            )

        self._stiffness_map = stiffness
        return stiffness

    def compute_compliance_map(self) -> np.ndarray:
        """
        Compute compliance map (inverse of stiffness).

        Compliance indicates how easily tissue deforms:
        - Higher compliance → more displacement for same force

        Returns
        -------
        compliance : np.ndarray
            3D compliance map
        """
        if self._stiffness_map is None:
            self.compute_stiffness_map()

        # Avoid division by zero
        return 1.0 / (self._stiffness_map + 0.1)

    def compute_anisotropy_tensor(self) -> np.ndarray:
        """
        Compute anisotropic stiffness tensor from DTI.

        For each voxel, computes a 3x3 tensor that describes
        directional stiffness based on the diffusion tensor orientation.

        Returns
        -------
        tensor : np.ndarray
            5D array of shape (nx, ny, nz, 3, 3)
        """
        if self.dti.v1 is None:
            raise ValueError("Principal eigenvector (v1) required for anisotropic constraints")

        # Base stiffness
        if self._stiffness_map is None:
            self.compute_stiffness_map()

        # Compute anisotropic tensor
        # Stiffness is higher along fiber direction (v1)
        tensor = np.zeros((*self.shape, 3, 3))

        # Identity component
        for i in range(3):
            tensor[..., i, i] = 1.0

        # Add anisotropic component along v1
        # Outer product of v1 with itself
        v1_outer = np.einsum('...i,...j->...ij', self.dti.v1, self.dti.v1)

        # Scale by FA (more anisotropic where FA is high)
        aniso_factor = (self.params.stiffness_ratio - 1.0) * self.dti.fa[..., np.newaxis, np.newaxis]
        tensor += aniso_factor * v1_outer

        # Scale by base stiffness
        tensor *= self._stiffness_map[..., np.newaxis, np.newaxis]

        self._anisotropy_tensor = tensor
        return tensor

    def modulate_displacement(
        self,
        displacement: np.ndarray,
        inverse: bool = True
    ) -> np.ndarray:
        """
        Modulate displacement field by tissue stiffness.

        Parameters
        ----------
        displacement : np.ndarray
            4D displacement field (nx, ny, nz, 3)
        inverse : bool
            If True, stiffer regions have less displacement

        Returns
        -------
        modulated : np.ndarray
            Modulated displacement field
        """
        if displacement.shape[:3] != self.shape:
            raise ValueError("Displacement shape must match DTI shape")

        if self._stiffness_map is None:
            self.compute_stiffness_map()

        if inverse:
            # Stiffer regions → less displacement
            modulation = self.compute_compliance_map()
        else:
            modulation = self._stiffness_map

        # Normalize modulation to preserve mean displacement magnitude
        modulation = modulation / (np.mean(modulation) + 1e-10)

        return displacement * modulation[..., np.newaxis]

    def modulate_displacement_anisotropic(
        self,
        displacement: np.ndarray
    ) -> np.ndarray:
        """
        Apply anisotropic modulation to displacement field.

        Displacement is reduced more in directions of higher stiffness
        (typically perpendicular to fiber bundles).

        Parameters
        ----------
        displacement : np.ndarray
            4D displacement field (nx, ny, nz, 3)

        Returns
        -------
        modulated : np.ndarray
            Anisotropically modulated displacement field
        """
        if self._anisotropy_tensor is None:
            self.compute_anisotropy_tensor()

        # Compute compliance tensor (inverse of stiffness tensor)
        # For simplicity, use pseudo-inverse element-wise
        tensor = self._anisotropy_tensor
        tensor_inv = np.zeros_like(tensor)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    try:
                        tensor_inv[i, j, k] = np.linalg.inv(
                            tensor[i, j, k] + 0.01 * np.eye(3)
                        )
                    except np.linalg.LinAlgError:
                        tensor_inv[i, j, k] = np.eye(3)

        # Apply compliance tensor to displacement
        modulated = np.einsum('...ij,...j->...i', tensor_inv, displacement)

        return modulated

    def compute_fiber_deviation_penalty(
        self,
        displacement: np.ndarray
    ) -> np.ndarray:
        """
        Compute penalty for displacement perpendicular to fibers.

        In highly anisotropic regions (white matter tracts), tissue
        tends to resist deformation perpendicular to fiber direction.

        Parameters
        ----------
        displacement : np.ndarray
            4D displacement field

        Returns
        -------
        penalty : np.ndarray
            3D penalty map (0-1, higher = more perpendicular)
        """
        if self.dti.v1 is None:
            return np.zeros(self.shape)

        # Normalize displacement
        disp_mag = np.linalg.norm(displacement, axis=-1, keepdims=True)
        disp_normalized = displacement / (disp_mag + 1e-10)

        # Compute angle between displacement and fiber direction
        # cos(theta) = |d . v1|
        cos_angle = np.abs(np.einsum('...i,...i', disp_normalized, self.dti.v1))

        # Penalty is higher when perpendicular (cos_angle near 0)
        # Scale by FA (only matters in anisotropic regions)
        penalty = (1 - cos_angle) * self.dti.fa

        return penalty

    def estimate_young_modulus(
        self,
        reference_e: float = 3.0
    ) -> np.ndarray:
        """
        Estimate Young's modulus map from DTI metrics.

        Uses empirical relationships between DTI metrics and
        tissue stiffness from literature.

        Parameters
        ----------
        reference_e : float
            Reference Young's modulus for average brain tissue (kPa)

        Returns
        -------
        young_modulus : np.ndarray
            3D Young's modulus map in kPa
        """
        if self._stiffness_map is None:
            self.compute_stiffness_map(normalize=False)

        # Scale stiffness map to physical Young's modulus
        # Based on typical range for brain tissue: 0.5-10 kPa
        young_modulus = reference_e * self._stiffness_map

        # Clip to physiological range
        young_modulus = np.clip(young_modulus, 0.5, 10.0)

        return young_modulus

    def get_tissue_mask(
        self,
        tissue_type: str = "white_matter"
    ) -> np.ndarray:
        """
        Generate tissue mask based on DTI characteristics.

        Parameters
        ----------
        tissue_type : str
            'white_matter', 'gray_matter', or 'csf'

        Returns
        -------
        mask : np.ndarray
            Binary tissue mask
        """
        if tissue_type == "white_matter":
            # High FA, moderate MD
            mask = (self.dti.fa > 0.2) & (self.dti.md < 1.2e-3)
        elif tissue_type == "gray_matter":
            # Low FA, moderate MD
            mask = (self.dti.fa < 0.25) & (self.dti.md > 0.6e-3) & (self.dti.md < 1.5e-3)
        elif tissue_type == "csf":
            # Very low FA, high MD
            mask = (self.dti.fa < 0.15) & (self.dti.md > 2.0e-3)
        else:
            raise ValueError(f"Unknown tissue type: {tissue_type}")

        return mask.astype(np.uint8)

    def compute_strain_energy_density(
        self,
        displacement: np.ndarray
    ) -> np.ndarray:
        """
        Compute strain energy density from displacement and stiffness.

        Parameters
        ----------
        displacement : np.ndarray
            4D displacement field

        Returns
        -------
        energy : np.ndarray
            3D strain energy density map
        """
        # Compute strain tensor
        strain = np.zeros((*self.shape, 3, 3))
        for i in range(3):
            for j in range(3):
                du_i_dx_j = np.gradient(displacement[..., i], axis=j)
                du_j_dx_i = np.gradient(displacement[..., j], axis=i)
                strain[..., i, j] = 0.5 * (du_i_dx_j + du_j_dx_i)

        # Get Young's modulus
        E = self.estimate_young_modulus()
        nu = self.params.poisson_ratio

        # Compute strain energy density (linear elasticity)
        # W = 0.5 * E / (1 + nu) * (epsilon_ij * epsilon_ij + nu/(1-2nu) * epsilon_kk^2)
        strain_sq = np.einsum('...ij,...ij', strain, strain)
        trace_strain = np.trace(strain, axis1=-2, axis2=-1)

        energy = 0.5 * E / (1 + nu) * (
            strain_sq + nu / (1 - 2*nu) * trace_strain**2
        )

        return energy

    def to_dict(self) -> Dict[str, Any]:
        """Export constraints configuration to dictionary."""
        return {
            "constraint_type": self.constraint_type.value,
            "tissue_params": {
                "young_modulus": self.params.young_modulus,
                "poisson_ratio": self.params.poisson_ratio,
                "stiffness_ratio": self.params.stiffness_ratio,
            },
            "reference_values": {
                "fa_wm": self.REFERENCE_FA_WM,
                "fa_gm": self.REFERENCE_FA_GM,
                "md_wm": self.REFERENCE_MD_WM,
                "md_gm": self.REFERENCE_MD_GM,
            },
        }


def create_synthetic_dti(
    shape: Tuple[int, int, int],
    tumor_center: Tuple[int, int, int],
    tumor_radius: float
) -> DTIData:
    """
    Create synthetic DTI data for testing.

    Parameters
    ----------
    shape : tuple
        Grid dimensions
    tumor_center : tuple
        Tumor center coordinates
    tumor_radius : float
        Tumor radius

    Returns
    -------
    dti : DTIData
        Synthetic DTI data
    """
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )

    # Distance from tumor
    distance = np.sqrt(
        (x - tumor_center[0])**2 +
        (y - tumor_center[1])**2 +
        (z - tumor_center[2])**2
    )

    # Create FA map (higher near center, lower near boundaries)
    # With reduced values near tumor
    fa = 0.3 + 0.2 * np.sin(x * 0.1) * np.cos(y * 0.1)
    fa = np.clip(fa, 0, 1)

    # Reduce FA near tumor (edema effect)
    tumor_effect = np.exp(-(distance - tumor_radius)**2 / (2 * 20**2))
    tumor_effect = np.clip(tumor_effect, 0, 1)
    fa = fa * (1 - 0.5 * tumor_effect)

    # Set FA to 0 inside tumor
    fa[distance < tumor_radius] = 0.05

    # Create MD map (higher near CSF/tumor)
    md = 0.8e-3 + 0.2e-3 * np.random.randn(*shape)
    md = np.clip(md, 0.5e-3, 3e-3)

    # Increase MD near tumor (edema)
    md = md * (1 + 0.5 * tumor_effect)
    md[distance < tumor_radius] = 1.5e-3

    # Create principal eigenvector field (pointing radially from tumor)
    v1 = np.zeros((*shape, 3))
    v1[..., 0] = (x - tumor_center[0])
    v1[..., 1] = (y - tumor_center[1])
    v1[..., 2] = (z - tumor_center[2])
    v1_mag = np.linalg.norm(v1, axis=-1, keepdims=True)
    v1 = v1 / (v1_mag + 1e-10)

    return DTIData(fa=fa, md=md, v1=v1)
