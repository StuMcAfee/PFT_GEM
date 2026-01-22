"""
Geometric Expansion Model for Tumor-Induced Brain Displacement

This module implements a simplified geometric expansion approach to model
how tumors displace surrounding brain tissue. Unlike computationally intensive
finite element methods, this approach uses analytical solutions based on
radial expansion with tissue-specific modulation.

Theory:
-------
The geometric expansion model assumes tumor growth creates radial displacement
that decays with distance from the tumor boundary. The displacement field is
modulated by:

1. Distance from tumor center/boundary
2. Tissue mechanical properties (derived from diffusion MRI)
3. Anatomical boundaries (ventricles, skull, tissue interfaces)

The displacement magnitude at point r from tumor center is:

    u(r) = u_0 * (R/r)^alpha * f(tissue) * g(boundaries)

Where:
    - u_0: displacement at tumor boundary
    - R: tumor radius
    - r: distance from tumor center
    - alpha: decay exponent (typically 1-2 for soft tissue)
    - f(tissue): tissue stiffness modulation from DTI
    - g(boundaries): boundary condition factor
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class TissueType(Enum):
    """Brain tissue types with characteristic mechanical properties."""
    WHITE_MATTER = "white_matter"
    GRAY_MATTER = "gray_matter"
    CSF = "csf"
    TUMOR = "tumor"
    VENTRICLE = "ventricle"
    BRAINSTEM = "brainstem"
    CEREBELLUM = "cerebellum"


@dataclass
class TumorParameters:
    """Parameters defining tumor geometry and growth characteristics.

    Attributes:
        center: (x, y, z) coordinates of tumor center in mm
        radius: Current tumor radius in mm
        growth_rate: Radial growth rate in mm/day (optional, for temporal modeling)
        shape: Tumor shape - 'spherical' or 'ellipsoidal'
        semi_axes: Semi-axes for ellipsoidal tumors (a, b, c) in mm
    """
    center: Tuple[float, float, float]
    radius: float
    growth_rate: float = 0.0
    shape: str = "spherical"
    semi_axes: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.shape == "ellipsoidal" and self.semi_axes is None:
            self.semi_axes = (self.radius, self.radius, self.radius)


@dataclass
class ModelParameters:
    """Parameters controlling the geometric expansion model behavior.

    Attributes:
        decay_exponent: Controls how quickly displacement decays with distance (1-3)
        boundary_stiffness: Stiffness factor at anatomical boundaries (0-1)
        csf_damping: Damping factor for CSF regions (0-1)
        max_displacement: Maximum allowed displacement in mm
        tissue_modulation: Whether to apply tissue-specific modulation
        use_dti_constraints: Whether to use DTI-derived constraints
    """
    decay_exponent: float = 2.0
    boundary_stiffness: float = 0.8
    csf_damping: float = 0.1
    max_displacement: float = 15.0
    tissue_modulation: bool = True
    use_dti_constraints: bool = True
    smoothing_sigma: float = 1.0


class GeometricExpansionModel:
    """
    Geometric expansion model for tumor-induced brain displacement.

    This model provides a computationally efficient alternative to FEM-based
    approaches for estimating displacement fields caused by tumor growth in
    the posterior fossa region.

    Parameters
    ----------
    tumor_params : TumorParameters
        Parameters defining tumor geometry
    model_params : ModelParameters, optional
        Parameters controlling model behavior
    grid_shape : tuple, optional
        Shape of the computational grid (nx, ny, nz)
    voxel_size : tuple, optional
        Voxel dimensions in mm (dx, dy, dz)

    Examples
    --------
    >>> tumor = TumorParameters(center=(45, 50, 30), radius=15.0)
    >>> model = GeometricExpansionModel(tumor)
    >>> displacement = model.compute_displacement_field()
    """

    def __init__(
        self,
        tumor_params: TumorParameters,
        model_params: Optional[ModelParameters] = None,
        grid_shape: Tuple[int, int, int] = (128, 128, 128),
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.tumor = tumor_params
        self.params = model_params or ModelParameters()
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.origin = origin

        # Initialize internal state
        self._tissue_map: Optional[np.ndarray] = None
        self._boundary_mask: Optional[np.ndarray] = None
        self._dti_constraints: Optional[np.ndarray] = None
        self._displacement_field: Optional[np.ndarray] = None

    def set_tissue_map(self, tissue_map: np.ndarray) -> None:
        """Set the tissue segmentation map.

        Parameters
        ----------
        tissue_map : np.ndarray
            3D array with tissue type labels matching TissueType enum values
        """
        if tissue_map.shape != self.grid_shape:
            raise ValueError(
                f"Tissue map shape {tissue_map.shape} does not match "
                f"grid shape {self.grid_shape}"
            )
        self._tissue_map = tissue_map

    def set_boundary_mask(self, boundary_mask: np.ndarray) -> None:
        """Set the anatomical boundary mask.

        Parameters
        ----------
        boundary_mask : np.ndarray
            3D binary array indicating boundary voxels
        """
        if boundary_mask.shape != self.grid_shape:
            raise ValueError(
                f"Boundary mask shape {boundary_mask.shape} does not match "
                f"grid shape {self.grid_shape}"
            )
        self._boundary_mask = boundary_mask

    def set_dti_constraints(self, fa_map: np.ndarray, md_map: np.ndarray) -> None:
        """Set DTI-derived biophysical constraints.

        Parameters
        ----------
        fa_map : np.ndarray
            Fractional anisotropy map (0-1)
        md_map : np.ndarray
            Mean diffusivity map (mm^2/s)
        """
        if fa_map.shape != self.grid_shape or md_map.shape != self.grid_shape:
            raise ValueError("DTI maps must match grid shape")

        # Compute stiffness modulation from DTI
        # Higher FA indicates more organized tissue (stiffer)
        # Lower MD indicates denser tissue (stiffer)
        fa_normalized = np.clip(fa_map, 0, 1)
        md_normalized = np.clip(md_map / 3e-3, 0, 1)  # Normalize to typical brain MD

        # Combined stiffness factor (higher = stiffer = less displacement)
        self._dti_constraints = 0.5 * fa_normalized + 0.5 * (1 - md_normalized)

    def _compute_distance_field(self) -> np.ndarray:
        """Compute distance from each voxel to tumor center/boundary."""
        # Create coordinate grids
        x = np.arange(self.grid_shape[0]) * self.voxel_size[0] + self.origin[0]
        y = np.arange(self.grid_shape[1]) * self.voxel_size[1] + self.origin[1]
        z = np.arange(self.grid_shape[2]) * self.voxel_size[2] + self.origin[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        if self.tumor.shape == "spherical":
            # Euclidean distance from tumor center
            distance = np.sqrt(
                (X - self.tumor.center[0])**2 +
                (Y - self.tumor.center[1])**2 +
                (Z - self.tumor.center[2])**2
            )
        else:
            # Ellipsoidal distance (scaled coordinates)
            a, b, c = self.tumor.semi_axes
            distance = np.sqrt(
                ((X - self.tumor.center[0]) / a)**2 +
                ((Y - self.tumor.center[1]) / b)**2 +
                ((Z - self.tumor.center[2]) / c)**2
            ) * self.tumor.radius

        return distance

    def _compute_direction_field(self) -> np.ndarray:
        """Compute unit direction vectors from tumor center to each voxel."""
        x = np.arange(self.grid_shape[0]) * self.voxel_size[0] + self.origin[0]
        y = np.arange(self.grid_shape[1]) * self.voxel_size[1] + self.origin[1]
        z = np.arange(self.grid_shape[2]) * self.voxel_size[2] + self.origin[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Direction vectors
        dx = X - self.tumor.center[0]
        dy = Y - self.tumor.center[1]
        dz = Z - self.tumor.center[2]

        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        magnitude = np.maximum(magnitude, 1e-10)  # Avoid division by zero

        directions = np.stack([dx/magnitude, dy/magnitude, dz/magnitude], axis=-1)
        return directions

    def _apply_tissue_modulation(
        self,
        displacement_magnitude: np.ndarray
    ) -> np.ndarray:
        """Apply tissue-specific displacement modulation."""
        if self._tissue_map is None or not self.params.tissue_modulation:
            return displacement_magnitude

        modulated = displacement_magnitude.copy()

        # Tissue-specific stiffness factors (relative to white matter = 1.0)
        tissue_stiffness = {
            TissueType.WHITE_MATTER.value: 1.0,
            TissueType.GRAY_MATTER.value: 0.8,
            TissueType.CSF.value: self.params.csf_damping,
            TissueType.VENTRICLE.value: self.params.csf_damping,
            TissueType.BRAINSTEM.value: 1.2,
            TissueType.CEREBELLUM.value: 0.9,
            TissueType.TUMOR.value: 0.0,  # No displacement inside tumor
        }

        for tissue_type, stiffness in tissue_stiffness.items():
            mask = self._tissue_map == tissue_type
            if isinstance(stiffness, (int, float)):
                # Higher stiffness = less displacement
                modulated[mask] *= (1.0 / max(stiffness, 0.1))

        return modulated

    def _apply_dti_modulation(
        self,
        displacement_magnitude: np.ndarray
    ) -> np.ndarray:
        """Apply DTI-derived stiffness modulation."""
        if self._dti_constraints is None or not self.params.use_dti_constraints:
            return displacement_magnitude

        # Inverse relationship: stiffer tissue = less displacement
        modulation = 1.0 / (0.5 + self._dti_constraints)
        return displacement_magnitude * modulation

    def _apply_boundary_conditions(
        self,
        displacement_magnitude: np.ndarray,
        directions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply anatomical boundary conditions."""
        if self._boundary_mask is None:
            return displacement_magnitude, directions

        # Reduce displacement near boundaries
        boundary_distance = self._compute_boundary_distance()
        boundary_factor = np.tanh(boundary_distance / 5.0)  # Smooth transition

        modulated_magnitude = displacement_magnitude * boundary_factor

        # Optionally redirect displacement parallel to boundaries
        # (simplified - full implementation would use boundary normals)

        return modulated_magnitude, directions

    def _compute_boundary_distance(self) -> np.ndarray:
        """Compute distance transform from boundary mask."""
        from scipy import ndimage

        if self._boundary_mask is None:
            return np.ones(self.grid_shape) * np.inf

        return ndimage.distance_transform_edt(
            ~self._boundary_mask.astype(bool),
            sampling=self.voxel_size
        )

    def compute_displacement_field(
        self,
        tumor_expansion: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute the displacement field caused by tumor expansion.

        Parameters
        ----------
        tumor_expansion : float, optional
            Amount of tumor expansion in mm. If None, uses tumor radius.

        Returns
        -------
        displacement_field : np.ndarray
            4D array of shape (nx, ny, nz, 3) containing displacement vectors
            in mm for each voxel.
        """
        expansion = tumor_expansion if tumor_expansion is not None else self.tumor.radius

        # Compute distance and direction fields
        distance = self._compute_distance_field()
        directions = self._compute_direction_field()

        # Create tumor mask (inside tumor)
        tumor_mask = distance < self.tumor.radius

        # Compute base displacement magnitude using geometric decay
        # Displacement is maximal at tumor boundary and decays with distance
        outside_tumor = distance >= self.tumor.radius

        displacement_magnitude = np.zeros(self.grid_shape)
        displacement_magnitude[outside_tumor] = (
            expansion *
            (self.tumor.radius / np.maximum(distance[outside_tumor], self.tumor.radius)) **
            self.params.decay_exponent
        )

        # Apply tissue modulation
        displacement_magnitude = self._apply_tissue_modulation(displacement_magnitude)

        # Apply DTI-derived constraints
        displacement_magnitude = self._apply_dti_modulation(displacement_magnitude)

        # Apply boundary conditions
        displacement_magnitude, directions = self._apply_boundary_conditions(
            displacement_magnitude, directions
        )

        # Clip to maximum displacement
        displacement_magnitude = np.clip(
            displacement_magnitude, 0, self.params.max_displacement
        )

        # Set displacement inside tumor to zero (tumor region moves as rigid body)
        displacement_magnitude[tumor_mask] = 0

        # Apply Gaussian smoothing for continuity
        if self.params.smoothing_sigma > 0:
            from scipy import ndimage
            displacement_magnitude = ndimage.gaussian_filter(
                displacement_magnitude,
                sigma=self.params.smoothing_sigma / np.array(self.voxel_size)
            )

        # Construct vector field
        self._displacement_field = directions * displacement_magnitude[..., np.newaxis]

        return self._displacement_field

    def compute_strain_field(self) -> np.ndarray:
        """
        Compute the strain tensor field from the displacement field.

        Returns
        -------
        strain_field : np.ndarray
            5D array of shape (nx, ny, nz, 3, 3) containing strain tensors
        """
        if self._displacement_field is None:
            self.compute_displacement_field()

        # Compute displacement gradients
        strain = np.zeros((*self.grid_shape, 3, 3))

        for i in range(3):
            for j in range(3):
                # Central differences for gradient
                gradient = np.gradient(
                    self._displacement_field[..., i],
                    self.voxel_size[j],
                    axis=j
                )
                # Strain tensor is symmetric part of displacement gradient
                strain[..., i, j] = gradient

        # Symmetrize: epsilon_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        strain = 0.5 * (strain + np.transpose(strain, (0, 1, 2, 4, 3)))

        return strain

    def compute_jacobian_determinant(self) -> np.ndarray:
        """
        Compute the Jacobian determinant of the deformation field.

        The Jacobian determinant indicates local volume change:
        - J > 1: expansion
        - J < 1: compression
        - J = 1: no volume change

        Returns
        -------
        jacobian : np.ndarray
            3D array of Jacobian determinant values
        """
        if self._displacement_field is None:
            self.compute_displacement_field()

        # Deformation gradient F = I + grad(u)
        F = np.zeros((*self.grid_shape, 3, 3))

        for i in range(3):
            F[..., i, i] = 1.0  # Identity
            for j in range(3):
                F[..., i, j] += np.gradient(
                    self._displacement_field[..., i],
                    self.voxel_size[j],
                    axis=j
                )

        # Compute determinant
        jacobian = np.linalg.det(F)

        return jacobian

    def get_displacement_at_point(
        self,
        point: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Get displacement vector at a specific point using interpolation.

        Parameters
        ----------
        point : tuple
            (x, y, z) coordinates in mm

        Returns
        -------
        displacement : np.ndarray
            3-element displacement vector in mm
        """
        if self._displacement_field is None:
            self.compute_displacement_field()

        from scipy import ndimage

        # Convert to voxel coordinates
        voxel_coords = [
            (point[i] - self.origin[i]) / self.voxel_size[i]
            for i in range(3)
        ]

        # Interpolate each component
        displacement = np.array([
            ndimage.map_coordinates(
                self._displacement_field[..., i],
                [[voxel_coords[0]], [voxel_coords[1]], [voxel_coords[2]]],
                order=1
            )[0]
            for i in range(3)
        ])

        return displacement

    def to_dict(self) -> Dict[str, Any]:
        """Export model configuration to dictionary."""
        return {
            "tumor": {
                "center": self.tumor.center,
                "radius": self.tumor.radius,
                "shape": self.tumor.shape,
                "semi_axes": self.tumor.semi_axes,
            },
            "parameters": {
                "decay_exponent": self.params.decay_exponent,
                "boundary_stiffness": self.params.boundary_stiffness,
                "csf_damping": self.params.csf_damping,
                "max_displacement": self.params.max_displacement,
            },
            "grid": {
                "shape": self.grid_shape,
                "voxel_size": self.voxel_size,
                "origin": self.origin,
            },
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GeometricExpansionModel":
        """Create model from configuration dictionary."""
        tumor = TumorParameters(
            center=tuple(config["tumor"]["center"]),
            radius=config["tumor"]["radius"],
            shape=config["tumor"].get("shape", "spherical"),
            semi_axes=config["tumor"].get("semi_axes"),
        )

        params = ModelParameters(
            decay_exponent=config["parameters"].get("decay_exponent", 2.0),
            boundary_stiffness=config["parameters"].get("boundary_stiffness", 0.8),
            csf_damping=config["parameters"].get("csf_damping", 0.1),
            max_displacement=config["parameters"].get("max_displacement", 15.0),
        )

        return cls(
            tumor_params=tumor,
            model_params=params,
            grid_shape=tuple(config["grid"]["shape"]),
            voxel_size=tuple(config["grid"]["voxel_size"]),
            origin=tuple(config["grid"].get("origin", (0, 0, 0))),
        )
