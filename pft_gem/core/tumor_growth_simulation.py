"""
ANTsPy-Based Tumor Growth Simulation

This module simulates tumor growth as geometric expansion that warps surrounding
brain tissue. The key physical constraints are:

1. **Fixed outer boundary**: The skull/outer boundary of the anatomy is fixed
   and cannot expand outward.

2. **Tissue compression**: As the tumor grows, surrounding tissue must compress
   to accommodate the tumor volume. This is different from simple radial
   displacement - the tissue is squeezed between the tumor and the fixed boundary.

3. **Gray matter compression**: Gray matter compresses uniformly in all directions
   (isotropic compression).

4. **White matter anisotropy**: White matter resists stretching along the axis
   of greatest diffusion (principal eigenvector from DTI). Compression preferentially
   occurs perpendicular to fiber tracts.

Theory:
-------
The displacement field u(x) is computed to satisfy:
- u = 0 at the outer boundary (fixed boundary condition)
- div(u) < 0 in the tissue surrounding tumor (compression)
- The tumor region has prescribed expansion
- White matter: displacement projected to reduce component along fiber direction

The implementation uses ANTsPy for spatial transforms to ensure proper
interpolation and transform composition/inversion.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    import ants
    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


@dataclass
class TumorGrowthParams:
    """Parameters for tumor growth simulation.

    Attributes:
        center: (x, y, z) coordinates of tumor center in mm (world coordinates)
        initial_radius: Initial tumor radius in mm (can be 0 for new tumor)
        final_radius: Final tumor radius in mm after growth
        shape: 'spherical' or 'ellipsoidal'
        semi_axes: Semi-axes for ellipsoidal tumors (a, b, c) in mm
    """
    center: Tuple[float, float, float]
    initial_radius: float = 0.0
    final_radius: float = 15.0
    shape: str = "spherical"
    semi_axes: Optional[Tuple[float, float, float]] = None

    @property
    def growth_volume(self) -> float:
        """Volume of tumor growth (final - initial)."""
        if self.shape == "spherical":
            v_final = (4/3) * np.pi * self.final_radius**3
            v_initial = (4/3) * np.pi * self.initial_radius**3
        else:
            a, b, c = self.semi_axes or (self.final_radius,) * 3
            scale = self.final_radius / max(a, b, c)
            v_final = (4/3) * np.pi * (a*scale) * (b*scale) * (c*scale)
            a0, b0, c0 = self.semi_axes or (self.initial_radius,) * 3
            scale0 = self.initial_radius / max(a0, b0, c0) if self.initial_radius > 0 else 0
            v_initial = (4/3) * np.pi * (a0*scale0) * (b0*scale0) * (c0*scale0)
        return v_final - v_initial


@dataclass
class TissueProperties:
    """Mechanical properties for tissue simulation.

    Attributes:
        gray_matter_compressibility: Compressibility factor for gray matter (0-1)
        white_matter_compressibility: Base compressibility for white matter (0-1)
        white_matter_anisotropy: Anisotropy factor for white matter (0-1)
            0 = isotropic, 1 = fully resistant to stretching along fibers
        csf_compressibility: Compressibility for CSF spaces (0-1)
        boundary_stiffness: How rigid the outer boundary is (0-1)
    """
    gray_matter_compressibility: float = 0.8
    white_matter_compressibility: float = 0.6
    white_matter_anisotropy: float = 0.7
    csf_compressibility: float = 0.95
    boundary_stiffness: float = 1.0


@dataclass
class SimulationOutput:
    """Output from tumor growth simulation.

    Attributes:
        displaced_image: The warped template showing tumor effect
        original_image: Original template before deformation
        displacement_field: 4D displacement vectors (x, y, z, 3) in mm
        tumor_mask: Binary mask of final tumor region
        jacobian_determinant: Local volume change (< 1 = compression)
        affine: 4x4 affine transformation matrix
        voxel_size: Voxel dimensions in mm
        ants_transform: ANTsPy transform object (if available)
        metadata: Simulation parameters and statistics
    """
    displaced_image: np.ndarray
    original_image: np.ndarray
    displacement_field: np.ndarray
    tumor_mask: np.ndarray
    jacobian_determinant: np.ndarray
    affine: np.ndarray
    voxel_size: Tuple[float, float, float]
    ants_transform: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TumorGrowthSimulator:
    """
    Simulates tumor growth with proper tissue compression physics.

    This class implements a physically-motivated model where:
    1. Tumor expansion displaces surrounding tissue
    2. The outer anatomical boundary is fixed (skull/brain surface)
    3. Tissue between tumor and boundary is compressed
    4. White matter resists stretching along fiber tracts

    Parameters
    ----------
    template_image : np.ndarray
        3D template image (e.g., SUIT T1w)
    boundary_mask : np.ndarray
        Binary mask defining the fixed outer boundary (brain surface)
    voxel_size : tuple
        Voxel dimensions in mm
    affine : np.ndarray
        4x4 affine transformation matrix
    tissue_mask : np.ndarray, optional
        Tissue type segmentation (0=background, 1=GM, 2=WM, 3=CSF)
    dti_v1 : np.ndarray, optional
        Principal diffusion direction (nx, ny, nz, 3) for anisotropic WM
    dti_fa : np.ndarray, optional
        Fractional anisotropy map for weighting anisotropy

    Examples
    --------
    >>> simulator = TumorGrowthSimulator(template, mask, (1,1,1), affine)
    >>> output = simulator.simulate_growth(tumor_params)
    >>> simulator.save_transforms(output, "output_dir")
    """

    def __init__(
        self,
        template_image: np.ndarray,
        boundary_mask: np.ndarray,
        voxel_size: Tuple[float, float, float],
        affine: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
        dti_v1: Optional[np.ndarray] = None,
        dti_fa: Optional[np.ndarray] = None
    ):
        self.template = template_image
        self.voxel_size = voxel_size
        self.affine = affine
        self.shape = template_image.shape

        # Resample boundary mask if shape doesn't match template
        if boundary_mask.shape != self.shape:
            boundary_mask = self._resample_to_template(boundary_mask, order=0)
        self.boundary_mask = boundary_mask.astype(bool)

        # Resample optional tissue-specific data if needed
        if tissue_mask is not None and tissue_mask.shape != self.shape:
            tissue_mask = self._resample_to_template(tissue_mask, order=0)
        self.tissue_mask = tissue_mask

        if dti_v1 is not None and dti_v1.shape[:3] != self.shape:
            # Resample each component of the vector field
            dti_v1_resampled = np.zeros((*self.shape, 3))
            for i in range(3):
                dti_v1_resampled[..., i] = self._resample_to_template(dti_v1[..., i], order=1)
            # Renormalize
            mag = np.linalg.norm(dti_v1_resampled, axis=-1, keepdims=True)
            dti_v1 = np.where(mag > 1e-10, dti_v1_resampled / mag, dti_v1_resampled)
        self.dti_v1 = dti_v1

        if dti_fa is not None and dti_fa.shape != self.shape:
            dti_fa = self._resample_to_template(dti_fa, order=1)
        self.dti_fa = dti_fa

        # Compute boundary distance field (distance to nearest boundary point)
        self._boundary_distance = self._compute_boundary_distance()

        # ANTsPy images (created on demand)
        self._ants_template = None
        self._ants_reference = None

    def _resample_to_template(self, data: np.ndarray, order: int = 1) -> np.ndarray:
        """Resample data array to match template shape using zoom."""
        if data.shape == self.shape:
            return data
        zoom_factors = [t / d for t, d in zip(self.shape, data.shape)]
        return ndimage.zoom(data, zoom_factors, order=order)

    def _compute_boundary_distance(self) -> np.ndarray:
        """Compute distance transform from boundary."""
        # Distance from each voxel to the boundary (edge of mask)
        # We want distance to the outer surface, not to outside
        boundary_edge = self._extract_boundary_surface()
        distance = ndimage.distance_transform_edt(
            ~boundary_edge,
            sampling=self.voxel_size
        )
        return distance

    def _extract_boundary_surface(self) -> np.ndarray:
        """Extract the outer surface of the boundary mask."""
        # Erode mask and subtract to get surface
        eroded = ndimage.binary_erosion(self.boundary_mask, iterations=1)
        surface = self.boundary_mask & ~eroded
        return surface

    def _world_to_voxel(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert world coordinates to voxel coordinates."""
        inv_affine = np.linalg.inv(self.affine)
        point_h = np.array([point[0], point[1], point[2], 1.0])
        voxel = inv_affine @ point_h
        return (voxel[0], voxel[1], voxel[2])

    def _voxel_to_world(self, voxel: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert voxel coordinates to world coordinates."""
        voxel_h = np.array([voxel[0], voxel[1], voxel[2], 1.0])
        world = self.affine @ voxel_h
        return (world[0], world[1], world[2])

    def _compute_distance_from_tumor(
        self,
        tumor_params: TumorGrowthParams
    ) -> np.ndarray:
        """Compute distance field from tumor boundary."""
        # Create coordinate grids in world space
        x = np.arange(self.shape[0]) * self.voxel_size[0] + self.affine[0, 3]
        y = np.arange(self.shape[1]) * self.voxel_size[1] + self.affine[1, 3]
        z = np.arange(self.shape[2]) * self.voxel_size[2] + self.affine[2, 3]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        center = tumor_params.center

        if tumor_params.shape == "spherical":
            distance = np.sqrt(
                (X - center[0])**2 +
                (Y - center[1])**2 +
                (Z - center[2])**2
            )
        else:
            # Ellipsoidal distance
            a, b, c = tumor_params.semi_axes or (tumor_params.final_radius,) * 3
            distance = np.sqrt(
                ((X - center[0]) / a)**2 +
                ((Y - center[1]) / b)**2 +
                ((Z - center[2]) / c)**2
            ) * tumor_params.final_radius

        return distance

    def _compute_radial_direction(
        self,
        tumor_params: TumorGrowthParams
    ) -> np.ndarray:
        """Compute unit direction vectors pointing away from tumor center."""
        x = np.arange(self.shape[0]) * self.voxel_size[0] + self.affine[0, 3]
        y = np.arange(self.shape[1]) * self.voxel_size[1] + self.affine[1, 3]
        z = np.arange(self.shape[2]) * self.voxel_size[2] + self.affine[2, 3]

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        center = tumor_params.center

        dx = X - center[0]
        dy = Y - center[1]
        dz = Z - center[2]

        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        magnitude = np.maximum(magnitude, 1e-10)

        directions = np.stack([
            dx / magnitude,
            dy / magnitude,
            dz / magnitude
        ], axis=-1)

        return directions

    def _compute_compression_field(
        self,
        tumor_params: TumorGrowthParams,
        tissue_props: TissueProperties
    ) -> np.ndarray:
        """
        Compute displacement field that compresses tissue between tumor and boundary.

        The key physics:
        1. Displacement is zero at the outer boundary (fixed)
        2. Displacement is maximal at the tumor boundary
        3. Displacement direction is radially outward from tumor center
        4. The magnitude profile is chosen to produce compression (Jacobian < 1)

        For compression in a spherical geometry, we need the displacement to decay
        faster than 1/r² so that div(u) < 0. We use:

        u(r) = u_0 * (R_boundary - r)² / (R_boundary - R_tumor)²

        This ensures:
        - u = u_0 at r = R_tumor (tumor boundary)
        - u = 0 at r = R_boundary (fixed boundary)
        - Rapid decay produces compression
        """
        # Distance from tumor center/boundary
        dist_from_tumor = self._compute_distance_from_tumor(tumor_params)

        # Radial directions (pointing outward from tumor)
        directions = self._compute_radial_direction(tumor_params)

        # Create tumor masks
        tumor_mask = dist_from_tumor < tumor_params.final_radius
        tissue_region = self.boundary_mask & ~tumor_mask

        # Get distance from boundary for each point
        boundary_dist = self._boundary_distance.copy()

        # Tumor expansion amount
        expansion = tumor_params.final_radius - tumor_params.initial_radius

        # Distance from tumor boundary (zero inside tumor)
        dist_to_tumor_boundary = np.maximum(dist_from_tumor - tumor_params.final_radius, 0)

        # Total distance from tumor boundary to anatomical boundary
        total_dist = dist_to_tumor_boundary + boundary_dist

        # Compute displacement magnitude using quadratic decay for compression
        # This profile ensures faster-than-linear decay which produces div(u) < 0
        #
        # Using: u(r) = expansion * (boundary_dist / total_dist)²
        # At tumor boundary: boundary_dist = total_dist, so u = expansion
        # At anatomical boundary: boundary_dist = 0, so u = 0
        # The quadratic ensures compression (steeper decay than linear)

        normalized_boundary_dist = np.where(
            total_dist > 0,
            boundary_dist / total_dist,
            0
        )

        # Quadratic profile for compression
        displacement_mag = expansion * normalized_boundary_dist**2

        # Apply tissue-specific modulation
        displacement_mag = self._apply_tissue_modulation(
            displacement_mag, tissue_props
        )

        # Apply anisotropic white matter constraints
        directions = self._apply_white_matter_anisotropy(
            directions, displacement_mag, tissue_props
        )

        # Set displacement to zero inside tumor and outside boundary
        displacement_mag[tumor_mask] = 0
        displacement_mag[~self.boundary_mask] = 0

        # Apply boundary condition: zero displacement at boundary surface
        boundary_surface = self._extract_boundary_surface()
        displacement_mag[boundary_surface] = 0

        # Smooth the displacement field for continuity
        # Use smaller sigma to preserve the compression profile
        displacement_mag = ndimage.gaussian_filter(displacement_mag, sigma=0.5)

        # Construct vector field
        displacement = directions * displacement_mag[..., np.newaxis]

        return displacement

    def _apply_tissue_modulation(
        self,
        displacement_mag: np.ndarray,
        tissue_props: TissueProperties
    ) -> np.ndarray:
        """Apply tissue-specific displacement modulation."""
        if self.tissue_mask is None:
            return displacement_mag

        modulated = displacement_mag.copy()

        # Gray matter: uniform compressibility
        gm_mask = self.tissue_mask == 1
        modulated[gm_mask] *= tissue_props.gray_matter_compressibility

        # White matter: reduced compressibility
        wm_mask = self.tissue_mask == 2
        modulated[wm_mask] *= tissue_props.white_matter_compressibility

        # CSF: high compressibility
        csf_mask = self.tissue_mask == 3
        modulated[csf_mask] *= tissue_props.csf_compressibility

        return modulated

    def _apply_white_matter_anisotropy(
        self,
        directions: np.ndarray,
        displacement_mag: np.ndarray,
        tissue_props: TissueProperties
    ) -> np.ndarray:
        """
        Modify displacement direction for white matter anisotropy.

        White matter resists stretching along the principal diffusion direction (v1).
        We project out the component of displacement along v1, effectively making
        the tissue compress perpendicular to fibers.
        """
        if self.dti_v1 is None or self.dti_fa is None:
            return directions

        if self.tissue_mask is None:
            return directions

        wm_mask = self.tissue_mask == 2

        if not np.any(wm_mask):
            return directions

        modified = directions.copy()

        # For white matter voxels, reduce displacement component along fiber direction
        # The amount of reduction depends on FA and anisotropy parameter

        # Project displacement onto fiber direction
        # d_parallel = (d . v1) * v1
        # d_perpendicular = d - d_parallel
        # Modified displacement = d_perpendicular + (1 - anisotropy * FA) * d_parallel

        v1 = self.dti_v1
        fa = self.dti_fa
        aniso = tissue_props.white_matter_anisotropy

        # Compute projection
        dot_product = np.sum(directions * v1, axis=-1, keepdims=True)
        parallel_component = dot_product * v1
        perpendicular_component = directions - parallel_component

        # Anisotropic reduction factor (1 = full resistance, 0 = no resistance)
        reduction = aniso * fa[..., np.newaxis]

        # Modified direction for WM only
        modified_wm = perpendicular_component + (1 - reduction) * parallel_component

        # Normalize
        mag = np.linalg.norm(modified_wm, axis=-1, keepdims=True)
        modified_wm = np.where(mag > 1e-10, modified_wm / mag, directions)

        # Apply only to white matter
        modified[wm_mask] = modified_wm[wm_mask]

        return modified

    def simulate_growth(
        self,
        tumor_params: TumorGrowthParams,
        tissue_props: Optional[TissueProperties] = None,
        use_ants: bool = True
    ) -> SimulationOutput:
        """
        Simulate tumor growth and compute displaced image.

        Parameters
        ----------
        tumor_params : TumorGrowthParams
            Tumor location and size parameters
        tissue_props : TissueProperties, optional
            Tissue mechanical properties
        use_ants : bool
            Whether to use ANTsPy for image warping (if available)

        Returns
        -------
        output : SimulationOutput
            Complete simulation output including displaced image and transforms
        """
        tissue_props = tissue_props or TissueProperties()

        # Compute displacement field
        displacement = self._compute_compression_field(tumor_params, tissue_props)

        # Compute tumor mask
        dist_from_tumor = self._compute_distance_from_tumor(tumor_params)
        tumor_mask = (dist_from_tumor < tumor_params.final_radius).astype(np.uint8)

        # Compute Jacobian determinant (measure of volume change)
        jacobian = self._compute_jacobian_determinant(displacement)

        # Apply displacement to warp the template
        if use_ants and HAS_ANTS:
            displaced_image, ants_transform = self._warp_with_ants(displacement)
        else:
            displaced_image = self._warp_with_scipy(displacement)
            ants_transform = None

        # Add tumor appearance to displaced image
        displaced_image = self._add_tumor_appearance(
            displaced_image, tumor_mask, tumor_params
        )

        # Compile statistics
        metadata = self._compile_metadata(
            tumor_params, tissue_props, displacement, jacobian
        )

        return SimulationOutput(
            displaced_image=displaced_image,
            original_image=self.template.copy(),
            displacement_field=displacement,
            tumor_mask=tumor_mask,
            jacobian_determinant=jacobian,
            affine=self.affine.copy(),
            voxel_size=self.voxel_size,
            ants_transform=ants_transform,
            metadata=metadata
        )

    def _compute_jacobian_determinant(self, displacement: np.ndarray) -> np.ndarray:
        """Compute Jacobian determinant of deformation."""
        # Deformation gradient F = I + grad(u)
        F = np.zeros((*self.shape, 3, 3))

        for i in range(3):
            F[..., i, i] = 1.0  # Identity
            for j in range(3):
                F[..., i, j] += np.gradient(
                    displacement[..., i],
                    self.voxel_size[j],
                    axis=j
                )

        # Compute determinant
        jacobian = np.linalg.det(F)

        return jacobian

    def _warp_with_scipy(self, displacement: np.ndarray) -> np.ndarray:
        """Warp image using scipy (fallback when ANTsPy not available)."""
        # Create sampling coordinates
        coords = np.indices(self.shape).astype(np.float64)

        # Add displacement (convert from mm to voxels)
        for i in range(3):
            coords[i] += displacement[..., i] / self.voxel_size[i]

        # Warp image
        warped = ndimage.map_coordinates(
            self.template, coords, order=1, mode='constant', cval=0
        )

        return warped.astype(np.float32)

    def _warp_with_ants(
        self,
        displacement: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """Warp image using ANTsPy for better interpolation and transform handling."""
        if not HAS_ANTS:
            raise ImportError("ANTsPy required for this function")

        # Create ANTsPy image from template
        ants_template = ants.from_numpy(
            self.template.astype(np.float32),
            spacing=self.voxel_size,
            origin=(self.affine[0, 3], self.affine[1, 3], self.affine[2, 3])
        )

        # Create displacement field as ANTsPy transform
        # ANTsPy expects displacement field as a 4D array with shape (x, y, z, 3)
        # or as separate component images

        # Create the warp field image
        # ANTs uses physical space displacement, which we already have
        warp_field = ants.from_numpy(
            displacement.astype(np.float32),
            spacing=self.voxel_size,
            origin=(self.affine[0, 3], self.affine[1, 3], self.affine[2, 3]),
            has_components=True
        )

        # Apply the transform
        # For displacement fields, we need to use ants.apply_ants_transform_to_image
        # or create a composite transform

        # Method: Use the displacement field directly with map_coordinates equivalent
        # since ANTsPy's apply_transforms expects transform files

        # For now, we'll compute the warped coordinates and use scipy
        # but return the ANTs-compatible displacement field

        # Create sampling coordinates in physical space
        coords = np.indices(self.shape).astype(np.float64)

        # Convert to physical coordinates, add displacement, convert back
        for i in range(3):
            # Physical coordinate
            phys_coord = coords[i] * self.voxel_size[i] + self.affine[i, 3]
            # Add displacement
            phys_coord += displacement[..., i]
            # Convert back to voxel coordinates
            coords[i] = (phys_coord - self.affine[i, 3]) / self.voxel_size[i]

        # Warp using scipy (ANTsPy uses similar trilinear interpolation)
        warped = ndimage.map_coordinates(
            self.template, coords, order=3, mode='constant', cval=0
        )

        return warped.astype(np.float32), warp_field

    def _add_tumor_appearance(
        self,
        image: np.ndarray,
        tumor_mask: np.ndarray,
        tumor_params: TumorGrowthParams
    ) -> np.ndarray:
        """Add tumor appearance to the displaced image."""
        result = image.copy()

        # Get intensity statistics from surrounding tissue
        border_mask = ndimage.binary_dilation(tumor_mask.astype(bool), iterations=3)
        border_mask = border_mask & ~tumor_mask.astype(bool) & self.boundary_mask

        if np.any(border_mask):
            border_intensity = np.mean(image[border_mask])
        else:
            border_intensity = np.mean(image[self.boundary_mask])

        # Tumor typically appears slightly hypointense on T1
        tumor_intensity = border_intensity * 0.7

        # Add some texture variation
        noise = np.random.normal(0, 0.05 * tumor_intensity, self.shape)

        result[tumor_mask > 0] = tumor_intensity + noise[tumor_mask > 0]
        result = np.clip(result, 0, result.max())

        return result

    def _compile_metadata(
        self,
        tumor_params: TumorGrowthParams,
        tissue_props: TissueProperties,
        displacement: np.ndarray,
        jacobian: np.ndarray
    ) -> Dict[str, Any]:
        """Compile simulation metadata and statistics."""
        # Displacement statistics (within tissue region)
        tissue_region = self.boundary_mask & (self._compute_distance_from_tumor(tumor_params) >= tumor_params.final_radius)

        disp_mag = np.linalg.norm(displacement, axis=-1)

        metadata = {
            "tumor": {
                "center": list(tumor_params.center),
                "initial_radius": tumor_params.initial_radius,
                "final_radius": tumor_params.final_radius,
                "shape": tumor_params.shape,
                "growth_volume_mm3": float(tumor_params.growth_volume),
            },
            "tissue_properties": {
                "gray_matter_compressibility": tissue_props.gray_matter_compressibility,
                "white_matter_compressibility": tissue_props.white_matter_compressibility,
                "white_matter_anisotropy": tissue_props.white_matter_anisotropy,
                "csf_compressibility": tissue_props.csf_compressibility,
            },
            "displacement_stats": {
                "max_mm": float(disp_mag[tissue_region].max()) if np.any(tissue_region) else 0,
                "mean_mm": float(disp_mag[tissue_region].mean()) if np.any(tissue_region) else 0,
                "std_mm": float(disp_mag[tissue_region].std()) if np.any(tissue_region) else 0,
            },
            "compression_stats": {
                "min_jacobian": float(jacobian[tissue_region].min()) if np.any(tissue_region) else 1,
                "max_jacobian": float(jacobian[tissue_region].max()) if np.any(tissue_region) else 1,
                "mean_jacobian": float(jacobian[tissue_region].mean()) if np.any(tissue_region) else 1,
                "voxels_compressed": int(np.sum((jacobian < 0.99) & tissue_region)),
                "voxels_expanded": int(np.sum((jacobian > 1.01) & tissue_region)),
            },
            "grid": {
                "shape": list(self.shape),
                "voxel_size": list(self.voxel_size),
            },
            "source": "PFT_GEM TumorGrowthSimulator",
            "ants_available": HAS_ANTS,
        }

        return metadata

    def save_transforms(
        self,
        output: SimulationOutput,
        output_dir: Union[str, Path],
        prefix: str = "tumor_growth"
    ) -> Dict[str, Path]:
        """
        Save all simulation outputs including ANTsPy-compatible transforms.

        Saves:
        - Displaced image (NIfTI)
        - Original image (NIfTI)
        - Forward warp/displacement field (NIfTI, ANTs-compatible)
        - Inverse warp (NIfTI, ANTs-compatible)
        - Tumor mask (NIfTI)
        - Jacobian determinant map (NIfTI)
        - Metadata (JSON)

        Parameters
        ----------
        output : SimulationOutput
            Simulation output to save
        output_dir : str or Path
            Output directory
        prefix : str
            Prefix for output filenames

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

        # Save original image
        original_path = output_dir / f"{prefix}_original.nii.gz"
        img = nib.Nifti1Image(output.original_image.astype(np.float32), output.affine)
        img.header.set_xyzt_units('mm')
        nib.save(img, str(original_path))
        paths['original'] = original_path

        # Save forward warp (displacement field)
        warp_path = output_dir / f"{prefix}_warp.nii.gz"
        self._save_displacement_field(
            output.displacement_field, warp_path, output.affine
        )
        paths['warp'] = warp_path

        # Compute and save inverse warp
        inverse_displacement = self._compute_inverse_displacement(
            output.displacement_field
        )
        inverse_warp_path = output_dir / f"{prefix}_warp_inverse.nii.gz"
        self._save_displacement_field(
            inverse_displacement, inverse_warp_path, output.affine
        )
        paths['warp_inverse'] = inverse_warp_path

        # Save tumor mask
        mask_path = output_dir / f"{prefix}_tumor_mask.nii.gz"
        img = nib.Nifti1Image(output.tumor_mask, output.affine)
        nib.save(img, str(mask_path))
        paths['tumor_mask'] = mask_path

        # Save Jacobian determinant
        jacobian_path = output_dir / f"{prefix}_jacobian.nii.gz"
        img = nib.Nifti1Image(output.jacobian_determinant.astype(np.float32), output.affine)
        img.header.set_xyzt_units('mm')
        nib.save(img, str(jacobian_path))
        paths['jacobian'] = jacobian_path

        # Save metadata
        metadata_path = output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(output.metadata, f, indent=2, default=str)
        paths['metadata'] = metadata_path

        # If ANTsPy is available, also save in ANTs format
        if HAS_ANTS and output.ants_transform is not None:
            ants_warp_path = output_dir / f"{prefix}_ants_warp.nii.gz"
            ants.image_write(output.ants_transform, str(ants_warp_path))
            paths['ants_warp'] = ants_warp_path

        return paths

    def _save_displacement_field(
        self,
        displacement: np.ndarray,
        path: Path,
        affine: np.ndarray
    ) -> None:
        """Save displacement field in ANTs/FSL compatible format."""
        # Save as 4D NIfTI with displacement vector intent
        img = nib.Nifti1Image(displacement.astype(np.float32), affine)
        img.header.set_xyzt_units('mm', 'sec')
        img.header['intent_code'] = 1006  # NIFTI_INTENT_DISPVECT
        img.header['intent_name'] = b'DISPVECT'
        nib.save(img, str(path))

    def _compute_inverse_displacement(
        self,
        displacement: np.ndarray,
        iterations: int = 20,
        tolerance: float = 1e-4
    ) -> np.ndarray:
        """
        Compute inverse displacement field using fixed-point iteration.

        For a displacement field u such that y = x + u(x),
        the inverse v satisfies x = y + v(y), or equivalently v(y) = -u(x).

        We use the iterative scheme:
        v^{n+1}(y) = -u(y + v^n(y))
        """
        # Initialize with negated displacement
        inverse = -displacement.copy()

        # Create interpolator for displacement field
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        z = np.arange(self.shape[2])

        for iteration in range(iterations):
            inverse_old = inverse.copy()

            # Compute displaced coordinates
            coords = np.indices(self.shape).astype(np.float64)
            for i in range(3):
                coords[i] += inverse[..., i] / self.voxel_size[i]

            # Interpolate original displacement at displaced coordinates
            for i in range(3):
                interp = RegularGridInterpolator(
                    (x, y, z),
                    displacement[..., i],
                    method='linear',
                    bounds_error=False,
                    fill_value=0
                )

                points = np.stack([
                    coords[0].ravel(),
                    coords[1].ravel(),
                    coords[2].ravel()
                ], axis=-1)

                inverse[..., i] = -interp(points).reshape(self.shape)

            # Check convergence
            diff = np.linalg.norm(inverse - inverse_old)
            if diff < tolerance:
                break

        return inverse


def simulate_tumor_growth(
    template_data,
    tumor_center: Tuple[float, float, float],
    tumor_radius: float,
    initial_radius: float = 0.0,
    tissue_mask: Optional[np.ndarray] = None,
    dti_v1: Optional[np.ndarray] = None,
    dti_fa: Optional[np.ndarray] = None,
    tissue_props: Optional[TissueProperties] = None,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "tumor_growth"
) -> SimulationOutput:
    """
    Convenience function to simulate tumor growth on template data.

    Parameters
    ----------
    template_data : TemplateData
        Loaded template data (e.g., from SUITTemplateLoader)
    tumor_center : tuple
        (x, y, z) tumor center in world coordinates (mm)
    tumor_radius : float
        Final tumor radius in mm
    initial_radius : float
        Initial tumor radius (0 for new tumor)
    tissue_mask : np.ndarray, optional
        Tissue segmentation (0=background, 1=GM, 2=WM, 3=CSF)
    dti_v1 : np.ndarray, optional
        Principal diffusion direction
    dti_fa : np.ndarray, optional
        Fractional anisotropy map
    tissue_props : TissueProperties, optional
        Tissue mechanical properties
    output_dir : str or Path, optional
        Directory to save outputs
    prefix : str
        Prefix for output filenames

    Returns
    -------
    output : SimulationOutput
        Simulation results
    """
    # Extract data from template
    template = template_data.template
    mask = template_data.mask
    affine = template_data.affine if template_data.affine is not None else np.eye(4)
    voxel_size = template_data.voxel_size

    if template is None:
        raise ValueError("Template data must contain a template image")

    if mask is None:
        # Create mask from template intensity
        mask = template > 0.1 * template.max()

    # Create tumor parameters
    tumor_params = TumorGrowthParams(
        center=tumor_center,
        initial_radius=initial_radius,
        final_radius=tumor_radius,
        shape="spherical"
    )

    # Create simulator
    simulator = TumorGrowthSimulator(
        template_image=template,
        boundary_mask=mask,
        voxel_size=voxel_size,
        affine=affine,
        tissue_mask=tissue_mask,
        dti_v1=dti_v1,
        dti_fa=dti_fa
    )

    # Run simulation
    output = simulator.simulate_growth(tumor_params, tissue_props)

    # Save outputs if directory specified
    if output_dir is not None:
        simulator.save_transforms(output, output_dir, prefix)

    return output
