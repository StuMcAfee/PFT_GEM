#!/usr/bin/env python3
"""
Visualization Example for PFT_GEM

This script demonstrates the visualization capabilities of PFT_GEM
for analyzing tumor-induced displacement fields.

Requires matplotlib: pip install matplotlib
"""

import numpy as np
import sys
sys.path.insert(0, '..')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

from pft_gem import GeometricExpansionModel
from pft_gem.core.geometric_model import TumorParameters, ModelParameters
from pft_gem.core.displacement import DisplacementField, FieldMetadata
from pft_gem.core.constraints import create_synthetic_dti, BiophysicalConstraints
from pft_gem.io import SUITTemplateLoader
from pft_gem.utils import create_spherical_tumor_mask


def main():
    if not HAS_MATPLOTLIB:
        print("This example requires matplotlib for visualization.")
        print("Please install it with: pip install matplotlib")
        return

    print("PFT_GEM: Visualization Example")
    print("=" * 50)

    # Setup
    grid_shape = (128, 128, 64)
    voxel_size = (1.0, 1.0, 1.0)

    tumor_params = TumorParameters(
        center=(64.0, 64.0, 32.0),
        radius=15.0,
        shape="spherical"
    )

    model_params = ModelParameters(
        decay_exponent=2.0,
        max_displacement=15.0,
        use_dti_constraints=True
    )

    # Create model
    model = GeometricExpansionModel(
        tumor_params=tumor_params,
        model_params=model_params,
        grid_shape=grid_shape,
        voxel_size=voxel_size
    )

    # Create synthetic data
    dti_data = create_synthetic_dti(
        shape=grid_shape,
        tumor_center=(64, 64, 32),
        tumor_radius=15.0
    )
    model.set_dti_constraints(dti_data.fa, dti_data.md)

    # Compute displacement
    print("Computing displacement field...")
    displacement = model.compute_displacement_field(tumor_expansion=5.0)
    magnitude = np.linalg.norm(displacement, axis=-1)

    # Create tumor mask
    tumor_mask = create_spherical_tumor_mask(
        grid_shape,
        tumor_params.center,
        tumor_params.radius
    )

    # Create synthetic template
    template_data = SUITTemplateLoader.create_synthetic_template(grid_shape)

    # -------------------------------------------------------------------------
    # Figure 1: Displacement magnitude in three views
    # -------------------------------------------------------------------------
    print("Creating visualization 1: Three orthogonal views...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slice_x = int(tumor_params.center[0])
    slice_y = int(tumor_params.center[1])
    slice_z = int(tumor_params.center[2])

    # Sagittal
    im0 = axes[0].imshow(magnitude[slice_x, :, :].T, origin='lower', cmap='hot')
    axes[0].contour(tumor_mask[slice_x, :, :].T, levels=[0.5], colors='cyan', linewidths=2)
    axes[0].set_title(f'Sagittal (x={slice_x})')
    axes[0].set_xlabel('Y (voxels)')
    axes[0].set_ylabel('Z (voxels)')

    # Coronal
    im1 = axes[1].imshow(magnitude[:, slice_y, :].T, origin='lower', cmap='hot')
    axes[1].contour(tumor_mask[:, slice_y, :].T, levels=[0.5], colors='cyan', linewidths=2)
    axes[1].set_title(f'Coronal (y={slice_y})')
    axes[1].set_xlabel('X (voxels)')
    axes[1].set_ylabel('Z (voxels)')

    # Axial
    im2 = axes[2].imshow(magnitude[:, :, slice_z].T, origin='lower', cmap='hot')
    axes[2].contour(tumor_mask[:, :, slice_z].T, levels=[0.5], colors='cyan', linewidths=2)
    axes[2].set_title(f'Axial (z={slice_z})')
    axes[2].set_xlabel('X (voxels)')
    axes[2].set_ylabel('Y (voxels)')

    fig.colorbar(im2, ax=axes, label='Displacement (mm)', shrink=0.8)
    fig.suptitle('Displacement Magnitude (tumor boundary in cyan)', fontsize=14)
    plt.tight_layout()
    plt.savefig('displacement_views.png', dpi=150, bbox_inches='tight')
    print("  Saved: displacement_views.png")

    # -------------------------------------------------------------------------
    # Figure 2: Vector field
    # -------------------------------------------------------------------------
    print("Creating visualization 2: Vector field...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Subsample for visualization
    step = 6
    y_range = np.arange(0, grid_shape[1], step)
    x_range = np.arange(0, grid_shape[0], step)
    Y, X = np.meshgrid(y_range, x_range)

    u = displacement[::step, ::step, slice_z, 0]
    v = displacement[::step, ::step, slice_z, 1]
    colors = np.sqrt(u**2 + v**2)

    # Background
    ax.imshow(template_data.template[:, :, slice_z].T, origin='lower', cmap='gray', alpha=0.5)

    # Vectors
    q = ax.quiver(X, Y, u.T, v.T, colors.T, cmap='hot', scale=50)
    ax.contour(tumor_mask[:, :, slice_z].T, levels=[0.5], colors='cyan', linewidths=2)

    ax.set_title(f'Displacement Vectors (axial slice z={slice_z})')
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    plt.colorbar(q, ax=ax, label='Magnitude (mm)')
    plt.savefig('displacement_vectors.png', dpi=150, bbox_inches='tight')
    print("  Saved: displacement_vectors.png")

    # -------------------------------------------------------------------------
    # Figure 3: Jacobian (volume change)
    # -------------------------------------------------------------------------
    print("Creating visualization 3: Jacobian determinant...")

    jacobian = model.compute_jacobian_determinant()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Jacobian map
    vmax = max(abs(jacobian.max() - 1), abs(jacobian.min() - 1))
    im0 = axes[0].imshow(jacobian[:, :, slice_z].T, origin='lower',
                         cmap='RdBu_r', vmin=1-vmax, vmax=1+vmax)
    axes[0].contour(tumor_mask[:, :, slice_z].T, levels=[0.5], colors='black', linewidths=2)
    axes[0].set_title('Jacobian Determinant')
    axes[0].set_xlabel('X (voxels)')
    axes[0].set_ylabel('Y (voxels)')
    plt.colorbar(im0, ax=axes[0], label='J (1=no change)')

    # Histogram
    j_outside = jacobian[tumor_mask == 0].flatten()
    axes[1].hist(j_outside, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[1].axvline(x=1.0, color='red', linestyle='--', label='J=1 (no change)')
    axes[1].set_xlabel('Jacobian Determinant')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution (outside tumor)')
    axes[1].legend()

    fig.suptitle('Volume Change Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('jacobian_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: jacobian_analysis.png")

    # -------------------------------------------------------------------------
    # Figure 4: Radial displacement profile
    # -------------------------------------------------------------------------
    print("Creating visualization 4: Radial profile...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute radial profile
    cx, cy, cz = int(tumor_params.center[0]), int(tumor_params.center[1]), int(tumor_params.center[2])

    # Profile along x-axis
    x_profile = magnitude[:, cy, cz]
    distances_x = np.arange(len(x_profile)) - cx

    # Profile along y-axis
    y_profile = magnitude[cx, :, cz]
    distances_y = np.arange(len(y_profile)) - cy

    # Profile along z-axis
    z_profile = magnitude[cx, cy, :]
    distances_z = np.arange(len(z_profile)) - cz

    ax.plot(distances_x, x_profile, 'r-', linewidth=2, label='X direction')
    ax.plot(distances_y, y_profile, 'g-', linewidth=2, label='Y direction')
    ax.plot(distances_z, z_profile, 'b-', linewidth=2, label='Z direction')

    ax.axvline(x=-tumor_params.radius, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=tumor_params.radius, color='gray', linestyle='--', alpha=0.5, label='Tumor boundary')
    ax.axvspan(-tumor_params.radius, tumor_params.radius, alpha=0.1, color='gray')

    ax.set_xlabel('Distance from tumor center (voxels)')
    ax.set_ylabel('Displacement magnitude (mm)')
    ax.set_title('Displacement Profile Through Tumor Center')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-60, 60)

    plt.savefig('displacement_profile.png', dpi=150, bbox_inches='tight')
    print("  Saved: displacement_profile.png")

    # -------------------------------------------------------------------------
    # Figure 5: DTI constraints effect
    # -------------------------------------------------------------------------
    print("Creating visualization 5: DTI constraint effect...")

    # Compute without DTI
    model_no_dti = GeometricExpansionModel(
        tumor_params=tumor_params,
        model_params=ModelParameters(use_dti_constraints=False),
        grid_shape=grid_shape,
        voxel_size=voxel_size
    )
    disp_no_dti = model_no_dti.compute_displacement_field(tumor_expansion=5.0)
    mag_no_dti = np.linalg.norm(disp_no_dti, axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = max(magnitude.max(), mag_no_dti.max())

    # Without DTI
    im0 = axes[0].imshow(mag_no_dti[:, :, slice_z].T, origin='lower', cmap='hot', vmax=vmax)
    axes[0].contour(tumor_mask[:, :, slice_z].T, levels=[0.5], colors='cyan', linewidths=2)
    axes[0].set_title('Without DTI Constraints')
    plt.colorbar(im0, ax=axes[0], label='mm')

    # With DTI
    im1 = axes[1].imshow(magnitude[:, :, slice_z].T, origin='lower', cmap='hot', vmax=vmax)
    axes[1].contour(tumor_mask[:, :, slice_z].T, levels=[0.5], colors='cyan', linewidths=2)
    axes[1].set_title('With DTI Constraints')
    plt.colorbar(im1, ax=axes[1], label='mm')

    # Difference
    diff = magnitude - mag_no_dti
    vmax_diff = np.abs(diff).max()
    im2 = axes[2].imshow(diff[:, :, slice_z].T, origin='lower', cmap='RdBu_r',
                         vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].contour(tumor_mask[:, :, slice_z].T, levels=[0.5], colors='black', linewidths=2)
    axes[2].set_title('Difference (with - without)')
    plt.colorbar(im2, ax=axes[2], label='mm')

    fig.suptitle('Effect of DTI Biophysical Constraints', fontsize=14)
    plt.tight_layout()
    plt.savefig('dti_effect.png', dpi=150, bbox_inches='tight')
    print("  Saved: dti_effect.png")

    plt.show()

    print("\nVisualization complete!")
    print("Generated files:")
    print("  - displacement_views.png")
    print("  - displacement_vectors.png")
    print("  - jacobian_analysis.png")
    print("  - displacement_profile.png")
    print("  - dti_effect.png")


if __name__ == "__main__":
    main()
