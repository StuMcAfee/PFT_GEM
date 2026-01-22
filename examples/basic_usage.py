#!/usr/bin/env python3
"""
Basic Usage Example for PFT_GEM

This script demonstrates the basic workflow for computing tumor-induced
displacement fields using the geometric expansion model.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pft_gem import GeometricExpansionModel
from pft_gem.core.geometric_model import TumorParameters, ModelParameters
from pft_gem.core.constraints import create_synthetic_dti, BiophysicalConstraints
from pft_gem.io import SUITTemplateLoader
from pft_gem.utils import create_spherical_tumor_mask


def main():
    print("=" * 60)
    print("PFT_GEM: Basic Usage Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Define the computational grid
    # -------------------------------------------------------------------------
    print("\n1. Setting up computational grid...")
    grid_shape = (128, 128, 64)
    voxel_size = (1.0, 1.0, 1.0)  # mm
    print(f"   Grid shape: {grid_shape}")
    print(f"   Voxel size: {voxel_size} mm")
    print(f"   Physical extent: {tuple(g*v for g,v in zip(grid_shape, voxel_size))} mm")

    # -------------------------------------------------------------------------
    # Step 2: Define tumor parameters
    # -------------------------------------------------------------------------
    print("\n2. Defining tumor parameters...")
    tumor_params = TumorParameters(
        center=(64.0, 64.0, 32.0),  # Center in mm
        radius=15.0,                 # 15mm radius (30mm diameter)
        shape="spherical"
    )
    print(f"   Tumor center: {tumor_params.center}")
    print(f"   Tumor radius: {tumor_params.radius} mm")
    print(f"   Tumor volume: {4/3 * np.pi * tumor_params.radius**3:.1f} mm³")

    # -------------------------------------------------------------------------
    # Step 3: Define model parameters
    # -------------------------------------------------------------------------
    print("\n3. Configuring model parameters...")
    model_params = ModelParameters(
        decay_exponent=2.0,        # How quickly displacement decays
        boundary_stiffness=0.8,    # Stiffness at boundaries
        csf_damping=0.1,           # CSF damping factor
        max_displacement=15.0,     # Maximum displacement (mm)
        tissue_modulation=True,    # Use tissue-specific modulation
        use_dti_constraints=True,  # Use DTI constraints
        smoothing_sigma=1.0        # Smoothing in mm
    )
    print(f"   Decay exponent: {model_params.decay_exponent}")
    print(f"   Max displacement: {model_params.max_displacement} mm")

    # -------------------------------------------------------------------------
    # Step 4: Create synthetic DTI data
    # -------------------------------------------------------------------------
    print("\n4. Creating synthetic DTI data...")
    dti_data = create_synthetic_dti(
        shape=grid_shape,
        tumor_center=(int(tumor_params.center[0]),
                      int(tumor_params.center[1]),
                      int(tumor_params.center[2])),
        tumor_radius=tumor_params.radius
    )
    print(f"   FA range: [{dti_data.fa.min():.3f}, {dti_data.fa.max():.3f}]")
    print(f"   MD range: [{dti_data.md.min()*1000:.2f}, {dti_data.md.max()*1000:.2f}] x 10⁻³ mm²/s")

    # -------------------------------------------------------------------------
    # Step 5: Create and configure the model
    # -------------------------------------------------------------------------
    print("\n5. Creating geometric expansion model...")
    model = GeometricExpansionModel(
        tumor_params=tumor_params,
        model_params=model_params,
        grid_shape=grid_shape,
        voxel_size=voxel_size
    )

    # Add DTI constraints
    model.set_dti_constraints(dti_data.fa, dti_data.md)
    print("   Model created with DTI constraints")

    # -------------------------------------------------------------------------
    # Step 6: Compute displacement field
    # -------------------------------------------------------------------------
    print("\n6. Computing displacement field...")
    tumor_expansion = 5.0  # mm of tumor expansion
    displacement = model.compute_displacement_field(tumor_expansion=tumor_expansion)
    print(f"   Tumor expansion: {tumor_expansion} mm")
    print(f"   Displacement field shape: {displacement.shape}")

    # Compute statistics
    magnitude = np.linalg.norm(displacement, axis=-1)
    print(f"\n   Displacement statistics:")
    print(f"   - Maximum: {magnitude.max():.2f} mm")
    print(f"   - Mean: {magnitude.mean():.2f} mm")
    print(f"   - Median: {np.median(magnitude):.2f} mm")
    print(f"   - Std dev: {magnitude.std():.2f} mm")

    # -------------------------------------------------------------------------
    # Step 7: Compute derived quantities
    # -------------------------------------------------------------------------
    print("\n7. Computing derived quantities...")

    # Jacobian determinant (volume change)
    jacobian = model.compute_jacobian_determinant()
    print(f"   Jacobian (volume change):")
    print(f"   - Min: {jacobian.min():.3f} (max compression)")
    print(f"   - Max: {jacobian.max():.3f} (max expansion)")
    print(f"   - Mean: {jacobian.mean():.3f}")
    print(f"   - Voxels with J < 0 (folding): {np.sum(jacobian < 0)}")

    # Strain field
    strain = model.compute_strain_field()
    max_principal_strain = np.max(np.abs(np.linalg.eigvalsh(strain)), axis=-1)
    print(f"   Maximum principal strain: {max_principal_strain.max():.4f}")

    # -------------------------------------------------------------------------
    # Step 8: Create tumor mask
    # -------------------------------------------------------------------------
    print("\n8. Creating tumor mask...")
    tumor_mask = create_spherical_tumor_mask(
        grid_shape,
        tumor_params.center,
        tumor_params.radius,
        voxel_size
    )
    print(f"   Tumor mask volume: {np.sum(tumor_mask)} voxels")
    print(f"   Tumor mask volume: {np.sum(tumor_mask) * np.prod(voxel_size):.1f} mm³")

    # -------------------------------------------------------------------------
    # Step 9: Analyze displacement at specific locations
    # -------------------------------------------------------------------------
    print("\n9. Displacement at key locations...")

    # At tumor boundary
    boundary_distance = tumor_params.radius
    boundary_point = (
        tumor_params.center[0] + boundary_distance,
        tumor_params.center[1],
        tumor_params.center[2]
    )
    disp_boundary = model.get_displacement_at_point(boundary_point)
    print(f"   At tumor boundary (+x): {np.linalg.norm(disp_boundary):.2f} mm")

    # At 10mm from boundary
    far_point = (
        tumor_params.center[0] + tumor_params.radius + 10,
        tumor_params.center[1],
        tumor_params.center[2]
    )
    disp_far = model.get_displacement_at_point(far_point)
    print(f"   At 10mm from boundary: {np.linalg.norm(disp_far):.2f} mm")

    # At 20mm from boundary
    farther_point = (
        tumor_params.center[0] + tumor_params.radius + 20,
        tumor_params.center[1],
        tumor_params.center[2]
    )
    disp_farther = model.get_displacement_at_point(farther_point)
    print(f"   At 20mm from boundary: {np.linalg.norm(disp_farther):.2f} mm")

    # -------------------------------------------------------------------------
    # Step 10: Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Tumor size: {tumor_params.radius * 2} mm diameter")
    print(f"Expansion simulated: {tumor_expansion} mm")
    print(f"Maximum displacement: {magnitude.max():.2f} mm")
    print(f"Mean displacement: {magnitude.mean():.2f} mm")
    print(f"Volume change range: [{jacobian.min():.3f}, {jacobian.max():.3f}]")
    print("\nDisplacement field ready for further analysis or export!")

    return displacement, model


if __name__ == "__main__":
    displacement, model = main()
