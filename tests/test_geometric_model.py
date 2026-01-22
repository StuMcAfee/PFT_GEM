"""
Unit tests for the geometric expansion model.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from pft_gem.core.geometric_model import (
    GeometricExpansionModel,
    TumorParameters,
    ModelParameters,
    TissueType,
)
from pft_gem.core.displacement import DisplacementField, FieldMetadata
from pft_gem.core.constraints import (
    BiophysicalConstraints,
    DTIData,
    create_synthetic_dti,
)
from pft_gem.utils.helpers import (
    create_spherical_tumor_mask,
    create_ellipsoidal_tumor_mask,
)


class TestTumorParameters:
    """Tests for TumorParameters dataclass."""

    def test_spherical_tumor(self):
        """Test spherical tumor creation."""
        tumor = TumorParameters(
            center=(64, 64, 32),
            radius=15.0,
            shape="spherical"
        )
        assert tumor.center == (64, 64, 32)
        assert tumor.radius == 15.0
        assert tumor.shape == "spherical"

    def test_ellipsoidal_tumor(self):
        """Test ellipsoidal tumor creation."""
        tumor = TumorParameters(
            center=(64, 64, 32),
            radius=15.0,
            shape="ellipsoidal",
            semi_axes=(20.0, 15.0, 10.0)
        )
        assert tumor.semi_axes == (20.0, 15.0, 10.0)

    def test_ellipsoidal_default_axes(self):
        """Test ellipsoidal tumor gets default semi_axes from radius."""
        tumor = TumorParameters(
            center=(64, 64, 32),
            radius=15.0,
            shape="ellipsoidal"
        )
        assert tumor.semi_axes == (15.0, 15.0, 15.0)


class TestGeometricExpansionModel:
    """Tests for GeometricExpansionModel class."""

    @pytest.fixture
    def basic_model(self):
        """Create a basic model for testing."""
        tumor = TumorParameters(center=(32, 32, 16), radius=10.0)
        return GeometricExpansionModel(
            tumor_params=tumor,
            grid_shape=(64, 64, 32),
            voxel_size=(1.0, 1.0, 1.0)
        )

    def test_model_creation(self, basic_model):
        """Test model creation."""
        assert basic_model.grid_shape == (64, 64, 32)
        assert basic_model.voxel_size == (1.0, 1.0, 1.0)

    def test_displacement_field_shape(self, basic_model):
        """Test displacement field has correct shape."""
        displacement = basic_model.compute_displacement_field()
        assert displacement.shape == (64, 64, 32, 3)

    def test_displacement_inside_tumor_is_zero(self, basic_model):
        """Test displacement is zero inside tumor."""
        displacement = basic_model.compute_displacement_field()
        magnitude = np.linalg.norm(displacement, axis=-1)

        # Create tumor mask
        tumor_mask = create_spherical_tumor_mask(
            (64, 64, 32),
            (32, 32, 16),
            10.0
        )

        # Check displacement inside tumor is near zero
        inside_displacement = magnitude[tumor_mask > 0]
        assert np.all(inside_displacement < 0.1)

    def test_displacement_decays_with_distance(self, basic_model):
        """Test displacement decays with distance from tumor."""
        displacement = basic_model.compute_displacement_field()
        magnitude = np.linalg.norm(displacement, axis=-1)

        # Get displacement at increasing distances from center
        cx, cy, cz = 32, 32, 16
        r = 10  # tumor radius

        # Just outside tumor boundary
        disp_near = magnitude[cx + r + 1, cy, cz]

        # Further from tumor
        disp_far = magnitude[cx + r + 10, cy, cz]

        # Even further
        disp_farther = magnitude[cx + r + 20, cy, cz]

        assert disp_near > disp_far > disp_farther

    def test_displacement_is_radial(self, basic_model):
        """Test displacement vectors point radially outward."""
        displacement = basic_model.compute_displacement_field()

        cx, cy, cz = 32, 32, 16
        r = 10

        # Check a point along +x direction
        point = (cx + r + 5, cy, cz)
        disp = displacement[point]

        # Displacement should be primarily in +x direction
        assert disp[0] > 0
        assert abs(disp[0]) > abs(disp[1])
        assert abs(disp[0]) > abs(disp[2])

    def test_tumor_expansion_scales_displacement(self, basic_model):
        """Test that larger tumor expansion gives larger displacement."""
        disp_small = basic_model.compute_displacement_field(tumor_expansion=2.0)
        disp_large = basic_model.compute_displacement_field(tumor_expansion=5.0)

        mag_small = np.linalg.norm(disp_small, axis=-1).max()
        mag_large = np.linalg.norm(disp_large, axis=-1).max()

        assert mag_large > mag_small

    def test_jacobian_determinant(self, basic_model):
        """Test Jacobian determinant computation."""
        basic_model.compute_displacement_field()
        jacobian = basic_model.compute_jacobian_determinant()

        assert jacobian.shape == (64, 64, 32)
        # Jacobian should be around 1 for small deformations
        assert jacobian.mean() > 0.5
        assert jacobian.mean() < 2.0

    def test_strain_field(self, basic_model):
        """Test strain field computation."""
        basic_model.compute_displacement_field()
        strain = basic_model.compute_strain_field()

        assert strain.shape == (64, 64, 32, 3, 3)
        # Strain tensor should be symmetric
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(
                    strain[..., i, j],
                    strain[..., j, i],
                    rtol=1e-5
                )

    def test_get_displacement_at_point(self, basic_model):
        """Test displacement interpolation at arbitrary points."""
        basic_model.compute_displacement_field()

        # Get displacement at a point
        point = (40.5, 32.0, 16.0)
        disp = basic_model.get_displacement_at_point(point)

        assert disp.shape == (3,)
        assert np.linalg.norm(disp) > 0

    def test_model_serialization(self, basic_model):
        """Test model can be converted to/from dict."""
        config = basic_model.to_dict()

        assert "tumor" in config
        assert "parameters" in config
        assert "grid" in config

        # Recreate model from config
        model2 = GeometricExpansionModel.from_dict(config)
        assert model2.grid_shape == basic_model.grid_shape
        assert model2.tumor.radius == basic_model.tumor.radius


class TestDisplacementField:
    """Tests for DisplacementField class."""

    @pytest.fixture
    def sample_field(self):
        """Create a sample displacement field."""
        shape = (32, 32, 16)
        data = np.random.randn(*shape, 3) * 0.1
        return DisplacementField(data)

    def test_magnitude(self, sample_field):
        """Test magnitude computation."""
        mag = sample_field.magnitude()
        assert mag.shape == (32, 32, 16)
        assert np.all(mag >= 0)

    def test_divergence(self, sample_field):
        """Test divergence computation."""
        div = sample_field.divergence()
        assert div.shape == (32, 32, 16)

    def test_jacobian_determinant(self, sample_field):
        """Test Jacobian determinant computation."""
        det_j = sample_field.jacobian_determinant()
        assert det_j.shape == (32, 32, 16)

    def test_smooth(self, sample_field):
        """Test smoothing."""
        smoothed = sample_field.smooth(sigma=1.0)
        assert smoothed.shape == sample_field.shape

    def test_add_fields(self, sample_field):
        """Test field addition."""
        result = sample_field + sample_field
        np.testing.assert_allclose(
            result.data,
            2 * sample_field.data
        )

    def test_scale_field(self, sample_field):
        """Test field scaling."""
        result = 2.0 * sample_field
        np.testing.assert_allclose(
            result.data,
            2 * sample_field.data
        )

    def test_statistics(self, sample_field):
        """Test statistics computation."""
        stats = sample_field.statistics()

        assert "magnitude" in stats
        assert "min" in stats["magnitude"]
        assert "max" in stats["magnitude"]
        assert "mean" in stats["magnitude"]


class TestBiophysicalConstraints:
    """Tests for BiophysicalConstraints class."""

    @pytest.fixture
    def dti_data(self):
        """Create sample DTI data."""
        shape = (32, 32, 16)
        fa = np.random.rand(*shape) * 0.5 + 0.1
        md = np.random.rand(*shape) * 1e-3 + 0.5e-3
        return DTIData(fa=fa, md=md)

    def test_stiffness_map(self, dti_data):
        """Test stiffness map computation."""
        constraints = BiophysicalConstraints(dti_data)
        stiffness = constraints.compute_stiffness_map()

        assert stiffness.shape == (32, 32, 16)
        assert stiffness.min() >= 0
        assert stiffness.max() <= 1

    def test_compliance_map(self, dti_data):
        """Test compliance map computation."""
        constraints = BiophysicalConstraints(dti_data)
        compliance = constraints.compute_compliance_map()

        assert compliance.shape == (32, 32, 16)
        assert np.all(compliance > 0)

    def test_modulate_displacement(self, dti_data):
        """Test displacement modulation."""
        constraints = BiophysicalConstraints(dti_data)

        displacement = np.random.randn(32, 32, 16, 3)
        modulated = constraints.modulate_displacement(displacement)

        assert modulated.shape == displacement.shape


class TestSyntheticDTI:
    """Tests for synthetic DTI generation."""

    def test_create_synthetic_dti(self):
        """Test synthetic DTI creation."""
        dti = create_synthetic_dti(
            shape=(32, 32, 16),
            tumor_center=(16, 16, 8),
            tumor_radius=5.0
        )

        assert dti.fa.shape == (32, 32, 16)
        assert dti.md.shape == (32, 32, 16)
        assert dti.v1.shape == (32, 32, 16, 3)

        # FA should be between 0 and 1
        assert dti.fa.min() >= 0
        assert dti.fa.max() <= 1

        # MD should be positive
        assert np.all(dti.md > 0)


class TestUtilities:
    """Tests for utility functions."""

    def test_spherical_mask(self):
        """Test spherical mask creation."""
        mask = create_spherical_tumor_mask(
            shape=(32, 32, 16),
            center=(16, 16, 8),
            radius=5.0
        )

        assert mask.shape == (32, 32, 16)
        assert mask.dtype == np.uint8

        # Center should be in mask
        assert mask[16, 16, 8] == 1

        # Far corner should not be in mask
        assert mask[0, 0, 0] == 0

    def test_ellipsoidal_mask(self):
        """Test ellipsoidal mask creation."""
        mask = create_ellipsoidal_tumor_mask(
            shape=(32, 32, 16),
            center=(16, 16, 8),
            semi_axes=(8.0, 6.0, 4.0)
        )

        assert mask.shape == (32, 32, 16)
        assert mask[16, 16, 8] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
