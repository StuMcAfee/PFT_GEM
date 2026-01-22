"""
Visualization utilities for displacement fields and brain imaging data.
"""

from .plotting import (
    plot_displacement_field,
    plot_displacement_magnitude,
    plot_slice_comparison,
    plot_vector_field,
    plot_jacobian,
    create_animation,
    VisualizationConfig,
)

__all__ = [
    "plot_displacement_field",
    "plot_displacement_magnitude",
    "plot_slice_comparison",
    "plot_vector_field",
    "plot_jacobian",
    "create_animation",
    "VisualizationConfig",
]
