"""
Visualization Functions for Displacement Fields

This module provides comprehensive visualization tools for examining
displacement fields, strain patterns, and brain deformation caused
by tumor growth.
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings.

    Attributes:
        figsize: Figure size in inches
        dpi: Figure resolution
        cmap_displacement: Colormap for displacement magnitude
        cmap_divergence: Colormap for divergence (compression/expansion)
        cmap_anatomy: Colormap for anatomical images
        vector_scale: Scaling factor for vector arrows
        vector_density: Density of vector arrows (1 = all, 2 = every other, etc.)
        alpha: Transparency for overlays
        colorbar: Whether to show colorbars
        title_fontsize: Font size for titles
    """
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    cmap_displacement: str = "hot"
    cmap_divergence: str = "RdBu_r"
    cmap_anatomy: str = "gray"
    vector_scale: float = 1.0
    vector_density: int = 4
    alpha: float = 0.7
    colorbar: bool = True
    title_fontsize: int = 12


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_displacement_magnitude(
    displacement: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    config: Optional[VisualizationConfig] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Displacement Magnitude"
) -> plt.Figure:
    """
    Plot displacement magnitude for a single slice.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3) or 3D magnitude
    slice_idx : int, optional
        Slice index to display. If None, uses middle slice.
    axis : int
        Axis perpendicular to slice (0=sagittal, 1=coronal, 2=axial)
    config : VisualizationConfig, optional
        Visualization settings
    ax : matplotlib.Axes, optional
        Axes to plot on. If None, creates new figure.
    vmin, vmax : float, optional
        Color scale limits
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    _check_matplotlib()
    config = config or VisualizationConfig()

    # Compute magnitude if 4D field provided
    if displacement.ndim == 4:
        magnitude = np.linalg.norm(displacement, axis=-1)
    else:
        magnitude = displacement

    # Select slice
    if slice_idx is None:
        slice_idx = magnitude.shape[axis] // 2

    # Extract slice based on axis
    if axis == 0:
        slice_data = magnitude[slice_idx, :, :]
    elif axis == 1:
        slice_data = magnitude[:, slice_idx, :]
    else:
        slice_data = magnitude[:, :, slice_idx]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure

    # Plot
    im = ax.imshow(
        slice_data.T,
        origin='lower',
        cmap=config.cmap_displacement,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title, fontsize=config.title_fontsize)
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')

    if config.colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Displacement (mm)')

    return fig


def plot_displacement_field(
    displacement: np.ndarray,
    anatomy: Optional[np.ndarray] = None,
    tumor_mask: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    config: Optional[VisualizationConfig] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot displacement field in three orthogonal views.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field (nx, ny, nz, 3)
    anatomy : np.ndarray, optional
        3D anatomical image for background
    tumor_mask : np.ndarray, optional
        3D tumor mask to overlay
    slice_indices : tuple, optional
        (sagittal, coronal, axial) slice indices
    config : VisualizationConfig, optional
        Visualization settings
    figsize : tuple, optional
        Override figure size

    Returns
    -------
    fig : matplotlib.Figure
        Figure with three subplot views
    """
    _check_matplotlib()
    config = config or VisualizationConfig()

    magnitude = np.linalg.norm(displacement, axis=-1)

    # Default to center slices
    if slice_indices is None:
        slice_indices = tuple(s // 2 for s in magnitude.shape)

    figsize = figsize or (15, 5)
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=config.dpi)

    views = [
        ("Sagittal", 0, slice_indices[0]),
        ("Coronal", 1, slice_indices[1]),
        ("Axial", 2, slice_indices[2]),
    ]

    for ax, (view_name, axis, idx) in zip(axes, views):
        # Extract slices
        if axis == 0:
            mag_slice = magnitude[idx, :, :]
            anat_slice = anatomy[idx, :, :] if anatomy is not None else None
            tumor_slice = tumor_mask[idx, :, :] if tumor_mask is not None else None
        elif axis == 1:
            mag_slice = magnitude[:, idx, :]
            anat_slice = anatomy[:, idx, :] if anatomy is not None else None
            tumor_slice = tumor_mask[:, idx, :] if tumor_mask is not None else None
        else:
            mag_slice = magnitude[:, :, idx]
            anat_slice = anatomy[:, :, idx] if anatomy is not None else None
            tumor_slice = tumor_mask[:, :, idx] if tumor_mask is not None else None

        # Plot anatomy background if available
        if anat_slice is not None:
            ax.imshow(anat_slice.T, origin='lower', cmap=config.cmap_anatomy, alpha=0.5)

        # Plot displacement magnitude
        im = ax.imshow(
            mag_slice.T,
            origin='lower',
            cmap=config.cmap_displacement,
            alpha=config.alpha if anat_slice is not None else 1.0
        )

        # Overlay tumor contour if available
        if tumor_slice is not None:
            ax.contour(tumor_slice.T, levels=[0.5], colors='cyan', linewidths=2)

        ax.set_title(f"{view_name} (slice {idx})", fontsize=config.title_fontsize)
        ax.set_aspect('equal')

    # Add colorbar
    if config.colorbar:
        fig.colorbar(im, ax=axes, label='Displacement (mm)', shrink=0.8)

    fig.suptitle("Displacement Field", fontsize=config.title_fontsize + 2)
    plt.tight_layout()

    return fig


def plot_vector_field(
    displacement: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    background: Optional[np.ndarray] = None,
    config: Optional[VisualizationConfig] = None,
    ax: Optional[plt.Axes] = None,
    color_by: str = "magnitude"
) -> plt.Figure:
    """
    Plot displacement vectors as a quiver plot.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field
    slice_idx : int, optional
        Slice index
    axis : int
        Axis perpendicular to slice
    background : np.ndarray, optional
        Background image
    config : VisualizationConfig, optional
        Visualization settings
    ax : matplotlib.Axes, optional
        Axes to plot on
    color_by : str
        'magnitude', 'direction', or 'component'

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    _check_matplotlib()
    config = config or VisualizationConfig()

    shape = displacement.shape[:3]
    if slice_idx is None:
        slice_idx = shape[axis] // 2

    # Extract slice data
    if axis == 0:
        u = displacement[slice_idx, :, :, 1]  # Y component
        v = displacement[slice_idx, :, :, 2]  # Z component
        bg = background[slice_idx, :, :] if background is not None else None
    elif axis == 1:
        u = displacement[:, slice_idx, :, 0]  # X component
        v = displacement[:, slice_idx, :, 2]  # Z component
        bg = background[:, slice_idx, :] if background is not None else None
    else:
        u = displacement[:, :, slice_idx, 0]  # X component
        v = displacement[:, :, slice_idx, 1]  # Y component
        bg = background[:, :, slice_idx] if background is not None else None

    # Create grid
    ny, nz = u.shape
    y = np.arange(0, ny, config.vector_density)
    z = np.arange(0, nz, config.vector_density)
    Y, Z = np.meshgrid(y, z, indexing='ij')

    # Subsample vectors
    u_sub = u[::config.vector_density, ::config.vector_density]
    v_sub = v[::config.vector_density, ::config.vector_density]

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure

    # Plot background
    if bg is not None:
        ax.imshow(bg.T, origin='lower', cmap=config.cmap_anatomy, alpha=0.5)

    # Color by magnitude or direction
    if color_by == "magnitude":
        colors = np.sqrt(u_sub**2 + v_sub**2)
    elif color_by == "direction":
        colors = np.arctan2(v_sub, u_sub)
    else:
        colors = u_sub  # Color by x-component

    # Plot vectors
    q = ax.quiver(
        Y, Z, u_sub.T, v_sub.T,
        colors.T,
        cmap=config.cmap_displacement,
        scale=50 / config.vector_scale,
        width=0.003
    )

    ax.set_title(f"Displacement Vectors (slice {slice_idx})", fontsize=config.title_fontsize)
    ax.set_aspect('equal')

    if config.colorbar:
        plt.colorbar(q, ax=ax, label='Magnitude (mm)')

    return fig


def plot_jacobian(
    jacobian: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    config: Optional[VisualizationConfig] = None,
    ax: Optional[plt.Axes] = None,
    symmetric_scale: bool = True
) -> plt.Figure:
    """
    Plot Jacobian determinant showing volume changes.

    Parameters
    ----------
    jacobian : np.ndarray
        3D Jacobian determinant field
    slice_idx : int, optional
        Slice index
    axis : int
        Axis perpendicular to slice
    config : VisualizationConfig, optional
        Visualization settings
    ax : matplotlib.Axes, optional
        Axes to plot on
    symmetric_scale : bool
        Whether to use symmetric color scale around 1

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    _check_matplotlib()
    config = config or VisualizationConfig()

    if slice_idx is None:
        slice_idx = jacobian.shape[axis] // 2

    # Extract slice
    if axis == 0:
        slice_data = jacobian[slice_idx, :, :]
    elif axis == 1:
        slice_data = jacobian[:, slice_idx, :]
    else:
        slice_data = jacobian[:, :, slice_idx]

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure

    # Set color scale
    if symmetric_scale:
        vmax = max(abs(slice_data.max() - 1), abs(slice_data.min() - 1))
        vmin, vmax = 1 - vmax, 1 + vmax
    else:
        vmin, vmax = slice_data.min(), slice_data.max()

    # Plot
    im = ax.imshow(
        slice_data.T,
        origin='lower',
        cmap=config.cmap_divergence,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title("Jacobian Determinant (J>1: expansion, J<1: compression)",
                 fontsize=config.title_fontsize)

    if config.colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Jacobian')

    return fig


def plot_slice_comparison(
    original: np.ndarray,
    warped: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    config: Optional[VisualizationConfig] = None,
    difference: bool = True
) -> plt.Figure:
    """
    Compare original and warped images side by side.

    Parameters
    ----------
    original : np.ndarray
        Original 3D image
    warped : np.ndarray
        Warped 3D image
    slice_idx : int, optional
        Slice index
    axis : int
        Axis perpendicular to slice
    config : VisualizationConfig, optional
        Visualization settings
    difference : bool
        Whether to show difference image

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    _check_matplotlib()
    config = config or VisualizationConfig()

    if slice_idx is None:
        slice_idx = original.shape[axis] // 2

    # Extract slices
    if axis == 0:
        orig_slice = original[slice_idx, :, :]
        warp_slice = warped[slice_idx, :, :]
    elif axis == 1:
        orig_slice = original[:, slice_idx, :]
        warp_slice = warped[:, slice_idx, :]
    else:
        orig_slice = original[:, :, slice_idx]
        warp_slice = warped[:, :, slice_idx]

    ncols = 3 if difference else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5), dpi=config.dpi)

    # Original
    axes[0].imshow(orig_slice.T, origin='lower', cmap=config.cmap_anatomy)
    axes[0].set_title("Original", fontsize=config.title_fontsize)

    # Warped
    axes[1].imshow(warp_slice.T, origin='lower', cmap=config.cmap_anatomy)
    axes[1].set_title("Warped", fontsize=config.title_fontsize)

    # Difference
    if difference:
        diff = warp_slice - orig_slice
        vmax = np.abs(diff).max()
        im = axes[2].imshow(
            diff.T,
            origin='lower',
            cmap=config.cmap_divergence,
            vmin=-vmax,
            vmax=vmax
        )
        axes[2].set_title("Difference", fontsize=config.title_fontsize)
        if config.colorbar:
            plt.colorbar(im, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def create_animation(
    displacement_sequence: List[np.ndarray],
    slice_idx: int,
    axis: int = 2,
    config: Optional[VisualizationConfig] = None,
    output_path: Optional[Union[str, Path]] = None,
    fps: int = 10,
    interval: int = 100
):
    """
    Create animation of displacement field evolution.

    Parameters
    ----------
    displacement_sequence : list
        List of 4D displacement fields
    slice_idx : int
        Slice index for visualization
    axis : int
        Axis perpendicular to slice
    config : VisualizationConfig, optional
        Visualization settings
    output_path : str or Path, optional
        Path to save animation (requires imageio or ffmpeg)
    fps : int
        Frames per second for saved animation
    interval : int
        Interval between frames in milliseconds

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    """
    _check_matplotlib()
    from matplotlib.animation import FuncAnimation

    config = config or VisualizationConfig()

    # Compute magnitudes
    magnitudes = [np.linalg.norm(d, axis=-1) for d in displacement_sequence]

    # Find global max for consistent scaling
    vmax = max(m.max() for m in magnitudes)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Extract first slice
    if axis == 0:
        first_slice = magnitudes[0][slice_idx, :, :]
    elif axis == 1:
        first_slice = magnitudes[0][:, slice_idx, :]
    else:
        first_slice = magnitudes[0][:, :, slice_idx]

    im = ax.imshow(first_slice.T, origin='lower', cmap=config.cmap_displacement,
                   vmin=0, vmax=vmax)
    title = ax.set_title("Frame 0", fontsize=config.title_fontsize)

    if config.colorbar:
        plt.colorbar(im, ax=ax, label='Displacement (mm)')

    def update(frame):
        if axis == 0:
            data = magnitudes[frame][slice_idx, :, :]
        elif axis == 1:
            data = magnitudes[frame][:, slice_idx, :]
        else:
            data = magnitudes[frame][:, :, slice_idx]

        im.set_array(data.T)
        title.set_text(f"Frame {frame}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(magnitudes),
                        interval=interval, blit=True)

    if output_path is not None:
        try:
            anim.save(str(output_path), fps=fps, writer='pillow')
        except Exception:
            try:
                anim.save(str(output_path), fps=fps, writer='ffmpeg')
            except Exception as e:
                print(f"Could not save animation: {e}")

    return anim


def plot_tumor_displacement_profile(
    displacement: np.ndarray,
    tumor_center: Tuple[int, int, int],
    directions: List[str] = ["x", "y", "z"],
    max_distance: int = 50,
    config: Optional[VisualizationConfig] = None
) -> plt.Figure:
    """
    Plot displacement magnitude as a function of distance from tumor.

    Parameters
    ----------
    displacement : np.ndarray
        4D displacement field
    tumor_center : tuple
        (x, y, z) tumor center coordinates
    directions : list
        Directions to plot profiles ('x', 'y', 'z', or 'radial')
    max_distance : int
        Maximum distance from tumor center
    config : VisualizationConfig, optional
        Visualization settings

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    _check_matplotlib()
    config = config or VisualizationConfig()

    magnitude = np.linalg.norm(displacement, axis=-1)
    cx, cy, cz = tumor_center

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    for direction in directions:
        if direction == "x":
            profile = magnitude[:, cy, cz]
            distances = np.arange(len(profile)) - cx
        elif direction == "y":
            profile = magnitude[cx, :, cz]
            distances = np.arange(len(profile)) - cy
        elif direction == "z":
            profile = magnitude[cx, cy, :]
            distances = np.arange(len(profile)) - cz
        elif direction == "radial":
            # Compute radial average
            x = np.arange(magnitude.shape[0]) - cx
            y = np.arange(magnitude.shape[1]) - cy
            z = np.arange(magnitude.shape[2]) - cz
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            R = np.sqrt(X**2 + Y**2 + Z**2).astype(int)

            distances = np.arange(max_distance)
            profile = np.array([
                magnitude[R == r].mean() if np.any(R == r) else 0
                for r in distances
            ])
        else:
            continue

        # Filter to max distance
        mask = np.abs(distances) <= max_distance
        ax.plot(distances[mask], profile[mask], label=f"{direction}-direction", linewidth=2)

    ax.set_xlabel("Distance from tumor center (voxels)", fontsize=config.title_fontsize)
    ax.set_ylabel("Displacement magnitude (mm)", fontsize=config.title_fontsize)
    ax.set_title("Displacement Profile", fontsize=config.title_fontsize + 2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
