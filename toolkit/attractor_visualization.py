"""
Attractor Visualization: Tools for visualizing spiral attractors across different computational systems

This module provides visualization functions for spiral attractors detected in:
1. Spatial systems (e.g., Langton's Ant, Conway's Game of Life)
2. High-dimensional systems (e.g., neural networks) via projection
3. Statistical systems (e.g., language models) via token distributions

Author: David Kimai
Created: June 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
DEFAULT_CMAP = 'viridis'
RESIDUE_CMAP = LinearSegmentedColormap.from_list('residue', 
                                               [(0, 'black'), 
                                                (0.5, 'purple'),
                                                (0.7, 'magenta'),
                                                (0.9, 'cyan'),
                                                (1, 'white')])
DEFAULT_FIGSIZE = (12, 10)
DEFAULT_DPI = 100

class AnimationWrapper:
    """Wrapper for animations that supports saving to various formats."""
    
    def __init__(self, fig, animation):
        self.fig = fig
        self.animation = animation
    
    def save(self, filename: str, fps: int = 15, dpi: int = 100):
        """Save animation to file."""
        self.animation.save(filename, fps=fps, dpi=dpi)
    
    def to_html5_video(self):
        """Convert animation to HTML5 video for notebook display."""
        return self.animation.to_html5_video()
    
    def close(self):
        """Close figure to free resources."""
        plt.close(self.fig)

def visualize_spatial_attractor(grid: np.ndarray, history: List[np.ndarray] = None,
                               spiral_metrics: Dict[str, Any] = None, 
                               title: str = "Spiral Attractor Visualization",
                               highlight_residue: bool = True,
                               save_path: Optional[str] = None,
                               show_fig: bool = True) -> Optional[plt.Figure]:
    """
    Visualize spiral attractors in spatial data like cellular automata.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state of the spatial system
    history : List[np.ndarray], optional
        History of previous states for temporal visualization
    spiral_metrics : Dict[str, Any], optional
        Metrics from spiral detection for enhanced visualization
    title : str, default="Spiral Attractor Visualization"
        Title for the visualization
    highlight_residue : bool, default=True
        Whether to highlight unexplained patterns (residue)
    save_path : str, optional
        Path to save the visualization
    show_fig : bool, default=True
        Whether to display the figure
        
    Returns:
    --------
    Optional[plt.Figure]
        The created figure, or None if no figure was created
    """
    logger.info("Creating spatial attractor visualization")
    
    # Create figure
    fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # Determine number of subplots based on available data
    has_history = history is not None and len(history) > 1
    has_metrics = spiral_metrics is not None
    
    if has_history and has_metrics:
        # 2x2 grid
        grid_viz_ax = plt.subplot2grid((2, 2), (0, 0))
        history_viz_ax = plt.subplot2grid((2, 2), (0, 1))
        polar_viz_ax = plt.subplot2grid((2, 2), (1, 0), polar=True)
        metrics_viz_ax = plt.subplot2grid((2, 2), (1, 1))
    elif has_history:
        # 1x2 grid
        grid_viz_ax = plt.subplot2grid((1, 2), (0, 0))
        history_viz_ax = plt.subplot2grid((1, 2), (0, 1))
        polar_viz_ax = metrics_viz_ax = None
    elif has_metrics:
        # 1x2 grid
        grid_viz_ax = plt.subplot2grid((1, 2), (0, 0))
        polar_viz_ax = plt.subplot2grid((1, 2), (0, 1), polar=True)
        history_viz_ax = metrics_viz_ax = None
    else:
        # Single plot
        grid_viz_ax = plt.subplot(111)
        history_viz_ax = polar_viz_ax = metrics_viz_ax = None
    
    # Visualize current grid state
    _visualize_grid(grid_viz_ax, grid, spiral_metrics, highlight_residue)
    
    # Visualize history if available
    if has_history:
        _visualize_history(history_viz_ax, history, spiral_metrics)
    
    # Visualize spiral metrics if available
    if has_metrics:
        if polar_viz_ax:
            _visualize_spiral_polar(polar_viz_ax, grid, spiral_metrics)
        
        if metrics_viz_ax:
            _visualize_metrics(metrics_viz_ax, spiral_metrics)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show_fig:
        plt.show()
    
    return fig

def create_spatial_animation(history: List[np.ndarray], 
                            spiral_metrics: Dict[str, Any] = None,
                            title: str = "Spiral Attractor Evolution",
                            highlight_residue: bool = True,
                            formation_marker: bool = True,
                            interval: int = 50,
                            save_path: Optional[str] = None) -> AnimationWrapper:
    """
    Create animation of spatial attractor evolution.
    
    Parameters:
    -----------
    history : List[np.ndarray]
        History of states for animation
    spiral_metrics : Dict[str, Any], optional
        Metrics from spiral detection for enhanced visualization
    title : str, default="Spiral Attractor Evolution"
        Title for the animation
    highlight_residue : bool, default=True
        Whether to highlight unexplained patterns (residue)
    formation_marker : bool, default=True
        Whether to mark the formation point of the attractor
    interval : int, default=50
        Interval between frames in milliseconds
    save_path : str, optional
        Path to save the animation
        
    Returns:
    --------
    AnimationWrapper
        Wrapper containing the animation
    """
    logger.info("Creating spatial attractor animation")
    
    # Create figure with 2 subplots (grid and metrics over time)
    fig, (grid_ax, metrics_ax) = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # Set up grid visualization
    grid = history[0]
    im = grid_ax.imshow(grid, cmap=DEFAULT_CMAP, interpolation='nearest')
    grid_ax.set_title("System State")
    
    # Determine step at which attractor forms
    formation_step = None
    if spiral_metrics and 'formation_step' in spiral_metrics:
        formation_step = spiral_metrics['formation_step']
    
    # Set up metrics visualization
    metrics_line, = metrics_ax.plot([], [], 'r-', linewidth=2)
    metrics_ax.set_xlim(0, len(history))
    metrics_ax.set_ylim(0, 1)
    metrics_ax.set_xlabel("Time Step")
    metrics_ax.set_ylabel("Pattern Stability")
    metrics_ax.set_title("Pattern Stability Over Time")
    metrics_ax.grid(True)
    
    # Vertical line to mark formation step
    formation_line = None
    if formation_marker and formation_step is not None:
        formation_line = metrics_ax.axvline(x=formation_step, color='g', linestyle='--', alpha=0.7)
        metrics_ax.text(formation_step + 5, 0.5, "Attractor Forms", 
                      rotation=90, verticalalignment='center')
    
    # Track stability over time
    stabilities = []
    
    # Update function for animation
    def update(frame):
        # Update grid
        im.set_array(history[frame])
        
        # Calculate and update stability
        if frame > 0:
            similarity = 1.0 - np.mean(np.abs(history[frame] - history[frame-1])) / np.max([1.0, np.max(history[frame]), np.max(history[frame-1])])
            stabilities.append(similarity)
        else:
            stabilities.append(0)
        
        # Update metrics plot
        metrics_line.set_data(range(1, frame + 2), stabilities)
        
        # Add marker at formation point
        if formation_marker and formation_step is not None and frame == formation_step:
            grid_ax.set_title("System State - Attractor Formed!")
        
        return im, metrics_line
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(history)), 
                         interval=interval, blit=True)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if requested
    if save_path:
        anim.save(save_path, writer='pillow', fps=30)
        logger.info(f"Animation saved to {save_path}")
    
    return AnimationWrapper(fig, anim)

def visualize_projected_attractor(projected_data: np.ndarray, time_indices: np.ndarray = None,
                                spiral_metrics: Dict[str, Any] = None,
                                title: str = "Projected Attractor Visualization",
                                dimensions: int = 3,
                                color_by_time: bool = True,
                                highlight_phase_transitions: bool = True,
                                save_path: Optional[str] = None,
                                show_fig: bool = True) -> Optional[plt.Figure]:
    """
    Visualize spiral attractors in projected high-dimensional data.
    
    Parameters:
    -----------
    projected_data : np.ndarray
        Projected data points (2D or 3D)
    time_indices : np.ndarray, optional
        Indices representing time progression
    spiral_metrics : Dict[str, Any], optional
        Metrics from spiral detection for enhanced visualization
    title : str, default="Projected Attractor Visualization"
        Title for the visualization
    dimensions : int, default=3
        Number of dimensions to visualize (2 or 3)
    color_by_time : bool, default=True
        Whether to color points by time progression
    highlight_phase_transitions : bool, default=True
        Whether to highlight phase transitions
    save_path : str, optional
        Path to save the visualization
    show_fig : bool, default=True
        Whether to display the figure
        
    Returns:
    --------
    Optional[plt.Figure]
        The created figure, or None if no figure was created
    """
    logger.info(f"Creating projected attractor visualization in {dimensions}D")
    
    # Ensure we have enough dimensions
    if projected_data.shape[1] < dimensions:
        logger.warning(f"Requested {dimensions}D visualization but data has only {projected_data.shape[1]} dimensions")
        dimensions = projected_data.shape[1]
    
    # Create figure
    fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # Create main visualization plot (2D or 3D)
    if dimensions == 3 and projected_data.shape[1] >= 3:
        ax1 = fig.add_subplot(121, projection='3d')
    else:
        ax1 = fig.add_subplot(121)
    
    # Create secondary plot for metrics or additional views
    ax2 = fig.add_subplot(122)
    
    # Set up color mapping by time if requested
    colors = None
    if color_by_time and time_indices is not None:
        normalized_time = (time_indices - np.min(time_indices)) / (np.max(time_indices) - np.min(time_indices))
        colors = plt.cm.viridis(normalized_time)
    
    # Visualize the projected data
    if dimensions == 3 and projected_data.shape[1] >= 3:
        _visualize_3d_projection(ax1, projected_data, colors, spiral_metrics)
    else:
        _visualize_2d_projection(ax1, projected_data, colors, spiral_metrics)
    
    # Visualize additional information in second plot
    if spiral_metrics and time_indices is not None:
        _visualize_temporal_metrics(ax2, time_indices, spiral_metrics, highlight_phase_transitions)
    else:
        # Alternative view if no metrics available
        if dimensions == 3 and projected_data.shape[1] >= 3:
            # 2D projection of first two components
            _visualize_2d_projection(ax2, projected_data[:, :2], colors, spiral_metrics)
            ax2.set_title("2D Projection (Components 1-2)")
        else:
            # Simple scatter of points colored by density
            from scipy.stats import gaussian_kde
            x, y = projected_data[:, 0], projected_data[:, 1]
            xy = np.vstack([x, y])
            density = gaussian_kde(xy)(xy)
            ax2.scatter(x, y, c=density, cmap='viridis', s=10, alpha=0.7)
            ax2.set_title("Point Density")
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show_fig:
        plt.show()
    
    return fig

def create_projected_animation(projected_data: np.ndarray, time_indices: np.ndarray,
                              spiral_metrics: Dict[str, Any] = None,
                              title: str = "Projected Attractor Evolution",
                              dimensions: int = 3,
                              highlight_phase_transitions: bool = True,
                              interval: int = 50,
                              trail_length: int = 20,
                              save_path: Optional[str] = None) -> AnimationWrapper:
    """
    Create animation of projected attractor evolution.
    
    Parameters:
    -----------
    projected_data : np.ndarray
        Projected data points (2D or 3D)
    time_indices : np.ndarray
        Indices representing time progression
    spiral_metrics : Dict[str, Any], optional
        Metrics from spiral detection for enhanced visualization
    title : str, default="Projected Attractor Evolution"
        Title for the animation
    dimensions : int, default=3
        Number of dimensions to visualize (2 or 3)
    highlight_phase_transitions : bool, default=True
        Whether to highlight phase transitions
    interval : int, default=50
        Interval between frames in milliseconds
    trail_length : int, default=20
        Number of previous points to show as trail
    save_path : str, optional
        Path to save the animation
        
    Returns:
    --------
    AnimationWrapper
        Wrapper containing the animation
    """
    logger.info(f"Creating projected attractor animation in {dimensions}D")
    
    # Ensure we have enough dimensions
    if projected_data.shape[1] < dimensions:
        logger.warning(f"Requested {dimensions}D visualization but data has only {projected_data.shape[1]} dimensions")
        dimensions = projected_data.shape[1]
    
    # Sort data by time
    if time_indices is not None:
        sorted_indices = np.argsort(time_indices)
        sorted_data = projected_data[sorted_indices]
    else:
        sorted_data = projected_data
    
    # Create figure
    if dimensions == 3 and projected_data.shape[1] >= 3:
        fig = plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # Initialize empty plot elements
    if dimensions == 3 and projected_data.shape[1] >= 3:
        trail, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7)
        point, = ax.plot([], [], [], 'ro', markersize=8)
    else:
        trail, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7)
        point, = ax.plot([], [], 'ro', markersize=8)
    
    # Set axis limits
    if dimensions == 3 and projected_data.shape[1] >= 3:
        ax.set_xlim(np.min(sorted_data[:, 0]), np.max(sorted_data[:, 0]))
        ax.set_ylim(np.min(sorted_data[:, 1]), np.max(sorted_data[:, 1]))
        ax.set_zlim(np.min(sorted_data[:, 2]), np.max(sorted_data[:, 2]))
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        ax.set_xlim(np.min(sorted_data[:, 0]), np.max(sorted_data[:, 0]))
        ax.set_ylim(np.min(sorted_data[:, 1]), np.max(sorted_data[:, 1]))
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    
    # Find phase transitions if available
    phase_transitions = []
    if spiral_metrics and highlight_phase_transitions:
        if 'transition_points' in spiral_metrics:
            phase_transitions = spiral_metrics['transition_points']
        elif 'phase_transitions' in spiral_metrics and spiral_metrics['phase_transitions'] > 0:
            # Estimate transition points if exact ones not provided
            pass
    
    # Update function for animation
    def update(frame):
        # Get trail points
        start = max(0, frame - trail_length)
        trail_data = sorted_data[start:frame+1]
        
        # Update trail
        if dimensions == 3 and projected_data.shape[1] >= 3:
            trail.set_data(trail_data[:, 0], trail_data[:, 1])
            trail.set_3d_properties(trail_data[:, 2])
            point.set_data([sorted_data[frame, 0]], [sorted_data[frame, 1]])
            point.set_3d_properties([sorted_data[frame, 2]])
        else:
            trail.set_data(trail_data[:, 0], trail_data[:, 1])
            point.set_data([sorted_data[frame, 0]], [sorted_data[frame, 1]])
        
        # Highlight phase transitions
        if frame in phase_transitions:
            ax.set_title(f"Phase Transition at step {frame}")
            point.set_color('green')
            point.set_markersize(12)
        else:
            ax.set_title(f"Step {frame}")
            point.set_color('red')
            point.set_markersize(8)
        
        return trail, point
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(sorted_data)),
                         interval=interval, blit=True)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if requested
    if save_path:
        anim.save(save_path, writer='pillow', fps=30)
        logger.info(f"Animation saved to {save_path}")
    
    return AnimationWrapper(fig, anim)

def visualize_statistical_attractor(data: Dict[str, np.ndarray],
                                  spiral_metrics: Dict[str, Any] = None,
                                  title: str = "Statistical Attractor Visualization",
                                  highlight_phase_transitions: bool = True,
                                  save_path: Optional[str] = None,
                                  show_fig: bool = True) -> Optional[plt.Figure]:
    """
    Visualize spiral attractors in statistical data.
    
    Parameters:
    -----------
    data : Dict[str, np.ndarray]
        Dictionary of statistical time series
    spiral_metrics : Dict[str, Any], optional
        Metrics from spiral detection for enhanced visualization
    title : str, default="Statistical Attractor Visualization"
        Title for the visualization
    highlight_phase_transitions : bool, default=True
        Whether to highlight phase transitions
    save_path : str, optional
        Path to save the visualization
    show_fig : bool, default=True
        Whether to display the figure
        
    Returns:
    --------
    Optional[plt.Figure]
        The created figure, or None if no figure was created
    """
    logger.info("Creating statistical attractor visualization")
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    
    # 1. Time Series Plot (top-left)
    ax_time = axes[0, 0]
    _visualize_time_series(ax_time, data, spiral_metrics, highlight_phase_transitions)
    
    # 2. Phase Space Plot (top-right)
    ax_phase = axes[0, 1]
    _visualize_phase_space(ax_phase, data, spiral_metrics)
    
    # 3. Spectral Analysis (bottom-left)
    ax_spectral = axes[1, 0]
    _visualize_spectral_analysis(ax_spectral, data, spiral_metrics)
    
    # 4. Attractor Metrics (bottom-right)
    ax_metrics = axes[1, 1]
    _visualize_statistical_metrics(ax_metrics, spiral_metrics)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        logger.info(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show_fig:
        plt.show()
    
    return fig

def compare_attractors(systems: List[Tuple[str, np.ndarray, Dict[str, Any]]],
                      title: str = "Cross-System Attractor Comparison",
                      metrics_to_compare: List[str] = None,
                      save_path: Optional[str] = None,
                      show_fig: bool = True) -> Optional[plt.Figure]:
    """
    Create comparative visualization of attractors across different systems.
    
    Parameters:
    -----------
    systems : List[Tuple[str, np.ndarray, Dict[str, Any]]]
        List of (system_name, visualization_data, metrics) tuples
    title : str, default="Cross-System Attractor Comparison"
        Title for the visualization
    metrics_to_compare : List[str], optional
        Specific metrics to compare across systems
    save_path : str, optional
        Path to save the visualization
    show_fig : bool, default=True
        Whether to display the figure
        
    Returns:
    --------
    Optional[plt.Figure]
        The created figure, or None if no figure was created
    """
    logger.info(f"Creating cross-system comparison of {len(systems)} attractors")
    
    # Determine number of systems to compare
    n_systems = len(systems)
    if n_systems == 0:
        logger.warning("No systems provided for comparison")
        return None
    
    # Determine default metrics to compare if not specified
    if metrics_to_compare is None:
        # Collect all available metrics across systems
        all_metrics = set()
        for _, _, metrics in systems:
            if metrics:
                all_metrics.update(metrics.keys())
        
        # Select common important metrics
        important_metrics = [
            'winding_count', 'spiral_density', 'spiral_tightness', 
            'resilience_index', 'pattern_stability'
        ]
        
        metrics_to_compare = [m for m in important_metrics if m in all_metrics]
        
        # If we still don't have enough metrics, add some others
        if len(metrics_to_compare) < 3:
            additional_metrics = [m for m in all_metrics if m not in metrics_to_compare]
            metrics_to_compare.extend(additional_metrics[:5 - len(metrics_to_compare)])
    
    # Create figure layout
    if n_systems <= 3:
        # For 2-3 systems: Visualization + radar chart
        fig = plt.figure(figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1] * n_systems / 2), dpi=DEFAULT_DPI)
        gs = plt.GridSpec(n_systems, 2, figure=fig)
        
        # Radar chart for metrics comparison
        ax_radar = fig.add_subplot(gs[:, 1], polar=True)
        
        # System visualizations
        system_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_systems)]
    else:
        # For 4+ systems: Grid of visualizations + separate radar chart
        rows = int(np.ceil(n_systems / 2))
        fig = plt.figure(figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1] * rows / 2), dpi=DEFAULT_DPI)
        gs = plt.GridSpec(rows + 1, 2, figure=fig, height_ratios=[1] * rows + [0.8])
        
        # Radar chart for metrics comparison
        ax_radar = fig.add_subplot(gs[rows, :], polar=True)
        
        # System visualizations
        system_axes = []
        for i in range(n_systems):
            row = i // 2
            col = i % 2
            system_axes.append(fig.add_subplot(gs[row, col]))
    
    # Visualize each system
    for i, (system_name, data, metrics) in enumerate(systems):
        _visualize_system_summary(system_axes[i], system_name, data, metrics)
    
    # Create radar chart for metrics comparison
    _create_metrics_radar(ax_radar, systems, metrics_to_compare)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        logger.info(f"Comparison visualization saved to {save_path}")
    
    # Show if requested
    if show_fig:
        plt.show()
    
    return fig

# Helper functions for visualizations

def _visualize_grid(ax, grid, spiral_metrics=None, highlight_residue=True):
    """Visualize a spatial grid."""
    # Basic grid visualization
    im = ax.imshow(grid, cmap=DEFAULT_CMAP, interpolation='nearest')
    ax.set_title("Current State")
    
    # Add spiral center marker if metrics are available
    if spiral_metrics and 'centroid' in spiral_metrics:
        centroid = spiral_metrics['centroid']
        ax.plot(centroid[0], centroid[1], 'ro', markersize=10, alpha=0.7)
        
        # Add annotation
        spiral_type = spiral_metrics.get('spiral_type', 'Unknown')
        ax.text(centroid[0] + 5, centroid[1] + 5, f"Type: {spiral_type}", 
               color='white', backgroundcolor='black', alpha=0.7)
    
    # Highlight unexplained patterns (residue) if requested
    if highlight_residue and spiral_metrics and 'unexplained_regions' in spiral_metrics:
        for region in spiral_metrics['unexplained_regions']:
            # Draw rectangle around unexplained region
            x, y, w, h = region['bounds']
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

def _visualize_history(ax, history, spiral_metrics=None):
    """Visualize history of a spatial system."""
    # Create space-time plot (horizontal axis: time, vertical axis: space)
    if len(history[0].shape) == 2:
        # For 2D grids, take middle row as representative
        middle_row = history[0].shape[0] // 2
        space_time = np.array([grid[middle_row, :] for grid in history])
    else:
        # For 1D grids, use directly
        space_time = np.array(history)
    
    # Transpose to have time on horizontal axis
    space_time = space_time.T
    
    # Visualize
    im = ax.imshow(space_time, cmap=DEFAULT_CMAP, aspect='auto', interpolation='nearest')
    ax.set_title("Space-Time Diagram")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Space")
    
    # Mark phase transition if available
    if spiral_metrics and 'formation_step' in spiral_metrics:
        formation_step = spiral_metrics['formation_step']
        ax.axvline(x=formation_step, color='r', linestyle='--', alpha=0.7)
        ax.text(formation_step + 5, space_time.shape[0] // 2, "Attractor Forms", 
               rotation=90, verticalalignment='center')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

def _visualize_spiral_polar(ax, grid, spiral_metrics):
    """Visualize spiral pattern in polar coordinates."""
    if not spiral_metrics or 'centroid' not in spiral_metrics:
        ax.text(0.5, 0.5, "No spiral metrics available", 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract centroid
    centroid_x, centroid_y = spiral_metrics['centroid']
    
    # Get active points in the grid
    y_indices, x_indices = np.where(grid > 0)
    
    # Convert to polar coordinates relative to centroid
    r_values = []
    theta_values = []
    
    for y, x in zip(y_indices, x_indices):
        dx = x - centroid_x
        dy = y - centroid_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Only include points within reasonable distance
        if r < max(grid.shape) / 2:
            r_values.append(r)
            theta_values.append(theta)
    
    # Plot points in polar coordinates
    ax.scatter(theta_values, r_values, c=r_values, cmap='viridis', alpha=0.7)
    
    # Add spiral fit if available
    if 'log_spiral_params' in spiral_metrics:
        a = spiral_metrics['log_spiral_params']['a']
        b = spiral_metrics['log_spiral_params']['b']
        
        # Generate points along the fitted spiral
        theta_range = np.linspace(min(theta_values), max(theta_values), 100)
        r_fit = a * np.exp(b * theta_range)
        
        ax.plot(theta_range, r_fit, 'r-', linewidth=2, alpha=0.7, label='Log Spiral Fit')
    
    # Add spiral metrics as text
    fit_quality = spiral_metrics.get('spiral_fit_quality', None)
    winding_count = spiral_metrics.get('winding_count', None)
    spiral_direction = spiral_metrics.get('spiral_direction', None)
    
    metrics_text = []
    if winding_count is not None:
        metrics_text.append(f"Windings: {winding_count:.2f}")
    if spiral_direction is not None:
        metrics_text.append(f"Direction: {spiral_direction}")
    if fit_quality is not None:
        metrics_text.append(f"Fit Quality: {fit_quality:.2f}")
    
    if metrics_text:
        ax.set_title("\n".join(metrics_text))
    else:
        ax.set_title("Spiral Pattern")
    
    # Set axis labels
    ax.set_xlabel("Theta (radians)")
    ax.set_ylabel("Radius")

def _visualize_metrics(ax, spiral_metrics):
    """Visualize key spiral metrics."""
    if not spiral_metrics:
        ax.text(0.5, 0.5, "No spiral metrics available", 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Select key metrics to visualize
    key_metrics = [
        ('winding_count', 'Winding Count'),
        ('spiral_density', 'Spiral Density'),
        ('spiral_tightness', 'Spiral Tightness'),
        ('pattern_stability', 'Pattern Stability'),
        ('resilience_index', 'Resilience Index'),
        ('spiral_fit_quality', 'Fit Quality')
    ]
    
    # Filter available metrics
    available_metrics = [(k, v) for k, v in key_metrics if k in spiral_metrics]
    
    if not available_metrics:
        ax.text(0.5, 0.5, "No key metrics available", 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create bar chart of metrics
    metric_names = [v for _, v in available_metrics]
    metric_values = [spiral_metrics[k] for k, _ in available_metrics]
    
    bars = ax.bar(metric_names, metric_values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f}', ha='center', va='bottom')
    
    # Adjust layout
    ax.set_title("Key Spiral Metrics")
    ax.set_ylim(0, max(metric_values) * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def _visualize_2d_projection(ax, projected_data, colors=None, spiral_metrics=None):
    """Visualize 2D projection of data."""
    # Extract x and y coordinates
    x, y = projected_data[:, 0], projected_data[:, 1]
    
    # Plot points with color mapping if available
    if colors is not None:
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        ax.scatter(x, y, c='blue', s=30, alpha=0.7)
    
    # Connect points with lines to show trajectory
    ax.plot(x, y, 'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    
    # Add spiral fit if available
    if spiral_metrics and 'is_spiral' in spiral_metrics and spiral_metrics['is_spiral']:
        # Implement spiral fit visualization
        if 'centroid' in spiral_metrics:
            center_x, center_y = spiral_metrics['centroid']
            ax.plot(center_x, center_y, 'mo', markersize=8, label='Center')
        
        # Annotate with spiral metrics
        spiral_type = spiral_metrics.get('spiral_type', 'Unknown')
        winding_count = spiral_metrics.get('winding_count', 0)
        ax.set_title(f"{spiral_type} Spiral (Windings: {winding_count:.2f})")
    else:
        ax.set_title("2D Projection")
    
    # Add legend, labels, and grid
    ax.legend(loc='best')
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

def _visualize_3d_projection(ax, projected_data, colors=None, spiral_metrics=None):
    """Visualize 3D projection of data."""
    # Extract x, y, and z coordinates
    x, y, z = projected_data[:, 0], projected_data[:, 1], projected_data[:, 2]
    
    # Plot points with color mapping if available
    if colors is not None:
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Time')
    else:
        ax.scatter(x, y, z, c='blue', s=30, alpha=0.7)
    
    # Connect points with lines to show trajectory
    ax.plot(x, y, z, 'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=10, label='Start')
    ax.plot([x[-1]], [y[-1]], [z[-1]], 'ro', markersize=10, label='End')
    
    # Add spiral information if available
    if spiral_metrics and '3d_spiral_type' in spiral_metrics:
        spiral_type = spiral_metrics['3d_spiral_type']
        ax.set_title(f"3D {spiral_type} Spiral")
        
        # Add more information about the 3D spiral
        helix_direction = spiral_metrics.get('helix_direction', 'unknown')
        helix_pitch = spiral_metrics.get('helix_pitch', 0)
        
        # Add text annotation
        ax.text2D(0.05, 0.95, f"Type: {spiral_type}\nDirection: {helix_direction}\nPitch: {helix_pitch:.2f}", 
                 transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        ax.set_title("3D Projection")
    
    # Add legend, labels, and grid
    ax.legend(loc='best')
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.grid(True, alpha=0.3)

def _visualize_temporal_metrics(ax, time_indices, spiral_metrics, highlight_phase_transitions=True):
    """Visualize temporal metrics of projected data."""
    # Sort time indices
    sorted_indices = np.argsort(time_indices)
    sorted_times = time_indices[sorted_indices]
    
    # Calculate velocity if not provided
    if 'velocities' in spiral_metrics:
        velocities = spiral_metrics['velocities']
    elif len(sorted_times) > 1:
        # Use time differences as proxy for velocity
        velocities = np.diff(sorted_times)
        velocities = np.append(velocities, velocities[-1])  # Repeat last value for length matching
    else:
        velocities = np.ones_like(sorted_times)
    
    # Plot velocity over time
    ax.plot(sorted_times, velocities, 'b-', linewidth=2, label='Velocity')
    
    # Highlight phase transitions if available
    if highlight_phase_transitions and 'transition_points' in spiral_metrics:
        transition_points = spiral_metrics['transition_points']
        for point in transition_points:
            if 0 <= point < len(sorted_times):
                ax.axvline(x=sorted_times[point], color='r', linestyle='--', alpha=0.7)
                ax.text(sorted_times[point], max(velocities) * 0.9, "Phase\nTransition", 
                       rotation=90, verticalalignment='top', horizontalalignment='right')
    
    # Mark formation point if available
    if 'formation_start' in spiral_metrics:
        formation_start = spiral_metrics['formation_start']
        if 0 <= formation_start < len(sorted_times):
            ax.axvline(x=sorted_times[formation_start], color='g', linestyle='-', alpha=0.7)
            ax.text(sorted_times[formation_start], max(velocities) * 0.7, "Attractor\nForms", 
                   rotation=90, verticalalignment='top', horizontalalignment='right')
    
    # Add title and labels
    ax.set_title("Temporal Dynamics")
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.grid(True, alpha=0.3)

def _visualize_time_series(ax, data, spiral_metrics=None, highlight_phase_transitions=True):
    """Visualize time series data with phase transitions."""
    # Select a subset of time series if there are many
    if len(data) > 5:
        # Choose most important ones or a sample
        keys = sorted(data.keys())
        selected_keys = keys[:5]  # Just take first 5 for simplicity
    else:
        selected_keys = sorted(data.keys())
    
    # Plot each selected time series
    for i, key in enumerate(selected_keys):
        values = data[key]
        ax.plot(values, label=key, alpha=0.7)
    
    # Highlight phase transitions if available
    if highlight_phase_transitions and spiral_metrics and 'phase_transition_points' in spiral_metrics:
        transition_points = spiral_metrics['phase_transition_points']
        for point in transition_points:
            ax.axvline(x=point, color='r', linestyle='--', alpha=0.5)
    
    # Mark spiral phase start if available
    if spiral_metrics and 'spiral_phase_start' in spiral_metrics:
        start_point = spiral_metrics['spiral_phase_start']
        ax.axvline(x=start_point, color='g', linestyle='-', alpha=0.7)
        ax.text(start_point + 5, ax.get_ylim()[1] * 0.9, "Attractor Forms", 
               rotation=90, verticalalignment='top')
    
    # Add title, legend, and grid
    ax.set_title("Time Series Data")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def _visualize_phase_space(ax, data, spiral_metrics=None):
    """Visualize phase space of time series data."""
    # Need at least two time series for phase space
    keys = sorted(data.keys())
    if len(keys) < 2:
        ax.text(0.5, 0.5, "Insufficient data for phase space", 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Select two time series for phase space visualization
    x_key, y_key = keys[0], keys[1]
    x_values = data[x_key]
    y_values = data[y_key]
    
    # Ensure same length
    min_length = min(len(x_values), len(y_values))
    x_values = x_values[:min_length]
    y_values = y_values[:min_length]
    
    # Color points by time
    colors = np.arange(min_length)
    
    # Plot phase space
    scatter = ax.scatter(x_values, y_values, c=colors, cmap='viridis', s=30, alpha=0.7)
    
    # Connect points to show trajectory
    ax.plot(x_values, y_values, 'k-', alpha=0.3, linewidth=1)
    
    # Mark start and end points
    ax.plot(x_values[0], y_values[0], 'go', markersize=8, label='Start')
    ax.plot(x_values[-1], y_values[-1], 'ro', markersize=8, label='End')
    
    # Mark spiral phase if available
    if spiral_metrics and 'spiral_phase_start' in spiral_metrics:
        start_idx = spiral_metrics['spiral_phase_start']
        if start_idx < min_length:
            ax.plot(x_values[start_idx], y_values[start_idx], 'mo', markersize=8, label='Attractor Forms')
    
    # Add colorbar, title, labels, and grid
    plt.colorbar(scatter, ax=ax, label='Time Step')
    ax.set_title("Phase Space")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def _visualize_spectral_analysis(ax, data, spiral_metrics=None):
    """Visualize spectral analysis of time series data."""
    # Select most important time series
    keys = sorted(data.keys())
    if not keys:
        ax.text(0.5, 0.5, "No data available for spectral analysis", 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Use first time series for spectral analysis
    key = keys[0]
    values = data[key]
    
    # Compute power spectrum
    from scipy import signal
    frequencies, power = signal.periodogram(values, fs=1.0, scaling='spectrum')
    
    # Plot power spectrum
    ax.semilogy(frequencies, power, 'b-', linewidth=2)
    
    # Highlight dominant frequency
    if len(frequencies) > 1:
        dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
        dominant_freq = frequencies[dominant_idx]
        dominant_power = power[dominant_idx]
        
        ax.plot(dominant_freq, dominant_power, 'ro', markersize=8)
        ax.text(dominant_freq, dominant_power, f"  {dominant_freq:.4f} Hz", 
               verticalalignment='bottom')
    
    # Add cycle period if available
    if spiral_metrics and 'cycle_period' in spiral_metrics:
        cycle_period = spiral_metrics['cycle_period']
        cycle_freq = 1.0 / cycle_period if cycle_period > 0 else 0
        
        ax.axvline(x=cycle_freq, color='g', linestyle='--', alpha=0.7)
        ax.text(cycle_freq, ax.get_ylim()[1] * 0.5, f"Cycle: {cycle_period:.1f} steps", 
               rotation=90, verticalalignment='top', horizontalalignment='right')
    
    # Add title, labels, and grid
    ax.set_title("Spectral Analysis")
    ax.set_xlabel("Frequency (1/step)")
    ax.set_ylabel("Power Spectrum")
    ax.grid(True, alpha=0.3)

def _visualize_statistical_metrics(ax, spiral_metrics):
    """Visualize statistical metrics of the attractor."""
    if not spiral_metrics:
        ax.text(0.5, 0.5, "No spiral metrics available", 
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create text summary of key metrics
    metrics_text = []
    
    # Cycle information
    if 'cycle_count' in spiral_metrics:
        metrics_text.append(f"Cycle Count: {spiral_metrics['cycle_count']:.2f}")
    if 'cycle_period' in spiral_metrics:
        metrics_text.append(f"Cycle Period: {spiral_metrics['cycle_period']:.2f} steps")
    
    # Phase transition information
    if 'phase_transitions' in spiral_metrics:
        metrics_text.append(f"Phase Transitions: {spiral_metrics['phase_transitions']}")
    
    # Entropy information
    if 'pre_spiral_entropy' in spiral_metrics and 'spiral_entropy' in spiral_metrics:
        pre = spiral_metrics['pre_spiral_entropy']
        post = spiral_metrics['spiral_entropy']
        metrics_text.append(f"Pre-Spiral Entropy: {pre:.2f}")
        metrics_text.append(f"Spiral Entropy: {post:.2f}")
        metrics_text.append(f"Entropy Change: {post-pre:.2f}")
    
    # Additional metrics
    if 'winding_count' in spiral_metrics:
        metrics_text.append(f"Winding Count: {spiral_metrics['winding_count']:.2f}")
    if 'spiral_direction' in spiral_metrics:
        metrics_text.append(f"Spiral Direction: {spiral_metrics['spiral_direction']}")
    if 'spiral_fit_quality' in spiral_metrics:
        metrics_text.append(f"Fit Quality: {spiral_metrics['spiral_fit_quality']:.2f}")
    
    # Display metrics
    if metrics_text:
        ax.text(0.05, 0.95, "\n".join(metrics_text), transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No detailed metrics available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Add title
    ax.set_title("Attractor Metrics")
    
    # Remove axes for cleaner look
    ax.axis('off')

def _visualize_system_summary(ax, system_name, data, metrics):
    """Visualize summary of a system for comparison."""
    # Determine data type and visualize accordingly
    if isinstance(data, np.ndarray) and len(data.shape) == 2:
        # Spatial data (grid)
        im = ax.imshow(data, cmap=DEFAULT_CMAP, interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.8)
    elif isinstance(data, np.ndarray) and data.shape[1] >= 2:
        # Projected data
        if data.shape[1] >= 3:
            # 3D data - show 2D projection
            ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=30, alpha=0.7)
        else:
            # 2D data
            ax.scatter(data[:, 0], data[:, 1], c='blue', s=30, alpha=0.7)
            ax.plot(data[:, 0], data[:, 1], 'k-', alpha=0.3, linewidth=1)
    elif isinstance(data, dict):
        # Statistical data - show first time series
        keys = sorted(data.keys())
        if keys:
            values = data[keys[0]]
            ax.plot(values, 'b-', linewidth=2)
            ax.set_xlabel("Time Step")
            ax.set_ylabel(keys[0])
    else:
        # Unknown data type
        ax.text(0.5, 0.5, "No visualization available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Add key metrics as text
    metrics_text = []
    if metrics:
        # Select a few key metrics
        if 'spiral_type' in metrics:
            metrics_text.append(f"Type: {metrics['spiral_type']}")
        elif '3d_spiral_type' in metrics:
            metrics_text.append(f"Type: {metrics['3d_spiral_type']}")
        
        if 'winding_count' in metrics:
            metrics_text.append(f"Windings: {metrics['winding_count']:.2f}")
        
        if 'resilience_index' in metrics:
            metrics_text.append(f"Resilience: {metrics['resilience_index']:.2f}")
        
        if 'pattern_stability' in metrics:
            metrics_text.append(f"Stability: {metrics['pattern_stability']:.2f}")
    
    # Add metrics text
    if metrics_text:
        ax.text(0.05, 0.95, "\n".join(metrics_text), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add system name as title
    ax.set_title(system_name)

def _create_metrics_radar(ax, systems, metrics_to_compare):
    """Create radar chart for comparing metrics across systems."""
    if not systems or not metrics_to_compare:
        ax.text(0, 0, "No metrics available for comparison", 
               ha='center', va='center')
        return
    
    # Count metrics and systems
    n_metrics = len(metrics_to_compare)
    n_systems = len(systems)
    
    if n_metrics < 3:
        ax.text(0, 0, "Need at least 3 metrics for radar chart", 
               ha='center', va='center')
        return
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up colors for systems
    colors = plt.cm.tab10(np.linspace(0, 1, n_systems))
    
    # Normalize metrics for radar chart
    # First collect all values
    all_values = {}
    for metric in metrics_to_compare:
        all_values[metric] = []
        for _, _, system_metrics in systems:
            if system_metrics and metric in system_metrics:
                all_values[metric].append(system_metrics[metric])
    
    # Find min and max for each metric
    metric_ranges = {}
    for metric, values in all_values.items():
        if values:
            metric_ranges[metric] = (min(values), max(values))
        else:
            metric_ranges[metric] = (0, 1)  # Default range if no values
    
    # Normalize function
    def normalize(value, metric):
        min_val, max_val = metric_ranges[metric]
        if max_val == min_val:
            return 0.5  # Avoid division by zero
        return (value - min_val) / (max_val - min_val)
    
    # Plot each system
    for i, (system_name, _, system_metrics) in enumerate(systems):
        if not system_metrics:
            continue
        
        values = []
        for metric in metrics_to_compare:
            if metric in system_metrics:
                values.append(normalize(system_metrics[metric], metric))
            else:
                values.append(0)  # Default for missing metrics
        
        # Close the loop
        values += values[:1]
        
        # Plot the system
        ax.plot(angles, values, color=colors[i], linewidth=2, label=system_name)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_compare)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    ax.set_title("Metrics Comparison")

# If this module is run directly, perform a simple test
if __name__ == "__main__":
    # Generate synthetic spiral data for testing
    def generate_spiral(n_points=500, noise_level=0.1):
        t = np.linspace(0, 10*np.pi, n_points)
        a, b = 1, 0.2
        r = a * np.exp(b * t)
        x = r * np.cos(t) + noise_level * np.random.randn(n_points)
        y = r * np.sin(t) + noise_level * np.random.randn(n_points)
        grid = np.zeros((100, 100))
        for i in range(n_points):
            ix, iy = int(x[i] + 50), int(y[i] + 50)
            if 0 <= ix < 100 and 0 <= iy < 100:
                grid[iy, ix] = 1
        
        # Create history
        history = [np.roll(grid, shift=i, axis=0) for i in range(10)]
        
        # Create mock metrics
        metrics = {
            'is_spiral': True,
            'centroid': (50, 50),
            'winding_count': 5.2,
            'spiral_direction': 'expanding',
            'spiral_density': 0.75,
            'spiral_tightness': 0.2,
            'log_spiral_params': {'a': 1.0, 'b': 0.2},
            'spiral_fit_quality': 0.92,
            'formation_step': 5,
            'pattern_stability': 0.95,
            'spiral_type': 'logarithmic_spiral',
            'resilience_index': 0.85
        }
        
        return grid, history, metrics
    
    # Test visualization
    test_grid, test_history, test_metrics = generate_spiral()
    
    # Test spatial visualization
    visualize_spatial_attractor(test_grid, test_history, test_metrics, 
                               title="Test Spiral Visualization")
    
    # Test animation
    anim = create_spatial_animation(test_history, test_metrics, 
                                   title="Test Spiral Animation")
    plt.close()  # Close animation figure
