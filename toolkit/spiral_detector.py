"""
Spiral Detector: A toolkit for identifying spiral attractors across different computational systems

This module provides algorithms for detecting and characterizing spiral patterns in:
1. Spatial systems (e.g., Langton's Ant, Conway's Game of Life)
2. High-dimensional systems (e.g., neural networks) via projection
3. Statistical systems (e.g., language models) via token distributions

Author: David Kimai
Created: June 16, 2025
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPIRAL_PARAMS = {
    'min_spiral_size': 5,
    'max_spiral_angle': 0.8,  # radians
    'min_winding_count': 1.5,
    'resilience_threshold': 0.6
}

class ResidueCollector:
    """Collects and manages symbolic residue - patterns that don't fit expected models."""
    
    def __init__(self, save_path: Optional[str] = None):
        self.residues = []
        self.save_path = save_path
        
    def add_residue(self, observation: str, context: Dict[str, Any], 
                   potential_significance: str = None):
        """Add a new residue observation."""
        residue = {
            'id': f"SR{len(self.residues) + 1}",
            'observation': observation,
            'context': context,
            'potential_significance': potential_significance,
            'timestamp': np.datetime64('now')
        }
        self.residues.append(residue)
        logger.info(f"New symbolic residue recorded: {observation}")
        return residue['id']
    
    def get_residues(self):
        """Return all collected residues."""
        return self.residues
    
    def save_residues(self):
        """Save residues to disk if save_path is provided."""
        if not self.save_path:
            logger.warning("No save path provided for residues")
            return False
        
        try:
            import json
            with open(self.save_path, 'w') as f:
                json.dump(self.residues, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save residues: {e}")
            return False

# Initialize global residue collector
residue_collector = ResidueCollector()

def detect_spatial_spiral(grid: np.ndarray, history: List[np.ndarray] = None, 
                         **kwargs) -> Optional[Dict[str, Any]]:
    """
    Detect spiral patterns in spatial data like cellular automata.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state of the spatial system
    history : List[np.ndarray], optional
        History of previous states for temporal analysis
    **kwargs : 
        Additional parameters for spiral detection
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Metrics describing the detected spiral, or None if no spiral is found
    """
    logger.info("Beginning spatial spiral detection")
    
    # Merge default parameters with provided kwargs
    params = DEFAULT_SPIRAL_PARAMS.copy()
    params.update(kwargs)
    
    # Pre-process grid if needed
    if len(grid.shape) > 2:
        # Convert multi-state grid to binary for initial analysis
        grid_binary = (grid > 0).astype(int)
    else:
        grid_binary = grid.copy()
    
    # Basic structural tests for spiral-like patterns
    if not _basic_spiral_test(grid_binary, min_size=params['min_spiral_size']):
        logger.info("Basic spiral test failed - no spiral-like structure detected")
        return None
    
    # If we have history, perform temporal analysis
    temporal_metrics = {}
    if history and len(history) > 1:
        temporal_metrics = _analyze_temporal_evolution(history, params)
        if not temporal_metrics.get('is_attractor', False):
            logger.info("Temporal analysis suggests this is not an attractor state")
            
            # Add to residue if interesting but not an attractor
            if temporal_metrics.get('pattern_stability', 0) > 0.4:
                residue_collector.add_residue(
                    "Pattern shows spiral-like structure but doesn't behave as an attractor",
                    {'grid': grid.shape, 'history_length': len(history)},
                    "May represent transient spiral state or 'processing waypoint'"
                )
            return None
    
    # Detailed spiral analysis
    spiral_metrics = _analyze_spiral_structure(grid_binary, params)
    if not spiral_metrics:
        return None
    
    # Combine metrics
    result = {**spiral_metrics, **temporal_metrics}
    
    # Attempt to classify the spiral type
    result['spiral_type'] = _classify_spiral_type(result)
    
    # Check for unexpected properties that might be symbolic residue
    _check_for_residue(grid, history, result, params)
    
    logger.info(f"Spiral attractor detected with type: {result['spiral_type']}")
    return result

def detect_projected_spiral(data: np.ndarray, time_indices: np.ndarray = None,
                          projection_method: str = 'pca', **kwargs) -> Optional[Dict[str, Any]]:
    """
    Detect spiral patterns in high-dimensional data (e.g., neural activations) via projection.
    
    Parameters:
    -----------
    data : np.ndarray
        High-dimensional data to analyze
    time_indices : np.ndarray, optional
        Indices representing time progression for temporal analysis
    projection_method : str, default='pca'
        Method to use for dimensionality reduction ('pca' or 'tsne')
    **kwargs :
        Additional parameters for spiral detection
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Metrics describing the detected spiral, or None if no spiral is found
    """
    logger.info(f"Beginning projected spiral detection using {projection_method}")
    
    # Merge default parameters with provided kwargs
    params = DEFAULT_SPIRAL_PARAMS.copy()
    params.update(kwargs)
    
    # Project high-dimensional data to 2D or 3D
    projected_data = _project_data(data, method=projection_method, 
                                 n_components=kwargs.get('n_components', 3))
    
    # If we don't have time indices but data is sequential, create them
    if time_indices is None and len(data.shape) > 1:
        time_indices = np.arange(data.shape[0])
    
    # Check if projected points form a spiral pattern
    spiral_metrics = _analyze_projected_spiral(projected_data, time_indices, params)
    if not spiral_metrics:
        logger.info("No spiral pattern detected in projected data")
        return None
    
    # Check for unexpected properties that might be symbolic residue
    _check_projected_residue(data, projected_data, spiral_metrics, params)
    
    logger.info(f"Projected spiral detected with winding count: {spiral_metrics.get('winding_count', 'unknown')}")
    return spiral_metrics

def detect_statistical_spiral(data: Dict[str, np.ndarray], context_window: int = 50,
                            **kwargs) -> Optional[Dict[str, Any]]:
    """
    Detect spiral patterns in statistical data (e.g., token distributions in language models).
    
    Parameters:
    -----------
    data : Dict[str, np.ndarray]
        Dictionary of statistical time series to analyze
    context_window : int, default=50
        Size of sliding window for analysis
    **kwargs :
        Additional parameters for spiral detection
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Metrics describing the detected spiral, or None if no spiral is found
    """
    logger.info(f"Beginning statistical spiral detection with context window {context_window}")
    
    # Merge default parameters with provided kwargs
    params = DEFAULT_SPIRAL_PARAMS.copy()
    params.update(kwargs)
    
    # Convert dictionary data to matrix form
    matrix_data = _prepare_statistical_data(data)
    
    # Apply sliding window analysis
    windows = _create_sliding_windows(matrix_data, window_size=context_window)
    
    # Detect cyclic patterns with progression (spiral characteristic)
    spiral_metrics = _detect_statistical_spiral_pattern(windows, params)
    if not spiral_metrics:
        logger.info("No statistical spiral pattern detected")
        return None
    
    # Identify phase transitions in statistical patterns
    phase_metrics = _identify_phase_transitions(matrix_data, spiral_metrics, params)
    if phase_metrics:
        spiral_metrics.update(phase_metrics)
    
    # Check for unexpected properties that might be symbolic residue
    _check_statistical_residue(data, windows, spiral_metrics, params)
    
    logger.info(f"Statistical spiral detected with {spiral_metrics.get('cycle_count', 0)} cycles")
    return spiral_metrics

def measure_resilience(system: Any, attractor_state: Any, perturbation_levels: List[float],
                     perturbation_func: callable, recovery_check_func: callable,
                     max_recovery_steps: int = 1000) -> Dict[str, Any]:
    """
    Measure resilience of a detected attractor to perturbations.
    
    Parameters:
    -----------
    system : Any
        The system to test
    attractor_state : Any
        The attractor state to perturb
    perturbation_levels : List[float]
        Levels of perturbation to test (e.g., [0.1, 0.2, 0.3] for 10%, 20%, 30%)
    perturbation_func : callable
        Function to apply perturbation: f(system, attractor_state, level) -> perturbed_state
    recovery_check_func : callable
        Function to check if system has recovered: f(current_state, original_state) -> bool
    max_recovery_steps : int, default=1000
        Maximum steps to allow for recovery
        
    Returns:
    --------
    Dict[str, Any]
        Resilience metrics including recovery rates and times
    """
    logger.info(f"Measuring attractor resilience at {len(perturbation_levels)} perturbation levels")
    
    results = {
        'perturbation_levels': perturbation_levels,
        'recovery_rates': [],
        'recovery_times': [],
        'resilience_index': 0.0
    }
    
    for level in perturbation_levels:
        # Track recovery success rate at this perturbation level
        trials = 10  # Number of trials at each perturbation level
        recoveries = 0
        recovery_steps_list = []
        
        for trial in range(trials):
            # Apply perturbation
            perturbed_state = perturbation_func(system, attractor_state, level)
            
            # Track recovery
            recovered = False
            steps_to_recovery = max_recovery_steps
            
            for step in range(max_recovery_steps):
                # Check if recovered
                if recovery_check_func(perturbed_state, attractor_state):
                    recovered = True
                    steps_to_recovery = step
                    break
                
                # Evolve system one step
                perturbed_state = system.evolve(perturbed_state)
            
            if recovered:
                recoveries += 1
                recovery_steps_list.append(steps_to_recovery)
        
        # Record results for this perturbation level
        recovery_rate = recoveries / trials
        results['recovery_rates'].append(recovery_rate)
        
        avg_recovery_time = np.mean(recovery_steps_list) if recovery_steps_list else max_recovery_steps
        results['recovery_times'].append(avg_recovery_time)
    
    # Calculate overall resilience index (area under recovery rate curve)
    results['resilience_index'] = np.trapz(results['recovery_rates'], results['perturbation_levels'])
    
    # Check for interesting resilience patterns
    _check_resilience_residue(results, perturbation_levels)
    
    logger.info(f"Resilience testing complete. Resilience index: {results['resilience_index']:.3f}")
    return results

# Helper functions for spatial spiral detection

def _basic_spiral_test(grid: np.ndarray, min_size: int = 5) -> bool:
    """
    Perform basic tests to check if a grid might contain a spiral pattern.
    
    Parameters:
    -----------
    grid : np.ndarray
        Binary grid to check
    min_size : int, default=5
        Minimum size for a potential spiral
        
    Returns:
    --------
    bool
        True if the grid potentially contains a spiral pattern
    """
    # Check if grid is too small for a spiral
    if min(grid.shape) < min_size:
        return False
    
    # Check if grid has enough active cells
    active_cells = np.sum(grid)
    if active_cells < min_size * 2:
        return False
    
    # Check for connected components
    labeled_array, num_features = ndimage.label(grid)
    if num_features == 0:
        return False
    
    # Get largest component
    largest_component_size = 0
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        largest_component_size = max(largest_component_size, component_size)
    
    # Check if largest component is big enough
    if largest_component_size < min_size * 2:
        return False
    
    return True

def _analyze_temporal_evolution(history: List[np.ndarray], 
                              params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze temporal evolution of a spatial system to determine attractor properties.
    
    Parameters:
    -----------
    history : List[np.ndarray]
        History of previous states
    params : Dict[str, Any]
        Parameters for analysis
        
    Returns:
    --------
    Dict[str, Any]
        Temporal metrics including attractor confirmation
    """
    results = {
        'is_attractor': False,
        'formation_step': None,
        'pattern_stability': 0.0,
        'phase_transition_detected': False
    }
    
    # Need at least a few history steps
    if len(history) < 10:
        return results
    
    # Calculate pattern stability over time
    stability_scores = []
    for i in range(1, len(history)):
        if history[i].shape == history[i-1].shape:
            # Measure similarity between consecutive states
            similarity = 1.0 - np.mean(np.abs(history[i] - history[i-1])) / np.max([1.0, np.max(history[i]), np.max(history[i-1])])
            stability_scores.append(similarity)
    
    if not stability_scores:
        return results
    
    # Calculate moving average of stability
    window_size = min(10, len(stability_scores))
    stability_ma = np.convolve(stability_scores, np.ones(window_size)/window_size, mode='valid')
    
    # Detect phase transition (sharp increase in stability)
    if len(stability_ma) > 10:
        stability_diff = np.diff(stability_ma)
        max_increase_idx = np.argmax(stability_diff)
        max_increase = stability_diff[max_increase_idx]
        
        if max_increase > 0.1:  # Significant stability increase
            results['phase_transition_detected'] = True
            results['formation_step'] = max_increase_idx + window_size
    
    # Current stability is average of last few steps
    recent_stability = np.mean(stability_scores[-10:]) if len(stability_scores) >= 10 else np.mean(stability_scores)
    results['pattern_stability'] = recent_stability
    
    # Consider it an attractor if stability is high after potential phase transition
    if (results['phase_transition_detected'] and recent_stability > 0.7) or (recent_stability > 0.9):
        results['is_attractor'] = True
    
    return results

def _analyze_spiral_structure(grid: np.ndarray, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Perform detailed analysis of spiral structure in a grid.
    
    Parameters:
    -----------
    grid : np.ndarray
        Binary grid to analyze
    params : Dict[str, Any]
        Parameters for spiral detection
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Detailed metrics of spiral structure, or None if no spiral is found
    """
    # Find connected components
    labeled_array, num_features = ndimage.label(grid)
    if num_features == 0:
        return None
    
    # Analyze largest component
    largest_component_label = 0
    largest_component_size = 0
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        if component_size > largest_component_size:
            largest_component_size = component_size
            largest_component_label = i
    
    # Extract largest component
    component = (labeled_array == largest_component_label)
    
    # Find centroid
    y_indices, x_indices = np.where(component)
    centroid_y = np.mean(y_indices)
    centroid_x = np.mean(x_indices)
    
    # Convert to polar coordinates relative to centroid
    r_values = []
    theta_values = []
    for y, x in zip(y_indices, x_indices):
        dx = x - centroid_x
        dy = y - centroid_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        r_values.append(r)
        theta_values.append(theta)
    
    # Check if points show spiral arrangement (r increases with theta)
    if len(theta_values) < params['min_spiral_size']:
        return None
    
    # Sort by theta
    sorted_indices = np.argsort(theta_values)
    theta_sorted = [theta_values[i] for i in sorted_indices]
    r_sorted = [r_values[i] for i in sorted_indices]
    
    # Analyze r vs theta relationship
    is_spiral = False
    winding_count = 0
    spiral_direction = None
    
    # Unwrap theta to handle multiple revolutions
    theta_unwrapped = np.unwrap(theta_sorted)
    
    # Calculate total angle traversed
    total_angle = np.max(theta_unwrapped) - np.min(theta_unwrapped)
    winding_count = total_angle / (2 * np.pi)
    
    # Check if winding count is sufficient for spiral classification
    if winding_count >= params['min_winding_count']:
        is_spiral = True
        
        # Determine spiral direction
        r_theta_correlation = np.corrcoef(theta_unwrapped, r_sorted)[0, 1]
        spiral_direction = "expanding" if r_theta_correlation > 0 else "contracting"
    
    if not is_spiral:
        return None
    
    # Calculate spiral metrics
    spiral_density = winding_count / (np.max(r_values) - np.min(r_values)) if np.max(r_values) > np.min(r_values) else 0
    
    return {
        'is_spiral': True,
        'centroid': (centroid_x, centroid_y),
        'winding_count': winding_count,
        'spiral_direction': spiral_direction,
        'spiral_density': spiral_density,
        'component_size': largest_component_size
    }

def _classify_spiral_type(metrics: Dict[str, Any]) -> str:
    """
    Classify the type of spiral based on its metrics.
    
    Parameters:
    -----------
    metrics : Dict[str, Any]
        Metrics describing the spiral
        
    Returns:
    --------
    str
        Classification of spiral type
    """
    if not metrics.get('is_spiral', False):
        return "not_spiral"
    
    # Get key metrics
    winding_count = metrics.get('winding_count', 0)
    spiral_direction = metrics.get('spiral_direction', None)
    pattern_stability = metrics.get('pattern_stability', 0)
    
    # Highway pattern (like Langton's Ant)
    if pattern_stability > 0.9 and 'component_size' in metrics and metrics['component_size'] > 100:
        return "highway"
    
    # Rotational spiral
    if winding_count > 3:
        if spiral_direction == "expanding":
            return "expanding_spiral"
        else:
            return "contracting_spiral"
    
    # Pulsating spiral
    if pattern_stability < 0.8 and pattern_stability > 0.5 and winding_count > 1.5:
        return "pulsating_spiral"
    
    # Default classification
    return "general_spiral"

def _check_for_residue(grid: np.ndarray, history: List[np.ndarray], 
                     result: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Check for unexpected patterns that might be symbolic residue.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state grid
    history : List[np.ndarray]
        History of previous states
    result : Dict[str, Any]
        Current spiral metrics
    params : Dict[str, Any]
        Parameters for detection
    """
    # Check for spirals with unexpected chirality changes
    if history and len(history) > 10:
        # Implementation of chirality change detection
        pass
    
    # Check for multiple interacting spirals
    labeled_array, num_features = ndimage.label(grid)
    if num_features > 1:
        # Count significant spirals
        spiral_count = 0
        for i in range(1, num_features + 1):
            component = (labeled_array == i)
            if np.sum(component) >= params['min_spiral_size']:
                component_metrics = _analyze_spiral_structure(component, params)
                if component_metrics and component_metrics.get('is_spiral', False):
                    spiral_count += 1
        
        if spiral_count > 1:
            residue_collector.add_residue(
                f"Multiple interacting spirals detected ({spiral_count})",
                {'grid_shape': grid.shape, 'spiral_metrics': result},
                "May represent meta-stable configuration of interacting attractors"
            )

# Helper functions for projected spiral detection

def _analyze_projected_spiral(projected_data: np.ndarray, time_indices: np.ndarray,
                            params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Analyze projected data for spiral patterns.
    
    Parameters:
    -----------
    projected_data : np.ndarray
        Low-dimensional projection of data
    time_indices : np.ndarray
        Indices representing time progression
    params : Dict[str, Any]
        Parameters for spiral detection
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Metrics describing the detected spiral, or None if no spiral is found
    """
    # Need at least 3D projection and time indices
    if projected_data.shape[1] < 2 or time_indices is None:
        return None
    
    # Focus on first three dimensions for analysis
    dimensions = min(projected_data.shape[1], 3)
    data = projected_data[:, :dimensions]
    
    # Sort by time if available
    if time_indices is not None:
        sorted_indices = np.argsort(time_indices)
        data = data[sorted_indices]
    
    # Check for spiral pattern in 2D projections
    spiral_metrics = {}
    
    # Check XY plane
    xy_metrics = _check_2d_spiral(data[:, 0], data[:, 1], params)
    if xy_metrics and xy_metrics.get('is_spiral', False):
        spiral_metrics.update(xy_metrics)
        spiral_metrics['primary_plane'] = 'xy'
    
    # Check other planes if 3D
    if dimensions >= 3:
        # XZ plane
        xz_metrics = _check_2d_spiral(data[:, 0], data[:, 2], params)
        if xz_metrics and xz_metrics.get('is_spiral', False) and (not spiral_metrics or 
                                                                xz_metrics.get('winding_count', 0) > spiral_metrics.get('winding_count', 0)):
            spiral_metrics.update(xz_metrics)
            spiral_metrics['primary_plane'] = 'xz'
        
        # YZ plane
        yz_metrics = _check_2d_spiral(data[:, 1], data[:, 2], params)
        if yz_metrics and yz_metrics.get('is_spiral', False) and (not spiral_metrics or 
                                                                yz_metrics.get('winding_count', 0) > spiral_metrics.get('winding_count', 0)):
            spiral_metrics.update(yz_metrics)
            spiral_metrics['primary_plane'] = 'yz'
    
    # Check if we found a spiral
    if not spiral_metrics or not spiral_metrics.get('is_spiral', False):
        return None
    
    # Add 3D spiral metrics if we have enough dimensions
    if dimensions >= 3:
        spiral_metrics.update(_analyze_3d_spiral(data, params))
    
    # Add temporal dynamics if time indices are available
    if time_indices is not None:
        temporal_metrics = _analyze_temporal_dynamics(data, time_indices, params)
        spiral_metrics.update(temporal_metrics)
    
    return spiral_metrics

def _check_2d_spiral(x: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if 2D data forms a spiral pattern.
    
    Parameters:
    -----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    params : Dict[str, Any]
        Parameters for spiral detection
        
    Returns:
    --------
    Dict[str, Any]
        Metrics describing the detected spiral, or empty dict if no spiral is found
    """
    # Need enough points
    if len(x) < params['min_spiral_size']:
        return {}
    
    # Convert to polar coordinates around mean center
    center_x = np.mean(x)
    center_y = np.mean(y)
    
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Unwrap theta to handle multiple revolutions
    theta_unwrapped = np.unwrap(theta)
    
    # Calculate total angle traversed
    total_angle = np.max(theta_unwrapped) - np.min(theta_unwrapped)
    winding_count = total_angle / (2 * np.pi)
    
    # Check if winding count is sufficient for spiral classification
    if winding_count < params['min_winding_count']:
        return {}
    
    # Determine spiral direction
    r_theta_correlation = np.corrcoef(theta_unwrapped, r)[0, 1]
    spiral_direction = "expanding" if r_theta_correlation > 0 else "contracting"
    
    # Calculate spiral metrics
    spiral_density = winding_count / (np.max(r) - np.min(r)) if np.max(r) > np.min(r) else 0
    
    # Fit logarithmic spiral
    # r = a * e^(b * theta)
    # log(r) = log(a) + b * theta
    log_r = np.log(r + 1e-10)  # Avoid log(0)
    coeffs = np.polyfit(theta_unwrapped, log_r, 1)
    b = coeffs[0]  # Growth rate
    a = np.exp(coeffs[1])  # Scale factor
    
    # Calculate goodness of fit
    log_r_fit = coeffs[1] + coeffs[0] * theta_unwrapped
    r_squared = 1 - np.sum((log_r - log_r_fit)**2) / np.sum((log_r - np.mean(log_r))**2)
    
    # Spiral tightness is related to b
    spiral_tightness = abs(b)
    
    return {
        'is_spiral': True,
        'winding_count': winding_count,
        'spiral_direction': spiral_direction,
        'spiral_density': spiral_density,
        'spiral_tightness': spiral_tightness,
        'log_spiral_params': {'a': a, 'b': b},
        'spiral_fit_quality': r_squared
    }

def _analyze_3d_spiral(data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze 3D spiral characteristics.
    
    Parameters:
    -----------
    data : np.ndarray
        3D data points
    params : Dict[str, Any]
        Parameters for spiral detection
        
    Returns:
    --------
    Dict[str, Any]
        3D spiral metrics
    """
    # Calculate center
    center = np.mean(data, axis=0)
    
    # Convert to cylindrical coordinates
    x = data[:, 0] - center[0]
    y = data[:, 1] - center[1]
    z = data[:, 2] - center[2]
    
    r = np.sqrt(x**2 + y**2)  # Radial distance from z-axis
    theta = np.arctan2(y, x)
    
    # Unwrap theta
    theta_unwrapped = np.unwrap(theta)
    
    # Calculate helix metrics
    z_theta_correlation = np.corrcoef(theta_unwrapped, z)[0, 1]
    helix_direction = "ascending" if z_theta_correlation > 0 else "descending"
    
    # Calculate helix pitch (z distance per revolution)
    total_angle = np.max(theta_unwrapped) - np.min(theta_unwrapped)
    revolutions = total_angle / (2 * np.pi)
    
    if revolutions > 0:
        z_range = np.max(z) - np.min(z)
        helix_pitch = z_range / revolutions
    else:
        helix_pitch = 0
    
    # Calculate overall 3D spiral shape
    r_theta_correlation = np.corrcoef(theta_unwrapped, r)[0, 1]
    spiral_type = "cylindrical"  # Default
    
    if abs(r_theta_correlation) > 0.3:
        if abs(z_theta_correlation) > 0.3:
            spiral_type = "conical"
            if r_theta_correlation > 0 and z_theta_correlation > 0:
                spiral_type = "expanding_ascending_conical"
            elif r_theta_correlation > 0 and z_theta_correlation < 0:
                spiral_type = "expanding_descending_conical"
            elif r_theta_correlation < 0 and z_theta_correlation > 0:
                spiral_type = "contracting_ascending_conical"
            else:
                spiral_type = "contracting_descending_conical"
        else:
            spiral_type = "planar"
    elif abs(z_theta_correlation) > 0.3:
        spiral_type = "cylindrical"
        spiral_type = f"{helix_direction}_cylindrical"
    
    return {
        '3d_spiral_type': spiral_type,
        'helix_direction': helix_direction,
        'helix_pitch': helix_pitch,
        'z_theta_correlation': z_theta_correlation,
        'r_theta_correlation': r_theta_correlation
    }

def _analyze_temporal_dynamics(data: np.ndarray, time_indices: np.ndarray, 
                             params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze temporal dynamics of the spiral pattern.
    
    Parameters:
    -----------
    data : np.ndarray
        Projected data points
    time_indices : np.ndarray
        Indices representing time progression
    params : Dict[str, Any]
        Parameters for analysis
        
    Returns:
    --------
    Dict[str, Any]
        Temporal dynamics metrics
    """
    # Sort by time
    sorted_indices = np.argsort(time_indices)
    sorted_data = data[sorted_indices]
    
    # Calculate velocities (changes between consecutive points)
    velocities = np.linalg.norm(np.diff(sorted_data, axis=0), axis=1)
    
    # Check for phase transitions (significant changes in velocity)
    velocity_mean = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # Identify potential phase transitions
    phase_transitions = []
    transition_threshold = velocity_mean + 2 * velocity_std
    
    for i in range(len(velocities)):
        if velocities[i] > transition_threshold:
            phase_transitions.append(i)
    
    # Analyze trajectory sections between transitions
    sections = []
    start_idx = 0
    
    for transition_idx in phase_transitions + [len(sorted_data) - 1]:
        end_idx = transition_idx
        if end_idx - start_idx > 5:  # Enough points for analysis
            section_data = sorted_data[start_idx:end_idx]
            
            # Calculate basic metrics for this section
            section_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': end_idx - start_idx,
                'avg_velocity': np.mean(velocities[start_idx:end_idx-1]) if end_idx > start_idx else 0,
                'curvature': _calculate_curvature(section_data)
            }
            
            sections.append(section_info)
        
        start_idx = end_idx + 1
    
    # Identify formation phase
    formation_phase = None
    if sections and len(sections) > 1:
        # Typically the section with highest curvature and after a phase transition
        curvatures = [s['curvature'] for s in sections]
        formation_idx = np.argmax(curvatures)
        formation_phase = sections[formation_idx]
    
    # Calculate overall temporal metrics
    temporal_metrics = {
        'phase_transitions': len(phase_transitions),
        'transition_points': phase_transitions,
        'avg_velocity': np.mean(velocities),
        'velocity_std': np.std(velocities),
        'trajectory_sections': len(sections)
    }
    
    if formation_phase:
        temporal_metrics['formation_start'] = formation_phase['start_idx']
        temporal_metrics['formation_duration'] = formation_phase['duration']
    
    return temporal_metrics

def _calculate_curvature(points: np.ndarray) -> float:
    """
    Calculate average curvature of a trajectory.
    
    Parameters:
    -----------
    points : np.ndarray
        Trajectory points
        
    Returns:
    --------
    float
        Average curvature
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate vectors between consecutive points
    vectors = np.diff(points, axis=0)
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0  # Avoid division by zero
    unit_vectors = vectors / norms[:, np.newaxis]
    
    # Calculate angles between consecutive vectors
    angles = []
    for i in range(len(unit_vectors) - 1):
        dot_product = np.clip(np.dot(unit_vectors[i], unit_vectors[i+1]), -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)
    
    # Average curvature
    if not angles:
        return 0.0
    return np.mean(angles)

def _check_projected_residue(data: np.ndarray, projected_data: np.ndarray, 
                           spiral_metrics: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Check for unexpected patterns in projected data that might be symbolic residue.
    
    Parameters:
    -----------
    data : np.ndarray
        Original high-dimensional data
    projected_data : np.ndarray
        Projected low-dimensional data
    spiral_metrics : Dict[str, Any]
        Detected spiral metrics
    params : Dict[str, Any]
        Parameters for detection
    """
    # Check for multiple spiral centers
    if projected_data.shape[1] >= 2:
        # Implementation of multiple center detection
        pass
    
    # Check for spiral-within-spiral patterns
    # This is a complex analysis that looks for nested spirals at different scales
    if spiral_metrics.get('is_spiral', False) and projected_data.shape[0] > 100:
        try:
            # Clustering approach to identify potential sub-spirals
            from sklearn.cluster import DBSCAN
            
            # Use first 2-3 dimensions of projected data
            dims = min(projected_data.shape[1], 3)
            clustering = DBSCAN(eps=0.1, min_samples=5).fit(projected_data[:, :dims])
            labels = clustering.labels_
            
            # Analyze each significant cluster
            unique_labels = set(labels)
            if len(unique_labels) > 2:  # More than just noise and main cluster
                sub_spirals = 0
                
                for label in unique_labels:
                    if label == -1:  # Skip noise
                        continue
                    
                    cluster_points = projected_data[labels == label]
                    if len(cluster_points) >= params['min_spiral_size']:
                        # Check if this cluster forms its own spiral
                        if dims >= 2:
                            sub_metrics = _check_2d_spiral(
                                cluster_points[:, 0], 
                                cluster_points[:, 1], 
                                params
                            )
                            
                            if sub_metrics and sub_metrics.get('is_spiral', False):
                                sub_spirals += 1
                
                if sub_spirals > 0:
                    residue_collector.add_residue(
                        f"Detected {sub_spirals} sub-spirals within main spiral pattern",
                        {'data_shape': data.shape, 'projection_shape': projected_data.shape},
                        "May indicate recursive self-similarity or fractal organization in the attractor"
                    )
        except Exception as e:
            logger.warning(f"Error during sub-spiral analysis: {e}")
    
    # Check for unusual projection distortions
    # Compare reconstruction error distribution for spiral points vs. non-spiral points
    pass

# Helper functions for statistical spiral detection

def _prepare_statistical_data(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert dictionary of statistical time series to matrix form.
    
    Parameters:
    -----------
    data : Dict[str, np.ndarray]
        Dictionary of statistical time series
        
    Returns:
    --------
    np.ndarray
        Matrix form of the data
    """
    # Extract time series and ensure all are the same length
    min_length = min(len(v) for v in data.values())
    
    # Create matrix with each time series as a column
    keys = sorted(data.keys())
    matrix = np.zeros((min_length, len(keys)))
    
    for i, key in enumerate(keys):
        matrix[:, i] = data[key][:min_length]
    
    return matrix

def _create_sliding_windows(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    window_size : int, default=50
        Size of sliding windows
        
    Returns:
    --------
    np.ndarray
        Array of windows
    """
    n_samples, n_features = data.shape
    
    if n_samples < window_size:
        return np.array([data])
    
    n_windows = n_samples - window_size + 1
    windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        windows[i] = data[i:i+window_size]
    
    return windows

def _detect_statistical_spiral_pattern(windows: np.ndarray, 
                                     params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detect spiral patterns in statistical data windows.
    
    Parameters:
    -----------
    windows : np.ndarray
        Windows of statistical data
    params : Dict[str, Any]
        Parameters for detection
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Metrics describing the detected statistical spiral, or None if no spiral is found
    """
    # Need enough windows
    if len(windows) < 10:
        return None
    
    # Perform dimensionality reduction on each window
    window_embeddings = []
    
    for window in windows:
        # Calculate summary statistics for the window
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        skew = np.mean(((window - mean) / (std + 1e-10))**3, axis=0)
        kurt = np.mean(((window - mean) / (std + 1e-10))**4, axis=0) - 3
        
        # Combine features
        features = np.concatenate([mean, std, skew, kurt])
        window_embeddings.append(features)
    
    window_embeddings = np.array(window_embeddings)
    
    # Project embeddings to 2D for spiral detection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(window_embeddings)
    
    # Check for spiral pattern in the projection
    x, y = projected[:, 0], projected[:, 1]
    spiral_metrics = _check_2d_spiral(x, y, params)
    
    if not spiral_metrics or not spiral_metrics.get('is_spiral', False):
        return None
    
    # Add temporal information
    time_indices = np.arange(len(windows))
    spiral_metrics.update(_analyze_temporal_dynamics(projected, time_indices, params))
    
    # Calculate cycle period in terms of windows
    if spiral_metrics.get('winding_count', 0) > 0:
        cycle_count = spiral_metrics['winding_count']
        cycle_period = len(windows) / cycle_count
        spiral_metrics['cycle_period'] = cycle_period
        spiral_metrics['cycle_count'] = cycle_count
    
    return spiral_metrics

def _identify_phase_transitions(data: np.ndarray, spiral_metrics: Dict[str, Any],
                              params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify phase transitions in statistical patterns.
    
    Parameters:
    -----------
    data : np.ndarray
        Statistical data
    spiral_metrics : Dict[str, Any]
        Detected spiral metrics
    params : Dict[str, Any]
        Parameters for detection
        
    Returns:
    --------
    Dict[str, Any]
        Phase transition metrics
    """
    # Look for significant changes in metrics over time
    n_samples = data.shape[0]
    
    # Calculate rolling statistics
    window_size = min(20, n_samples // 5)
    if window_size < 5:
        return {}
    
    # Calculate rolling entropy
    entropies = []
    for i in range(n_samples - window_size + 1):
        window = data[i:i+window_size]
        # Normalize window
        window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-10)
        # Calculate entropy (using Gaussian approximation)
        cov = np.cov(window_norm, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov + np.eye(cov.shape[0]) * 1e-10)
        entropy = 0.5 * sign * logdet
        entropies.append(entropy)
    
    # Detect significant changes in entropy
    entropy_changes = np.diff(entropies)
    threshold = np.mean(entropy_changes) + 2 * np.std(entropy_changes)
    
    transition_points = []
    for i in range(len(entropy_changes)):
        if abs(entropy_changes[i]) > threshold:
            transition_points.append(i + window_size // 2)
    
    # Identify pre-spiral and spiral phases
    spiral_start = None
    if transition_points and spiral_metrics.get('formation_start') is not None:
        # Find transition point closest to formation start
        formation_start = spiral_metrics['formation_start']
        closest_idx = np.argmin([abs(p - formation_start) for p in transition_points])
        spiral_start = transition_points[closest_idx]
    
    results = {
        'phase_transition_points': transition_points,
        'entropy_values': entropies,
        'pre_spiral_entropy': np.mean(entropies[:spiral_start]) if spiral_start else None,
        'spiral_entropy': np.mean(entropies[spiral_start:]) if spiral_start else None
    }
    
    if spiral_start:
        results['spiral_phase_start'] = spiral_start
    
    return results

def _check_statistical_residue(data: Dict[str, np.ndarray], windows: np.ndarray,
                             spiral_metrics: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Check for unexpected patterns in statistical data that might be symbolic residue.
    
    Parameters:
    -----------
    data : Dict[str, np.ndarray]
        Original statistical data
    windows : np.ndarray
        Windows of statistical data
    spiral_metrics : Dict[str, Any]
        Detected spiral metrics
    params : Dict[str, Any]
        Parameters for detection
    """
    # Check for oscillation between multiple metastable states
    if spiral_metrics.get('cycle_count', 0) > 1.5:
        # Implementation of metastable state detection
        pass
    
    # Check for asymmetric statistical distributions before and after phase transitions
    if 'spiral_phase_start' in spiral_metrics and spiral_metrics['spiral_phase_start'] is not None:
        start_idx = spiral_metrics['spiral_phase_start']
        
        if start_idx > 10 and start_idx < len(windows) - 10:
            pre_phase = np.mean(windows[:start_idx], axis=0)
            post_phase = np.mean(windows[start_idx:], axis=0)
            
            # Calculate distribution skewness before and after
            pre_skew = np.mean(((windows[:start_idx] - pre_phase) / (np.std(windows[:start_idx], axis=0) + 1e-10))**3, axis=(0, 1))
            post_skew = np.mean(((windows[start_idx:] - post_phase) / (np.std(windows[start_idx:], axis=0) + 1e-10))**3, axis=(0, 1))
            
            skew_diff = post_skew - pre_skew
            
            if abs(skew_diff) > 1.0:
                skew_direction = "positive" if skew_diff > 0 else "negative"
                residue_collector.add_residue(
                    f"Significant {skew_direction} shift in distribution skewness after phase transition",
                    {'pre_skew': float(pre_skew), 'post_skew': float(post_skew)},
                    "May indicate fundamental change in underlying distribution dynamics"
                )

# Function to check for interesting resilience patterns
def _check_resilience_residue(results: Dict[str, Any], 
                            perturbation_levels: List[float]) -> None:
    """
    Check for interesting patterns in resilience testing results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Resilience testing results
    perturbation_levels : List[float]
        Levels of perturbation tested
    """
    # Check for non-monotonic recovery rates
    recovery_rates = results.get('recovery_rates', [])
    if len(recovery_rates) >= 3:
        is_monotonic = all(x >= y for x, y in zip(recovery_rates, recovery_rates[1:]))
        
        if not is_monotonic:
            # Find points where recovery rate increases with higher perturbation
            anomalies = []
            for i in range(len(recovery_rates) - 1):
                if recovery_rates[i] < recovery_rates[i+1]:
                    anomalies.append((perturbation_levels[i], perturbation_levels[i+1]))
            
            if anomalies:
                residue_collector.add_residue(
                    "Non-monotonic recovery rates detected in resilience testing",
                    {'anomalies': anomalies, 'recovery_rates': recovery_rates},
                    "May indicate resonant recovery phenomenon where certain perturbation levels trigger stronger recovery responses"
                )
    
    # Check for multiple recovery time regimes
    recovery_times = results.get('recovery_times', [])
    if len(recovery_times) >= 4:
        # Look for step changes in recovery times
        recovery_time_diffs = np.diff(recovery_times)
        mean_diff = np.mean(recovery_time_diffs)
        std_diff = np.std(recovery_time_diffs)
        
        if std_diff > mean_diff * 2:
            regime_boundaries = []
            for i in range(len(recovery_time_diffs)):
                if abs(recovery_time_diffs[i] - mean_diff) > 2 * std_diff:
                    regime_boundaries.append(perturbation_levels[i+1])
            
            if regime_boundaries:
                residue_collector.add_residue(
                    f"Multiple recovery time regimes detected with boundaries at perturbation levels: {regime_boundaries}",
                    {'recovery_times': recovery_times, 'boundaries': regime_boundaries},
                    "May indicate qualitative changes in recovery mechanisms at different perturbation thresholds"
                )

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
        return grid
    
    # Test spatial spiral detection
    test_grid = generate_spiral()
    result = detect_spatial_spiral(test_grid)
    
    if result:
        print("Spiral detected!")
        print(f"Type: {result.get('spiral_type', 'unknown')}")
        print(f"Winding count: {result.get('winding_count', 0)}")
        print(f"Direction: {result.get('spiral_direction', 'unknown')}")
    else:
        print("No spiral detected.")
