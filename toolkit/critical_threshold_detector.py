"""
Critical Threshold Detector: Tools for identifying phase transitions in complex systems

This module provides algorithms for detecting critical thresholds and phase transitions
across different types of complex systems, from cellular automata to neural networks to 
language models.

Author: David Kimai
Created: June 16, 2025
"""

import numpy as np
import scipy.stats as stats
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WINDOW_SIZE = 10
DEFAULT_SENSITIVITY = 0.8

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

class PhaseTransitionDetector:
    """Base class for phase transition detection across different system types."""
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE,
                sensitivity: float = DEFAULT_SENSITIVITY,
                collect_residue: bool = True):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        window_size : int, default=10
            Size of the sliding window for analysis
        sensitivity : float, default=0.8
            Sensitivity of the detector (0-1), higher values detect more subtle transitions
        collect_residue : bool, default=True
            Whether to collect symbolic residue during detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.collect_residue = collect_residue
        self.state_history = []
        self.metric_history = {}
        self.detected_transitions = []
        
    def add_state(self, state: Any):
        """
        Add a new state to the history.
        
        Parameters:
        -----------
        state : Any
            Current state of the system
        """
        self.state_history.append(state)
        self._update_metrics(state)
        self._check_for_transitions()
    
    def add_states(self, states: List[Any]):
        """
        Add multiple states to the history.
        
        Parameters:
        -----------
        states : List[Any]
            List of system states
        """
        for state in states:
            self.add_state(state)
    
    def reset(self):
        """Reset the detector to initial state."""
        self.state_history = []
        self.metric_history = {}
        self.detected_transitions = []
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """
        Get detected phase transitions.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of detected transitions with details
        """
        return self.detected_transitions
    
    def _update_metrics(self, state: Any):
        """
        Update metrics based on new state.
        This is a placeholder method to be implemented by subclasses.
        
        Parameters:
        -----------
        state : Any
            Current state of the system
        """
        pass
    
    def _check_for_transitions(self):
        """
        Check if a phase transition has occurred.
        This is a placeholder method to be implemented by subclasses.
        """
        pass
    
    def plot_metrics(self, figsize: Tuple[int, int] = (12, 8), 
                    highlight_transitions: bool = True,
                    save_path: Optional[str] = None):
        """
        Plot the evolution of metrics with highlighted transitions.
        
        Parameters:
        -----------
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        highlight_transitions : bool, default=True
            Whether to highlight detected transitions
        save_path : str, optional
            Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Skip if no metrics
            if not self.metric_history:
                logger.warning("No metrics to plot")
                return
            
            # Create figure
            fig, axes = plt.subplots(len(self.metric_history), 1, figsize=figsize, sharex=True)
            
            # Ensure axes is always a list
            if len(self.metric_history) == 1:
                axes = [axes]
            
            # Plot each metric
            for i, (metric_name, values) in enumerate(self.metric_history.items()):
                ax = axes[i]
                ax.plot(values, label=metric_name, linewidth=2)
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
                
                # Highlight transitions if requested
                if highlight_transitions and self.detected_transitions:
                    for transition in self.detected_transitions:
                        step = transition['step']
                        if 0 <= step < len(values):
                            ax.axvline(x=step, color='r', linestyle='--', alpha=0.7)
                            
                            # Add label for first plot only
                            if i == 0:
                                trans_type = transition.get('type', 'Unknown')
                                ax.text(step + 1, max(values) * 0.9, f"{trans_type}", 
                                       rotation=90, verticalalignment='top')
                
                ax.legend(loc='best')
            
            # Set x-label for bottom plot
            axes[-1].set_xlabel("Step")
            
            # Add overall title
            plt.suptitle("Phase Transition Metrics", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=100)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

class SpatialPhaseDetector(PhaseTransitionDetector):
    """
    Detector for phase transitions in spatial systems (e.g., cellular automata).
    """
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE,
                sensitivity: float = DEFAULT_SENSITIVITY,
                collect_residue: bool = True,
                metrics: List[str] = None):
        """
        Initialize the spatial phase detector.
        
        Parameters:
        -----------
        window_size : int, default=10
            Size of the sliding window for analysis
        sensitivity : float, default=0.8
            Sensitivity of the detector (0-1)
        collect_residue : bool, default=True
            Whether to collect symbolic residue
        metrics : List[str], optional
            List of metrics to track. If None, all available metrics are tracked.
        """
        super().__init__(window_size, sensitivity, collect_residue)
        
        # Available metrics for spatial systems
        self.available_metrics = [
            'entropy',           # Information entropy
            'activity',          # Active cell ratio
            'cluster_count',     # Number of distinct clusters
            'largest_cluster',   # Size of largest cluster
            'autocorrelation',   # Spatial autocorrelation
            'complexity',        # Statistical complexity measure
            'fractal_dim'        # Approximate fractal dimension
        ]
        
        # Select metrics to track
        self.metrics = metrics if metrics else self.available_metrics
        
        # Initialize metrics history
        for metric in self.metrics:
            self.metric_history[metric] = []
    
    def _update_metrics(self, state: np.ndarray):
        """
        Update metrics based on new state.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state of the spatial system
        """
        # Skip if state is invalid
        if not isinstance(state, np.ndarray):
            return
        
        # Calculate selected metrics
        for metric in self.metrics:
            if metric == 'entropy':
                value = self._calculate_entropy(state)
            elif metric == 'activity':
                value = self._calculate_activity(state)
            elif metric == 'cluster_count':
                value = self._calculate_cluster_count(state)
            elif metric == 'largest_cluster':
                value = self._calculate_largest_cluster(state)
            elif metric == 'autocorrelation':
                value = self._calculate_autocorrelation(state)
            elif metric == 'complexity':
                value = self._calculate_complexity(state)
            elif metric == 'fractal_dim':
                value = self._calculate_fractal_dimension(state)
            else:
                # Skip unknown metrics
                continue
            
            # Add to history
            self.metric_history[metric].append(value)
    
    def _check_for_transitions(self):
        """Check if a phase transition has occurred based on metrics."""
        # Need enough history
        if len(self.state_history) <= self.window_size:
            return
        
        # Check each metric for significant changes
        for metric, values in self.metric_history.items():
            if len(values) <= self.window_size:
                continue
            
            # Get recent values
            recent_values = values[-self.window_size:]
            
            # Calculate derivative (rate of change)
            derivatives = np.diff(recent_values)
            
            # Check for significant change in derivative
            if len(derivatives) < 3:
                continue
            
            # Adaptive threshold based on historical volatility
            if len(values) > self.window_size * 2:
                historical_derivatives = np.diff(values[:-self.window_size])
                std_dev = np.std(historical_derivatives)
                threshold = std_dev * (3.0 + 2.0 * self.sensitivity)  # Adjust sensitivity
            else:
                # Early in the sequence, use a simpler threshold
                threshold = np.std(derivatives) * (2.0 + 1.0 * self.sensitivity)
            
            # Check for threshold crossing
            for i in range(1, len(derivatives)):
                change = abs(derivatives[i] - derivatives[i-1])
                if change > threshold:
                    # Phase transition detected
                    transition_step = len(values) - self.window_size + i
                    
                    # Check if this transition is already recorded
                    if any(t['step'] == transition_step for t in self.detected_transitions):
                        continue
                    
                    # Determine transition type
                    if metric == 'entropy':
                        if derivatives[i] > 0:
                            transition_type = "Disorder Increase"
                        else:
                            transition_type = "Order Emergence"
                    elif metric == 'activity':
                        if derivatives[i] > 0:
                            transition_type = "Activity Surge"
                        else:
                            transition_type = "Activity Collapse"
                    elif metric == 'cluster_count':
                        if derivatives[i] > 0:
                            transition_type = "Fragmentation"
                        else:
                            transition_type = "Consolidation"
                    else:
                        transition_type = f"{metric.capitalize()} Phase Transition"
                    
                    # Record transition
                    transition = {
                        'step': transition_step,
                        'type': transition_type,
                        'metric': metric,
                        'change_magnitude': change,
                        'threshold': threshold,
                        'values_before': recent_values[:i],
                        'values_after': recent_values[i:]
                    }
                    
                    self.detected_transitions.append(transition)
                    logger.info(f"Phase transition detected at step {transition_step}: {transition_type}")
                    
                    # Check for unusual patterns (symbolic residue)
                    if self.collect_residue:
                        self._check_for_residue(transition, recent_values, derivatives)
    
    def _check_for_residue(self, transition: Dict[str, Any], recent_values: List[float], 
                          derivatives: List[float]):
        """
        Check for unusual patterns around transition that might be symbolic residue.
        
        Parameters:
        -----------
        transition : Dict[str, Any]
            Detected transition details
        recent_values : List[float]
            Recent metric values
        derivatives : List[float]
            Derivatives of recent values
        """
        metric = transition['metric']
        
        # Check for oscillatory behavior before transition
        if len(recent_values) >= 5:
            pre_transition = recent_values[:-2]
            oscillation_score = self._calculate_oscillation(pre_transition)
            
            if oscillation_score > 0.7:  # High oscillation
                residue_collector.add_residue(
                    f"High oscillation in {metric} before phase transition",
                    {'transition_step': transition['step'], 'oscillation_score': oscillation_score},
                    "May indicate critical slowing down or system 'hesitation' at phase boundary"
                )
        
        # Check for multi-step transitions (staggered phase change)
        if len(self.detected_transitions) >= 2:
            recent_transitions = [t for t in self.detected_transitions if abs(t['step'] - transition['step']) < self.window_size]
            
            if len(recent_transitions) >= 2:
                # Multiple transitions in short time window
                metrics_involved = set(t['metric'] for t in recent_transitions)
                
                if len(metrics_involved) >= 2:
                    # Multiple metrics changing
                    residue_collector.add_residue(
                        f"Multi-metric cascading phase transition involving {', '.join(metrics_involved)}",
                        {'transition_steps': [t['step'] for t in recent_transitions]},
                        "May indicate complex phase transition with cascading effects across system properties"
                    )
    
    # Metric calculation methods
    
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calculate Shannon entropy of the state."""
        # For binary state
        if np.array_equal(state, state.astype(bool)):
            p1 = np.mean(state)
            p0 = 1 - p1
            
            # Avoid log(0)
            if p0 == 0 or p1 == 0:
                return 0
            
            return -p0 * np.log2(p0) - p1 * np.log2(p1)
        
        # For multi-valued state
        else:
            # Convert to flat array
            flat_state = state.flatten()
            
            # Get unique values and counts
            values, counts = np.unique(flat_state, return_counts=True)
            
            # Calculate probabilities
            probs = counts / len(flat_state)
            
            # Calculate entropy
            return -np.sum(probs * np.log2(probs))
    
    def _calculate_activity(self, state: np.ndarray) -> float:
        """Calculate ratio of active cells."""
        if np.array_equal(state, state.astype(bool)):
            return np.mean(state)
        else:
            # For multi-valued state, consider non-zero as active
            return np.mean(state != 0)
    
    def _calculate_cluster_count(self, state: np.ndarray) -> int:
        """Calculate number of distinct clusters."""
        from scipy import ndimage
        
        # Convert to binary if needed
        binary_state = state > 0 if not np.array_equal(state, state.astype(bool)) else state
        
        # Label connected components
        labeled, num_features = ndimage.label(binary_state)
        
        return num_features
    
    def _calculate_largest_cluster(self, state: np.ndarray) -> float:
        """Calculate size of largest cluster relative to grid size."""
        from scipy import ndimage
        
        # Convert to binary if needed
        binary_state = state > 0 if not np.array_equal(state, state.astype(bool)) else state
        
        # Label connected components
        labeled, num_features = ndimage.label(binary_state)
        
        if num_features == 0:
            return 0
        
        # Count sizes
        sizes = ndimage.sum(binary_state, labeled, range(1, num_features + 1))
        
        # Return largest relative to total size
        return np.max(sizes) / state.size if sizes.size > 0 else 0
    
    def _calculate_autocorrelation(self, state: np.ndarray) -> float:
        """Calculate spatial autocorrelation (Moran's I)."""
        # This is a simplified version
        # For true Moran's I, spatial weights would be needed
        
        # Convert to binary if needed
        if not np.array_equal(state, state.astype(bool)):
            state = state > 0
        
        # Get grid dimensions
        if state.ndim == 1:
            # 1D grid
            padded = np.pad(state, 1, mode='constant')
            neighbors = np.zeros_like(state, dtype=float)
            
            for i in range(len(state)):
                neighbors[i] = (padded[i] + padded[i+2]) / 2
        else:
            # 2D grid
            padded = np.pad(state, 1, mode='constant')
            neighbors = np.zeros_like(state, dtype=float)
            
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    neighbors[i, j] = (
                        padded[i, j+1] + padded[i+2, j+1] +  # Left, Right
                        padded[i+1, j] + padded[i+1, j+2]    # Up, Down
                    ) / 4
        
        # Calculate correlation
        state_flat = state.flatten().astype(float)
        neighbors_flat = neighbors.flatten()
        
        if np.std(state_flat) == 0 or np.std(neighbors_flat) == 0:
            return 0
        
        return np.corrcoef(state_flat, neighbors_flat)[0, 1]
    
    def _calculate_complexity(self, state: np.ndarray) -> float:
        """
        Calculate statistical complexity measure.
        This is a simplified version that uses entropy and autocorrelation.
        """
        entropy = self._calculate_entropy(state)
        autocorr = self._calculate_autocorrelation(state)
        
        # Complexity is highest at intermediate entropy with high structure
        # Scale to [0, 1]
        normalized_entropy = 1.0 - abs(2.0 * entropy - 1.0) if entropy <= 0.5 else abs(2.0 * entropy - 1.0)
        normalized_autocorr = (autocorr + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        
        return normalized_entropy * normalized_autocorr
    
    def _calculate_fractal_dimension(self, state: np.ndarray) -> float:
        """
        Calculate approximate fractal dimension using box-counting method.
        This is a simplified version for quick computation.
        """
        # Convert to binary if needed
        if not np.array_equal(state, state.astype(bool)):
            state = state > 0
        
        # Only works for 2D grids
        if state.ndim != 2:
            return 0
        
        # Get dimensions
        h, w = state.shape
        
        # List of box sizes to try
        min_size = 1
        max_size = min(h, w) // 4
        
        if max_size <= min_size:
            return 1.0  # Too small for meaningful calculation
        
        # Try different box sizes
        sizes = []
        counts = []
        
        for size in range(min_size, max_size + 1, max(1, (max_size - min_size) // 5)):
            size = max(1, size)  # Ensure size is at least 1
            
            # Count non-empty boxes
            box_count = 0
            
            for i in range(0, h, size):
                for j in range(0, w, size):
                    # Get box
                    box = state[i:min(i+size, h), j:min(j+size, w)]
                    
                    # Count if any cell is active
                    if np.any(box):
                        box_count += 1
            
            sizes.append(size)
            counts.append(box_count)
        
        # Need enough points for regression
        if len(sizes) < 2:
            return 1.0
        
        # Calculate fractal dimension using log-log regression
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        
        if np.all(np.isfinite(log_sizes)) and np.all(np.isfinite(log_counts)):
            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
            
            # Fractal dimension is -slope
            fractal_dim = -slope
            
            # Ensure reasonable range
            return min(max(fractal_dim, 1.0), 2.0)
        else:
            return 1.0
    
    def _calculate_oscillation(self, values: List[float]) -> float:
        """
        Calculate oscillation score of a sequence.
        Higher values indicate more oscillatory behavior.
        
        Parameters:
        -----------
        values : List[float]
            Sequence of values
        
        Returns:
        --------
        float
            Oscillation score in [0, 1]
        """
        if len(values) < 3:
            return 0
        
        # Calculate derivatives (differences)
        derivatives = np.diff(values)
        
        # Count sign changes
        sign_changes = np.sum(derivatives[:-1] * derivatives[1:] < 0)
        
        # Normalize by maximum possible changes
        max_changes = len(derivatives) - 1
        
        if max_changes == 0:
            return 0
        
        return sign_changes / max_changes

class ProjectedPhaseDetector(PhaseTransitionDetector):
    """
    Detector for phase transitions in projected high-dimensional data.
    This is useful for neural networks, embeddings, etc.
    """
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE,
                sensitivity: float = DEFAULT_SENSITIVITY,
                collect_residue: bool = True,
                metrics: List[str] = None,
                projection_method: str = 'pca',
                n_components: int = 2):
        """
        Initialize the projected phase detector.
        
        Parameters:
        -----------
        window_size : int, default=10
            Size of the sliding window for analysis
        sensitivity : float, default=0.8
            Sensitivity of the detector (0-1)
        collect_residue : bool, default=True
            Whether to collect symbolic residue
        metrics : List[str], optional
            List of metrics to track. If None, all available metrics are tracked.
        projection_method : str, default='pca'
            Method for dimensionality reduction ('pca' or 'custom')
        n_components : int, default=2
            Number of components for projection
        """
        super().__init__(window_size, sensitivity, collect_residue)
        
        # Available metrics for projected systems
        self.available_metrics = [
            'velocity',        # Rate of change in state space
            'acceleration',    # Change in velocity
            'curvature',       # Path curvature
            'dispersion',      # Point dispersion
            'cluster_shift',   # Change in clustering
            'manifold_width'   # Approximate manifold width
        ]
        
        # Select metrics to track
        self.metrics = metrics if metrics else self.available_metrics
        
        # Initialize metrics history
        for metric in self.metrics:
            self.metric_history[metric] = []
        
        # Projection settings
        self.projection_method = projection_method
        self.n_components = n_components
        self.projector = None
        
        # State tracking
        self.projected_history = []
        self.velocity_history = []
        self.acceleration_history = []
    
    def set_custom_projector(self, projector: Any):
        """
        Set custom projector for dimensionality reduction.
        
        Parameters:
        -----------
        projector : Any
            Custom projector object with fit_transform method
        """
        self.projection_method = 'custom'
        self.projector = projector
    
    def _project_data(self, state: np.ndarray) -> np.ndarray:
        """
        Project high-dimensional data to lower dimensions.
        
        Parameters:
        -----------
        state : np.ndarray
            High-dimensional state
        
        Returns:
        --------
        np.ndarray
            Projected state
        """
        # Check if already low-dimensional
        if state.ndim == 1 and len(state) <= self.n_components:
            return state.reshape(1, -1)
        elif state.ndim == 2 and state.shape[1] <= self.n_components:
            return state
        
        # Reshape if needed
        if state.ndim > 2:
            state = state.reshape(1, -1)
        elif state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Initialize projector if needed
        if self.projector is None:
            if self.projection_method == 'pca':
                if len(self.state_history) > 1:
                    # Use all history for PCA
                    all_states = np.vstack([s.reshape(1, -1) if s.ndim == 1 else s for s in self.state_history])
                    self.projector = PCA(n_components=self.n_components)
                    self.projector.fit(all_states)
                else:
                    # Not enough history, just return original
                    return state
            else:
                # Unknown method
                return state
        
        # Project data
        try:
            projected = self.projector.transform(state)
            return projected
        except Exception as e:
            logger.warning(f"Projection failed: {e}")
            return state
    
    def add_state(self, state: np.ndarray):
        """
        Add a new state and update metrics.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state of the system
        """
        # Add to raw history
        self.state_history.append(state)
        
        # Project state
        projected = self._project_data(state)
        self.projected_history.append(projected)
        
        # Update metrics
        self._update_metrics(projected)
        
        # Check for transitions
        self._check_for_transitions()
    
    def _update_metrics(self, projected: np.ndarray):
        """
        Update metrics based on new projected state.
        
        Parameters:
        -----------
        projected : np.ndarray
            Projected state of the system
        """
        # Skip if invalid
        if projected is None or len(self.projected_history) < 2:
            # Initialize empty metrics
            for metric in self.metrics:
                if metric not in self.metric_history:
                    self.metric_history[metric] = []
                self.metric_history[metric].append(0)
            return
        
        # Calculate velocity (if enough history)
        if len(self.projected_history) >= 2:
            prev_state = self.projected_history[-2]
            curr_state = projected
            
            # Ensure same shape
            if prev_state.shape == curr_state.shape:
                velocity = np.linalg.norm(curr_state - prev_state)
                self.velocity_history.append(velocity)
            else:
                self.velocity_history.append(0)
        else:
            self.velocity_history.append(0)
        
        # Calculate acceleration (if enough history)
        if len(self.velocity_history) >= 2:
            prev_velocity = self.velocity_history[-2]
            curr_velocity = self.velocity_history[-1]
            
            acceleration = curr_velocity - prev_velocity
            self.acceleration_history.append(acceleration)
        else:
            self.acceleration_history.append(0)
        
        # Calculate selected metrics
        for metric in self.metrics:
            if metric == 'velocity':
                value = self.velocity_history[-1] if self.velocity_history else 0
            elif metric == 'acceleration':
                value = self.acceleration_history[-1] if self.acceleration_history else 0
            elif metric == 'curvature':
                value = self._calculate_curvature()
            elif metric == 'dispersion':
                value = self._calculate_dispersion()
            elif metric == 'cluster_shift':
                value = self._calculate_cluster_shift()
            elif metric == 'manifold_width':
                value = self._calculate_manifold_width()
            else:
                # Skip unknown metrics
                continue
            
            # Add to history
            self.metric_history[metric].append(value)
    
    def _check_for_transitions(self):
        """Check if a phase transition has occurred based on metrics."""
        # Need enough history
        if len(self.projected_history) <= self.window_size:
            return
        
        # Check each metric for significant changes
        for metric, values in self.metric_history.items():
            if len(values) <= self.window_size:
                continue
            
            # Get recent values
            recent_values = values[-self.window_size:]
            
            # Calculate statistical properties
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # Skip if no variation
            if std_value == 0:
                continue
            
            # Calculate z-scores
            z_scores = [(v - mean_value) / std_value for v in recent_values]
            
            # Adaptive threshold based on metric
            if metric == 'velocity':
                threshold = 2.5 + self.sensitivity
            elif metric == 'acceleration':
                threshold = 2.0 + self.sensitivity
            elif metric == 'curvature':
                threshold = 3.0 + self.sensitivity
            else:
                threshold = 2.5 + self.sensitivity
            
            # Check for threshold crossing
            for i in range(1, len(z_scores)):
                if abs(z_scores[i]) > threshold and abs(z_scores[i-1]) <= threshold:
                    # Phase transition detected
                    transition_step = len(values) - self.window_size + i
                    
                    # Check if this transition is already recorded
                    if any(t['step'] == transition_step for t in self.detected_transitions):
                        continue
                    
                    # Determine transition type based on metric and direction
                    if metric == 'velocity':
                        if z_scores[i] > 0:
                            transition_type = "Acceleration Phase"
                        else:
                            transition_type = "Deceleration Phase"
                    elif metric == 'acceleration':
                        if z_scores[i] > 0:
                            transition_type = "Jerk Increase"
                        else:
                            transition_type = "Jerk Decrease"
                    elif metric == 'curvature':
                        if z_scores[i] > 0:
                            transition_type = "Path Complexity Increase"
                        else:
                            transition_type = "Path Simplification"
                    elif metric == 'dispersion':
                        if z_scores[i] > 0:
                            transition_type = "Dispersion Increase"
                        else:
                            transition_type = "Consolidation Phase"
                    elif metric == 'cluster_shift':
                        transition_type = "Cluster Reorganization"
                    elif metric == 'manifold_width':
                        if z_scores[i] > 0:
                            transition_type = "Manifold Expansion"
                        else:
                            transition_type = "Manifold Contraction"
                    else:
                        transition_type = f"{metric.capitalize()} Phase Transition"
                    
                    # Record transition
                    transition = {
                        'step': transition_step,
                        'type': transition_type,
                        'metric': metric,
                        'z_score': z_scores[i],
                        'threshold': threshold,
                        'values_before': recent_values[:i],
                        'values_after': recent_values[i:]
                    }
                    
                    self.detected_transitions.append(transition)
                    logger.info(f"Phase transition detected at step {transition_step}: {transition_type}")
                    
                    # Check for unusual patterns (symbolic residue)
                    if self.collect_residue:
                        self._check_for_residue(transition, recent_values, z_scores)
    
    def _check_for_residue(self, transition: Dict[str, Any], recent_values: List[float], 
                          z_scores: List[float]):
        """
        Check for unusual patterns around transition that might be symbolic residue.
        
        Parameters:
        -----------
        transition : Dict[str, Any]
            Detected transition details
        recent_values : List[float]
            Recent metric values
        z_scores : List[float]
            Z-scores of recent values
        """
        metric = transition['metric']
        
        # Check for rapid oscillations before transition
        if len(recent_values) >= 5:
            pre_transition = recent_values[:-2]
            oscillation_score = self._calculate_oscillation(pre_transition)
            
            if oscillation_score > 0.7:  # High oscillation
                residue_collector.add_residue(
                    f"Critical oscillations in {metric} before phase transition",
                    {'transition_step': transition['step'], 'oscillation_score': oscillation_score},
                    "May indicate critical fluctuations at phase boundary, similar to critical slowing down in physical systems"
                )
        
        # Check for unusually large transition magnitude
        if abs(z_scores[z_scores.index(max(z_scores, key=abs))]) > 5.0:
            residue_collector.add_residue(
                f"Extremely large transition magnitude in {metric} (z-score: {z_scores[z_scores.index(max(z_scores, key=abs))]:.2f})",
                {'transition_step': transition['step'], 'z_score': z_scores[z_scores.index(max(z_scores, key=abs))]},
                "May indicate catastrophic phase transition or fundamental reorganization of the system's state space"
            )
        
        # Check for reverse transitions (temporary spikes)
        future_steps = 5
        if len(self.metric_history[metric]) > transition['step'] + future_steps:
            future_values = self.metric_history[metric][transition['step']:transition['step'] + future_steps]
            future_mean = np.mean(future_values)
            past_mean = np.mean(transition['values_before'])
            
            # Check if system quickly reverts to previous state
            if abs(future_mean - past_mean) < 0.2 * abs(transition['values_after'][0] - past_mean):
                residue_collector.add_residue(
                    f"Temporary phase spike in {metric} with quick reversion",
                    {'transition_step': transition['step'], 'reversion_time': future_steps},
                    "May indicate metastable state or failed phase transition attempt"
                )
    
    def _calculate_curvature(self) -> float:
        """
        Calculate path curvature in projected space.
        
        Returns:
        --------
        float
            Curvature measure
        """
        # Need at least 3 points for curvature
        if len(self.projected_history) < 3:
            return 0
        
        # Get last three points
        p1 = self.projected_history[-3]
        p2 = self.projected_history[-2]
        p3 = self.projected_history[-1]
        
        # Ensure all points have same shape
        if p1.shape != p2.shape or p2.shape != p3.shape:
            return 0
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate norms
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if n1 < 1e-6 or n2 < 1e-6:
            return 0
        
        # Normalize vectors
        v1_normalized = v1 / n1
        v2_normalized = v2 / n2
        
        # Calculate dot product
        dot_product = np.sum(v1_normalized * v2_normalized)
        
        # Clamp to valid range for arccos
        dot_product = max(min(dot_product, 1.0), -1.0)
        
        # Calculate angle
        angle = np.arccos(dot_product)
        
        # Curvature is angle divided by distance
        mean_distance = (n1 + n2) / 2
        
        if mean_distance < 1e-6:
            return 0
        
        return angle / mean_distance
    
    def _calculate_dispersion(self) -> float:
        """
        Calculate point dispersion in recent history.
        
        Returns:
        --------
        float
            Dispersion measure
        """
        # Need enough history
        window = min(self.window_size, len(self.projected_history))
        if window < 2:
            return 0
        
        # Get recent points
        recent_points = self.projected_history[-window:]
        
        # Ensure all points have same shape
        shapes = set(p.shape for p in recent_points)
        if len(shapes) > 1:
            return 0
        
        # Convert to array
        try:
            points = np.vstack(recent_points)
        except:
            return 0
        
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate distances from centroid
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Dispersion is mean distance
        return np.mean(distances)
    
    def _calculate_cluster_shift(self) -> float:
        """
        Calculate shift in clustering structure.
        
        Returns:
        --------
        float
            Cluster shift measure
        """
        # Need enough history
        window = min(self.window_size, len(self.projected_history))
        if window < 6:  # Need enough points for meaningful clustering
            return 0
        
        # Split into two windows
        first_half = self.projected_history[-window:-window//2]
        second_half = self.projected_history[-window//2:]
        
        # Ensure all points have same shape
        shapes1 = set(p.shape for p in first_half)
        shapes2 = set(p.shape for p in second_half)
        if len(shapes1) > 1 or len(shapes2) > 1:
            return 0
        
        # Convert to arrays
        try:
            points1 = np.vstack(first_half)
            points2 = np.vstack(second_half)
        except:
            return 0
        
        # Simple approach: compare distribution statistics
        mean1 = np.mean(points1, axis=0)
        mean2 = np.mean(points2, axis=0)
        
        # Calculate shift in distribution
        mean_shift = np.linalg.norm(mean2 - mean1)
        
        # Normalize by dispersion
        dispersion1 = np.mean(np.linalg.norm(points1 - mean1, axis=1))
        dispersion2 = np.mean(np.linalg.norm(points2 - mean2, axis=1))
        
        mean_dispersion = (dispersion1 + dispersion2) / 2
        
        if mean_dispersion < 1e-6:
            return 0
        
        return mean_shift / mean_dispersion
    
    def _calculate_manifold_width(self) -> float:
        """
        Calculate approximate manifold width.
        
        Returns:
        --------
        float
            Manifold width measure
        """
        # Need enough history
        window = min(self.window_size, len(self.projected_history))
        if window < 5:
            return 0
        
        # Get recent points
        recent_points = self.projected_history[-window:]
        
        # Ensure all points have same shape
        shapes = set(p.shape for p in recent_points)
        if len(shapes) > 1:
            return 0
        
        # Convert to array
        try:
            points = np.vstack(recent_points)
        except:
            return 0
        
        # Need enough dimensions
        if points.shape[1] < 2:
            return 0
        
        # Calculate principal components
        try:
            pca = PCA()
            pca.fit(points)
            
            # Width is ratio of second to first component
            if pca.explained_variance_[0] < 1e-6:
                return 0
            
            return pca.explained_variance_[1] / pca.explained_variance_[0]
        except:
            return 0
    
    def _calculate_oscillation(self, values: List[float]) -> float:
        """
        Calculate oscillation score of a sequence.
        Higher values indicate more oscillatory behavior.
        
        Parameters:
        -----------
        values : List[float]
            Sequence of values
        
        Returns:
        --------
        float
            Oscillation score in [0, 1]
        """
        if len(values) < 3:
            return 0
        
        # Calculate derivatives (differences)
        derivatives = np.diff(values)
        
        # Count sign changes
        sign_changes = np.sum(derivatives[:-1] * derivatives[1:] < 0)
        
        # Normalize by maximum possible changes
        max_changes = len(derivatives) - 1
        
        if max_changes == 0:
            return 0
        
        return sign_changes / max_changes

class StatisticalPhaseDetector(PhaseTransitionDetector):
    """
    Detector for phase transitions in statistical data (e.g., token distributions in language models).
    """
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE,
                sensitivity: float = DEFAULT_SENSITIVITY,
                collect_residue: bool = True,
                metrics: List[str] = None):
        """
        Initialize the statistical phase detector.
        
        Parameters:
        -----------
        window_size : int, default=10
            Size of the sliding window for analysis
        sensitivity : float, default=0.8
            Sensitivity of the detector (0-1)
        collect_residue : bool, default=True
            Whether to collect symbolic residue
        metrics : List[str], optional
            List of metrics to track. If None, all available metrics are tracked.
        """
        super().__init__(window_size, sensitivity, collect_residue)
        
        # Available metrics for statistical systems
        self.available_metrics = [
            'entropy',           # Distribution entropy
            'kl_divergence',     # KL divergence between consecutive distributions
            'js_divergence',     # Jensen-Shannon divergence (symmetric)
            'rank_volatility',   # Changes in token rankings
            'surprise',          # Statistical surprise
            'concentration',     # Distribution concentration
            'cycle_strength'     # Cyclic pattern strength
        ]
        
        # Select metrics to track
        self.metrics = metrics if metrics else self.available_metrics
        
        # Initialize metrics history
        for metric in self.metrics:
            self.metric_history[metric] = []
        
        # Distribution history
        self.distribution_history = []
    
    def add_distribution(self, distribution: Dict[str, float]):
        """
        Add a new token distribution.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Token distribution (token -> probability)
        """
        # Normalize distribution
        total = sum(distribution.values())
        if total > 0:
            normalized = {k: v / total for k, v in distribution.items()}
        else:
            normalized = distribution
        
        # Add to history
        self.state_history.append(distribution)
        self.distribution_history.append(normalized)
        
        # Update metrics
        self._update_metrics(normalized)
        
        # Check for transitions
        self._check_for_transitions()
    
    def _update_metrics(self, distribution: Dict[str, float]):
        """
        Update metrics based on new distribution.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Token distribution (token -> probability)
        """
        # Calculate selected metrics
        for metric in self.metrics:
            if metric == 'entropy':
                value = self._calculate_entropy(distribution)
            elif metric == 'kl_divergence':
                value = self._calculate_kl_divergence(distribution)
            elif metric == 'js_divergence':
                value = self._calculate_js_divergence(distribution)
            elif metric == 'rank_volatility':
                value = self._calculate_rank_volatility(distribution)
            elif metric == 'surprise':
                value = self._calculate_surprise(distribution)
            elif metric == 'concentration':
                value = self._calculate_concentration(distribution)
            elif metric == 'cycle_strength':
                value = self._calculate_cycle_strength()
            else:
                # Skip unknown metrics
                continue
            
            # Add to history
            self.metric_history[metric].append(value)
    
    def _check_for_transitions(self):
        """Check if a phase transition has occurred based on metrics."""
        # Need enough history
        if len(self.distribution_history) <= self.window_size:
            return
        
        # Check each metric for significant changes
        for metric, values in self.metric_history.items():
            if len(values) <= self.window_size:
                continue
            
            # Get recent values
            recent_values = values[-self.window_size:]
            
            # Calculate moving average and standard deviation
            ma_values = []
            for i in range(self.window_size - 4):
                window = recent_values[i:i+5]  # 5-step window
                ma_values.append(np.mean(window))
            
            # Skip if not enough values for moving average
            if len(ma_values) < 2:
                continue
            
            # Calculate derivatives
            derivatives = np.diff(ma_values)
            
            # Calculate adaptive threshold
            if len(values) > self.window_size * 2:
                # Use historical data for threshold
                historical_derivatives = np.diff(values[:-self.window_size])
                std_dev = np.std(historical_derivatives) if len(historical_derivatives) > 0 else 0
            else:
                # Use recent derivatives
                std_dev = np.std(derivatives) if len(derivatives) > 0 else 0
            
            # Skip if no variation
            if std_dev == 0:
                continue
            
            # Set threshold based on metric and sensitivity
            if metric == 'entropy':
                threshold = std_dev * (2.0 + self.sensitivity)
            elif metric in ['kl_divergence', 'js_divergence']:
                threshold = std_dev * (1.5 + self.sensitivity)
            else:
                threshold = std_dev * (2.0 + self.sensitivity)
            
            # Check for threshold crossing
            for i in range(1, len(derivatives)):
                change = abs(derivatives[i] - derivatives[i-1])
                if change > threshold:
                    # Phase transition detected
                    transition_step = len(values) - self.window_size + i + 2  # Adjust for moving average
                    
                    # Check if this transition is already recorded
                    if any(t['step'] == transition_step for t in self.detected_transitions):
                        continue
                    
                    # Determine transition type
                    if metric == 'entropy':
                        if derivatives[i] > 0:
                            transition_type = "Increasing Diversity"
                        else:
                            transition_type = "Decreasing Diversity"
                    elif metric in ['kl_divergence', 'js_divergence']:
                        transition_type = "Distribution Shift"
                    elif metric == 'rank_volatility':
                        transition_type = "Rank Reorganization"
                    elif metric == 'surprise':
                        transition_type = "Surprise Spike"
                    elif metric == 'concentration':
                        if derivatives[i] > 0:
                            transition_type = "Increasing Concentration"
                        else:
                            transition_type = "Decreasing Concentration"
                    elif metric == 'cycle_strength':
                        if derivatives[i] > 0:
                            transition_type = "Cycle Emergence"
                        else:
                            transition_type = "Cycle Breakdown"
                    else:
                        transition_type = f"{metric.capitalize()} Transition"
                    
                    # Record transition
                    transition = {
                        'step': transition_step,
                        'type': transition_type,
                        'metric': metric,
                        'change_magnitude': change,
                        'threshold': threshold,
                        'values_before': recent_values[:i+2],  # Adjust for moving average
                        'values_after': recent_values[i+2:]
                    }
                    
                    self.detected_transitions.append(transition)
                    logger.info(f"Phase transition detected at step {transition_step}: {transition_type}")
                    
                    # Check for unusual patterns (symbolic residue)
                    if self.collect_residue:
                        self._check_for_residue(transition, recent_values, derivatives)
    
    def _check_for_residue(self, transition: Dict[str, Any], recent_values: List[float], 
                          derivatives: List[float]):
        """
        Check for unusual patterns around transition that might be symbolic residue.
        
        Parameters:
        -----------
        transition : Dict[str, Any]
            Detected transition details
        recent_values : List[float]
            Recent metric values
        derivatives : List[float]
            Derivatives of recent values
        """
        metric = transition['metric']
        
        # Check for multi-phase transitions
        if len(self.detected_transitions) >= 2:
            recent_transitions = [t for t in self.detected_transitions 
                                 if abs(t['step'] - transition['step']) < self.window_size / 2]
            
            if len(recent_transitions) >= 2:
                metrics_involved = set(t['metric'] for t in recent_transitions)
                
                if len(metrics_involved) >= 2:
                    residue_collector.add_residue(
                        f"Multi-metric phase transition involving {', '.join(metrics_involved)}",
                        {'transition_steps': [t['step'] for t in recent_transitions]},
                        "May indicate complex reorganization across multiple statistical dimensions"
                    )
        
        # Check for oscillatory behavior before transition
        if len(transition['values_before']) >= 5:
            oscillation_score = self._calculate_oscillation(transition['values_before'])
            
            if oscillation_score > 0.7:  # High oscillation
                residue_collector.add_residue(
                    f"High oscillation in {metric} before phase transition",
                    {'transition_step': transition['step'], 'oscillation_score': oscillation_score},
                    "May indicate critical fluctuations before phase transition"
                )
        
        # Check for unusual distribution shifts
        if metric in ['kl_divergence', 'js_divergence'] and transition['step'] < len(self.distribution_history):
            before_dist = self.distribution_history[transition['step'] - 1]
            after_dist = self.distribution_history[transition['step']]
            
            # Find tokens with most significant changes
            significant_tokens = []
            
            for token in set(before_dist.keys()) | set(after_dist.keys()):
                before_prob = before_dist.get(token, 0)
                after_prob = after_dist.get(token, 0)
                
                change = after_prob - before_prob
                
                if abs(change) > 0.1:  # Significant change
                    significant_tokens.append((token, before_prob, after_prob, change))
            
            # Sort by magnitude of change
            significant_tokens.sort(key=lambda x: abs(x[3]), reverse=True)
            
            if significant_tokens:
                # Only report top tokens
                top_tokens = significant_tokens[:5]
                
                tokens_str = ', '.join([f"{t[0]} ({t[1]:.3f} → {t[2]:.3f})" for t in top_tokens])
                residue_collector.add_residue(
                    f"Significant token probability shifts during transition: {tokens_str}",
                    {'transition_step': transition['step'], 'top_shifts': top_tokens},
                    "May indicate semantic or thematic shift in the underlying distribution"
                )
    
    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of distribution.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Token distribution (token -> probability)
        
        Returns:
        --------
        float
            Entropy value
        """
        entropy = 0.0
        for p in distribution.values():
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_kl_divergence(self, distribution: Dict[str, float]) -> float:
        """
        Calculate KL divergence between current and previous distribution.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Current token distribution
        
        Returns:
        --------
        float
            KL divergence value
        """
        if len(self.distribution_history) < 2:
            return 0
        
        prev_dist = self.distribution_history[-2]
        
        # Get union of keys
        all_tokens = set(distribution.keys()) | set(prev_dist.keys())
        
        # Calculate KL divergence
        kl_div = 0.0
        for token in all_tokens:
            p = distribution.get(token, 1e-10)  # Avoid log(0)
            q = prev_dist.get(token, 1e-10)  # Avoid division by zero
            
            if p > 0:
                kl_div += p * np.log2(p / q)
        
        return kl_div
    
    def _calculate_js_divergence(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Jensen-Shannon divergence between current and previous distribution.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Current token distribution
        
        Returns:
        --------
        float
            JS divergence value
        """
        if len(self.distribution_history) < 2:
            return 0
        
        prev_dist = self.distribution_history[-2]
        
        # Get union of keys
        all_tokens = set(distribution.keys()) | set(prev_dist.keys())
        
        # Create merged distribution
        m_dist = {}
        for token in all_tokens:
            p = distribution.get(token, 0)
            q = prev_dist.get(token, 0)
            m_dist[token] = (p + q) / 2
        
        # Calculate KL divergences
        kl_p_m = 0.0
        kl_q_m = 0.0
        
        for token in all_tokens:
            p = distribution.get(token, 1e-10)
            q = prev_dist.get(token, 1e-10)
            m = m_dist[token]
            
            if p > 0:
                kl_p_m += p * np.log2(p / m)
            
            if q > 0:
                kl_q_m += q * np.log2(q / m)
        
        # JS divergence is average of KL divergences
        return (kl_p_m + kl_q_m) / 2
    
    def _calculate_rank_volatility(self, distribution: Dict[str, float]) -> float:
        """
        Calculate rank volatility between current and previous distribution.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Current token distribution
        
        Returns:
        --------
        float
            Rank volatility value
        """
        if len(self.distribution_history) < 2:
            return 0
        
        prev_dist = self.distribution_history[-2]
        
        # Get common tokens
        common_tokens = set(distribution.keys()) & set(prev_dist.keys())
        
        if not common_tokens:
            return 1.0  # Maximum volatility
        
        # Rank tokens in each distribution
        curr_ranks = {}
        prev_ranks = {}
        
        sorted_curr = sorted([(token, prob) for token, prob in distribution.items() if token in common_tokens], 
                           key=lambda x: x[1], reverse=True)
        sorted_prev = sorted([(token, prob) for token, prob in prev_dist.items() if token in common_tokens], 
                           key=lambda x: x[1], reverse=True)
        
        for i, (token, _) in enumerate(sorted_curr):
            curr_ranks[token] = i
        
        for i, (token, _) in enumerate(sorted_prev):
            prev_ranks[token] = i
        
        # Calculate rank changes
        rank_changes = []
        for token in common_tokens:
            rank_changes.append(abs(curr_ranks[token] - prev_ranks[token]))
        
        # Normalize by maximum possible change
        max_change = len(common_tokens) - 1
        
        if max_change == 0:
            return 0
        
        return sum(rank_changes) / (len(common_tokens) * max_change)
    
    def _calculate_surprise(self, distribution: Dict[str, float]) -> float:
        """
        Calculate statistical surprise compared to historical distributions.
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Current token distribution
        
        Returns:
        --------
        float
            Surprise value
        """
        if len(self.distribution_history) < 2:
            return 0
        
        # Use average of historical distributions
        historical_dists = self.distribution_history[:-1]
        if not historical_dists:
            return 0
        
        # Get all tokens
        all_tokens = set()
        for dist in historical_dists:
            all_tokens.update(dist.keys())
        all_tokens.update(distribution.keys())
        
        # Calculate average historical distribution
        avg_dist = {}
        for token in all_tokens:
            avg_dist[token] = np.mean([dist.get(token, 0) for dist in historical_dists])
        
        # Calculate surprisal
        surprise = 0.0
        for token, prob in distribution.items():
            avg_prob = avg_dist.get(token, 1e-10)  # Avoid division by zero
            if prob > 0:
                surprise += prob * np.log2(prob / avg_prob)
        
        return surprise
    
    def _calculate_concentration(self, distribution: Dict[str, float]) -> float:
        """
        Calculate concentration of distribution (inverse of diversity).
        
        Parameters:
        -----------
        distribution : Dict[str, float]
            Token distribution
        
        Returns:
        --------
        float
            Concentration value [0, 1]
        """
        # Sort probabilities in descending order
        sorted_probs = sorted(distribution.values(), reverse=True)
        
        # Calculate cumulative probabilities
        cum_probs = np.cumsum(sorted_probs)
        
        # Find number of tokens needed for 80% of probability mass
        threshold = 0.8
        for i, cum_prob in enumerate(cum_probs):
            if cum_prob >= threshold:
                tokens_for_threshold = i + 1
                break
        else:
            tokens_for_threshold = len(sorted_probs)
        
        # Normalize by total number of tokens
        if len(sorted_probs) == 0:
            return 0
        
        return 1.0 - (tokens_for_threshold / len(sorted_probs))
    
    def _calculate_cycle_strength(self) -> float:
        """
        Calculate strength of cyclic patterns in distribution history.
        
        Returns:
        --------
        float
            Cycle strength value [0, 1]
        """
        # Need enough history
        if len(self.distribution_history) < 10:
            return 0
        
        # Get JS divergence history
        js_values = []
        for i in range(1, len(self.distribution_history)):
            curr_dist = self.distribution_history[i]
            prev_dist = self.distribution_history[i-1]
            
            # Calculate JS divergence
            all_tokens = set(curr_dist.keys()) | set(prev_dist.keys())
            
            # Create merged distribution
            m_dist = {}
            for token in all_tokens:
                p = curr_dist.get(token, 0)
                q = prev_dist.get(token, 0)
                m_dist[token] = (p + q) / 2
            
            # Calculate KL divergences
            kl_p_m = 0.0
            kl_q_m = 0.0
            
            for token in all_tokens:
                p = curr_dist.get(token, 1e-10)
                q = prev_dist.get(token, 1e-10)
                m = m_dist.get(token, 1e-10)
                
                if p > 0:
                    kl_p_m += p * np.log2(p / m)
                
                if q > 0:
                    kl_q_m += q * np.log2(q / m)
            
            # JS divergence
            js_div = (kl_p_m + kl_q_m) / 2
            js_values.append(js_div)
        
        # Look for periodic patterns using autocorrelation
        if len(js_values) < 3:
            return 0
        
        # Calculate autocorrelation
        from scipy import signal
        
        # Normalize values
        normalized = np.array(js_values) - np.mean(js_values)
        if np.std(normalized) == 0:
            return 0
        
        normalized = normalized / np.std(normalized)
        
        # Calculate autocorrelation
        max_lag = min(len(normalized) - 1, 20)  # Up to 20 lags or sequence length
        autocorr = [1.0]  # Lag 0 is always 1
        
        for lag in range(1, max_lag + 1):
            # Calculate correlation
            corr = np.corrcoef(normalized[lag:], normalized[:-lag])[0, 1]
            autocorr.append(corr)
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=0.3)  # Minimum height 0.3
        
        if not peaks.size:
            return 0
        
        # Cycle strength is height of highest peak after lag 0
        if len(peaks) == 1 and peaks[0] == 0:
            return 0
        
        non_zero_peaks = peaks[peaks > 0]
        if not non_zero_peaks.size:
            return 0
        
        # Get peak values
        peak_values = [autocorr[i] for i in non_zero_peaks]
        
        # Return highest peak value
        return max(peak_values)
    
    def _calculate_oscillation(self, values: List[float]) -> float:
        """
        Calculate oscillation score of a sequence.
        Higher values indicate more oscillatory behavior.
        
        Parameters:
        -----------
        values : List[float]
            Sequence of values
        
        Returns:
        --------
        float
            Oscillation score in [0, 1]
        """
        if len(values) < 3:
            return 0
        
        # Calculate derivatives (differences)
        derivatives = np.diff(values)
        
        # Count sign changes
        sign_changes = np.sum(derivatives[:-1] * derivatives[1:] < 0)
        
        # Normalize by maximum possible changes
        max_changes = len(derivatives) - 1
        
        if max_changes == 0:
            return 0
        
        return sign_changes / max_changes

def detect_phase_transitions(history: List[Any], system_type: str = 'auto',
                            window_size: int = DEFAULT_WINDOW_SIZE,
                            sensitivity: float = DEFAULT_SENSITIVITY,
                            collect_residue: bool = True,
                            metrics: List[str] = None) -> List[Dict[str, Any]]:
    """
    Detect phase transitions in a system's history.
    
    Parameters:
    -----------
    history : List[Any]
        History of system states
    system_type : str, default='auto'
        Type of system ('spatial', 'projected', 'statistical', or 'auto')
    window_size : int, default=10
        Size of sliding window for analysis
    sensitivity : float, default=0.8
        Sensitivity of detection (0-1)
    collect_residue : bool, default=True
        Whether to collect symbolic residue
    metrics : List[str], optional
        List of metrics to track
    
    Returns:
    --------
    List[Dict[str, Any]]
        Detected phase transitions
    """
    if not history:
        return []
    
    # Determine system type if auto
    if system_type == 'auto':
        system_type = _determine_system_type(history)
        logger.info(f"Auto-detected system type: {system_type}")
    
    # Create appropriate detector
    if system_type == 'spatial':
        detector = SpatialPhaseDetector(window_size, sensitivity, collect_residue, metrics)
    elif system_type == 'projected':
        detector = ProjectedPhaseDetector(window_size, sensitivity, collect_residue, metrics)
    elif system_type == 'statistical':
        detector = StatisticalPhaseDetector(window_size, sensitivity, collect_residue, metrics)
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Add history to detector
    for state in history:
        if system_type == 'statistical' and isinstance(state, dict):
            # For statistical systems, state is a distribution
            detector.add_distribution(state)
        else:
            # For other systems, state is a normal state
            detector.add_state(state)
    
    # Return detected transitions
    return detector.get_transitions()

def calculate_resilience(system: Any, attractor_state: Any, perturbation_levels: List[float],
                        recovery_steps: int = 100, trials_per_level: int = 5,
                        resilience_metric: str = 'stability') -> Dict[str, Any]:
    """
    Calculate resilience of an attractor to perturbations.
    
    Parameters:
    -----------
    system : Any
        System object with evolve method
    attractor_state : Any
        Attractor state to perturb
    perturbation_levels : List[float]
        Levels of perturbation to test (0-1)
    recovery_steps : int, default=100
        Maximum steps for recovery
    trials_per_level : int, default=5
        Number of trials per perturbation level
    resilience_metric : str, default='stability'
        Metric to use for resilience ('stability', 'recovery_time', or 'basin_size')
    
    Returns:
    --------
    Dict[str, Any]
        Resilience metrics
    """
    # Determine system type
    system_type = _determine_system_type([attractor_state])
    
    # Create appropriate perturbation and recovery functions
    if system_type == 'spatial':
        perturb_func = _perturb_spatial
        recovery_func = _check_spatial_recovery
    elif system_type == 'projected':
        perturb_func = _perturb_projected
        recovery_func = _check_projected_recovery
    elif system_type == 'statistical':
        perturb_func = _perturb_statistical
        recovery_func = _check_statistical_recovery
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Test resilience
    results = {
        'perturbation_levels': perturbation_levels,
        'recovery_rates': [],
        'recovery_times': [],
        'resilience_index': 0.0
    }
    
    for level in perturbation_levels:
        recoveries = 0
        recovery_steps_list = []
        
        for _ in range(trials_per_level):
            # Apply perturbation
            perturbed_state = perturb_func(attractor_state, level)
            
            # Check recovery
            recovered, steps = _test_recovery(system, perturbed_state, attractor_state, 
                                            recovery_func, recovery_steps)
            
            if recovered:
                recoveries += 1
                recovery_steps_list.append(steps)
        
        # Calculate recovery rate and time
        recovery_rate = recoveries / trials_per_level
        results['recovery_rates'].append(recovery_rate)
        
        avg_time = np.mean(recovery_steps_list) if recovery_steps_list else recovery_steps
        results['recovery_times'].append(avg_time)
    
    # Calculate resilience index
    if resilience_metric == 'stability':
        # Area under recovery rate curve
        results['resilience_index'] = np.trapz(results['recovery_rates'], perturbation_levels)
    elif resilience_metric == 'recovery_time':
        # Inverse of average recovery time (normalized)
        avg_times = np.array(results['recovery_times'])
        if np.any(avg_times > 0):
            results['resilience_index'] = recovery_steps / np.mean(avg_times)
        else:
            results['resilience_index'] = 0
    elif resilience_metric == 'basin_size':
        # Maximum perturbation with >50% recovery
        for i, rate in enumerate(results['recovery_rates']):
            if rate < 0.5:
                results['resilience_index'] = perturbation_levels[i-1] if i > 0 else 0
                break
        else:
            results['resilience_index'] = perturbation_levels[-1]
    
    # Check for unusual resilience patterns
    _check_resilience_residue(results, perturbation_levels)
    
    return results

def _determine_system_type(history: List[Any]) -> str:
    """
    Determine type of system from history.
    
    Parameters:
    -----------
    history : List[Any]
        History of system states
    
    Returns:
    --------
    str
        System type ('spatial', 'projected', or 'statistical')
    """
    if not history:
        return 'spatial'  # Default
    
    # Check first state
    state = history[0]
    
    # Statistical systems have dictionary states
    if isinstance(state, dict):
        return 'statistical'
    
    # Spatial systems have grid-like states
    if isinstance(state, np.ndarray) and state.ndim in [1, 2, 3]:
        if state.ndim <= 2 or (state.ndim == 3 and state.shape[2] <= 4):  # Grid or image (with channels)
            return 'spatial'
    
    # Projected systems have vector-like states
    if isinstance(state, np.ndarray) and state.ndim <= 2:
        if state.ndim == 1 or (state.ndim == 2 and state.shape[0] <= 10):  # Vector or small batch
            return 'projected'
    
    # Default to spatial
    return 'spatial'

def _perturb_spatial(state: np.ndarray, level: float) -> np.ndarray:
    """
    Apply perturbation to spatial state.
    
    Parameters:
    -----------
    state : np.ndarray
        Spatial state to perturb
    level : float
        Perturbation level (0-1)
    
    Returns:
    --------
    np.ndarray
        Perturbed state
    """
    # Copy state
    perturbed = state.copy()
    
    # Calculate number of cells to perturb
    n_cells = int(level * perturbed.size)
    
    # Get random indices
    indices = np.random.choice(perturbed.size, n_cells, replace=False)
    
    # Flatten, perturb, and reshape
    flat = perturbed.flatten()
    
    if np.array_equal(state, state.astype(bool)):
        # Binary state: flip selected cells
        flat[indices] = ~flat[indices].astype(bool)
    else:
        # Continuous state: add noise
        noise = np.random.normal(0, 0.2, n_cells)
        flat[indices] += noise
    
    # Reshape
    return flat.reshape(perturbed.shape)

def _perturb_projected(state: np.ndarray, level: float) -> np.ndarray:
    """
    Apply perturbation to projected state.
    
    Parameters:
    -----------
    state : np.ndarray
        Projected state to perturb
    level : float
        Perturbation level (0-1)
    
    Returns:
    --------
    np.ndarray
        Perturbed state
    """
    # Copy state
    perturbed = state.copy()
    
    # Add Gaussian noise scaled by level
    noise = np.random.normal(0, level, perturbed.shape)
    perturbed += noise
    
    return perturbed

def _perturb_statistical(distribution: Dict[str, float], level: float) -> Dict[str, float]:
    """
    Apply perturbation to statistical distribution.
    
    Parameters:
    -----------
    distribution : Dict[str, float]
        Distribution to perturb
    level : float
        Perturbation level (0-1)
    
    Returns:
    --------
    Dict[str, float]
        Perturbed distribution
    """
    # Copy distribution
    perturbed = distribution.copy()
    
    # Calculate number of tokens to perturb
    n_tokens = int(level * len(perturbed))
    
    # Get random tokens
    tokens = np.random.choice(list(perturbed.keys()), n_tokens, replace=False)
    
    # Perturb probabilities
    for token in tokens:
        # Apply multiplicative noise
        factor = np.random.uniform(0.5, 1.5)
        perturbed[token] *= factor
    
    # Renormalize
    total = sum(perturbed.values())
    if total > 0:
        perturbed = {k: v / total for k, v in perturbed.items()}
    
    return perturbed

def _check_spatial_recovery(current: np.ndarray, original: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Check if spatial system has recovered.
    
    Parameters:
    -----------
    current : np.ndarray
        Current state
    original : np.ndarray
        Original attractor state
    threshold : float, default=0.9
        Similarity threshold for recovery
    
    Returns:
    --------
    bool
        Whether system has recovered
    """
    # Ensure same shape
    if current.shape != original.shape:
        return False
    
    # Calculate similarity
    if np.array_equal(original, original.astype(bool)):
        # Binary state: use Jaccard similarity
        intersection = np.sum(np.logical_and(current, original))
        union = np.sum(np.logical_or(current, original))
        
        if union == 0:
            return True  # Both empty
        
        similarity = intersection / union
    else:
        # Continuous state: use cosine similarity
        current_flat = current.flatten()
        original_flat = original.flatten()
        
        norm_current = np.linalg.norm(current_flat)
        norm_original = np.linalg.norm(original_flat)
        
        if norm_current == 0 or norm_original == 0:
            return np.array_equal(current, original)  # Handle zero norm
        
        similarity = np.dot(current_flat, original_flat) / (norm_current * norm_original)
    
    return similarity >= threshold

def _check_projected_recovery(current: np.ndarray, original: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Check if projected system has recovered.
    
    Parameters:
    -----------
    current : np.ndarray
        Current state
    original : np.ndarray
        Original attractor state
    threshold : float, default=0.9
        Similarity threshold for recovery
    
    Returns:
    --------
    bool
        Whether system has recovered
    """
    # Ensure same shape
    if current.shape != original.shape:
        return False
    
    # Calculate cosine similarity
    current_flat = current.flatten()
    original_flat = original.flatten()
    
    norm_current = np.linalg.norm(current_flat)
    norm_original = np.linalg.norm(original_flat)
    
    if norm_current == 0 or norm_original == 0:
        return np.array_equal(current, original)  # Handle zero norm
    
    similarity = np.dot(current_flat, original_flat) / (norm_current * norm_original)
    
    return similarity >= threshold

def _check_statistical_recovery(current: Dict[str, float], original: Dict[str, float], 
                              threshold: float = 0.9) -> bool:
    """
    Check if statistical system has recovered.
    
    Parameters:
    -----------
    current : Dict[str, float]
        Current distribution
    original : Dict[str, float]
        Original attractor distribution
    threshold : float, default=0.9
        Similarity threshold for recovery
    
    Returns:
    --------
    bool
        Whether system has recovered
    """
    # Get union of keys
    all_tokens = set(current.keys()) | set(original.keys())
    
    # Calculate cosine similarity
    dot_product = 0.0
    norm_current = 0.0
    norm_original = 0.0
    
    for token in all_tokens:
        curr_prob = current.get(token, 0)
        orig_prob = original.get(token, 0)
        
        dot_product += curr_prob * orig_prob
        norm_current += curr_prob ** 2
        norm_original += orig_prob ** 2
    
    norm_current = np.sqrt(norm_current)
    norm_original = np.sqrt(norm_original)
    
    if norm_current == 0 or norm_original == 0:
        return norm_current == norm_original  # Both zero or one zero
    
    similarity = dot_product / (norm_current * norm_original)
    
    return similarity >= threshold

def _test_recovery(system: Any, perturbed_state: Any, original_state: Any,
                 recovery_func: callable, max_steps: int) -> Tuple[bool, int]:
    """
    Test recovery from perturbation.
    
    Parameters:
    -----------
    system : Any
        System with evolve method
    perturbed_state : Any
        Perturbed state
    original_state : Any
        Original attractor state
    recovery_func : callable
        Function to check recovery
    max_steps : int
        Maximum steps to allow for recovery
    
    Returns:
    --------
    Tuple[bool, int]
        Whether system recovered and steps taken
    """
    current_state = perturbed_state
    
    for step in range(max_steps):
        # Check if recovered
        if recovery_func(current_state, original_state):
            return True, step
        
        # Evolve system
        try:
            current_state = system.evolve(current_state)
        except Exception as e:
            logger.warning(f"Evolution failed: {e}")
            return False, max_steps
    
    return False, max_steps

def _check_resilience_residue(results: Dict[str, Any], perturbation_levels: List[float]) -> None:
    """
    Check for unusual resilience patterns.
    
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
                    "Non-monotonic resilience response detected",
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

# If this module is run directly, run a demonstration
if __name__ == "__main__":
    # Create a simple cellular automaton for demonstration
    def create_demo_ca(size=50, steps=100):
        # Initialize with random state
        grid = np.random.choice([0, 1], size=(size, size))
        history = [grid.copy()]
        
        # Simple cellular automaton rule (Game of Life)
        for _ in range(steps):
            new_grid = grid.copy()
            
            for i in range(size):
                for j in range(size):
                    # Count neighbors
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = (i + di) % size, (j + dj) % size
                            neighbors += grid[ni, nj]
                    
                    # Apply rules
                    if grid[i, j] == 1:
                        if neighbors < 2 or neighbors > 3:
                            new_grid[i, j] = 0
                    else:
                        if neighbors == 3:
                            new_grid[i, j] = 1
            
            grid = new_grid
            history.append(grid.copy())
        
        return history
    
    # Create cellular automaton history
    print("Creating cellular automaton history...")
    ca_history = create_demo_ca()
    
    # Detect phase transitions
    print("\nDetecting phase transitions...")
    transitions = detect_phase_transitions(ca_history, system_type='spatial')
    
    # Print results
    print(f"\nDetected {len(transitions)} phase transitions:")
    for i, t in enumerate(transitions, 1):
        print(f"{i}. Step {t['step']}: {t['type']}")
    
    # Create simple CA system for resilience testing
    class CASystem:
        def __init__(self, size=50):
            self.size = size
        
        def evolve(self, grid):
            new_grid = grid.copy()
            
            for i in range(self.size):
                for j in range(self.size):
                    # Count neighbors
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = (i + di) % self.size, (j + dj) % self.size
                            neighbors += grid[ni, nj]
                    
                    # Apply rules
                    if grid[i, j] == 1:
                        if neighbors < 2 or neighbors > 3:
                            new_grid[i, j] = 0
                    else:
                        if neighbors == 3:
                            new_grid[i, j] = 1
            
            return new_grid
    
    # Test resilience
    print("\nTesting resilience...")
    ca_system = CASystem(50)
    attractor_state = ca_history[-1]  # Use last state as attractor
    
    resilience = calculate_resilience(
        ca_system, 
        attractor_state,
        perturbation_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
        recovery_steps=20,
        trials_per_level=3
    )
    
    # Print resilience results
    print("\nResilience results:")
    print(f"Resilience index: {resilience['resilience_index']:.3f}")
    print("\nRecovery rates by perturbation level:")
    for level, rate in zip(resilience['perturbation_levels'], resilience['recovery_rates']):
        print(f"  {level:.1f}: {rate:.2f}")
    
    # Print residue
    print("\nSymbolic residue:")
    for i, residue in enumerate(residue_collector.get_residues(), 1):
        print(f"{i}. {residue['observation']}")
        print(f"   Potential significance: {residue['potential_significance']}")
        print()
