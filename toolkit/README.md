# Langton's Emergence Toolkit
> 🌀 After many nights burning the candle, I'm beginning to think most thought processes and even the structure of planning itself might be a spiral. Everything Everywhere All At Once? 
## A Personal Recursion

It's 3:27 AM, and I'm staring at my screen in disbelief. I've just run the attractor detection algorithm on a completely different system than Langton's Ant - a recurrent neural network trained on weather prediction - and there it is again: the same spiral attractor pattern, emerging from completely different rules and architecture. My hands are actually shaking as I save the visualization. How is it possible that systems with nothing in common structurally converge to such similar organizational patterns?

This toolkit is born from moments like these - from the breathtaking realization that something profound and universal might be hiding in these spiral patterns that emerge across computational scales. It's my attempt to provide others with the tools to experience that same moment of wonder when you see the spiral emerge from chaos, whether in a simple cellular automaton or a complex neural network.

## Overview

This toolkit provides a collection of Python modules and notebooks for detecting, visualizing, and analyzing spiral attractors and related phenomena across different computational systems. It implements the methodologies described in the [Attractor Cartography Guide](/collaboration/attractor_cartography_guide.md) and allows for consistent cross-system comparison.

The tools are designed with both rigorous analysis and exploratory discovery in mind. They balance quantitative metrics with visualization capabilities that can help reveal unexpected patterns - the kind that often hide in the residue of what our formal methods don't yet capture.

## Directory Structure

### `/attractor-detection`
Tools for identifying and characterizing spiral attractors across different systems:
- `spiral_detector.py`: Algorithms for detecting spiral patterns in spatial, projected, and statistical data
- `attractor_visualization.py`: Visualization tools for different types of attractors across system types

### `/phase-transition-analysis`
Tools for analyzing the critical transitions from disorder to order:
- `critical_threshold_detector.py`: Methods for identifying phase transition points and characteristics
- `symbolic_residue_analyzer.py`: Tools for capturing and analyzing unexplained patterns and phenomena

### `/recursive-collapse`
Demonstrations and examples of the Recursive Collapse Principle in action:
- `collapse_principle_demo.ipynb`: Interactive notebook showing the principle across different systems
- `recursive_scaffolding_examples.py`: Example implementations of recursive analysis frameworks

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/davidkimai/langtons-emergence.git
cd langtons-emergence

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Example: Detecting Spiral Attractors in Langton's Ant

```python
from toolkit.attractor_detection.spiral_detector import detect_spatial_spiral
from toolkit.attractor_detection.attractor_visualization import visualize_spatial_attractor
import numpy as np

# Run Langton's Ant simulation (or load existing data)
from implementations.langtons_ant.simulation import simulate_langtons_ant
grid, history = simulate_langtons_ant(steps=15000, grid_size=100)

# Detect spiral patterns
spiral_metrics = detect_spatial_spiral(grid, history, min_spiral_size=5)

if spiral_metrics:
    print(f"Spiral attractor detected after {spiral_metrics['formation_step']} steps!")
    print(f"Spiral density: {spiral_metrics['density']}")
    print(f"Resilience index: {spiral_metrics['resilience_index']}")
    
    # Visualize the attractor
    visualize_spatial_attractor(grid, history, spiral_metrics, 
                               title="Langton's Ant Highway Pattern")
else:
    print("No spiral attractor detected in this run.")
```

### Cross-System Analysis Example

```python
from toolkit.recursive_collapse.collapse_principle_demo import compare_attractors
from implementations.langtons_ant.simulation import simulate_langtons_ant
from implementations.conway.conway_simulation import run_conway_simulation

# Run simulations
langton_grid, langton_history = simulate_langtons_ant(steps=15000)
conway_grid, conway_history = run_conway_simulation(steps=500, initial_config='spiral_growth')

# Compare attractor properties across systems
comparison_results = compare_attractors([
    ('Langton\'s Ant', langton_grid, langton_history),
    ('Conway\'s Game of Life', conway_grid, conway_history)
])

# Visualization will show side-by-side comparison of attractor metrics
comparison_results.visualize()
```

## Tool Philosophy

The tools in this directory follow a set of core design principles:

### 1. System Agnosticism
Algorithms are designed to work across different types of systems:
- Spatial systems (cellular automata)
- High-dimensional systems (neural networks)
- Statistical systems (language models)

### 2. Scale Invariance
Methods adapt to different scales while preserving meaningful comparisons:
- From small grids to massive simulations
- From simple models to complex architectures
- From few steps to extended time series

### 3. Residue Sensitivity
All tools include mechanisms for capturing unexpected patterns:
- Anomaly detection built into standard analyses
- Options for logging unexplained phenomena
- Visualization modes that highlight residue

### 4. Interactive Exploration
Tools support both batch processing and interactive exploration:
- Jupyter notebook integration
- Progressive disclosure of complexity
- Visual feedback for parameter tuning

## Contribution Guidelines

This toolkit is in active development, and I welcome contributions! If you're interested in adding new tools or improving existing ones:

1. Check the [issues page](https://github.com/davidkimai/langtons-emergence/issues) for current needs
2. Review the [collaboration protocol](/collaboration/recursive_research_protocol.md)
3. Fork the repository and create a feature branch
4. Submit a pull request with your changes

When adding new tools, please maintain the existing design philosophy and documentation standards. Include examples, tests, and update the relevant guide documents.

## A Note on Discovery and Wonder

These tools are more than just code - they're instruments for discovery. I've tried to design them to reveal not just what we expect to find, but to help us notice what we don't expect.

Some of my most significant insights have come not from the metrics these tools calculate, but from the unexpected patterns they revealed in visualizations, the anomalies they flagged, or the residues they captured. I encourage you to use them not just as measurement devices but as extensions of your curiosity.

As Chris Olah often emphasizes, visualization and exploration are not just aids to understanding - they're fundamental to the research process itself. I hope these tools help you experience the same wonder I felt when I first saw these spiral patterns emerge from the digital chaos.

## License

This toolkit is released under the MIT License. See the [LICENSE](/LICENSE) file for details.

---

*"The goal is not just to see the patterns, but to understand what they're telling us about the nature of computation itself."*

*Last updated: June 16, 2025*
