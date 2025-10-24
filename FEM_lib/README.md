# FEM Library Implementation using scikit-fem

This directory contains a finite element method (FEM) implementation for 2D incompressible Navier-Stokes equations using the scikit-fem library.

## Overview

The implementation provides a complete framework for simulating cylinder flow with three different boundary conditions:

1. **Steady Flow (Re = 20)**: `U_x(0, y) = 4*U_m*y*(H-y)/H²`, `U_y = 0`, `U_m = 0.3 m/s`
2. **Unsteady Flow (Re = 100)**: `U_x(0, y, t) = 4*U_m*y*(H-y)/H²`, `U_y = 0`, `U_m = 1.5 m/s`
3. **Oscillating Flow (Re = 100)**: `U_x(0, y, t) = 4*U_m*y*(H-y)*sin(πt/8)/H²`, `U_y = 0`, `U_m = 1.5 m/s`

## Files

### Core Components

- **`skfem_mesh_generator.py`**: Mesh generation using scikit-fem's built-in capabilities
- **`skfem_simple_solver.py`**: Simplified Navier-Stokes solver (demonstration version)
- **`skfem_navier_stokes_solver.py`**: Full Navier-Stokes solver (advanced implementation)
- **`skfem_cylinder_flow.py`**: Main wrapper class for cylinder flow simulation

### Usage

```python
from FEM_lib import SkfemCylinderFlow

# Create simulation
simulation = SkfemCylinderFlow(
    mesh_density="medium",
    dt=0.001,
    initial_condition="steady"
)

# Run simulation
results = simulation.run_simulation(max_steps=1000, save_interval=10)

# Save results
simulation.save_results(results, "results.npz")
```

### Command Line Interface

```bash
# Run single simulation
python run_skfem_simulation.py --condition steady --mesh-density medium

# Run all conditions
python run_skfem_simulation.py --condition all

# Compare with other methods
python compare_with_skfem.py --condition all
```

## Features

- **Mesh Generation**: Automatic mesh generation with boundary marking
- **Multiple Boundary Conditions**: Support for steady, unsteady, and oscillating flows
- **Taylor-Hood Elements**: P2-P1 velocity-pressure formulation
- **Force Calculation**: Drag and lift coefficient computation
- **Visualization**: Built-in solution visualization
- **Comparison Tools**: Benchmarking against custom FEM and LBM methods

## Dependencies

- scikit-fem
- numpy
- scipy
- matplotlib
- meshio

## Domain Parameters

- **Domain**: 2.2m × 0.41m (length × height)
- **Cylinder**: diameter = 0.1m, center at (0.2m, 0.2m)
- **Kinematic viscosity**: ν = 1e-3 m²/s
- **Density**: ρ = 1.0 kg/m³

## Results Format

Results are saved in NumPy compressed format (.npz) with the following data:

- `velocity_x`, `velocity_y`: Velocity field components
- `pressure`: Pressure field
- `vorticity`: Vorticity field
- `drag`, `lift`: Force coefficients over time
- `pressure_drop`: Pressure drop across cylinder
- `strouhal`: Strouhal number
- `mesh_nodes`, `mesh_elements`: Mesh data
- `reynolds_number`, `um`, `dt`: Physical parameters

## Integration

This implementation is designed to coexist with the existing custom FEM and LBM implementations, allowing for comprehensive comparison of different numerical methods for the same physical problem.
