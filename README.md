# FEM vs LBM Comparison for 2D Cylinder Flow

This project implements and compares two numerical methods for simulating 2D flow around a circular cylinder:

- **Lattice Boltzmann Method (LBM)** with D2Q9 lattice and MRT collision model
- **Finite Element Method (FEM)** using FEniCS with incompressible Navier-Stokes equations

## Problem Description

The simulation considers 2D flow around a circular cylinder with:
- **Domain**: 2.2 × 0.41 m rectangular domain
- **Cylinder**: Diameter 0.1 m, centered at (0.2, 0.2) m
- **Reynolds numbers**: 20, 40, 100, 200
- **Focus**: Pressure field visualization and Strouhal number calculation

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **No additional FEM libraries needed** - uses pure scipy implementation

3. **Mesh generation** (simplified - no additional libraries needed):
   - Uses built-in structured mesh generator
   - No gmsh or OpenGL dependencies required

## Usage

### 1. Generate Meshes (FEM)

Meshes are generated automatically when running FEM simulations. If you want to generate them manually:

```bash
cd FEM
python generate_simple_mesh.py
```

This creates meshes in the `meshes/` directory.

### 2. Run LBM Simulations

```bash
cd LBM
python cylinder_flow_lbm.py
```

This runs LBM simulations for all Reynolds numbers and saves results.

### 3. Run FEM Simulations

```bash
cd FEM
python cylinder_flow_fem.py
```

This runs FEM simulations for all Reynolds numbers and saves results.

### 4. Visualize Results

**LBM visualization**:
```bash
cd LBM
python visualize_lbm.py
```

**FEM visualization**:
```bash
cd FEM
python visualize_fem.py
```

### 5. Compare Methods

```bash
python compare_methods.py
```

This generates comparison plots and statistics.

## Project Structure

```
FEM_LBM_compare/
├── LBM/                          # Lattice Boltzmann Method
│   ├── lbm_solver.py            # Core LBM solver with MRT
│   ├── cylinder_flow_lbm.py     # Cylinder flow simulation
│   └── visualize_lbm.py         # Visualization tools
├── FEM/                         # Finite Element Method
│   ├── fem_solver.py            # Core FEM solver
│   ├── generate_mesh.py         # Mesh generation
│   ├── cylinder_flow_fem.py    # Cylinder flow simulation
│   └── visualize_fem.py         # Visualization tools
├── compare_methods.py           # Comparison framework
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Key Parameters

### LBM Parameters
- **Lattice**: D2Q9 (2D, 9 velocities)
- **Collision model**: MRT (Multiple Relaxation Time)
- **Grid resolution**: 200 × 50
- **Relaxation time**: Computed from Reynolds number

### FEM Parameters
- **Implementation**: Pure scipy (no FEniCS required)
- **Elements**: Linear triangular elements
- **Time scheme**: Incremental Pressure Correction Scheme (IPCS)
- **Mesh**: Generated with gmsh, refined near cylinder

### Physical Parameters
- **Domain**: 2.2 × 0.41 m
- **Cylinder**: D = 0.1 m, center at (0.2, 0.2) m
- **Reynolds numbers**: 20, 40, 100, 200
- **Inlet velocity**: Parabolic profile, max 0.1 m/s

## Expected Results

### Strouhal Numbers
- **Re = 20**: St ≈ 0.0 (steady flow)
- **Re = 40**: St ≈ 0.0 (steady flow)
- **Re = 100**: St ≈ 0.2 (vortex shedding)
- **Re = 200**: St ≈ 0.2 (vortex shedding)

### Drag Coefficients
- **Re = 20**: Cd ≈ 2.0
- **Re = 40**: Cd ≈ 1.5
- **Re = 100**: Cd ≈ 1.2
- **Re = 200**: Cd ≈ 1.0

## Output Files

### LBM Results
- `lbm_results_Re*.npz`: Time series data
- `lbm_results_Re*_fields.npz`: Field data
- `lbm_plots/`: Visualization plots

### FEM Results
- `fem_results_Re*.npz`: Time series data
- `fem_results_Re*_fields.npz`: Field data
- `fem_plots/`: Visualization plots

### Comparison Results
- `comparison_results/`: Comparison plots and statistics
- `comparison_results/summary_report.json`: Summary statistics

## Validation

The results are validated against:
- **Literature values** for Strouhal numbers
- **Benchmark drag coefficients** for cylinder flow
- **Cross-validation** between FEM and LBM methods

## Troubleshooting

### Common Issues

1. **FEniCS installation**: Make sure FEniCS is properly installed and accessible
2. **Mesh generation**: Ensure gmsh is installed and accessible
3. **Memory issues**: For high Reynolds numbers, consider reducing grid resolution
4. **Convergence**: Adjust time step size or relaxation parameters if needed

### Performance Tips

- **LBM**: Use smaller time steps for high Reynolds numbers
- **FEM**: Use finer meshes near the cylinder for better accuracy
- **Visualization**: Reduce field data resolution for faster plotting

## References

1. **LBM**: Lattice Boltzmann Method for Fluid Dynamics
2. **FEM**: Finite Element Method for Navier-Stokes Equations
3. **Cylinder Flow**: Benchmark case for CFD validation

## License

This project is open source and available under the MIT License.
