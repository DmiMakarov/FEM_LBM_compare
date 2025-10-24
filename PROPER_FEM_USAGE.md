# Proper FEM Cylinder Flow Simulation

This implementation provides a proper finite element method solver for cylinder flow using scikit-fem, solving the actual Navier-Stokes equations (not fake physics).

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `scikit-fem` - For finite element method
- `numpy`, `scipy` - For numerical computations
- `matplotlib` - For visualization

## Usage

### Command Line Interface

Run simulations using the standalone script:

```bash
# Steady flow (Re=20)
python run_proper_skfem_simulation.py --condition steady --max-steps 2000

# Unsteady flow (Re=100)
python run_proper_skfem_simulation.py --condition unsteady --max-steps 5000

# Oscillating flow (Re=100)
python run_proper_skfem_simulation.py --condition oscillating --max-steps 5000
```

### Available Options

- `--condition`: Flow type ("steady", "unsteady", "oscillating")
- `--mesh-density`: Mesh resolution ("coarse", "medium", "fine")
- `--dt`: Time step size (default: 0.001)
- `--max-steps`: Maximum time steps (default: 1000)
- `--save-interval`: Save results every N steps (default: 10)
- `--output-dir`: Output directory (default: "results/proper_skfem")
- `--visualize`: Generate visualization plots

### Python API

```python
from FEM_lib import ProperSkfemCylinderFlow

# Create simulation
simulation = ProperSkfemCylinderFlow(
    mesh_density="medium",
    dt=0.001,
    initial_condition="steady"
)

# Run simulation
results = simulation.run_simulation(
    max_steps=1000,
    save_interval=10
)

# Save results
simulation.save_results(results, "results/proper_skfem/steady_flow.npz")

# Visualize
simulation.visualize_solution("solution.png")
```

## Boundary Conditions

The implementation supports three boundary conditions as specified:

1. **Steady flow (Re=20)**:
   - U_x(0,y) = 4*U_m*y*(H-y)/H², U_y=0, U_m=0.3 m/s

2. **Unsteady flow (Re=100)**:
   - U_x(0,y,t) = 4*U_m*y*(H-y)/H², U_y=0, U_m=1.5 m/s

3. **Oscillating flow (Re=100)**:
   - U_x(0,y,t) = 4*U_m*y*(H-y)*sin(πt/8)/H², U_y=0, U_m=1.5 m/s

## Validation

Run the validation tests:

```bash
# Test installation
python test_proper_skfem_installation.py

# Test boundary conditions and physics
python test_fenics_solver.py
```

This will test:
- scikit-fem installation
- Boundary condition application
- Physics parameters
- Mesh generation
- Time stepping

## Output Format

Results are saved in NumPy compressed format (.npz) containing:

- **Field data**: velocity_x, velocity_y, pressure, vorticity
- **Time series**: drag, lift, pressure_drop, strouhal
- **Field time series**: velocity_x_fields, velocity_y_fields, pressure_fields, vorticity_fields
- **Mesh data**: nodes, elements, boundary_nodes
- **Physical parameters**: reynolds_number, um, dt, nu, rho, etc.
- **Timing information**: total_time, time_per_step, force_calculation_time

## Key Features

- **Real FEM solver**: Uses scikit-fem with proper Taylor-Hood elements (P2-P1)
- **Actual Navier-Stokes**: Solves real NS equations using IPCS scheme
- **Boundary conditions**: Correctly implements specified inlet profiles
- **Force computation**: Drag and lift calculation using stress tensor integration
- **Time stepping**: Stable time integration with proper linearization
- **Mesh generation**: Automatic mesh generation with cylinder hole
- **Visualization**: Solution plotting and animation support

## Technical Details

### Solver Method
- **Elements**: Taylor-Hood (P2-P1) for velocity-pressure
- **Time scheme**: IPCS (Incremental Pressure Correction Scheme)
- **Linearization**: Newton-Raphson for nonlinear terms
- **Boundary conditions**: DirichletBC for velocity, natural for pressure

### Physics Implementation
- Incompressible Navier-Stokes equations
- Proper weak form using scikit-fem forms
- Mass, stiffness, gradient, and divergence operators
- Convection term linearization for stability

## Comparison with Previous Implementation

This proper scikit-fem implementation replaces the previous "fake physics" solver with:

- ✅ **Real Navier-Stokes equations** (not hardcoded flow patterns)
- ✅ **Proper finite element method** (Taylor-Hood elements)
- ✅ **Correct boundary conditions** (parabolic inlet profiles)
- ✅ **Stable time integration** (IPCS scheme)
- ✅ **Force computation** (stress tensor integration)
- ✅ **Mass conservation** (divergence-free velocity)

## Troubleshooting

### Common Issues

1. **scikit-fem installation**: Make sure scikit-fem is properly installed
2. **Memory usage**: Fine meshes require significant RAM
3. **Convergence**: May need smaller time steps for high Re
4. **Boundary conditions**: Ensure proper boundary detection

### Performance Tips

- Use "coarse" mesh for testing
- Start with larger time steps
- Monitor convergence for steady cases
- Use appropriate mesh density for Reynolds number

## Alternative: FEniCS Implementation

If you prefer FEniCS (requires conda installation):

```bash
# Install FEniCS via conda
conda install -c conda-forge fenicsx

# Then use the FEniCS implementation
python run_fenics_simulation.py --condition steady
```

The FEniCS implementation provides the same functionality but requires conda installation.
