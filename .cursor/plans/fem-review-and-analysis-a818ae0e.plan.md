<!-- a818ae0e-ec25-4442-8e53-7cff94cc8f31 32da3a6f-ed4d-4959-91dd-8e3d2a311e27 -->
# Add FEM Results Saving Functionality

## Problem

The FEM solver (`true_cylinder_flow_fem.py`) runs simulations but doesn't save the results to disk. The `run_optimized_comparison.py` script doesn't call any save method, so all simulation data is lost after the run completes.

## Solution

Add a `save_results()` method to `TrueCylinderFlowFEM` class and call it from `run_optimized_comparison.py`.

## Implementation Steps

### 1. Add save_results() method to true_cylinder_flow_fem.py

Location: After `get_timing_info()` method (around line 265)

Add method to save:

- Velocity fields (ux, uy)
- Pressure fields
- Vorticity fields  
- Time series data (drag, lift, Strouhal, pressure drop)
- Mesh data (nodes, elements)
- Timing information
- Physical parameters (Re, U_m, dt, etc.)

Save format: `.npz` file (compressed NumPy format)

### 2. Call save_results() in run_optimized_comparison.py

Location: After running FEM simulation (around line 120)

Add:

```python
# Save FEM results
output_filename = f"FEM/results/fem_solution_Re{int(re)}_{condition}.npz"
sim.save_results(fem_results, output_filename)
```

### 3. Create results directory structure

Ensure `FEM/results/` directory exists for saving files

## Files to Modify

- `/home/lama/FEM_LBM_compare/FEM/true_cylinder_flow_fem.py` - add save_results() method
- `/home/lama/FEM_LBM_compare/run_optimized_comparison.py` - call save method after simulation

## Expected Output

After running simulation, create files like:

- `FEM/results/fem_solution_Re30_steady.npz`
- `FEM/results/fem_solution_Re150_unsteady.npz`
- `FEM/results/fem_solution_Re150_oscillating.npz`

Each file containing all simulation data for post-processing and visualization.

### To-dos

- [ ] Add save_results() method to TrueCylinderFlowFEM class
- [ ] Call save_results() in run_optimized_comparison.py after FEM simulation
- [ ] Test that results are saved correctly