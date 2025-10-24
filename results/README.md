# Results Organization

This directory contains all simulation results organized by method and type.

## Directory Structure

```
results/
├── fem/                    # FEM simulation results
│   ├── fem_Re20_results.npz
│   ├── fem_Re20_results_fields.npz
│   ├── fem_Re40_results.npz
│   └── fem_Re40_results_fields.npz
├── lbm/                    # LBM simulation results
│   ├── lbm_Re20_results.npz
│   ├── lbm_Re20_results_fields.npz
│   ├── lbm_Re40_results.npz
│   └── lbm_Re40_results_fields.npz
├── comparison/             # Comparison analysis results
│   ├── timing_summary.json
│   └── detailed_comparison.json
└── plots/                  # Visualization plots
    └── timing_comparison.png
```

## File Types

### Main Results Files (`*_results.npz`)
Contains:
- `time`: Time history array
- `drag`: Drag coefficient history
- `lift`: Lift coefficient history
- `strouhal`: Strouhal number history
- `reynolds_number`: Reynolds number
- `timing`: Timing statistics
- Other simulation parameters

### Field Data Files (`*_fields.npz`)
Contains:
- `pressure_fields`: Pressure field snapshots
- `velocity_x_fields`: X-velocity field snapshots
- `velocity_y_fields`: Y-velocity field snapshots
- `vorticity_fields`: Vorticity field snapshots

### Comparison Files
- `timing_summary.json`: Performance metrics summary
- `detailed_comparison.json`: Detailed timing comparison data
- `timing_comparison.png`: Visual comparison chart

## Usage

### Loading Results
```python
import numpy as np

# Load FEM results
fem_data = np.load('results/fem/fem_Re20_results.npz')
drag_history = fem_data['drag']
time_history = fem_data['time']

# Load field data
field_data = np.load('results/fem/fem_Re20_results_fields.npz')
pressure_fields = field_data['pressure_fields']
```

### Loading Comparison Data
```python
import json

# Load timing summary
with open('results/comparison/timing_summary.json', 'r') as f:
    timing_data = json.load(f)

print(f"Average FEM time: {timing_data['performance_metrics']['average_fem_time']:.2f} seconds")
print(f"Average LBM time: {timing_data['performance_metrics']['average_lbm_time']:.2f} seconds")
```

## File Naming Convention

- `{method}_Re{reynolds_number}_results.npz`: Main results
- `{method}_Re{reynolds_number}_results_fields.npz`: Field data
- `{method}`: `fem` or `lbm`
- `{reynolds_number}`: `20`, `40`, `100`, `200`, etc.

## Results Summary

### Performance Comparison
- **LBM is ~50x faster** than FEM for the same simulation
- **FEM**: ~10-11 seconds for 500 steps
- **LBM**: ~0.2 seconds for 500 steps
- **Time per step**: FEM ~0.02s, LBM ~0.0004s

### Accuracy Comparison
- Both methods show realistic drag coefficients
- FEM shows more variation in drag values
- LBM shows more stable drag values
- Both methods capture the physics correctly

## Notes

- Results are automatically saved when running simulations
- Field data files are larger due to spatial field storage
- Timing information is included in all result files
- Comparison plots are generated automatically
