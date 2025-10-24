# 🚀 Launch Guide for FEM vs LBM Comparison

## 📋 **Quick Start Commands**

### **1. Test Initial Conditions**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python test_initial_conditions.py
```
**What it does:**
- Tests all 3 initial conditions (steady, unsteady, oscillating)
- Generates velocity profile plots
- Runs short simulations for both FEM and LBM

---

### **2. Run Complete Comparison**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python compare_methods.py
```
**What it does:**
- Runs FEM and LBM simulations for Re = 20, 40, 100, 200
- Generates timing comparison
- Creates pressure drop analysis
- Saves all results to organized folders

---

### **3. Generate Animations**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python animate_solutions.py
```
**What it does:**
- Creates animated GIFs for pressure, velocity, and vorticity
- Generates method-specific animations (FEM, LBM, comparison)
- Saves animations to `results/animations/`

---

## 🎯 **Individual Method Launches**

### **FEM Only**
```python
from FEM.cylinder_flow_fem import CylinderFlowFEM

# Steady condition (Re=20)
fem = CylinderFlowFEM(
    'meshes/cylinder_mesh_Re20_data.npz',
    reynolds_number=20, dt=0.001, max_velocity=0.3,
    initial_condition="steady", um=0.3
)
results = fem.run_simulation(max_steps=500, save_interval=50)
```

### **LBM Only**
```python
from LBM.cylinder_flow_lbm import CylinderFlowLBM

# Oscillating condition (Re=100)
lbm = CylinderFlowLBM(
    nx=100, ny=25, reynolds_number=100,
    cylinder_diameter=0.1, cylinder_x=0.2, cylinder_y=0.2,
    initial_condition="oscillating", um=1.5
)
results = lbm.run_simulation(max_steps=500, save_interval=50)
```

---

## 📊 **Generated Outputs**

### **Results Structure**
```
results/
├── fem/                    # FEM simulation results
│   ├── fem_Re20_results.npz
│   ├── fem_Re40_results.npz
│   ├── fem_Re100_results.npz
│   └── fem_Re200_results.npz
├── lbm/                    # LBM simulation results
│   ├── lbm_Re20_results.npz
│   ├── lbm_Re40_results.npz
│   ├── lbm_Re100_results.npz
│   └── lbm_Re200_results.npz
├── comparison/             # Comparison analysis
│   ├── timing_summary.json
│   └── detailed_comparison.json
├── plots/                  # Analysis plots
│   ├── timing_comparison.png
│   └── pressure_drop_comparison.png
├── animations/             # Flow field animations
│   ├── fem_pressure_Re20_animation.gif
│   ├── lbm_pressure_Re20_animation.gif
│   ├── comparison_pressure_Re20_animation.gif
│   └── ... (for all Reynolds numbers and field types)
└── initial_conditions/     # Initial condition analysis
    ├── initial_conditions_comparison.png
    ├── steady_profile_Re20.png
    ├── unsteady_profile_Re100.png
    └── oscillating_profile_Re100.png
```

---

## 🎨 **Custom Launches**

### **Specific Initial Condition**
```python
# Test only oscillating condition
from test_initial_conditions import test_initial_conditions

# Modify the test_initial_conditions function to test only one condition
initial_conditions = [
    ("oscillating", 1.5, "Condition 3: Oscillating")
]
```

### **Custom Reynolds Numbers**
```python
# Modify compare_methods.py
comparison = MethodComparison(reynolds_numbers=[50, 150, 300])
```

### **Custom Animation Settings**
```python
from animate_solutions import FlowAnimator

animator = FlowAnimator()
# Create custom animation
animator.create_fem_animation(100, 'pressure', duration=10.0, fps=15)
```

---

## ⚙️ **Configuration Options**

### **Initial Conditions**
- **"steady"**: `U_x(0,y) = 4U_m y(H-y)/H^2`, `U_m = 0.3 m/s` (Re=20)
- **"unsteady"**: `U_x(0,y,t) = 4U_m y(H-y)/H^2`, `U_m = 1.5 m/s` (Re=100)
- **"oscillating"**: `U_x(0,y,t) = 4U_m y(H-y)sin(πt/8)/H^2`, `U_m = 1.5 m/s` (Re=100)

### **Simulation Parameters**
- **Time step**: `dt = 0.001`
- **Max steps**: `500` (configurable)
- **Save interval**: `50` (configurable)
- **Reynolds numbers**: `[20, 40, 100, 200]`

### **Grid Settings**
- **FEM**: Structured mesh with ~5000 nodes
- **LBM**: 100×25 grid points
- **Domain**: 2.2×0.41 m
- **Cylinder**: D=0.1 m at (0.2, 0.2)

---

## 🔧 **Troubleshooting**

### **Common Issues**
1. **Missing mesh files**: Run mesh generation first
2. **Memory issues**: Reduce grid size or max steps
3. **Convergence problems**: Adjust time step or relaxation parameters

### **Performance Tips**
- **FEM**: Slower but more accurate
- **LBM**: Faster but less accurate at high Re
- **Animations**: Reduce fps or duration for faster generation

---

## 📈 **Expected Results**

### **Performance Comparison**
- **FEM**: ~10-30 seconds per simulation
- **LBM**: ~0.3 seconds per simulation
- **Speedup**: 30-80x faster for LBM

### **Physical Results**
- **Drag coefficients**: Vary with Reynolds number
- **Strouhal number**: Should be ~0.2 for Re > 40
- **Pressure drop**: Higher for higher Reynolds numbers

---

## 🎉 **Summary**

The framework provides:

1. **✅ Three Initial Conditions** - Steady, unsteady, oscillating
2. **✅ Both Methods** - FEM and LBM comparison
3. **✅ Complete Analysis** - Timing, forces, pressure drop
4. **✅ Visualizations** - Plots and animations
5. **✅ Organized Output** - Structured results folder

**Ready to launch!** 🚀
