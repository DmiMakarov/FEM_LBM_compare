# 🚀 Complete Launch Guide for FEM vs LBM Comparison

## 📋 **All Available Launch Options**

### **1. Standard Comparison (Original)**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python compare_methods.py
```
**What it does:**
- Runs FEM vs LBM comparison for Re = 20, 40, 100, 200
- Generates timing analysis
- Creates pressure drop analysis
- Saves results to `results/comparison/` and `results/plots/`

---

### **2. Initial Condition Comparison (NEW!)**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python compare_initial_conditions.py
```
**What it does:**
- Compares FEM vs LBM for all three initial conditions:
  - **Steady** (Re=20): `U_x(0,y) = 4U_m y(H-y)/H^2`, `U_m = 0.3 m/s`
  - **Unsteady** (Re=100): `U_x(0,y,t) = 4U_m y(H-y)/H^2`, `U_m = 1.5 m/s`
  - **Oscillating** (Re=100): `U_x(0,y,t) = 4U_m y(H-y)sin(πt/8)/H^2`, `U_m = 1.5 m/s`
- Creates individual comparison plots for each condition
- Generates timing analysis for each condition
- Saves results to `results/initial_condition_comparison/`

---

### **3. Test Initial Conditions**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python test_initial_conditions.py
```
**What it does:**
- Tests all three initial conditions with short simulations
- Generates velocity profile plots
- Validates the initial condition implementations
- Saves results to `results/initial_conditions/`

---

### **4. Standard Animations**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python animate_solutions.py
```
**What it does:**
- Creates flow field animations for standard comparison
- Generates pressure, velocity, and vorticity animations
- Creates method-specific and comparison animations
- Saves animations to `results/animations/`

---

### **5. Initial Condition Animations (NEW!)**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python animate_initial_conditions.py
```
**What it does:**
- Creates animations for all three initial conditions
- Generates pressure, velocity, and vorticity animations for each condition
- Creates side-by-side FEM vs LBM comparison animations
- Saves animations to `results/animations/initial_conditions/`

---

### **6. Comprehensive Launcher (NEW!)**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python run_comparisons.py --mode all
```
**Available modes:**
- `all` - Run everything (default)
- `standard` - Standard FEM vs LBM comparison
- `initial_conditions` - Initial condition comparison
- `animations` - Standard animations
- `ic_animations` - Initial condition animations
- `test_ic` - Test initial conditions
- `help` - Show help information

---

## 🎯 **Quick Start Commands**

### **Run Everything**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python run_comparisons.py
```

### **Run Only Initial Condition Comparison**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python run_comparisons.py --mode initial_conditions
```

### **Run Only Animations**
```bash
cd /home/lama/FEM_LBM_compare
source .venv/bin/activate
python run_comparisons.py --mode ic_animations
```

---

## 📊 **Generated Outputs**

### **Results Structure**
```
results/
├── comparison/                    # Standard comparison results
│   ├── timing_summary.json
│   ├── detailed_comparison.json
│   └── pressure_drop_comparison.json
├── initial_condition_comparison/ # Initial condition comparison
│   ├── plots/
│   │   ├── steady_comparison_Re20.png
│   │   ├── unsteady_comparison_Re100.png
│   │   ├── oscillating_comparison_Re100.png
│   │   ├── steady_timing_Re20.png
│   │   ├── unsteady_timing_Re100.png
│   │   ├── oscillating_timing_Re100.png
│   │   └── overall_comparison.png
│   └── data/
│       └── summary.json
├── plots/                        # Analysis plots
│   ├── timing_comparison.png
│   └── pressure_drop_comparison.png
├── animations/                   # Standard animations
│   ├── fem_pressure_Re20_animation.gif
│   ├── lbm_pressure_Re20_animation.gif
│   ├── comparison_pressure_Re20_animation.gif
│   └── ... (for all Reynolds numbers and field types)
├── animations/initial_conditions/ # Initial condition animations
│   ├── fem_pressure_steady_Re20_animation.gif
│   ├── lbm_pressure_steady_Re20_animation.gif
│   ├── comparison_pressure_steady_Re20_animation.gif
│   ├── fem_pressure_unsteady_Re100_animation.gif
│   ├── lbm_pressure_unsteady_Re100_animation.gif
│   ├── comparison_pressure_unsteady_Re100_animation.gif
│   ├── fem_pressure_oscillating_Re100_animation.gif
│   ├── lbm_pressure_oscillating_Re100_animation.gif
│   ├── comparison_pressure_oscillating_Re100_animation.gif
│   └── ... (for all field types)
└── initial_conditions/          # Initial condition testing
    ├── initial_conditions_comparison.png
    ├── steady_profile_Re20.png
    ├── unsteady_profile_Re100.png
    └── oscillating_profile_Re100.png
```

---

## 🎨 **Initial Conditions Explained**

### **1. Steady Condition (Re=20)**
- **Formula**: `U_x(0,y) = 4U_m y(H-y)/H^2`, `U_y = 0`
- **Parameters**: `U_m = 0.3 m/s`, `H = 0.41 m`
- **Purpose**: Steady flow for low Reynolds number
- **Expected**: Laminar flow, no vortex shedding

### **2. Unsteady Condition (Re=100)**
- **Formula**: `U_x(0,y,t) = 4U_m y(H-y)/H^2`, `U_y = 0`
- **Parameters**: `U_m = 1.5 m/s`, `H = 0.41 m`
- **Purpose**: Unsteady flow for higher Reynolds number
- **Expected**: Vortex shedding, oscillating forces

### **3. Oscillating Condition (Re=100)**
- **Formula**: `U_x(0,y,t) = 4U_m y(H-y)sin(πt/8)/H^2`, `U_y = 0`
- **Parameters**: `U_m = 1.5 m/s`, `H = 0.41 m`
- **Purpose**: Time-dependent oscillating inlet
- **Expected**: Forced oscillations, complex flow patterns

---

## ⚙️ **Configuration Options**

### **Reynolds Numbers**
- **Standard comparison**: `[20, 40, 100, 200]`
- **Initial condition comparison**: `[20, 100]`
- **Custom**: Modify the scripts to use different values

### **Simulation Parameters**
- **Time step**: `dt = 0.001`
- **Max steps**: `500` (standard), `200` (initial conditions)
- **Save interval**: `50` (standard), `20` (initial conditions)
- **Animation duration**: `5.0` seconds
- **Animation fps**: `10`

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
4. **Animation errors**: Check matplotlib backend

### **Performance Tips**
- **FEM**: Slower but more accurate
- **LBM**: Faster but less accurate at high Re
- **Animations**: Reduce fps or duration for faster generation

---

## 📈 **Expected Results**

### **Performance Comparison**
- **FEM**: ~5-10 seconds per simulation
- **LBM**: ~0.1-0.2 seconds per simulation
- **Speedup**: 30-80x faster for LBM

### **Physical Results**
- **Drag coefficients**: Vary with Reynolds number and initial condition
- **Lift coefficients**: Higher for unsteady/oscillating conditions
- **Strouhal number**: Should be ~0.2 for Re > 40
- **Pressure drop**: Higher for higher Reynolds numbers

---

## 🎉 **Summary**

The framework now provides:

1. **✅ Three Initial Conditions** - Steady, unsteady, oscillating
2. **✅ Both Methods** - FEM and LBM comparison
3. **✅ Complete Analysis** - Timing, forces, pressure drop
4. **✅ Visualizations** - Plots and animations
5. **✅ Organized Output** - Structured results folder
6. **✅ Multiple Launch Options** - Standard, initial conditions, animations
7. **✅ Comprehensive Launcher** - Single command for everything

**Ready to launch!** 🚀

---

## 🚀 **Quick Reference**

| Command | Purpose | Output |
|---------|---------|---------|
| `python compare_methods.py` | Standard comparison | `results/comparison/` |
| `python compare_initial_conditions.py` | Initial condition comparison | `results/initial_condition_comparison/` |
| `python test_initial_conditions.py` | Test initial conditions | `results/initial_conditions/` |
| `python animate_solutions.py` | Standard animations | `results/animations/` |
| `python animate_initial_conditions.py` | Initial condition animations | `results/animations/initial_conditions/` |
| `python run_comparisons.py` | Everything | All results |
| `python run_comparisons.py --mode help` | Show help | Help information |
