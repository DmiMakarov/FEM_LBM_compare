<!-- 68ff6129-cdbc-4d94-b237-379fba413f20 52db1f0d-22cd-434d-bda8-baaf73f98ca3 -->
# Create Proper FEM Solver for 2D Cylinder Flow

## Problem Analysis

Current issues with existing FEM solvers:

1. **Incorrect physics**: Finite difference approximations on unstructured meshes are fundamentally wrong
2. **Numerical instability**: The `working_fem_solver.py` uses ad-hoc neighbor searches and incorrect gradient computations leading to divergence
3. **Missing proper FEM formulation**: No element-level assembly, no shape functions, no proper integration

## Solution Approach

Build a proper FEM solver using:

- **IPCS (Incremental Pressure Correction Scheme)** for time discretization
- **Linear triangular elements (P1)** for velocity and pressure
- **Proper element assembly** with shape functions and Gauss quadrature
- **Stabilization** for equal-order velocity-pressure formulation (PSPG/SUPG if needed)

## Key Implementation Steps

### 1. Clean Old Files

Remove problematic FEM solvers:

- `FEM/fem_solver.py`
- `FEM/working_fem_solver.py`
- `FEM/proper_fem_solver.py`
- `FEM/realistic_fem_solver.py`
- `FEM/simple_fem_solver.py`
- `FEM/exact_fem_solver.py`

Keep only:

- `FEM/generate_simple_mesh.py` (mesh generation)
- `FEM/visualize_fem.py` (visualization)
- `FEM/cylinder_flow_fem.py` (will be updated)

### 2. Create New FEM Solver (`FEM/fem_solver.py`)

Core components:

**Element Assembly:**

```python
- Shape functions for P1 triangular elements
- Gradient computation in reference element
- Jacobian transformation
- Gauss quadrature integration
```

**System Matrices:**

- **Mass matrix M**: ∫ φᵢ φⱼ dx
- **Stiffness matrix K**: ∫ ∇φᵢ · ∇φⱼ dx
- **Gradient matrices Gx, Gy**: ∫ φᵢ ∂φⱼ/∂x dx
- **Divergence matrix D**: Gx^T + Gy^T

**IPCS Time Stepping:**

1. **Tentative velocity**: (M + νΔt K) u* = M uⁿ - Δt Gp^n - Δt N(uⁿ)
2. **Pressure correction**: D M⁻¹ G Δp = D u* / Δt
3. **Velocity correction**: u^(n+1) = u* - Δt M⁻¹ G Δp
4. **Pressure update**: p^(n+1) = p^n + Δp

**Boundary Conditions:**

**Inlet (x = 0):** Three different velocity profiles available:

1. **Steady (Re = 20)**: U_x(0, y) = 4U_m y(H - y)/H², U_y = 0, U_m = 0.3 m/s
2. **Unsteady (Re = 100)**: U_x(0, y, t) = 4U_m y(H - y)/H², U_y = 0, U_m = 1.5 m/s
3. **Oscillating**: U_x(0, y, t) = 4U_m y(H - y)sin(πt/8)/H², U_y = 0, U_m = 1.5 m/s

Where H = 0.41 m (channel height)

**Cylinder:** No-slip boundary condition (u = 0, v = 0)

**Outlet (x = L):** Do-nothing / zero normal stress condition (∂u/∂x = 0, ∂v/∂x = 0, p = 0)

**Walls (y = 0, y = H):** No-slip (u = 0, v = 0) or slip depending on setup

**Fixed Physical Parameters:**

- Kinematic viscosity: ν = 10⁻³ m²/s
- Fluid density: ρ = 1.0 kg/m³
- Cylinder diameter: D = 0.1 m
- Domain: 2.2 × 0.41 m

**Reynolds Number Calculation:**

Re = U_m × D / ν

For the three conditions:

- Condition 1: Re = 0.3 × 0.1 / 10⁻³ = 30 (not 20, should adjust U_m or use Re = 20)
- Condition 2: Re = 1.5 × 0.1 / 10⁻³ = 150 (not 100, should adjust)
- Condition 3: Re = 1.5 × 0.1 / 10⁻³ = 150

**Force Computation:**

Proper integration over cylinder surface:

```python
Drag = ∫_cylinder (-p nx + ν ∂u/∂n) ds
Lift = ∫_cylinder (-p ny + ν ∂v/∂n) ds
```

### 3. Update Main Simulation Script

Modify `FEM/cylinder_flow_fem.py`:

- Import new `FEM_Solver` class
- Remove dependency on broken solvers
- Add proper result saving
- Implement convergence checking

### 4. Key Files to Create/Modify

**New file:** `FEM/fem_solver.py` (~500-700 lines)

- Complete FEM implementation with IPCS

**Update:** `FEM/cylinder_flow_fem.py`

- Use new solver
- Clean up imports

**Update:** `FEM/visualize_fem.py` (if needed)

- Ensure compatibility with new solver output

## Physics Correctness

The new solver will properly implement:

1. **Incompressible Navier-Stokes**: ∂u/∂t + (u·∇)u = -∇p + ν∇²u, ∇·u = 0
2. **Proper weak formulation** with test functions
3. **Consistent boundary conditions** matching LBM setup
4. **Accurate force computation** using stress tensor integration

## Expected Outcomes

- Stable simulation for Re = 20, 40, 100, 200
- Correct vortex shedding for Re ≥ 100
- Strouhal numbers matching literature (St ≈ 0.2 for Re = 100)
- Comparable results to LBM implementation

### To-dos

- [ ] Remove old/broken FEM solver files
- [ ] Create new fem_solver.py with proper FEM formulation and IPCS scheme
- [ ] Update cylinder_flow_fem.py to use new solver
- [ ] Run test simulation to verify solver works correctly