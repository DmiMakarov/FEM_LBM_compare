<!-- 3f2975c2-bae3-41c1-9dd6-4c568bd8cd0b c1aa56fb-88e8-460b-be89-257fc657a270 -->
# Fix FEM Laplacian Calculation Bug

## Problem

The FEM simulation shows velocity becoming maximum (10.0 m/s) everywhere very quickly, while pressure stays near zero. Root cause: **Laplacian calculation is not normalized**.

### Current Buggy Code

File: `FEM/fem_solver.py`, lines 302-305

```python
for j, xj, yj, dist in neighbors:
    weight = 1.0 / (dist**2)
    laplacian_x += weight * (self.u[2*j] - self.u[2*i])
    laplacian_y += weight * (self.u[2*j+1] - self.u[2*i+1])
```

**Problem**: Weights sum to ~50,000+ (8-16 neighbors × weights 565-9048), making viscous term explode.

### Impact

- Velocity explodes from 0.3 m/s to 10.0 m/s (clipping limit) in ~50 steps
- 97% of nodes hit maximum velocity
- Pressure solver cannot compensate
- Simulation is completely wrong

## Solution

### Fix 1: Normalize Laplacian Weights

File: `FEM/fem_solver.py`, lines 297-312

Replace the unnormalized Laplacian with proper normalization:

```python
if len(neighbors) >= 4:
    # Compute Laplacian with proper weight normalization
    laplacian_x = 0.0
    laplacian_y = 0.0
    weight_sum = 0.0
    
    for j, xj, yj, dist in neighbors:
        weight = 1.0 / (dist**2)
        weight_sum += weight
        laplacian_x += weight * (self.u[2*j] - self.u[2*i])
        laplacian_y += weight * (self.u[2*j+1] - self.u[2*i+1])
    
    # Normalize by total weight
    if weight_sum > 0:
        laplacian_x /= weight_sum
        laplacian_y /= weight_sum
    
    # Add artificial viscosity for numerical stability
    artificial_visc = 0.01
    total_visc = self.nu + artificial_visc
    
    viscous_term[2*i] = total_visc * laplacian_x
    viscous_term[2*i+1] = total_visc * laplacian_y
```

### Fix 2: Reduce Time Step (Already Applied)

The 5x speedup (dt=0.005) is too aggressive with the buggy Laplacian. After fixing the Laplacian, we can test if dt=0.005 is stable or needs adjustment.

### Fix 3: Increase Pressure Relaxation

File: `FEM/fem_solver.py`, line 396

After fixing the Laplacian, increase omega from 0.01 to allow faster pressure convergence:

```python
# Adaptive relaxation based on Reynolds number
if self.reynolds_number < 50:
    omega = 0.1  # Faster convergence for low Re
else:
    omega = 0.05  # More stable for high Re
```

## Expected Results

After fixes:

- Velocity stays around 0.3 m/s (inlet velocity), not 10.0 m/s
- Flow develops properly from inlet to outlet
- Pressure builds up correctly (~1-5 Pa range)
- Simulation converges to steady state

## Files to Modify

1. `FEM/fem_solver.py`:

   - Lines 297-312: Fix Laplacian normalization
   - Line 396: Increase omega to 0.1 (low Re) or 0.05 (high Re)

### To-dos

- [ ] Update time_scale from 0.1 to 1.0/8.0 in simple_fem_solver.py (lines 134, 172)
- [ ] Update time_scale from 0.1 to 1.0/8.0 in proper_fem_solver.py (line 444)
- [ ] Update time_scale from 0.1 to 1.0/8.0 in robust_fem_solver.py (line 298)
- [ ] Verify fem_solver.py already has correct implementation (line 162)
- [ ] Create summary document showing corrected boundary conditions