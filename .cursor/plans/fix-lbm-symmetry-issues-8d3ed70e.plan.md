<!-- 8d3ed70e-6582-41a3-8f4d-d38dbb8a8b19 621fc78f-62d9-4536-8ef5-9f9737f7a821 -->
# Fix LBM Unsteady Case Divergence

## Critical Problem

**Pressure: -300 to 2400** → Simulation is DIVERGING!

### Root Cause

The inlet is applying **physical velocity (1.5 m/s)** directly to the lattice, but:

- Max velocity limit is 0.05 (lattice units)
- Inlet is trying to apply 1.5 (way too high!)
- **Mismatch**: Inlet forces high velocity → Collision clamps it → Creates massive pressure waves → DIVERGENCE

## The Fix

### **Scale Physical Velocity to Lattice Units**

**File:** `LBM/cylinder_flow_lbm.py` (after line 74)

The problem is here:

```python
self.u_inlet = self.um  # WRONG: Using 1.5 m/s directly!
```

Should be:

```python
# Scale physical velocity to safe lattice units
# For Re=150, we need MUCH smaller lattice velocity
if self.reynolds_number <= 50:
    self.u_lattice = 0.08  # Lattice velocity for low Re
elif self.reynolds_number <= 100:
    self.u_lattice = 0.06  # Lattice velocity for medium Re  
else:
    self.u_lattice = 0.04  # Lattice velocity for high Re (very conservative)

self.u_inlet = self.u_lattice  # Use lattice velocity, not physical!
```

### **Why This Works**

1. **Physical velocity (U_m)**: 1.5 m/s (for Re calculation only)
2. **Lattice velocity (u_lattice)**: 0.04 (for simulation)
3. **Reynolds number**: Still Re=150 (calculated from physical values)
4. **Stability**: Lattice Ma = 0.04/0.577 ≈ 0.07 << 1 ✓

### **Alternative: Adjust Grid Resolution**

If you want to keep physical velocity scaling, you need:

- **Finer grid**: Smaller dx → Lower lattice velocity
- **Current**: nx=100, dx=0.022
- **Needed for Re=150**: nx=500+, dx=0.0044

But this is computationally expensive!

## Implementation

### Step 1: Add Lattice Velocity Scaling

**File:** `LBM/cylinder_flow_lbm.py` (lines 73-75)

```python
# Scale physical velocity to safe lattice units
if self.reynolds_number <= 50:
    self.u_lattice = 0.08  # Lattice velocity for low Re
elif self.reynolds_number <= 100:
    self.u_lattice = 0.06  # Lattice velocity for medium Re  
else:
    self.u_lattice = 0.04  # Lattice velocity for high Re

# Use lattice velocity for simulation (not physical velocity!)
self.u_inlet = self.u_lattice
```

### Step 2: Update Inlet Profile Scaling

**File:** `LBM/cylinder_flow_lbm.py` (lines 302-308)

The parabolic profile should use `self.u_inlet` (lattice), not `self.um` (physical):

```python
if self.initial_condition == "steady":
    u_inlet_profile[j] = 4 * self.u_inlet * y * (H - y) / (H**2)  # Use u_inlet!
elif self.initial_condition == "unsteady":
    u_inlet_profile[j] = 4 * self.u_inlet * y * (H - y) / (H**2)  # Use u_inlet!
elif self.initial_condition == "oscillating":
    time_scale = 0.1
    u_inlet_profile[j] = 4 * self.u_inlet * y * (H - y) * np.sin(np.pi * self.lbm.simulation_time * time_scale) / (H**2)  # Use u_inlet!
```

### Step 3: Keep Velocity Limits Aligned

**File:** `LBM/simple_lbm_solver.py` (lines 144-151)

The max_u should match the inlet velocity:

```python
# Adjust max velocity based on Reynolds number
# MUST match the inlet lattice velocity!
if self.reynolds_number <= 50:
    max_u = 0.1  # Match inlet
elif self.reynolds_number <= 100:
    max_u = 0.08  # Match inlet
else:
    max_u = 0.06  # Slightly higher than inlet (0.04) for safety
```

## Expected Results

- **Pressure range**: Should be ~0.3-0.4 (reasonable)
- **Stability**: No divergence
- **Physics**: Still represents Re=150 (Reynolds calculated from physical values)
- **Accuracy**: Reduced (due to low Ma), but stable

## Key Insight

**LBM requires**:

- Physical velocity (U_m) for Reynolds number calculation
- **Separate** lattice velocity (u_lattice) for simulation
- Lattice velocity MUST be << 0.1 for stability
- For high Re, use VERY small lattice velocity (0.04-0.05)

## Trade-off

- **Stability**: ✓ Excellent
- **Accuracy**: Reduced (low Ma = more compressibility errors)
- **Alternative**: Use much finer grid (5x more points) for better accuracy

### To-dos

- [ ] Add lattice velocity scaling (u_lattice = 0.04 for Re>100)
- [ ] Use u_inlet = u_lattice instead of um
- [ ] Update parabolic inlet profile to use u_inlet instead of um
- [ ] Align max_u with inlet lattice velocity