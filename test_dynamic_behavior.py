#!/usr/bin/env python3
"""
Test dynamic behavior of scikit-fem solver.
"""

import sys
import numpy as np
from pathlib import Path

# Add FEM_lib to path
sys.path.append('FEM_lib')

from FEM_lib import SkfemCylinderFlow

def test_dynamic_behavior():
    """Test that the solver produces time-varying results."""
    print("Testing dynamic behavior of scikit-fem solver...")

    # Create simulation
    simulation = SkfemCylinderFlow(
        mesh_density="coarse",
        dt=0.01,
        initial_condition="steady"
    )

    # Run a few time steps and check if values change
    print("Running 10 time steps...")

    initial_u = simulation.solver.u.copy()
    initial_v = simulation.solver.v.copy()
    initial_p = simulation.solver.p.copy()

    # Run 10 time steps
    for i in range(10):
        u_new, v_new, p_new = simulation.solver.solve_time_step()

    final_u = simulation.solver.u.copy()
    final_v = simulation.solver.v.copy()
    final_p = simulation.solver.p.copy()

    # Check if values changed
    u_changed = not np.allclose(initial_u, final_u, atol=1e-10)
    v_changed = not np.allclose(initial_v, final_v, atol=1e-10)
    p_changed = not np.allclose(initial_p, final_p, atol=1e-10)

    print(f"  Velocity u changed: {u_changed}")
    print(f"  Velocity v changed: {v_changed}")
    print(f"  Pressure changed: {p_changed}")

    if u_changed or v_changed or p_changed:
        print("  ✓ Dynamic behavior detected!")

        # Show some statistics
        print(f"  Initial u range: [{np.min(initial_u):.6f}, {np.max(initial_u):.6f}]")
        print(f"  Final u range: [{np.min(final_u):.6f}, {np.max(final_u):.6f}]")
        print(f"  Initial v range: [{np.min(initial_v):.6f}, {np.max(initial_v):.6f}]")
        print(f"  Final v range: [{np.min(final_v):.6f}, {np.max(final_v):.6f}]")

        return True
    else:
        print("  ✗ No dynamic behavior detected - values are static")
        return False

def test_oscillating_behavior():
    """Test oscillating boundary condition."""
    print("\nTesting oscillating boundary condition...")

    # Create oscillating simulation
    simulation = SkfemCylinderFlow(
        mesh_density="coarse",
        dt=0.01,
        initial_condition="oscillating"
    )

    # Run several time steps and check for oscillation
    print("Running 20 time steps...")

    u_values = []
    for i in range(20):
        u_new, v_new, p_new = simulation.solver.solve_time_step()
        # Get average velocity at inlet (first few nodes)
        inlet_avg_u = np.mean([u_new[j] for j in range(min(20, len(u_new)))])
        u_values.append(inlet_avg_u)
        # Check oscillation factor
        oscillation_factor = np.sin(2 * np.pi * simulation.solver.simulation_time / 1.0)
        print(f"    Step {i+1}: inlet_avg_u = {inlet_avg_u:.6f}, time = {simulation.solver.simulation_time:.3f}, osc_factor = {oscillation_factor:.6f}")

        # Check if oscillating flag is set
        if hasattr(simulation.solver, 'oscillating'):
            print(f"      Oscillating flag: {simulation.solver.oscillating}")

    # Check for oscillation
    u_std = np.std(u_values)
    u_range = np.max(u_values) - np.min(u_values)

    print(f"  Velocity std: {u_std:.6f}")
    print(f"  Velocity range: {u_range:.6f}")

    if u_std > 0.001:  # Some variation
        print("  ✓ Oscillating behavior detected!")
        return True
    else:
        print("  ✗ No oscillating behavior detected")
        return False

def main():
    """Run dynamic behavior tests."""
    print("=" * 60)
    print("TESTING DYNAMIC BEHAVIOR")
    print("=" * 60)

    # Test 1: Basic dynamic behavior
    test1_passed = test_dynamic_behavior()

    # Test 2: Oscillating behavior
    test2_passed = test_oscillating_behavior()

    print(f"\n" + "=" * 60)
    print("DYNAMIC BEHAVIOR TEST RESULTS")
    print("=" * 60)
    print(f"Basic dynamic behavior: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Oscillating behavior: {'✓ PASS' if test2_passed else '✗ FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 All dynamic behavior tests passed!")
        print("The scikit-fem solver now produces time-varying results.")
        print("Animations should show dynamic behavior instead of static strips.")
        return True
    else:
        print("\n❌ Some tests failed - solver may still be static")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
