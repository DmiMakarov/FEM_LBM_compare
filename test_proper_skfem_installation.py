#!/usr/bin/env python3
"""
Test proper scikit-fem installation and basic functionality.

This script tests whether the proper scikit-fem Navier-Stokes solver
can be imported and used for cylinder flow simulation.
"""

import sys
from pathlib import Path

def test_scikit_fem_import():
    """Test scikit-fem imports."""
    print("Testing scikit-fem imports...")

    try:
        import skfem
        print("  ✓ skfem imported successfully")
    except ImportError as e:
        print(f"  ❌ skfem import failed: {e}")
        return False

    try:
        import skfem
        from skfem import Basis, ElementTriP1, ElementTriP2
        print("  ✓ skfem components imported successfully")
    except ImportError as e:
        print(f"  ❌ skfem components import failed: {e}")
        return False

    return True

def test_our_implementation():
    """Test our proper scikit-fem implementation."""
    print("Testing our proper scikit-fem implementation...")

    try:
        # Add FEM_lib to path
        sys.path.append(str(Path(__file__).parent / "FEM_lib"))

        from FEM_lib import ProperSkfemCylinderFlow
        print("  ✓ ProperSkfemCylinderFlow imported successfully")

        # Test initialization (coarse mesh for speed)
        simulation = ProperSkfemCylinderFlow(
            mesh_density="coarse",
            dt=0.001,
            initial_condition="steady"
        )
        print("  ✓ ProperSkfemCylinderFlow initialized successfully")

        # Test one time step
        ux, uy, p = simulation.solver.solve_time_step()
        print(f"  ✓ Time step completed: ux shape={ux.shape}, uy shape={uy.shape}, p shape={p.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Our implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_boundary_conditions():
    """Test that boundary conditions are correctly set."""
    print("Testing boundary conditions...")

    try:
        from FEM_lib import ProperSkfemCylinderFlow

        # Test steady flow
        sim = ProperSkfemCylinderFlow(
            mesh_density="coarse",
            dt=0.001,
            initial_condition="steady"
        )

        expected_re = 0.3 * 0.1 / 1e-3  # U_m * D / nu
        assert abs(sim.reynolds_number - expected_re) < 1e-6, f"Reynolds number mismatch"
        print(f"  ✓ Steady flow Re = {sim.reynolds_number:.2f}")

        # Test unsteady flow
        sim = ProperSkfemCylinderFlow(
            mesh_density="coarse",
            dt=0.001,
            initial_condition="unsteady"
        )

        expected_re = 1.5 * 0.1 / 1e-3  # U_m * D / nu
        assert abs(sim.reynolds_number - expected_re) < 1e-6, f"Reynolds number mismatch"
        print(f"  ✓ Unsteady flow Re = {sim.reynolds_number:.2f}")

        # Test oscillating flow
        sim = ProperSkfemCylinderFlow(
            mesh_density="coarse",
            dt=0.001,
            initial_condition="oscillating"
        )

        expected_re = 1.5 * 0.1 / 1e-3  # U_m * D / nu
        assert abs(sim.reynolds_number - expected_re) < 1e-6, f"Reynolds number mismatch"
        print(f"  ✓ Oscillating flow Re = {sim.reynolds_number:.2f}")

        return True

    except Exception as e:
        print(f"  ❌ Boundary conditions test failed: {e}")
        return False

def main():
    """Run all installation tests."""
    print("=" * 60)
    print("Proper Scikit-fem Installation Test")
    print("=" * 60)

    # Test imports
    if not test_scikit_fem_import():
        print("\n❌ scikit-fem imports failed. Please install scikit-fem.")
        print("Run: pip install scikit-fem")
        sys.exit(1)

    # Test our implementation
    if not test_our_implementation():
        print("\n❌ Our proper scikit-fem implementation failed.")
        sys.exit(1)

    # Test boundary conditions
    if not test_boundary_conditions():
        print("\n❌ Boundary conditions test failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Proper scikit-fem solver is ready!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python run_proper_skfem_simulation.py --condition steady --max-steps 100")
    print("  python test_fenics_solver.py")

if __name__ == "__main__":
    main()
