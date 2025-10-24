#!/usr/bin/env python3
"""
Test FEniCS installation and basic functionality.

This script tests whether FEniCS is properly installed and can be used
for the cylinder flow simulation.
"""

import sys
from pathlib import Path

def test_fenics_import():
    """Test FEniCS imports."""
    print("Testing FEniCS imports...")

    try:
        import dolfinx
        print("  ✓ dolfinx imported successfully")
    except ImportError as e:
        print(f"  ❌ dolfinx import failed: {e}")
        return False

    try:
        import ufl
        print("  ✓ ufl imported successfully")
    except ImportError as e:
        print(f"  ❌ ufl import failed: {e}")
        return False

    try:
        from mpi4py import MPI
        print("  ✓ mpi4py imported successfully")
    except ImportError as e:
        print(f"  ❌ mpi4py import failed: {e}")
        return False

    try:
        import petsc4py
        print("  ✓ petsc4py imported successfully")
    except ImportError as e:
        print(f"  ❌ petsc4py import failed: {e}")
        return False

    return True

def test_fenics_basic():
    """Test basic FEniCS functionality."""
    print("Testing basic FEniCS functionality...")

    try:
        import dolfinx
        from dolfinx import mesh
        from dolfinx.mesh import create_rectangle, CellType
        from mpi4py import MPI

        # Create a simple mesh
        mesh_obj = create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
            [10, 10],
            CellType.triangle
        )

        print(f"  ✓ Mesh created: {mesh_obj.topology.index_map(0).size_global} vertices")

        # Test function space creation
        from dolfinx import fem
        V = fem.FunctionSpace(mesh_obj, ("Lagrange", 1))
        print(f"  ✓ Function space created: {V.dofmap.index_map.size_global} DOFs")

        return True

    except Exception as e:
        print(f"  ❌ FEniCS basic test failed: {e}")
        return False

def test_our_implementation():
    """Test our FEniCS implementation."""
    print("Testing our FEniCS implementation...")

    try:
        # Add FEM_lib to path
        sys.path.append(str(Path(__file__).parent / "FEM_lib"))

        from FEM_lib import FenicsCylinderFlow
        print("  ✓ FenicsCylinderFlow imported successfully")

        # Test initialization (coarse mesh for speed)
        simulation = FenicsCylinderFlow(
            mesh_density="coarse",
            dt=0.001,
            initial_condition="steady"
        )
        print("  ✓ FenicsCylinderFlow initialized successfully")

        # Test one time step
        ux, uy, p = simulation.solver.solve_time_step()
        print(f"  ✓ Time step completed: ux shape={ux.shape}, uy shape={uy.shape}, p shape={p.shape}")

        return True

    except Exception as e:
        print(f"  ❌ Our implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all installation tests."""
    print("=" * 60)
    print("FEniCS Installation Test")
    print("=" * 60)

    # Test imports
    if not test_fenics_import():
        print("\n❌ FEniCS imports failed. Please install FEniCS properly.")
        print("See: https://fenicsproject.org/download/")
        sys.exit(1)

    # Test basic functionality
    if not test_fenics_basic():
        print("\n❌ FEniCS basic functionality failed.")
        sys.exit(1)

    # Test our implementation
    if not test_our_implementation():
        print("\n❌ Our FEniCS implementation failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - FEniCS is ready to use!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python run_fenics_simulation.py --condition steady --max-steps 100")
    print("  python test_fenics_solver.py")

if __name__ == "__main__":
    import numpy as np
    main()
