#!/usr/bin/env python3
"""
Basic test script for scikit-fem implementation.
"""

import sys
import os
from pathlib import Path

# Add FEM_lib to path
sys.path.append(str(Path(__file__).parent / "FEM_lib"))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")

    try:
        from FEM_lib.skfem_mesh_generator import SkfemMeshGenerator
        print("  ✓ SkfemMeshGenerator import successful")
    except Exception as e:
        print(f"  ✗ SkfemMeshGenerator import failed: {e}")
        return False

    try:
        from FEM_lib.skfem_cylinder_flow import SkfemCylinderFlow
        print("  ✓ SkfemCylinderFlow import successful")
    except Exception as e:
        print(f"  ✗ SkfemCylinderFlow import failed: {e}")
        return False

    return True

def test_mesh_generation():
    """Test mesh generation."""
    print("\nTesting mesh generation...")

    try:
        from FEM_lib.skfem_mesh_generator import SkfemMeshGenerator

        # Create mesh generator
        generator = SkfemMeshGenerator(mesh_density="coarse")
        print("  ✓ Mesh generator created")

        # Generate mesh
        mesh = generator.generate_mesh()
        print(f"  ✓ Mesh generated: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} elements")

        # Get mesh info
        info = generator.get_mesh_info(mesh)
        print(f"  ✓ Mesh info: {info}")

        # Check boundary nodes
        if hasattr(generator, 'boundary_nodes'):
            print(f"  ✓ Boundary nodes available: {len(generator.boundary_nodes)} types")
        else:
            print("  ⚠ No boundary nodes found")

        return True

    except Exception as e:
        print(f"  ✗ Mesh generation failed: {e}")
        return False

def test_cylinder_flow():
    """Test cylinder flow simulation."""
    print("\nTesting cylinder flow simulation...")

    try:
        from FEM_lib.skfem_cylinder_flow import SkfemCylinderFlow

        # Create simulation
        simulation = SkfemCylinderFlow(
            mesh_density="coarse",
            dt=0.001,
            initial_condition="steady"
        )
        print("  ✓ Cylinder flow simulation created")

        # Run short simulation
        results = simulation.run_simulation(max_steps=10, save_interval=5)
        print(f"  ✓ Simulation completed: {len(results.get('time', []))} time steps")

        return True

    except Exception as e:
        print(f"  ✗ Cylinder flow simulation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Scikit-fem Implementation")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\nImport tests failed. Exiting.")
        return False

    # Test mesh generation
    if not test_mesh_generation():
        print("\nMesh generation tests failed. Exiting.")
        return False

    # Test cylinder flow
    if not test_cylinder_flow():
        print("\nCylinder flow tests failed. Exiting.")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
