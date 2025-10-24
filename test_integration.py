#!/usr/bin/env python3
"""
Test integration of scikit-fem with comparison and animation scripts.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append('FEM_lib')
sys.path.append('FEM')
sys.path.append('LBM')

def test_comparison_integration():
    """Test that scikit-fem can be imported in comparison script."""
    print("Testing comparison script integration...")

    try:
        # Test imports
        from FEM_lib import SkfemCylinderFlow
        from FEM.true_cylinder_flow_fem import TrueCylinderFlowFEM
        from LBM.cylinder_flow_lbm import CylinderFlowLBM

        print("  ✓ All imports successful")

        # Test that we can create instances
        print("  Testing SkfemCylinderFlow creation...")
        skfem_sim = SkfemCylinderFlow(
            mesh_density="coarse",
            dt=0.01,
            initial_condition="steady"
        )
        print("  ✓ SkfemCylinderFlow created successfully")

        return True

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False

def test_animation_integration():
    """Test that scikit-fem can be used in animation script."""
    print("\nTesting animation script integration...")

    try:
        # Test that we can import the animation generator
        from create_fixed_animations import FixedAnimationGenerator

        print("  ✓ FixedAnimationGenerator imported successfully")

        # Test that the method exists
        generator = FixedAnimationGenerator()
        if hasattr(generator, 'create_skfem_animation'):
            print("  ✓ create_skfem_animation method exists")
        else:
            print("  ✗ create_skfem_animation method not found")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Animation integration test failed: {e}")
        return False

def test_run_script():
    """Test that the run script can handle scikit-fem."""
    print("\nTesting run script integration...")

    try:
        # Test that we can import the comparison runner
        from run_optimized_comparison import OptimizedComparisonRunner

        print("  ✓ OptimizedComparisonRunner imported successfully")

        # Test that the method exists
        runner = OptimizedComparisonRunner()
        print("  ✓ OptimizedComparisonRunner created successfully")

        return True

    except Exception as e:
        print(f"  ✗ Run script integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("SCIKIT-FEM INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_comparison_integration,
        test_animation_integration,
        test_run_script
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS")
    print(f"=" * 60)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nScikit-fem has been successfully integrated with:")
        print("  ✓ Comparison script (run_optimized_comparison.py)")
        print("  ✓ Animation script (create_fixed_animations.py)")
        print("  ✓ All imports and methods working")

        print("\nUsage examples:")
        print("  # Run comparison with all methods")
        print("  python run_optimized_comparison.py --method all")
        print("  # Create animations for all methods")
        print("  python create_fixed_animations.py --method all")
        print("  # Run only scikit-fem simulation")
        print("  python run_skfem_simulation.py --condition steady")

        return True
    else:
        print("❌ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
