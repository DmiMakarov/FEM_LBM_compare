#!/usr/bin/env python3
"""
Debug script to check where LBM velocity is being applied.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.append('LBM')

def debug_lbm_velocity():
    """Debug LBM velocity application."""
    print("🔍 Debugging LBM velocity application...")

    try:
        from cylinder_flow_lbm import CylinderFlowLBM

        # Create a small LBM simulation
        print("Creating LBM simulation...")
        lbm = CylinderFlowLBM(nx=10, ny=5, initial_condition='steady')

        print(f"Grid size: {lbm.nx} x {lbm.ny}")
        print(f"Domain: {lbm.domain_length} x {lbm.domain_height}")
        print(f"Grid spacing: dx={lbm.dx:.3f}, dy={lbm.dy:.3f}")

        # Check inlet mask
        print(f"\nInlet mask shape: {lbm.inlet_mask.shape}")
        print(f"Inlet mask (left boundary): {lbm.inlet_mask[0, :]}")

        # Check parabolic profile
        profile = lbm._get_parabolic_inlet_profile()
        print(f"\nParabolic profile: {profile}")
        print(f"Profile shape: {profile.shape}")

        # Check cylinder mask
        print(f"\nCylinder mask shape: {lbm.cylinder_mask.shape}")
        print(f"Cylinder at inlet: {lbm.cylinder_mask[0, :]}")

        # Run a few steps and check velocity
        print("\nRunning LBM steps...")
        for step in range(5):
            profile = lbm._get_parabolic_inlet_profile()
            lbm.lbm.step(lbm.cylinder_mask, lbm.inlet_mask, lbm.outlet_mask, profile)

            # Get velocity field
            ux, uy = lbm.lbm.get_velocity()
            print(f"Step {step}: Velocity shape: {ux.shape}")
            print(f"  Max velocity: {np.max(np.sqrt(ux**2 + uy**2)):.4f}")
            print(f"  Max velocity location: {np.unravel_index(np.argmax(np.sqrt(ux**2 + uy**2)), ux.shape)}")

            # Check inlet velocity
            inlet_velocity = ux[0, :]
            print(f"  Inlet velocity: {inlet_velocity}")

            # Check bottom velocity
            bottom_velocity = ux[:, 0]
            print(f"  Bottom velocity: {bottom_velocity}")

            print()

        print("✅ LBM debugging completed successfully!")

    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lbm_velocity()
