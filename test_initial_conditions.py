"""
Test script for the three different initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths for imports
sys.path.append('FEM')
sys.path.append('LBM')

from FEM.cylinder_flow_fem import CylinderFlowFEM
from LBM.cylinder_flow_lbm import CylinderFlowLBM


def test_initial_conditions():
    """Test the three different initial conditions."""

    print("Testing Three Different Initial Conditions")
    print("=" * 50)

    # Test parameters
    reynolds_numbers = [20, 100]
    initial_conditions = [
        ("steady", 0.3, "Condition 1: U_x(0,y) = 4U_m y(H-y)/H^2, U_m=0.3 m/s (Re=20)"),
        ("unsteady", 1.5, "Condition 2: U_x(0,y,t) = 4U_m y(H-y)/H^2, U_m=1.5 m/s (Re=100)"),
        ("oscillating", 1.5, "Condition 3: U_x(0,y,t) = 4U_m y(H-y)sin(πt/8)/H^2, U_m=1.5 m/s")
    ]

    # Create results directory
    os.makedirs("results/initial_conditions", exist_ok=True)

    for i, (condition, um, description) in enumerate(initial_conditions):
        print(f"\n{i+1}. Testing {description}")
        print("-" * 40)

        # Choose appropriate Reynolds number
        if condition == "steady":
            re = 20
        else:
            re = 100

        print(f"Reynolds number: {re}")
        print(f"Max velocity: {um} m/s")
        print(f"Initial condition: {condition}")

        # Test FEM
        try:
            print("\nFEM Simulation:")
            fem_sim = CylinderFlowFEM(
                'meshes/cylinder_mesh_Re20_data.npz',
                reynolds_number=re,
                dt=0.001,
                max_velocity=um,
                initial_condition=condition,
                um=um
            )

            # Run short simulation
            fem_results = fem_sim.run_simulation(max_steps=50, save_interval=10)
            print(f"FEM completed successfully")

        except Exception as e:
            print(f"FEM failed: {e}")

        # Test LBM
        try:
            print("\nLBM Simulation:")
            lbm_sim = CylinderFlowLBM(
                nx=100, ny=25,
                reynolds_number=re,
                cylinder_diameter=0.1,
                cylinder_x=0.2, cylinder_y=0.2,
                initial_condition=condition,
                um=um
            )

            # Run short simulation
            lbm_results = lbm_sim.run_simulation(max_steps=50, save_interval=10)
            print(f"LBM completed successfully")

        except Exception as e:
            print(f"LBM failed: {e}")

        # Plot velocity profiles
        plot_velocity_profiles(condition, um, re)

    print("\n" + "=" * 50)
    print("Initial condition testing completed!")


def plot_velocity_profiles(condition: str, um: float, re: int):
    """Plot velocity profiles for different initial conditions."""

    # Create y-coordinates
    H = 0.41  # Domain height
    y = np.linspace(0, H, 100)

    # Calculate velocity profiles
    if condition == "steady":
        # Condition 1: U_x(0, y) = 4U_m y (H − y)/H^2
        ux = 4 * um * y * (H - y) / (H**2)
        title = f"Steady Profile (Re={re}, U_m={um} m/s)"

    elif condition == "unsteady":
        # Condition 2: U_x(0, y, t) = 4U_m y (H − y)/H^2
        ux = 4 * um * y * (H - y) / (H**2)
        title = f"Unsteady Profile (Re={re}, U_m={um} m/s)"

    elif condition == "oscillating":
        # Condition 3: U_x(0, y, t) = 4 U_m y (H − y) sin(πt/8)/H^2
        # Plot at different times
        times = [0, 2, 4, 6, 8]
        plt.figure(figsize=(10, 6))

        for t in times:
            ux = 4 * um * y * (H - y) * np.sin(np.pi * t / 8.0) / (H**2)
            plt.plot(ux, y, label=f't = {t}s', linewidth=2)

        plt.xlabel('Velocity U_x (m/s)')
        plt.ylabel('y (m)')
        plt.title(f'Oscillating Profile (Re={re}, U_m={um} m/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = f"results/initial_conditions/oscillating_profile_Re{re}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Oscillating profile plot saved to: {plot_path}")
        return

    # Plot for steady and unsteady conditions
    plt.figure(figsize=(8, 6))
    plt.plot(ux, y, 'b-', linewidth=2, label='U_x profile')
    plt.xlabel('Velocity U_x (m/s)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = f"results/initial_conditions/{condition}_profile_Re{re}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Velocity profile plot saved to: {plot_path}")


def create_comparison_plot():
    """Create a comparison plot of all three initial conditions."""

    H = 0.41  # Domain height
    y = np.linspace(0, H, 100)

    plt.figure(figsize=(12, 8))

    # Condition 1: Steady (Re=20, U_m=0.3)
    ux1 = 4 * 0.3 * y * (H - y) / (H**2)
    plt.subplot(2, 2, 1)
    plt.plot(ux1, y, 'b-', linewidth=2)
    plt.xlabel('Velocity U_x (m/s)')
    plt.ylabel('y (m)')
    plt.title('Condition 1: Steady (Re=20, U_m=0.3 m/s)')
    plt.grid(True, alpha=0.3)

    # Condition 2: Unsteady (Re=100, U_m=1.5)
    ux2 = 4 * 1.5 * y * (H - y) / (H**2)
    plt.subplot(2, 2, 2)
    plt.plot(ux2, y, 'r-', linewidth=2)
    plt.xlabel('Velocity U_x (m/s)')
    plt.ylabel('y (m)')
    plt.title('Condition 2: Unsteady (Re=100, U_m=1.5 m/s)')
    plt.grid(True, alpha=0.3)

    # Condition 3: Oscillating (Re=100, U_m=1.5) at different times
    plt.subplot(2, 2, 3)
    times = [0, 2, 4, 6, 8]
    for t in times:
        ux3 = 4 * 1.5 * y * (H - y) * np.sin(np.pi * t / 8.0) / (H**2)
        plt.plot(ux3, y, label=f't = {t}s', linewidth=2)
    plt.xlabel('Velocity U_x (m/s)')
    plt.ylabel('y (m)')
    plt.title('Condition 3: Oscillating (Re=100, U_m=1.5 m/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Comparison of all conditions
    plt.subplot(2, 2, 4)
    plt.plot(ux1, y, 'b-', linewidth=2, label='Condition 1 (Steady)')
    plt.plot(ux2, y, 'r-', linewidth=2, label='Condition 2 (Unsteady)')
    # Plot oscillating at t=4s
    ux3_t4 = 4 * 1.5 * y * (H - y) * np.sin(np.pi * 4 / 8.0) / (H**2)
    plt.plot(ux3_t4, y, 'g--', linewidth=2, label='Condition 3 (Oscillating, t=4s)')
    plt.xlabel('Velocity U_x (m/s)')
    plt.ylabel('y (m)')
    plt.title('Comparison of All Conditions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save comparison plot
    plot_path = "results/initial_conditions/initial_conditions_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    # Create comparison plot first
    create_comparison_plot()

    # Test initial conditions
    test_initial_conditions()
