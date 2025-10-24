"""
Optimized comparison using different meshes for FEM and LBM.
FEM: Fine mesh for accuracy, LBM: Coarse mesh for speed.
"""

import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Tuple, List

# Import solvers
import sys
sys.path.append('FEM')
sys.path.append('LBM')
sys.path.append('FEM_lib')

from FEM.true_cylinder_flow_fem import TrueCylinderFlowFEM
from cylinder_flow_lbm import CylinderFlowLBM
from FEM_lib import ProperSkfemCylinderFlow
from FEM_lib.fast_skfem_solver import FastSkfemCylinderFlow

class OptimizedComparisonRunner:
    """
    Compare FEM and LBM using optimized meshes for each method.
    FEM: Fine mesh for accuracy, LBM: Coarse mesh for speed.
    """

    def __init__(self):
        """
        Initialize optimized comparison runner.
        """
        # FEM: Use fine mesh for accuracy
        self.fem_mesh = "fine"  # 80x40 grid, 3191 nodes, 16 cylinder nodes

        # LBM: Use coarse mesh for speed
        self.lbm_nx = 120
        self.lbm_ny = 60

        print(f"Optimized Comparison Runner:")
        print(f"  FEM mesh: {self.fem_mesh} (fine for accuracy)")
        print(f"  LBM mesh: {self.lbm_nx}x{self.lbm_ny} (coarse for speed)")
        print(f"  Strategy: FEM accuracy vs LBM speed")

    def run_simulation(self, method: str, reynolds_number: float,
                      initial_condition: str, max_steps: int) -> Tuple[Dict, float, object]:
        """
        Run simulation with optimized mesh for each method.
        """
        print(f"\n{'='*50}")
        print(f"Running {method.upper()} - Re={reynolds_number}, {initial_condition}")
        if method.lower() == "fem":
            print(f"Mesh: {self.fem_mesh} (optimized for accuracy)")
        elif method.lower() == "skfem":
            print(f"Mesh: scikit-fem generated (medium density)")
        else:
            print(f"Mesh: {self.lbm_nx}x{self.lbm_ny} (optimized for speed)")
        print(f"{'='*50}")

        start_time = time.time()

        if method.lower() == "fem":
            # Use fine mesh for FEM accuracy
            sim = TrueCylinderFlowFEM(
                mesh_data_file="meshes/true_fem_mesh_30x15_data.npz",
                dt=0.01,  # Increased from 0.001 to 0.01 for more meaningful simulation time
                initial_condition=initial_condition
            )
        elif method.lower() == "skfem":
            # Use FAST scikit-fem solver for speed
            sim = FastSkfemCylinderFlow(
                mesh_density="coarse",  # Use coarse for speed
                dt=0.01,
                initial_condition=initial_condition
            )
        elif method.lower() == "lbm":
            # Use coarse mesh for LBM speed
            sim = CylinderFlowLBM(
                nx=self.lbm_nx,
                ny=self.lbm_ny,
                initial_condition=initial_condition
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run simulation with more frames for animations
        if method.lower() == "fem":
            # FEM needs more steps and smaller save interval for animations
            results = sim.run_simulation(max_steps=max_steps, save_interval=1)
        elif method.lower() == "skfem":
            # Proper scikit-fem can use medium save interval
            results = sim.run_simulation(max_steps=max_steps, save_interval=5)
        else:
            # LBM can use larger save interval
            results = sim.run_simulation(max_steps=max_steps, save_interval=10)

        elapsed_time = time.time() - start_time

        print(f"\n{method.upper()} completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {results.get('timing', {}).get('time_per_step', 0.0):.4f} seconds")
        print(f"Final drag: {results['drag'][-1]:.4f}")
        print(f"Final lift: {results['lift'][-1]:.4f}")
        print(f"Final Strouhal: {results['strouhal'][-1]:.4f}")

        return results, elapsed_time, sim

    def run_comparison(self, max_steps: int = 2000, method: str = 'all'):
        """
        Run optimized comparison between FEM, Scikit-fem, and LBM.
        """
        print(f"\n{'#'*80}")
        print(f"OPTIMIZED FEM vs SKFEM vs LBM COMPARISON")
        print(f"FEM: True proper FEM with optimized mesh (30x15)")
        print(f"SKFEM: Proper scikit-fem with real Navier-Stokes solver")
        print(f"LBM: Coarse mesh for speed")
        print(f"Steps: {max_steps}")
        print(f"{'#'*80}")

        # Test cases
        test_cases = [
            ("steady", 30.0),      # Re = 30
            ("unsteady", 150.0),   # Re = 150
            ("oscillating", 150.0)  # Re = 150
        ]

        comparison_results = {}

        for condition, re in test_cases:
            print(f"\n{'#'*60}")
            print(f"TESTING: {condition.title()} flow (Re={re})")
            print(f"{'#'*60}")

            if method == 'fem' or method == 'all':
                # Run FEM simulation (fine mesh)
                fem_results, fem_time, fem_sim = self.run_simulation("fem", re, condition, max_steps)

                # Save FEM results
                output_filename = f"results/fem/fem_solution_Re{int(re)}_{condition}.npz"
                fem_sim.save_results(fem_results, output_filename)

            if method == 'skfem' or method == 'all':
                # Run Proper Scikit-fem simulation
                skfem_results, skfem_time, skfem_sim = self.run_simulation("skfem", re, condition, max_steps)

                # Save Proper Scikit-fem results
                output_filename = f"results/proper_skfem/proper_skfem_solution_Re{int(re)}_{condition}.npz"
                skfem_sim.save_results(skfem_results, output_filename)

            if method == 'lbm' or method == 'all':
                # Run LBM simulation (coarse mesh)
                lbm_results, lbm_time, lbm_sim = self.run_simulation("lbm", re, condition, max_steps)

            # Compare results
            print(f"\n{'='*50}")
            print(f"OPTIMIZED COMPARISON RESULTS")
            print(f"{'='*50}")

            print(f"\n⏱️  PERFORMANCE:")
            if method == 'fem' or method == 'all':
                print(f"  FEM time: {fem_time:.2f} seconds (fine mesh)")
            if method == 'skfem' or method == 'all':
                print(f"  SKFEM time: {skfem_time:.2f} seconds (proper Navier-Stokes solver)")
            if method == 'lbm' or method == 'all':
                print(f"  LBM time: {lbm_time:.2f} seconds (coarse mesh)")

            if method == 'all':
                print(f"  Speed comparison:")
                if 'fem_time' in locals() and 'lbm_time' in locals():
                    print(f"    LBM vs FEM: {fem_time/lbm_time:.1f}x faster")
                if 'skfem_time' in locals() and 'lbm_time' in locals():
                    print(f"    LBM vs SKFEM: {skfem_time/lbm_time:.1f}x faster")
                if 'fem_time' in locals() and 'skfem_time' in locals():
                    print(f"    SKFEM vs FEM: {fem_time/skfem_time:.1f}x faster")

            print(f"\n📊 RESULTS:")
            if method == 'fem' or method == 'all':
                print(f"  FEM - Drag: {fem_results['drag'][-1]:.4f}, Lift: {fem_results['lift'][-1]:.4f}, St: {fem_results['strouhal'][-1]:.4f}")
            if method == 'skfem' or method == 'all':
                print(f"  SKFEM - Drag: {skfem_results['drag'][-1]:.4f}, Lift: {skfem_results['lift'][-1]:.4f}, St: {skfem_results['strouhal'][-1]:.4f} (proper NS solver)")
            if method == 'lbm' or method == 'all':
                print(f"  LBM - Drag: {lbm_results['drag'][-1]:.4f}, Lift: {lbm_results['lift'][-1]:.4f}, St: {lbm_results['strouhal'][-1]:.4f}")

            print(f"\n🔍 DIFFERENCES:")
            if method == 'all':
                if 'fem_results' in locals() and 'lbm_results' in locals():
                    drag_diff = abs(fem_results['drag'][-1] - lbm_results['drag'][-1])
                    lift_diff = abs(fem_results['lift'][-1] - lbm_results['lift'][-1])
                    st_diff = abs(fem_results['strouhal'][-1] - lbm_results['strouhal'][-1])
                    print(f"  FEM vs LBM - Drag: {drag_diff:.4f}, Lift: {lift_diff:.4f}, St: {st_diff:.4f}")

                if 'skfem_results' in locals() and 'lbm_results' in locals():
                    drag_diff = abs(skfem_results['drag'][-1] - lbm_results['drag'][-1])
                    lift_diff = abs(skfem_results['lift'][-1] - lbm_results['lift'][-1])
                    st_diff = abs(skfem_results['strouhal'][-1] - lbm_results['strouhal'][-1])
                    print(f"  SKFEM vs LBM - Drag: {drag_diff:.4f}, Lift: {lift_diff:.4f}, St: {st_diff:.4f} (proper NS vs LBM)")

                if 'fem_results' in locals() and 'skfem_results' in locals():
                    drag_diff = abs(fem_results['drag'][-1] - skfem_results['drag'][-1])
                    lift_diff = abs(fem_results['lift'][-1] - skfem_results['lift'][-1])
                    st_diff = abs(fem_results['strouhal'][-1] - skfem_results['strouhal'][-1])
                    print(f"  FEM vs SKFEM - Drag: {drag_diff:.4f}, Lift: {lift_diff:.4f}, St: {st_diff:.4f} (both proper NS solvers)")

            # Store results
            key = f"Re{re}_{condition}"
            comparison_results[key] = {
                "reynolds": re,
                "condition": condition,
                "fem_results": fem_results if method == 'fem' or method == 'all' else None,
                "skfem_results": skfem_results if method == 'skfem' or method == 'all' else None,
                "lbm_results": lbm_results if method == 'lbm' or method == 'all' else None,
                "fem_time": fem_time if method == 'fem' or method == 'all' else None,
                "skfem_time": skfem_time if method == 'skfem' or method == 'all' else None,
                "lbm_time": lbm_time if method == 'lbm' or method == 'all' else None,
                "speedup": fem_time / lbm_time if method == 'all' else None,
                "drag_difference": drag_diff if method == 'all' else None,
                "lift_difference": lift_diff if method == 'all' else None,
                "strouhal_difference": st_diff if method == 'all' else None,
                "fem_mesh": self.fem_mesh,
                "skfem_mesh": "medium (proper NS solver)",
                "lbm_mesh": f"{self.lbm_nx}x{self.lbm_ny}"
            }

        # Save comparison results
        self.save_results(comparison_results, method)

        # Create summary plots
        if method == 'both':
            self.create_summary_plots(comparison_results)

        # Create animations
        #self.create_animations(comparison_results)

        print(f"\n{'='*80}")
        print(f"OPTIMIZED COMPARISON COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: results/comparison/")
        print(f"Summary plots saved to: results/comparison/")

        return comparison_results

    def save_results(self, results: Dict, method: str = 'both'):
        """Save optimized comparison results."""
        os.makedirs('results/comparison', exist_ok=True)

        # Create simplified results
        simplified_results = {}
        for key, value in results.items():
            simplified_results[key] = {
                "reynolds": value["reynolds"],
                "condition": value["condition"],
                "fem_time": value["fem_time"],
                "skfem_time": value.get("skfem_time"),
                "lbm_time": value["lbm_time"],
                "speedup": value["speedup"],
                "fem_drag": value["fem_results"]["drag"][-1] if method == 'fem' or method == 'all' else None,
                "fem_lift": value["fem_results"]["lift"][-1] if method == 'fem' or method == 'all' else None,
                "fem_strouhal": value["fem_results"]["strouhal"][-1] if method == 'fem' or method == 'all' else None,
                "skfem_drag": value["skfem_results"]["drag"][-1] if method == 'skfem' or method == 'all' else None,
                "skfem_lift": value["skfem_results"]["lift"][-1] if method == 'skfem' or method == 'all' else None,
                "skfem_strouhal": value["skfem_results"]["strouhal"][-1] if method == 'skfem' or method == 'all' else None,
                "lbm_drag": value["lbm_results"]["drag"][-1] if method == 'lbm' or method == 'all' else None,
                "lbm_lift": value["lbm_results"]["lift"][-1] if method == 'lbm' or method == 'all' else None,
                "lbm_strouhal": value["lbm_results"]["strouhal"][-1] if method == 'lbm' or method == 'all' else None,
                "drag_difference": value["drag_difference"] if method == 'all' else None,
                "lift_difference": value["lift_difference"] if method == 'all' else None,
                "strouhal_difference": value["strouhal_difference"] if method == 'all' else None,
                "fem_mesh": value["fem_mesh"],
                "skfem_mesh": value.get("skfem_mesh", "medium (proper NS solver)"),
                "lbm_mesh": value["lbm_mesh"]
            }

        filename = "results/comparison/optimized_comparison_results.json"
        with open(filename, 'w') as f:
            json.dump(simplified_results, f, indent=2)

        print(f"💾 Results saved to {filename}")

    def create_summary_plots(self, results: Dict):
        """Create summary comparison plots."""
        os.makedirs('results/comparison', exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data
        cases = list(results.keys())
        fem_drag = [results[case]['fem_results']['drag'][-1] for case in cases]
        lbm_drag = [results[case]['lbm_results']['drag'][-1] for case in cases]
        fem_lift = [results[case]['fem_results']['lift'][-1] for case in cases]
        lbm_lift = [results[case]['lbm_results']['lift'][-1] for case in cases]

        # Extract skfem data if available
        skfem_drag = [results[case].get('skfem_results', {}).get('drag', [0])[-1] for case in cases if 'skfem_results' in results[case]]
        skfem_lift = [results[case].get('skfem_results', {}).get('lift', [0])[-1] for case in cases if 'skfem_results' in results[case]]

        # Plot 1: Drag comparison
        x = np.arange(len(cases))
        width = 0.25

        axes[0,0].bar(x - width, fem_drag, width, label='FEM (Fine)', alpha=0.8, color='blue')
        axes[0,0].bar(x, lbm_drag, width, label='LBM (Coarse)', alpha=0.8, color='red')
        if skfem_drag:
            axes[0,0].bar(x + width, skfem_drag, width, label='SKFEM (Proper NS)', alpha=0.8, color='green')
        axes[0,0].set_xlabel('Test Case')
        axes[0,0].set_ylabel('Drag Force')
        axes[0,0].set_title('Drag Force Comparison (Optimized Meshes)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(cases, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2: Lift comparison
        axes[0,1].bar(x - width, fem_lift, width, label='FEM (Fine)', alpha=0.8, color='blue')
        axes[0,1].bar(x, lbm_lift, width, label='LBM (Coarse)', alpha=0.8, color='red')
        if skfem_lift:
            axes[0,1].bar(x + width, skfem_lift, width, label='SKFEM (Proper NS)', alpha=0.8, color='green')
        axes[0,1].set_xlabel('Test Case')
        axes[0,1].set_ylabel('Lift Force')
        axes[0,1].set_title('Lift Force Comparison (Optimized Meshes)')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(cases, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Performance comparison
        fem_times = [results[case]['fem_time'] for case in cases]
        lbm_times = [results[case]['lbm_time'] for case in cases]
        skfem_times = [results[case].get('skfem_time', 0) for case in cases if 'skfem_time' in results[case]]
        speedups = [results[case]['speedup'] for case in cases]

        axes[1,0].bar(x - width, fem_times, width, label='FEM (Fine)', alpha=0.8, color='blue')
        axes[1,0].bar(x, lbm_times, width, label='LBM (Coarse)', alpha=0.8, color='red')
        if skfem_times:
            axes[1,0].bar(x + width, skfem_times, width, label='SKFEM (Proper NS)', alpha=0.8, color='green')
        axes[1,0].set_xlabel('Test Case')
        axes[1,0].set_ylabel('Time (seconds)')
        axes[1,0].set_title('Performance Comparison (Optimized Meshes)')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(cases, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Speedup
        axes[1,1].bar(x, speedups, alpha=0.8, color='green')
        axes[1,1].set_xlabel('Test Case')
        axes[1,1].set_ylabel('LBM Speedup (x)')
        axes[1,1].set_title('LBM Speedup over FEM (Optimized Meshes)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(cases, rotation=45)
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = "results/comparison/optimized_summary_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Creating summary plots...")
        print(f"  Created: {filename}")

        plt.close()

    def create_animations(self, results: Dict):
        """Create animations for both FEM and LBM results."""
        print(f"\n🎬 Creating optimized animations...")

        # Create animations directory
        os.makedirs('results/animations', exist_ok=True)

        for key, result in results.items():
            re = result["reynolds"]
            condition = result["condition"]

            print(f"\n🎬 Creating animations for {key}...")

            # Create FEM animations
            self.create_fem_animation(key, result)

            # Create LBM animations
            self.create_lbm_animation(key, result)

    def create_fem_animation(self, key: str, result: Dict):
        """Create FEM animation using fine mesh data."""
        re = result["reynolds"]
        condition = result["condition"]
        fem_results = result["fem_results"]

        # Load FEM field data
        try:
            fem_file = f"results/fem/fem_Re{re:.1f}_results_fields.npz"
            fem_data = np.load(fem_file)

            for field_type in ["pressure", "velocity", "vorticity"]:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Domain parameters (consistent with mesh)
                    domain_length = 2.2
                    domain_height = 0.41
                    cylinder_x = 0.2
                    cylinder_y = 0.2
                    cylinder_radius = 0.05

                    # Create visualization grid (simplified for FEM)
                    nx_vis, ny_vis = 40, 20  # Visualization grid
                    x_grid = np.linspace(0, domain_length, nx_vis)
                    y_grid = np.linspace(0, domain_height, ny_vis)
                    X, Y = np.meshgrid(x_grid, y_grid)

                    cbar = None

                    def animate(frame):
                        nonlocal cbar
                        ax.clear()

                        # Get the correct field data
                        field_key = field_type + '_field'
                        if field_key in fem_data:
                            field_data = fem_data[field_key]
                            if frame < len(field_data):
                                data = field_data[frame]

                                # Reshape data for visualization (column-major order)
                                if len(data) >= nx_vis * ny_vis:
                                    Z = data[:nx_vis*ny_vis].reshape(nx_vis, ny_vis).T
                                else:
                                    Z_temp = np.zeros(nx_vis * ny_vis)
                                    Z_temp[:len(data)] = data
                                    Z = Z_temp.reshape(nx_vis, ny_vis).T
                            else:
                                return
                        else:
                            return

                        # Create contour plot
                        im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', extend='both')

                        # Add cylinder
                        cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                            color='black', zorder=10, alpha=0.9)
                        ax.add_patch(cylinder)
                        cylinder_outline = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                                    fill=False, color='white',
                                                    linewidth=3, zorder=11)
                        ax.add_patch(cylinder_outline)

                        ax.set_title(f'FEM - {field_type.title()} (Re={re:.0f}, {condition})\nFine Mesh (3191 nodes)',
                                    fontsize=14, fontweight='bold')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(0, domain_length)
                        ax.set_ylim(0, domain_height)

                        # Create colorbar only once
                        if cbar is None:
                            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                            cbar.set_label(f'{field_type.title()}', fontsize=12)

                    field_key = field_type + '_field'
                    max_frames = len(fem_data[field_key]) if field_key in fem_data else 0
                    if max_frames > 0:
                        anim = FuncAnimation(fig, animate, frames=max_frames, interval=300, repeat=True)
                        filename = f"results/animations/fem_{field_type}_{condition}_Re{re:.0f}_optimized.gif"
                        anim.save(filename, writer='pillow', fps=3)
                        print(f"  Created: {filename}")

                    plt.close()

                except Exception as e:
                    print(f"  Failed to create FEM {field_type} animation: {e}")

        except Exception as e:
            print(f"  Failed to load FEM data for {key}: {e}")

    def create_lbm_animation(self, key: str, result: Dict):
        """Create LBM animation using coarse mesh data."""
        re = result["reynolds"]
        condition = result["condition"]
        lbm_results = result["lbm_results"]

        # Load LBM field data
        try:
            # Try different possible filenames due to floating point precision
            possible_files = [
                f"results/lbm/lbm_Re{re:.1f}_{condition}_results_fields.npz",
                f"results/lbm/lbm_Re{re}_{condition}_results_fields.npz",
                f"results/lbm/lbm_Re{re:.6f}_{condition}_results_fields.npz"
            ]

            lbm_data = None
            for lbm_file in possible_files:
                if os.path.exists(lbm_file):
                    lbm_data = np.load(lbm_file)
                    break

            if lbm_data is None:
                print(f"  No LBM field data found for Re={re}")
                return

            for field_type in ["pressure", "velocity", "vorticity"]:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Domain parameters (consistent with LBM)
                    domain_length = 2.2
                    domain_height = 0.41
                    cylinder_x = 0.2
                    cylinder_y = 0.2
                    cylinder_radius = 0.05

                    # LBM grid parameters
                    nx_lbm = self.lbm_nx
                    ny_lbm = self.lbm_ny
                    dx = domain_length / nx_lbm
                    dy = domain_height / ny_lbm

                    # Create LBM grid
                    x_lbm = np.linspace(dx/2, domain_length - dx/2, nx_lbm)
                    y_lbm = np.linspace(dy/2, domain_height - dy/2, ny_lbm)
                    X_lbm, Y_lbm = np.meshgrid(x_lbm, y_lbm)

                    # Cylinder in grid coordinates
                    cylinder_ix = int(cylinder_x / dx)
                    cylinder_iy = int(cylinder_y / dy)
                    cylinder_radius_grid = max(1, int(cylinder_radius / dx))

                    # Ensure cylinder is within bounds
                    cylinder_ix = max(1, min(nx_lbm-2, cylinder_ix))
                    cylinder_iy = max(1, min(ny_lbm-2, cylinder_iy))

                    # Get the correct field data based on field type
                    if field_type == "pressure":
                        field_data = lbm_data['pressure_fields']
                    elif field_type == "velocity":
                        field_data = lbm_data['velocity_x_fields']  # We'll compute magnitude
                    elif field_type == "vorticity":
                        field_data = lbm_data['vorticity_fields']
                    else:
                        continue

                    cbar = None

                    def animate(frame):
                        nonlocal cbar
                        ax.clear()

                        if frame < len(field_data):
                            if field_type == "velocity":
                                # For velocity, compute magnitude from x and y components
                                ux = lbm_data['velocity_x_fields'][frame]
                                uy = lbm_data['velocity_y_fields'][frame]
                                Z = np.sqrt(ux**2 + uy**2)
                            else:
                                Z = field_data[frame]

                            # Transpose for proper orientation
                            Z = Z.T

                            # Create contour plot
                            im = ax.contourf(X_lbm, Y_lbm, Z, levels=20, cmap='viridis', extend='both')

                            # Add velocity vectors for velocity field
                            if field_type == "velocity":
                                ux = lbm_data['velocity_x_fields'][frame]
                                uy = lbm_data['velocity_y_fields'][frame]
                                # Subsample for cleaner visualization
                                step = max(1, min(nx_lbm//8, ny_lbm//8))
                                ax.quiver(X_lbm[::step, ::step], Y_lbm[::step, ::step],
                                        ux[::step, ::step].T, uy[::step, ::step].T,
                                        alpha=0.7, scale=20, width=0.003)

                            # Add cylinder
                            cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                                color='black', zorder=10, alpha=0.9)
                            ax.add_patch(cylinder)
                            cylinder_outline = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                                        fill=False, color='white',
                                                        linewidth=3, zorder=11)
                            ax.add_patch(cylinder_outline)

                            ax.set_title(f'LBM - {field_type.title()} (Re={re:.0f}, {condition})\nCoarse Mesh ({nx_lbm}x{ny_lbm})',
                                        fontsize=14, fontweight='bold')
                            ax.set_xlabel('x')
                            ax.set_ylabel('y')
                            ax.set_aspect('equal')
                            ax.grid(True, alpha=0.3)
                            ax.set_xlim(0, domain_length)
                            ax.set_ylim(0, domain_height)

                            # Create colorbar only once
                            if cbar is None:
                                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                                cbar.set_label(f'{field_type.title()}', fontsize=12)

                    max_frames = len(field_data)
                    if max_frames > 0:
                        anim = FuncAnimation(fig, animate, frames=max_frames, interval=300, repeat=True)
                        filename = f"results/animations/lbm_{field_type}_{condition}_Re{re:.0f}_optimized.gif"
                        anim.save(filename, writer='pillow', fps=3)
                        print(f"  Created: {filename}")

                    plt.close()

                except Exception as e:
                    print(f"  Failed to create LBM {field_type} animation: {e}")

        except Exception as e:
            print(f"  Failed to load LBM data for {key}: {e}")

def main():
    """Main function to run optimized comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Run optimized FEM vs LBM comparison')
    parser.add_argument('--steps', type=int, default=100,
                       help='Maximum number of time steps')

    parser.add_argument('--method', type=str, default='both',
                       choices=['fem', 'lbm', 'skfem', "all"],
                       help='Method to use for comparison')

    args = parser.parse_args()

    # Run optimized comparison
    runner = OptimizedComparisonRunner()
    results = runner.run_comparison(max_steps=args.steps, method=args.method)

    print(f"\n🎉 Optimized comparison completed successfully!")
    print(f"Check results/comparison/ for detailed results and plots.")

if __name__ == "__main__":
    main()
