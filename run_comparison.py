#!/usr/bin/env python3
"""
Comprehensive FEM vs LBM comparison script.
Runs both methods on three initial conditions with coarse mesh,
compares results, and creates animations.
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
from datetime import datetime

# Add paths for imports
sys.path.append('FEM')
sys.path.append('LBM')

from FEM.cylinder_flow_fem import CylinderFlowFEM
from LBM.cylinder_flow_lbm import CylinderFlowLBM


class ComparisonRunner:
    """Main class for running FEM vs LBM comparisons."""

    def __init__(self, mesh_type="coarse"):
        """Initialize comparison runner."""
        self.mesh_type = mesh_type
        self.mesh_configs = {
            "very_coarse": (20, 5),
            "coarse": (40, 10),
            "fine": (100, 25)
        }

        if mesh_type not in self.mesh_configs:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

        self.nx, self.ny = self.mesh_configs[mesh_type]
        self.results = {}

        # Create output directories
        os.makedirs("results/comparison", exist_ok=True)
        os.makedirs("results/animations", exist_ok=True)

    def run_simulation(self, method, reynolds_number, initial_condition, max_steps=100):
        """Run a single simulation."""
        print(f"\n{'='*50}")
        print(f"Running {method.upper()} - Re={reynolds_number}, {initial_condition}")
        print(f"Mesh: {self.mesh_type} ({self.nx}x{self.ny})")
        print(f"{'='*50}")

        start_time = time.time()

        if method.lower() == "fem":
            sim = CylinderFlowFEM(
                mesh_data_file=f"meshes/{self.mesh_type}_mesh_data.npz",
                dt=0.001,
                initial_condition=initial_condition
            )

        elif method.lower() == "lbm":
            sim = CylinderFlowLBM(
                nx=self.nx,
                ny=self.ny,
                initial_condition=initial_condition
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run simulation
        results = sim.run_simulation(max_steps=max_steps, save_interval=10)

        elapsed_time = time.time() - start_time

        print(f"\n{method.upper()} completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {elapsed_time/max_steps:.4f} seconds")
        print(f"Final drag: {results['drag'][-1]:.4f}")
        print(f"Final lift: {results['lift'][-1]:.4f}")
        print(f"Final Strouhal: {results['strouhal'][-1]:.4f}")

        return results, elapsed_time

    def run_all_comparisons(self, max_steps=100):
        """Run all FEM vs LBM comparisons."""
        print(f"\n{'#'*80}")
        print(f"COMPREHENSIVE FEM vs LBM COMPARISON")
        print(f"Mesh: {self.mesh_type} ({self.nx}x{self.ny})")
        print(f"Steps: {max_steps}")
        print(f"{'#'*80}")

        # Test configurations
        test_configs = [
            {"reynolds": 20, "condition": "steady", "description": "Steady flow (Re=20)"},
            {"reynolds": 100, "condition": "unsteady", "description": "Unsteady flow (Re=100)"},
            {"reynolds": 100, "condition": "oscillating", "description": "Oscillating flow (Re=100)"}
        ]

        all_results = {}

        for config in test_configs:
            re = config["reynolds"]
            condition = config["condition"]
            desc = config["description"]

            print(f"\n{'#'*60}")
            print(f"TESTING: {desc}")
            print(f"{'#'*60}")

            # Run FEM
            fem_results, fem_time = self.run_simulation("fem", re, condition, max_steps)

            # Run LBM
            lbm_results, lbm_time = self.run_simulation("lbm", re, condition, max_steps)

            # Store results
            key = f"Re{re}_{condition}"
            all_results[key] = {
                "description": desc,
                "reynolds": re,
                "condition": condition,
                "fem_results": fem_results,
                "lbm_results": lbm_results,
                "fem_time": fem_time,
                "lbm_time": lbm_time,
                "speedup": fem_time / lbm_time if lbm_time > 0 else float('inf')
            }

            # Print comparison
            self.print_comparison(all_results[key])

        self.results = all_results
        self.save_comparison_results()
        self.create_animations()
        self.create_summary_plots()

        return all_results

    def print_comparison(self, result):
        """Print comparison results."""
        print(f"\n{'='*50}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*50}")

        print(f"\n⏱️  PERFORMANCE:")
        print(f"  FEM time: {result['fem_time']:.2f} seconds")
        print(f"  LBM time: {result['lbm_time']:.2f} seconds")
        print(f"  LBM speedup: {result['speedup']:.1f}x faster")

        print(f"\n📊 RESULTS:")
        fem_drag = result['fem_results']['drag'][-1]
        fem_lift = result['fem_results']['lift'][-1]
        fem_st = result['fem_results']['strouhal'][-1]

        lbm_drag = result['lbm_results']['drag'][-1]
        lbm_lift = result['lbm_results']['lift'][-1]
        lbm_st = result['lbm_results']['strouhal'][-1]

        print(f"  FEM - Drag: {fem_drag:.4f}, Lift: {fem_lift:.4f}, St: {fem_st:.4f}")
        print(f"  LBM - Drag: {lbm_drag:.4f}, Lift: {lbm_lift:.4f}, St: {lbm_st:.4f}")

        print(f"\n🔍 DIFFERENCES:")
        print(f"  Drag difference: {abs(fem_drag - lbm_drag):.4f}")
        print(f"  Lift difference: {abs(fem_lift - lbm_lift):.4f}")
        print(f"  Strouhal difference: {abs(fem_st - lbm_st):.4f}")

    def save_comparison_results(self):
        """Save comparison results to JSON."""
        # Prepare data for JSON serialization
        json_data = {}
        for key, result in self.results.items():
            json_data[key] = {
                "description": result["description"],
                "reynolds": result["reynolds"],
                "condition": result["condition"],
                "fem_time": result["fem_time"],
                "lbm_time": result["lbm_time"],
                "speedup": result["speedup"],
                "fem_final_drag": result["fem_results"]["drag"][-1],
                "fem_final_lift": result["fem_results"]["lift"][-1],
                "fem_final_strouhal": result["fem_results"]["strouhal"][-1],
                "lbm_final_drag": result["lbm_results"]["drag"][-1],
                "lbm_final_lift": result["lbm_results"]["lift"][-1],
                "lbm_final_strouhal": result["lbm_results"]["strouhal"][-1]
            }

        # Save to file
        filename = f"results/comparison/comparison_results_{self.mesh_type}.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"\n💾 Results saved to {filename}")

    def create_animations(self):
        """Create animations for all simulations."""
        print(f"\n🎬 Creating animations...")

        # First, let's debug what data we have
        self.debug_field_data()

        # Create both FEM and LBM animations
        print(f"\n🎬 Creating FEM and LBM animations...")

        for key, result in self.results.items():
            re = result["reynolds"]
            condition = result["condition"]

            # Create FEM animations
            try:
                self.create_fem_animation(key, result)
            except Exception as e:
                print(f"Failed to create FEM animation for {key}: {e}")

            # Create LBM animations
            try:
                self.create_lbm_animation(key, result)
            except Exception as e:
                print(f"Failed to create LBM animation for {key}: {e}")

    def create_lbm_animation(self, key, result):
        """Create improved LBM animation with velocity, cylinder, and transposed data."""
        re = result["reynolds"]
        condition = result["condition"]

        # Get LBM field data
        lbm_results = result["lbm_results"]

        # Get mesh dimensions
        nx, ny = self.nx, self.ny

        # Cylinder parameters (physical coordinates) - consistent with FEM
        cylinder_x_phys = 0.2  # Physical position
        cylinder_y_phys = 0.2
        cylinder_diameter = 0.1
        cylinder_radius_phys = cylinder_diameter / 2  # 0.05

        # Domain dimensions - consistent with FEM
        domain_length = 2.2
        domain_height = 0.41

        # Convert to grid coordinates
        dx = domain_length / nx  # Grid spacing in x
        dy = domain_height / ny  # Grid spacing in y
        cylinder_ix = int(cylinder_x_phys / dx)
        cylinder_iy = int(cylinder_y_phys / dy)
        cylinder_radius_grid = max(1, int((cylinder_radius_phys / dx)))  # Minimum radius of 1

        # Ensure cylinder is within bounds
        cylinder_ix = max(1, min(nx-2, cylinder_ix))
        cylinder_iy = max(1, min(ny-2, cylinder_iy))

        # Debug info (can be uncommented for debugging)
        # print(f"  Debug: Grid {nx}x{ny}, dx={dx:.3f}, dy={dy:.3f}")
        # print(f"  Debug: Cylinder at grid ({cylinder_ix}, {cylinder_iy}), radius={cylinder_radius_grid}")

        for field_type in ["pressure", "velocity", "vorticity"]:
            try:
                if field_type == "pressure":
                    lbm_data = lbm_results["pressure_field"]
                elif field_type == "velocity":
                    # Get velocity components
                    vel_x_data = lbm_results["velocity_field"]
                    vel_y_data = lbm_results["velocity_field"]  # Assuming same structure
                    # Calculate velocity magnitude
                    lbm_data = []
                    for i in range(len(vel_x_data)):
                        if isinstance(vel_x_data[i], tuple) and len(vel_x_data[i]) == 2:
                            ux, uy = vel_x_data[i]
                            vel_mag = np.sqrt(ux**2 + uy**2)
                            lbm_data.append(vel_mag)
                        else:
                            # Fallback if velocity data is not in expected format
                            lbm_data.append(np.zeros((ny, nx)))
                elif field_type == "vorticity":
                    lbm_data = lbm_results["vorticity_field"]

                if not lbm_data or len(lbm_data) == 0:
                    continue

                # Create improved LBM animation
                fig, ax = plt.subplots(figsize=(10, 8))

                # Create colorbar once outside the animation function
                cbar = None

                def animate(frame):
                    ax.clear()
                    if frame < len(lbm_data):
                        # Transpose the data for better visualization
                        data = lbm_data[frame].T  # Transpose: (nx, ny) -> (ny, nx)

                        # Create contour plot
                        im = ax.contourf(data, levels=20, cmap='viridis', extend='both')

                        # Add cylinder (ensure it's visible)
                        if cylinder_radius_grid > 0:
                            cylinder = plt.Circle((cylinder_ix, cylinder_iy),
                                                cylinder_radius_grid,
                                                color='black',
                                                zorder=10,
                                                alpha=0.9)
                            ax.add_patch(cylinder)

                            # Add cylinder outline
                            cylinder_outline = plt.Circle((cylinder_ix, cylinder_iy),
                                                        cylinder_radius_grid,
                                                        fill=False,
                                                        color='white',
                                                        linewidth=3,
                                                        zorder=11)
                            ax.add_patch(cylinder_outline)

                        # Add velocity vectors if velocity field
                        if field_type == "velocity" and frame < len(vel_x_data):
                            try:
                                if isinstance(vel_x_data[frame], tuple) and len(vel_x_data[frame]) == 2:
                                    ux, uy = vel_x_data[frame]
                                    # Create grid for quiver plot
                                    x_grid = np.arange(0, nx, 3)  # Every 3rd point for clarity
                                    y_grid = np.arange(0, ny, 3)
                                    X, Y = np.meshgrid(x_grid, y_grid)

                                    # Sample velocity at grid points
                                    U = ux[::3, ::3]  # Downsample
                                    V = uy[::3, ::3]

                                    # Add quiver plot
                                    ax.quiver(X, Y, U, V, alpha=0.7, color='white', scale=30, width=0.003)
                            except:
                                pass  # Skip velocity vectors if data format is unexpected

                        # Formatting
                        ax.set_title(f'LBM - {field_type.title()} (Re={re}, {condition})',
                                    fontsize=14, fontweight='bold')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)

                        # Add colorbar only once
                        nonlocal cbar
                        if cbar is None:
                            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                            cbar.set_label(f'{field_type.title()}', fontsize=12)

                        # Set axis limits
                        ax.set_xlim(0, nx)
                        ax.set_ylim(0, ny)

                # Create animation
                max_frames = len(lbm_data)
                if max_frames > 0:
                    anim = FuncAnimation(fig, animate, frames=max_frames, interval=300, repeat=True)

                    # Save animation
                    filename = f"results/animations/lbm_{field_type}_{condition}_Re{re}_improved.gif"
                    anim.save(filename, writer='pillow', fps=3)
                    print(f"  Created: {filename}")

                plt.close()

            except Exception as e:
                print(f"  Failed to create {field_type} animation for {key}: {e}")

    def create_fem_animation(self, key, result):
        """Create FEM animation using FEMVisualizer."""
        re = result["reynolds"]
        condition = result["condition"]

        # Get FEM field data
        fem_results = result["fem_results"]

        # Create FEM animations for each field type
        for field_type in ["pressure", "velocity", "vorticity"]:
            try:
                if field_type == "pressure":
                    fem_data = fem_results["pressure_field"]
                elif field_type == "velocity":
                    fem_data = fem_results["velocity_field"]
                elif field_type == "vorticity":
                    fem_data = fem_results["vorticity_field"]

                if not fem_data or len(fem_data) == 0:
                    continue

                # Create FEM animation with consistent domain and cylinder
                fig, ax = plt.subplots(figsize=(10, 8))

                # Use same domain as LBM for consistency
                domain_length = 2.2
                domain_height = 0.41
                cylinder_x = 0.2
                cylinder_y = 0.2
                cylinder_radius = 0.05

                # Create colorbar once outside animation
                cbar = None

                def animate(frame):
                    ax.clear()
                    if frame < len(fem_data):
                        # FEM data is 1D node arrays, create scatter plot
                        data = fem_data[frame]

                        # Use same grid as LBM for consistency
                        nx, ny = 20, 10  # Match LBM grid
                        x_grid = np.linspace(0, domain_length, nx)
                        y_grid = np.linspace(0, domain_height, ny)
                        X, Y = np.meshgrid(x_grid, y_grid)

                        # Reshape 1D data to 2D for visualization
                        if len(data) >= nx * ny:
                            Z = data[:nx*ny].reshape(nx, ny)
                        else:
                            # Pad with zeros if not enough data
                            Z = np.zeros((nx, ny))
                            Z.flat[:len(data)] = data

                        # Create contour plot
                        im = ax.contourf(X, Y, Z.T, levels=20, cmap='viridis', extend='both')

                        # Add cylinder with same size as LBM
                        cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                            color='black', zorder=10, alpha=0.9)
                        ax.add_patch(cylinder)

                        cylinder_outline = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                                    fill=False, color='white',
                                                    linewidth=3, zorder=11)
                        ax.add_patch(cylinder_outline)

                        # Formatting
                        ax.set_title(f'FEM - {field_type.title()} (Re={re}, {condition})',
                                    fontsize=14, fontweight='bold')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(0, domain_length)
                        ax.set_ylim(0, domain_height)

                        # Add colorbar only once
                        nonlocal cbar
                        if cbar is None:
                            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                            cbar.set_label(f'{field_type.title()}', fontsize=12)

                # Create animation
                max_frames = len(fem_data)
                if max_frames > 0:
                    anim = FuncAnimation(fig, animate, frames=max_frames, interval=300, repeat=True)

                    # Save animation
                    filename = f"results/animations/fem_{field_type}_{condition}_Re{re}_improved.gif"
                    anim.save(filename, writer='pillow', fps=3)
                    print(f"  Created: {filename}")

                plt.close()

            except Exception as e:
                print(f"  Failed to create FEM {field_type} animation for {key}: {e}")

    def debug_field_data(self):
        """Debug what field data is available."""
        print(f"\n🔍 Debugging field data...")

        for key, result in self.results.items():
            print(f"\n{key}:")

            # Check FEM data
            fem_results = result["fem_results"]
            print(f"  FEM fields available: {list(fem_results.keys())}")

            for field in ["pressure_field", "velocity_field", "vorticity_field"]:
                if field in fem_results:
                    data = fem_results[field]
                    print(f"    {field}: {len(data) if data else 0} frames")
                    if data and len(data) > 0:
                        print(f"      Shape: {data[0].shape if hasattr(data[0], 'shape') else 'No shape'}")
                else:
                    print(f"    {field}: Not available")

            # Check LBM data
            lbm_results = result["lbm_results"]
            print(f"  LBM fields available: {list(lbm_results.keys())}")

            for field in ["pressure_field", "velocity_field", "vorticity_field"]:
                if field in lbm_results:
                    data = lbm_results[field]
                    print(f"    {field}: {len(data) if data else 0} frames")
                    if data and len(data) > 0:
                        print(f"      Shape: {data[0].shape if hasattr(data[0], 'shape') else 'No shape'}")
                else:
                    print(f"    {field}: Not available")

    def create_animation(self, key, field_type, result):
        """Create animation for a specific field."""
        re = result["reynolds"]
        condition = result["condition"]

        # Get field data with proper error checking
        try:
            if field_type == "pressure":
                fem_data = result["fem_results"]["pressure_field"]
                lbm_data = result["lbm_results"]["pressure_field"]
            elif field_type == "velocity":
                fem_vel = result["fem_results"]["velocity_field"]
                lbm_vel = result["lbm_results"]["velocity_field"]
                fem_data = [np.sqrt(v[0]**2 + v[1]**2) for v in fem_vel] if fem_vel else []
                lbm_data = [np.sqrt(v[0]**2 + v[1]**2) for v in lbm_vel] if lbm_vel else []
            elif field_type == "vorticity":
                fem_data = result["fem_results"]["vorticity_field"]
                lbm_data = result["lbm_results"]["vorticity_field"]

            # Check if we have valid data
            if not fem_data or not lbm_data:
                print(f"  Skipping {field_type} animation for {key}: No field data")
                return

            if len(fem_data) == 0 or len(lbm_data) == 0:
                print(f"  Skipping {field_type} animation for {key}: Empty field data")
                return

            # Check data shapes - handle both 1D (FEM) and 2D (LBM) data
            fem_shape = fem_data[0].shape if hasattr(fem_data[0], 'shape') else (1,)
            lbm_shape = lbm_data[0].shape if hasattr(lbm_data[0], 'shape') else (1,)

            # For 1D FEM data, we need to reshape it to 2D for visualization
            if len(fem_shape) == 1:
                # FEM data is 1D (node-based), need to reshape to 2D grid
                # This is a simplified approach - in practice you'd need the mesh connectivity
                print(f"  Skipping {field_type} animation for {key}: FEM data is 1D, needs mesh connectivity for 2D visualization")
                return

        except (KeyError, IndexError, TypeError) as e:
            print(f"  Skipping {field_type} animation for {key}: Data access error - {e}")
            return

        # Create side-by-side animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        def animate(frame):
            ax1.clear()
            ax2.clear()

            # FEM plot
            if frame < len(fem_data) and fem_data[frame] is not None:
                try:
                    im1 = ax1.contourf(fem_data[frame], levels=20, cmap='viridis')
                    ax1.set_title(f'FEM - {field_type.title()} (Re={re}, {condition})')
                    ax1.set_aspect('equal')
                except Exception as e:
                    ax1.text(0.5, 0.5, f'FEM Data Error: {e}', transform=ax1.transAxes, ha='center')

            # LBM plot
            if frame < len(lbm_data) and lbm_data[frame] is not None:
                try:
                    im2 = ax2.contourf(lbm_data[frame], levels=20, cmap='viridis')
                    ax2.set_title(f'LBM - {field_type.title()} (Re={re}, {condition})')
                    ax2.set_aspect('equal')
                except Exception as e:
                    ax2.text(0.5, 0.5, f'LBM Data Error: {e}', transform=ax2.transAxes, ha='center')

        # Create animation with proper frame count
        max_frames = min(len(fem_data), len(lbm_data))
        if max_frames == 0:
            print(f"  Skipping {field_type} animation for {key}: No valid frames")
            return

        try:
            anim = FuncAnimation(fig, animate, frames=max_frames, interval=200, repeat=True)

            # Save animation
            filename = f"results/animations/{field_type}_{condition}_Re{re}_comparison.gif"
            anim.save(filename, writer='pillow', fps=5)
            print(f"  Created: {filename}")

        except Exception as e:
            print(f"  Failed to create {field_type} animation for {key}: {e}")
        finally:
            plt.close()

    def create_summary_plots(self):
        """Create summary comparison plots."""
        print(f"\n📊 Creating summary plots...")

        # Performance comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data
        tests = list(self.results.keys())
        fem_times = [self.results[t]["fem_time"] for t in tests]
        lbm_times = [self.results[t]["lbm_time"] for t in tests]
        speedups = [self.results[t]["speedup"] for t in tests]

        fem_drags = [self.results[t]["fem_results"]["drag"][-1] for t in tests]
        lbm_drags = [self.results[t]["lbm_results"]["drag"][-1] for t in tests]

        # Performance plot
        x = np.arange(len(tests))
        width = 0.35

        ax1.bar(x - width/2, fem_times, width, label='FEM', alpha=0.8)
        ax1.bar(x + width/2, lbm_times, width, label='LBM', alpha=0.8)
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.replace('_', '\n') for t in tests], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Speedup plot
        ax2.bar(x, speedups, alpha=0.8, color='green')
        ax2.set_xlabel('Test Case')
        ax2.set_ylabel('LBM Speedup (x)')
        ax2.set_title('LBM Speedup over FEM')
        ax2.set_xticks(x)
        ax2.set_xticklabels([t.replace('_', '\n') for t in tests], rotation=45)
        ax2.grid(True, alpha=0.3)

        # Drag comparison
        ax3.bar(x - width/2, fem_drags, width, label='FEM', alpha=0.8)
        ax3.bar(x + width/2, lbm_drags, width, label='LBM', alpha=0.8)
        ax3.set_xlabel('Test Case')
        ax3.set_ylabel('Final Drag Coefficient')
        ax3.set_title('Drag Coefficient Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([t.replace('_', '\n') for t in tests], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Difference plot
        drag_diffs = [abs(f - l) for f, l in zip(fem_drags, lbm_drags)]
        ax4.bar(x, drag_diffs, alpha=0.8, color='red')
        ax4.set_xlabel('Test Case')
        ax4.set_ylabel('Drag Difference')
        ax4.set_title('FEM vs LBM Drag Difference')
        ax4.set_xticks(x)
        ax4.set_xticklabels([t.replace('_', '\n') for t in tests], rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"results/comparison/summary_comparison_{self.mesh_type}.png",
                   dpi=300, bbox_inches='tight')
        print(f"  Created: results/comparison/summary_comparison_{self.mesh_type}.png")
        plt.close()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='FEM vs LBM Comparison')
    parser.add_argument('--mesh', choices=['very_coarse', 'coarse', 'fine'],
                       default='coarse', help='Mesh size to use')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of time steps')

    args = parser.parse_args()

    # Create comparison runner
    runner = ComparisonRunner(mesh_type=args.mesh)

    # Run all comparisons
    results = runner.run_all_comparisons(max_steps=args.steps)

    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: results/comparison/")
    print(f"Animations saved to: results/animations/")
    print(f"Summary plots saved to: results/comparison/")


if __name__ == "__main__":
    main()
