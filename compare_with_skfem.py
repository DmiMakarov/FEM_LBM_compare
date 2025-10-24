#!/usr/bin/env python3
"""
Compare scikit-fem results with existing custom FEM and LBM implementations.

This script runs simulations using all three methods and generates comparison
plots and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "FEM_lib"))
sys.path.append(str(Path(__file__).parent / "FEM"))
sys.path.append(str(Path(__file__).parent / "LBM"))

# Import all methods
from FEM_lib import SkfemCylinderFlow
from FEM.true_cylinder_flow_fem import TrueCylinderFlowFEM
from LBM.cylinder_flow_lbm import CylinderFlowLBM


class MethodComparison:
    """
    Compare different numerical methods for cylinder flow simulation.
    """

    def __init__(self, mesh_density: str = "medium", dt: float = 0.001):
        """
        Initialize comparison.

        Args:
            mesh_density: Mesh density for FEM methods
            dt: Time step size
        """
        self.mesh_density = mesh_density
        self.dt = dt

        # Results storage
        self.results = {
            'skfem': {},
            'custom_fem': {},
            'lbm': {}
        }

        # Timing information
        self.timing = {
            'skfem': {},
            'custom_fem': {},
            'lbm': {}
        }

        # Create output directory
        self.output_dir = "results/comparison"
        os.makedirs(self.output_dir, exist_ok=True)

    def run_all_methods(self, condition: str = "steady", max_steps: int = 500):
        """
        Run all three methods for the given condition.

        Args:
            condition: Initial condition ("steady", "unsteady", "oscillating")
            max_steps: Maximum number of time steps
        """
        print(f"\n{'='*60}")
        print(f"COMPARING METHODS FOR {condition.upper()} FLOW")
        print(f"{'='*60}")

        # Run scikit-fem
        print(f"\n--- Running Scikit-fem ---")
        start_time = time.time()
        try:
            skfem_sim = SkfemCylinderFlow(
                mesh_density=self.mesh_density,
                dt=self.dt,
                initial_condition=condition
            )
            skfem_results = skfem_sim.run_simulation(max_steps=max_steps, save_interval=50)
            skfem_time = time.time() - start_time

            self.results['skfem'][condition] = skfem_results
            self.timing['skfem'][condition] = {
                'total_time': skfem_time,
                'time_per_step': skfem_time / max_steps
            }

            print(f"  Scikit-fem completed in {skfem_time:.2f} seconds")

        except Exception as e:
            print(f"  Scikit-fem failed: {e}")
            self.results['skfem'][condition] = None
            self.timing['skfem'][condition] = None

        # Run custom FEM
        print(f"\n--- Running Custom FEM ---")
        start_time = time.time()
        try:
            custom_fem_sim = TrueCylinderFlowFEM(
                mesh_data_file="meshes/small_mesh_20x10_data.npz",
                dt=self.dt,
                initial_condition=condition
            )
            custom_fem_results = custom_fem_sim.run_simulation(max_steps=max_steps, save_interval=50)
            custom_fem_time = time.time() - start_time

            self.results['custom_fem'][condition] = custom_fem_results
            self.timing['custom_fem'][condition] = {
                'total_time': custom_fem_time,
                'time_per_step': custom_fem_time / max_steps
            }

            print(f"  Custom FEM completed in {custom_fem_time:.2f} seconds")

        except Exception as e:
            print(f"  Custom FEM failed: {e}")
            self.results['custom_fem'][condition] = None
            self.timing['custom_fem'][condition] = None

        # Run LBM
        print(f"\n--- Running LBM ---")
        start_time = time.time()
        try:
            lbm_sim = CylinderFlowLBM(
                nx=100, ny=25,  # Match resolution
                initial_condition=condition
            )
            lbm_results = lbm_sim.run_simulation(max_steps=max_steps, save_interval=50)
            lbm_time = time.time() - start_time

            self.results['lbm'][condition] = lbm_results
            self.timing['lbm'][condition] = {
                'total_time': lbm_time,
                'time_per_step': lbm_time / max_steps
            }

            print(f"  LBM completed in {lbm_time:.2f} seconds")

        except Exception as e:
            print(f"  LBM failed: {e}")
            self.results['lbm'][condition] = None
            self.timing['lbm'][condition] = None

    def generate_comparison_plots(self, condition: str = "steady"):
        """Generate comparison plots for the given condition."""
        print(f"\nGenerating comparison plots for {condition} flow...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Method Comparison: {condition.title()} Flow', fontsize=16)

        # Plot drag coefficient
        ax = axes[0, 0]
        for method, results in self.results.items():
            if results.get(condition) and results[condition].get('drag'):
                times = results[condition]['time']
                drag = results[condition]['drag']
                ax.plot(times, drag, label=f'{method.upper()}', linewidth=2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Drag Coefficient')
        ax.set_title('Drag Coefficient vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot lift coefficient
        ax = axes[0, 1]
        for method, results in self.results.items():
            if results.get(condition) and results[condition].get('lift'):
                times = results[condition]['time']
                lift = results[condition]['lift']
                ax.plot(times, lift, label=f'{method.upper()}', linewidth=2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Lift Coefficient')
        ax.set_title('Lift Coefficient vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot pressure drop
        ax = axes[0, 2]
        for method, results in self.results.items():
            if results.get(condition) and results[condition].get('pressure_drop'):
                times = results[condition]['time']
                pressure_drop = results[condition]['pressure_drop']
                ax.plot(times, pressure_drop, label=f'{method.upper()}', linewidth=2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Pressure Drop')
        ax.set_title('Pressure Drop vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot Strouhal number
        ax = axes[1, 0]
        for method, results in self.results.items():
            if results.get(condition) and results[condition].get('strouhal'):
                times = results[condition]['time']
                strouhal = results[condition]['strouhal']
                ax.plot(times, strouhal, label=f'{method.upper()}', linewidth=2)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Strouhal Number')
        ax.set_title('Strouhal Number vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot timing comparison
        ax = axes[1, 1]
        methods = []
        times = []
        for method, timing in self.timing.items():
            if timing.get(condition):
                methods.append(method.upper())
                times.append(timing[condition]['total_time'])

        if methods:
            bars = ax.bar(methods, times, color=['blue', 'green', 'red'])
            ax.set_ylabel('Total Time (seconds)')
            ax.set_title('Computational Time Comparison')
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{time_val:.1f}s', ha='center', va='bottom')

        # Plot time per step comparison
        ax = axes[1, 2]
        methods = []
        time_per_step = []
        for method, timing in self.timing.items():
            if timing.get(condition):
                methods.append(method.upper())
                time_per_step.append(timing[condition]['time_per_step'])

        if methods:
            bars = ax.bar(methods, time_per_step, color=['blue', 'green', 'red'])
            ax.set_ylabel('Time per Step (seconds)')
            ax.set_title('Time per Step Comparison')
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, time_val in zip(bars, time_per_step):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{time_val:.4f}s', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        plot_filename = f"{self.output_dir}/comparison_{condition}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"  Comparison plot saved to: {plot_filename}")

        plt.show()

    def generate_summary_table(self):
        """Generate summary table of results."""
        print(f"\nGenerating summary table...")

        # Create summary data
        summary_data = []

        for condition in ["steady", "unsteady", "oscillating"]:
            for method in ["skfem", "custom_fem", "lbm"]:
                if (self.results.get(method, {}).get(condition) and
                    self.timing.get(method, {}).get(condition)):

                    results = self.results[method][condition]
                    timing = self.timing[method][condition]

                    # Get final values
                    final_drag = results['drag'][-1] if results['drag'] else 0.0
                    final_lift = results['lift'][-1] if results['lift'] else 0.0
                    final_pressure_drop = results['pressure_drop'][-1] if results['pressure_drop'] else 0.0
                    final_strouhal = results['strouhal'][-1] if results['strouhal'] else 0.0

                    summary_data.append({
                        'condition': condition,
                        'method': method.upper(),
                        'total_time': timing['total_time'],
                        'time_per_step': timing['time_per_step'],
                        'final_drag': final_drag,
                        'final_lift': final_lift,
                        'final_pressure_drop': final_pressure_drop,
                        'final_strouhal': final_strouhal
                    })

        # Create DataFrame and save
        import pandas as pd
        df = pd.DataFrame(summary_data)

        if not df.empty:
            # Save to CSV
            csv_filename = f"{self.output_dir}/comparison_summary.csv"
            df.to_csv(csv_filename, index=False)
            print(f"  Summary table saved to: {csv_filename}")

            # Print table
            print(f"\nSummary Table:")
            print(df.to_string(index=False, float_format='%.6f'))

        return df

    def save_comparison_data(self):
        """Save all comparison data to files."""
        print(f"\nSaving comparison data...")

        # Save results
        results_filename = f"{self.output_dir}/comparison_results.npz"
        np.savez_compressed(results_filename, **self.results)
        print(f"  Results saved to: {results_filename}")

        # Save timing
        timing_filename = f"{self.output_dir}/comparison_timing.json"
        with open(timing_filename, 'w') as f:
            json.dump(self.timing, f, indent=2)
        print(f"  Timing data saved to: {timing_filename}")

    def run_full_comparison(self):
        """Run full comparison for all conditions."""
        print("Running full comparison of all methods...")
        print("=" * 60)

        # Run all conditions
        for condition in ["steady", "unsteady", "oscillating"]:
            self.run_all_methods(condition, max_steps=300)  # Shorter for comparison

        # Generate plots for each condition
        for condition in ["steady", "unsteady", "oscillating"]:
            self.generate_comparison_plots(condition)

        # Generate summary
        summary_df = self.generate_summary_table()

        # Save all data
        self.save_comparison_data()

        print(f"\nFull comparison completed!")
        print(f"Results saved to: {self.output_dir}/")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare numerical methods for cylinder flow")
    parser.add_argument("--condition", "-c",
                       choices=["steady", "unsteady", "oscillating", "all"],
                       default="all",
                       help="Initial condition to test")
    parser.add_argument("--max-steps", "-s", type=int, default=300,
                       help="Maximum number of time steps")
    parser.add_argument("--mesh-density", "-m",
                       choices=["coarse", "medium", "fine"],
                       default="medium",
                       help="Mesh density for FEM methods")

    args = parser.parse_args()

    # Create comparison object
    comparison = MethodComparison(mesh_density=args.mesh_density)

    if args.condition == "all":
        # Run full comparison
        comparison.run_full_comparison()
    else:
        # Run single condition
        comparison.run_all_methods(args.condition, max_steps=args.max_steps)
        comparison.generate_comparison_plots(args.condition)
        comparison.generate_summary_table()
        comparison.save_comparison_data()


if __name__ == "__main__":
    main()
