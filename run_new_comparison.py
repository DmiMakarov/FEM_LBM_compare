"""
Run comparison between new simple FEM solver and LBM solver.
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

from simple_cylinder_flow_fem import SimpleCylinderFlowFEM
from cylinder_flow_lbm import CylinderFlowLBM

class NewComparisonRunner:
    """
    Compare new simple FEM solver with LBM solver.
    """

    def __init__(self, mesh_type: str = "coarse"):
        """
        Initialize comparison runner.

        Args:
            mesh_type: Type of mesh to use ('coarse', 'medium', 'fine')
        """
        self.mesh_type = mesh_type

        # Mesh parameters
        mesh_sizes = {
            'coarse': {'nx': 40, 'ny': 20},
            'medium': {'nx': 60, 'ny': 30},
            'fine': {'nx': 80, 'ny': 40}
        }

        if mesh_type not in mesh_sizes:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

        self.nx = mesh_sizes[mesh_type]['nx']
        self.ny = mesh_sizes[mesh_type]['ny']

        print(f"New Comparison Runner initialized:")
        print(f"  Mesh type: {mesh_type}")
        print(f"  Grid size: {self.nx}x{self.ny}")

    def run_simulation(self, method: str, reynolds_number: float,
                      initial_condition: str, max_steps: int) -> Tuple[Dict, float]:
        """
        Run simulation with specified method.

        Args:
            method: 'fem' or 'lbm'
            reynolds_number: Reynolds number
            initial_condition: Type of initial condition
            max_steps: Maximum number of time steps

        Returns:
            Tuple of (results, elapsed_time)
        """
        print(f"\n{'='*50}")
        print(f"Running {method.upper()} - Re={reynolds_number}, {initial_condition}")
        print(f"Mesh: {self.mesh_type} ({self.nx}x{self.ny})")
        print(f"{'='*50}")

        start_time = time.time()

        if method.lower() == "fem":
            # Use new simple FEM solver
            sim = SimpleCylinderFlowFEM(
                mesh_data_file=f"meshes/fem_{self.mesh_type}_mesh_data.npz",
                dt=0.001,
                initial_condition=initial_condition
            )
        elif method.lower() == "lbm":
            # Use existing LBM solver
            sim = CylinderFlowLBM(
                nx=self.nx,
                ny=self.ny,
                initial_condition=initial_condition
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Run simulation
        results = sim.run_simulation(max_steps=max_steps, save_interval=5)

        elapsed_time = time.time() - start_time

        print(f"\n{method.upper()} completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {results['timing']['time_per_step']:.4f} seconds")
        print(f"Final drag: {results['drag'][-1]:.4f}")
        print(f"Final lift: {results['lift'][-1]:.4f}")
        print(f"Final Strouhal: {results['strouhal'][-1]:.4f}")

        return results, elapsed_time

    def run_comparison(self, max_steps: int = 50):
        """
        Run complete comparison between FEM and LBM.
        """
        print(f"\n{'#'*80}")
        print(f"NEW FEM vs LBM COMPARISON")
        print(f"Mesh: {self.mesh_type} ({self.nx}x{self.ny})")
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

            # Run FEM simulation
            fem_results, fem_time = self.run_simulation("fem", re, condition, max_steps)

            # Run LBM simulation
            lbm_results, lbm_time = self.run_simulation("lbm", re, condition, max_steps)

            # Compare results
            print(f"\n{'='*50}")
            print(f"COMPARISON RESULTS")
            print(f"{'='*50}")

            print(f"\n⏱️  PERFORMANCE:")
            print(f"  FEM time: {fem_time:.2f} seconds")
            print(f"  LBM time: {lbm_time:.2f} seconds")
            print(f"  LBM speedup: {fem_time/lbm_time:.1f}x faster")

            print(f"\n📊 RESULTS:")
            print(f"  FEM - Drag: {fem_results['drag'][-1]:.4f}, Lift: {fem_results['lift'][-1]:.4f}, St: {fem_results['strouhal'][-1]:.4f}")
            print(f"  LBM - Drag: {lbm_results['drag'][-1]:.4f}, Lift: {lbm_results['lift'][-1]:.4f}, St: {lbm_results['strouhal'][-1]:.4f}")

            print(f"\n🔍 DIFFERENCES:")
            drag_diff = abs(fem_results['drag'][-1] - lbm_results['drag'][-1])
            lift_diff = abs(fem_results['lift'][-1] - lbm_results['lift'][-1])
            st_diff = abs(fem_results['strouhal'][-1] - lbm_results['strouhal'][-1])
            print(f"  Drag difference: {drag_diff:.4f}")
            print(f"  Lift difference: {lift_diff:.4f}")
            print(f"  Strouhal difference: {st_diff:.4f}")

            # Store results
            key = f"Re{re}_{condition}"
            comparison_results[key] = {
                "reynolds": re,
                "condition": condition,
                "fem_results": fem_results,
                "lbm_results": lbm_results,
                "fem_time": fem_time,
                "lbm_time": lbm_time,
                "speedup": fem_time / lbm_time,
                "drag_difference": drag_diff,
                "lift_difference": lift_diff,
                "strouhal_difference": st_diff
            }

        # Save simplified comparison results
        self.save_simplified_results(comparison_results)

        # Create summary plots
        self.create_summary_plots(comparison_results)

        print(f"\n{'='*80}")
        print(f"COMPARISON COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: results/comparison/")
        print(f"Summary plots saved to: results/comparison/")

        return comparison_results

    def save_comparison_results(self, results: Dict):
        """Save comparison results to JSON file."""
        os.makedirs('results/comparison', exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        json_results = convert_to_serializable(results)

        filename = f"results/comparison/new_comparison_results_{self.mesh_type}.json"
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"💾 Results saved to {filename}")

    def save_simplified_results(self, results: Dict):
        """Save simplified comparison results."""
        os.makedirs('results/comparison', exist_ok=True)

        # Create simplified results
        simplified_results = {}
        for key, value in results.items():
            simplified_results[key] = {
                "reynolds": value["reynolds"],
                "condition": value["condition"],
                "fem_time": value["fem_time"],
                "lbm_time": value["lbm_time"],
                "speedup": value["speedup"],
                "fem_drag": value["fem_results"]["drag"][-1],
                "fem_lift": value["fem_results"]["lift"][-1],
                "fem_strouhal": value["fem_results"]["strouhal"][-1],
                "lbm_drag": value["lbm_results"]["drag"][-1],
                "lbm_lift": value["lbm_results"]["lift"][-1],
                "lbm_strouhal": value["lbm_results"]["strouhal"][-1],
                "drag_difference": value["drag_difference"],
                "lift_difference": value["lift_difference"],
                "strouhal_difference": value["strouhal_difference"]
            }

        filename = f"results/comparison/new_comparison_results_{self.mesh_type}.json"
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

        # Plot 1: Drag comparison
        x = np.arange(len(cases))
        width = 0.35

        axes[0,0].bar(x - width/2, fem_drag, width, label='FEM', alpha=0.8)
        axes[0,0].bar(x + width/2, lbm_drag, width, label='LBM', alpha=0.8)
        axes[0,0].set_xlabel('Test Case')
        axes[0,0].set_ylabel('Drag Force')
        axes[0,0].set_title('Drag Force Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(cases, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Plot 2: Lift comparison
        axes[0,1].bar(x - width/2, fem_lift, width, label='FEM', alpha=0.8)
        axes[0,1].bar(x + width/2, lbm_lift, width, label='LBM', alpha=0.8)
        axes[0,1].set_xlabel('Test Case')
        axes[0,1].set_ylabel('Lift Force')
        axes[0,1].set_title('Lift Force Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(cases, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Performance comparison
        fem_times = [results[case]['fem_time'] for case in cases]
        lbm_times = [results[case]['lbm_time'] for case in cases]
        speedups = [results[case]['speedup'] for case in cases]

        axes[1,0].bar(x - width/2, fem_times, width, label='FEM', alpha=0.8)
        axes[1,0].bar(x + width/2, lbm_times, width, label='LBM', alpha=0.8)
        axes[1,0].set_xlabel('Test Case')
        axes[1,0].set_ylabel('Time (seconds)')
        axes[1,0].set_title('Performance Comparison')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(cases, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Speedup
        axes[1,1].bar(x, speedups, alpha=0.8, color='green')
        axes[1,1].set_xlabel('Test Case')
        axes[1,1].set_ylabel('LBM Speedup (x)')
        axes[1,1].set_title('LBM Speedup over FEM')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(cases, rotation=45)
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = f"results/comparison/new_summary_comparison_{self.mesh_type}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Creating summary plots...")
        print(f"  Created: {filename}")

        plt.close()

def main():
    """Main function to run comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Run new FEM vs LBM comparison')
    parser.add_argument('--mesh', type=str, default='coarse',
                       choices=['coarse', 'medium', 'fine'],
                       help='Mesh type to use')
    parser.add_argument('--steps', type=int, default=50,
                       help='Maximum number of time steps')

    args = parser.parse_args()

    # Run comparison
    runner = NewComparisonRunner(mesh_type=args.mesh)
    results = runner.run_comparison(max_steps=args.steps)

    print(f"\n🎉 Comparison completed successfully!")
    print(f"Check results/comparison/ for detailed results and plots.")

if __name__ == "__main__":
    main()
