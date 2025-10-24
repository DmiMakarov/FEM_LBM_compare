#!/usr/bin/env python3
"""
Standalone script to run proper scikit-fem cylinder flow simulations.

This script provides a command-line interface for running cylinder flow
simulations using the proper scikit-fem Navier-Stokes solver.
"""

import argparse
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add FEM_lib to path
sys.path.append(str(Path(__file__).parent / "FEM_lib"))

from FEM_lib import ProperSkfemCylinderFlow


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run proper scikit-fem cylinder flow simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation parameters
    parser.add_argument("--condition", "-c",
                       choices=["steady", "unsteady", "oscillating"],
                       default="steady",
                       help="Initial condition type")

    parser.add_argument("--mesh-density", "-m",
                       choices=["coarse", "medium", "fine"],
                       default="medium",
                       help="Mesh density")

    parser.add_argument("--dt", "-t", type=float, default=0.001,
                       help="Time step size")

    parser.add_argument("--max-steps", "-s", type=int, default=1000,
                       help="Maximum number of time steps")

    parser.add_argument("--save-interval", "-i", type=int, default=10,
                       help="Save results every N steps")

    parser.add_argument("--convergence-threshold", "-e", type=float, default=1e-6,
                       help="Convergence threshold for steady state")

    # Output parameters
    parser.add_argument("--output-dir", "-o", default="results/proper_skfem",
                       help="Output directory for results")

    parser.add_argument("--save-fields", action="store_true",
                       help="Save field data (velocity, pressure, vorticity)")

    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Generate visualization plots")

    # Performance parameters
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    return parser.parse_args()


def run_simulation(args):
    """Run the cylinder flow simulation."""
    print("=" * 60)
    print("Proper Scikit-fem Cylinder Flow Simulation")
    print("=" * 60)
    print(f"Initial condition: {args.condition}")
    print(f"Mesh density: {args.mesh_density}")
    print(f"Time step: {args.dt}")
    print(f"Max steps: {args.max_steps}")
    print(f"Save interval: {args.save_interval}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize simulation
    print("\nInitializing simulation...")
    start_time = time.time()

    simulation = ProperSkfemCylinderFlow(
        mesh_density=args.mesh_density,
        dt=args.dt,
        initial_condition=args.condition
    )

    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f} seconds")

    # Run simulation
    print(f"\nRunning simulation...")
    sim_start = time.time()

    results = simulation.run_simulation(
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        convergence_threshold=args.convergence_threshold
    )

    sim_time = time.time() - sim_start
    print(f"Simulation completed in {sim_time:.2f} seconds")

    # Save results
    print(f"\nSaving results...")
    output_filename = f"{args.output_dir}/proper_skfem_{args.condition}_flow.npz"
    simulation.save_results(results, output_filename)

    # Generate visualization if requested
    if args.visualize:
        print(f"\nGenerating visualizations...")
        viz_filename = f"{args.output_dir}/proper_skfem_{args.condition}_solution.png"
        simulation.visualize_solution(viz_filename)

    # Print summary
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Initialization: {init_time:.2f} seconds ({init_time/total_time*100:.1f}%)")
    print(f"Simulation: {sim_time:.2f} seconds ({sim_time/total_time*100:.1f}%)")
    print(f"Time per step: {sim_time/args.max_steps:.4f} seconds")

    # Print results summary
    if results['drag']:
        print(f"\nResults:")
        print(f"  Final drag coefficient: {results['drag'][-1]:.6f}")
        print(f"  Final lift coefficient: {results['lift'][-1]:.6f}")
        print(f"  Final pressure drop: {results['pressure_drop'][-1]:.6f}")
        if results['strouhal']:
            print(f"  Strouhal number: {results['strouhal'][-1]:.6f}")

    print(f"\nResults saved to: {output_filename}")
    print("=" * 60)

    return results, simulation


def run_all_conditions():
    """Run simulations for all three boundary conditions."""
    conditions = ["steady", "unsteady", "oscillating"]

    print("Running all boundary conditions...")
    print("=" * 60)

    for condition in conditions:
        print(f"\n--- Running {condition} flow ---")

        # Create simulation
        simulation = ProperSkfemCylinderFlow(
            mesh_density="medium",
            dt=0.001,
            initial_condition=condition
        )

        # Run simulation
        results = simulation.run_simulation(
            max_steps=500,  # Shorter for testing
            save_interval=50
        )

        # Save results
        output_filename = f"results/proper_skfem/proper_skfem_{condition}_flow.npz"
        simulation.save_results(results, output_filename)

        print(f"  {condition} simulation completed")
        print(f"  Results saved to: {output_filename}")

    print("\nAll simulations completed!")


def main():
    """Main function."""
    args = parse_arguments()

    try:
        if args.condition == "all":
            run_all_conditions()
        else:
            results, simulation = run_simulation(args)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
