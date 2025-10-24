"""
2D flow around a circular cylinder using Finite Element Method (FEM).
Uses FEniCS for the finite element formulation.
"""

import numpy as np
from typing import Tuple, List, Dict
import time
import os

from fem_solver import FEM_Solver


class CylinderFlowFEM:
    """
    FEM simulation of 2D flow around a circular cylinder.
    """

    def __init__(self, mesh_data_file: str, dt: float = 0.001,
                 initial_condition: str = "steady"):
        """
        Initialize cylinder flow simulation.

        Args:
            mesh_data_file: Path to mesh data file (.npz)
            dt: Time step size
            initial_condition: Type of initial condition ("steady", "unsteady", "oscillating")
        """
        # Use fine mesh for better accuracy
        self.mesh_data_file = "meshes/fine_mesh_80x40_data.npz"
        self.dt = dt / 100  # Use extremely small time step for stability
        self.initial_condition = initial_condition

        # Physical parameters (fixed)
        self.cylinder_diameter = 0.1
        self.cylinder_x = 0.2
        self.cylinder_y = 0.2
        self.nu = 1e-3  # Fixed kinematic viscosity (m²/s)
        self.rho = 1.0  # Fixed density (kg/m³)

        # Define initial conditions and calculate derived parameters
        if initial_condition == "steady":
            # Condition 1: U_m = 0.3 m/s
            self.um = 0.3
            self.reynolds_number = self.um * self.cylinder_diameter / self.nu  # Re = 30
        elif initial_condition == "unsteady":
            # Condition 2: U_m = 1.5 m/s
            self.um = 1.5
            self.reynolds_number = self.um * self.cylinder_diameter / self.nu  # Re = 150
        elif initial_condition == "oscillating":
            # Condition 3: U_m = 1.5 m/s with time variation
            self.um = 1.5
            self.reynolds_number = self.um * self.cylinder_diameter / self.nu  # Re = 150
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")

        # Load unified mesh data
        mesh_data = np.load(mesh_data_file)
        mesh_dict = {
            'nodes': mesh_data['nodes'],
            'elements': mesh_data['elements'],
            'boundary_nodes': mesh_data['boundary_nodes'],
            'cylinder_nodes': mesh_data['cylinder_nodes'],
            'inlet_nodes': mesh_data['inlet_nodes'],
            'outlet_nodes': mesh_data['outlet_nodes'],
            'nx': mesh_data.get('nx', 80),
            'ny': mesh_data.get('ny', 40)
        }

        # Initialize FEM solver with calculated Reynolds number
        self.fem = FEM_Solver(mesh_dict, self.reynolds_number, self.dt, self.nu,
                              initial_condition=initial_condition, um=self.um)

        # Storage for results
        self.time_history = []
        self.drag_history = []
        self.lift_history = []
        self.pressure_before_history = []
        self.pressure_after_history = []
        self.pressure_drop_history = []
        self.timing_info = {
            'total_time': 0.0,
            'time_per_step': 0.0,
            'force_calculation_time': 0.0,
            'solver_time': 0.0
        }
        self.strouhal_history = []

        print(f"FEM Setup:")
        print(f"  Mesh file: {mesh_data_file}")
        print(f"  Initial condition: {initial_condition}")
        print(f"  Max velocity (U_m): {self.um:.2f} m/s")
        print(f"  Calculated Reynolds number: {self.reynolds_number:.2f}")
        print(f"  Time step: {dt}")
        print(f"  Viscosity: {self.nu:.6f}")

    def run_simulation(self, max_steps: int = 1000,
                      save_interval: int = 10,
                      convergence_threshold: float = 1e-6) -> Dict:
        """
        Run the FEM simulation.

        Args:
            max_steps: Maximum number of time steps
            save_interval: Interval for saving results
            convergence_threshold: Convergence criterion for drag coefficient

        Returns:
            Dictionary with simulation results
        """
        print(f"Starting FEM simulation for {max_steps} steps...")
        start_time = time.time()

        # Initialize results storage
        results = {
            'time': [],
            'drag': [],
            'lift': [],
            'strouhal': [],
            'pressure_drop': [],
            'pressure_field': [],
            'velocity_field': [],
            'vorticity_field': [],
            'timing': {}
        }

        # Timing variables
        solver_times = []
        force_times = []

        # Main simulation loop
        for step in range(max_steps):
            step_start = time.time()

            # Solve one time step
            solver_start = time.time()
            u_new, p_new = self.fem.solve_time_step()
            solver_end = time.time()
            solver_times.append(solver_end - solver_start)

            # Update solution
            self.fem.u = u_new
            self.fem.p = p_new

            # Compute forces every 5 steps
            if step % 5 == 0:
                force_start = time.time()
                drag, lift = self.fem.compute_forces()
                pressure_before, pressure_after, pressure_drop = self.fem.compute_pressure_drop()
                force_end = time.time()
                force_times.append(force_end - force_start)

                self.drag_history.append(drag)
                self.lift_history.append(lift)
                self.pressure_before_history.append(pressure_before)
                self.pressure_after_history.append(pressure_after)
                self.pressure_drop_history.append(pressure_drop)
                self.time_history.append(step * self.dt)

                # Compute Strouhal number
                strouhal = self.fem.compute_strouhal_number(
                    self.lift_history, self.time_history)
                self.strouhal_history.append(strouhal)

            # Save results at intervals
            if step % save_interval == 0:
                # Get field data
                pressure = self.fem.get_pressure_field()
                ux, uy = self.fem.get_velocity_field()
                vorticity = self.fem.get_vorticity_field()

                results['time'].append(step * self.dt)
                results['drag'].append(self.drag_history[-1] if self.drag_history else 0)
                results['lift'].append(self.lift_history[-1] if self.lift_history else 0)
                results['strouhal'].append(self.strouhal_history[-1] if self.strouhal_history else 0)
                results['pressure_drop'].append(self.pressure_drop_history[-1] if self.pressure_drop_history else 0)
                results['pressure_field'].append(pressure.copy())
                results['velocity_field'].append((ux.copy(), uy.copy()))
                results['vorticity_field'].append(vorticity.copy())

                print(f"Step {step:6d}: Drag={drag:.4f}, Lift={lift:.4f}, St={strouhal:.4f}")

            # Check convergence
            if len(self.drag_history) > 50:
                recent_drag = np.array(self.drag_history[-50:])
                drag_std = np.std(recent_drag)
                if drag_std < convergence_threshold:
                    print(f"Converged at step {step}")
                    break

        elapsed_time = time.time() - start_time

        # Calculate timing statistics
        self.timing_info['total_time'] = elapsed_time
        self.timing_info['time_per_step'] = elapsed_time / max_steps
        self.timing_info['force_calculation_time'] = np.mean(force_times) if force_times else 0.0
        self.timing_info['solver_time'] = np.mean(solver_times)

        # Store timing in results
        results['timing'] = self.timing_info.copy()

        print(f"FEM simulation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {elapsed_time/max_steps:.4f} seconds")
        print(f"Average solver time per step: {np.mean(solver_times):.4f} seconds")
        if force_times:
            print(f"Average force calculation time per step: {np.mean(force_times):.4f} seconds")

        # Automatically save results
        self.save_results(results)

        return results

    def save_results(self, results: Dict, filename: str = None):
        """Save simulation results to organized folder structure."""
        import os

        # Create results directory if it doesn't exist
        results_dir = "results/fem"
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            filename = f"fem_Re{self.reynolds_number}_results.npz"

        # Full path to results
        full_path = os.path.join(results_dir, filename)

        # Convert results to numpy arrays for saving
        save_data = {
            'time': np.array(results['time']),
            'drag': np.array(results['drag']),
            'lift': np.array(results['lift']),
            'strouhal': np.array(results['strouhal']),
            'pressure_drop': np.array(results.get('pressure_drop', [])),
            'reynolds_number': self.reynolds_number,
            'cylinder_diameter': self.cylinder_diameter,
            'dt': self.dt,
            'nu': self.nu,
            'timing': results.get('timing', {})
        }

        # Save main results
        np.savez(full_path, **save_data)
        print(f"FEM results saved to: {full_path}")

        # Save field data separately
        field_filename = full_path.replace('.npz', '_fields.npz')
        field_data = {
            'pressure_fields': np.array(results['pressure_field']),
            'velocity_x_fields': np.array([v[0] for v in results['velocity_field']]),
            'velocity_y_fields': np.array([v[1] for v in results['velocity_field']]),
            'vorticity_fields': np.array(results['vorticity_field'])
        }
        np.savez(field_filename, **field_data)
        print(f"FEM field data saved to: {field_filename}")

        print(f"Results saved to {filename} and {field_filename}")

    def save_solution(self, filename: str):
        """Save FEM solution to file."""
        self.fem.save_solution(filename)


def main():
    """Main function to run cylinder flow simulation."""
    # Check if mesh files exist
    mesh_dir = "meshes"
    if not os.path.exists(mesh_dir):
        print("Mesh directory not found. Please run generate_mesh.py first.")
        return

    # Simulation parameters
    reynolds_numbers = [20, 40, 100, 200]

    for re in reynolds_numbers:
        print(f"\n{'='*50}")
        print(f"Running FEM simulation for Re = {re}")
        print(f"{'='*50}")

        # Check if mesh data file exists
        mesh_data_file = os.path.join(mesh_dir, f"cylinder_mesh_Re{re:.0f}_data.npz")
        if not os.path.exists(mesh_data_file):
            print(f"Mesh data file {mesh_data_file} not found. Generating mesh...")
            # Generate mesh using simple mesh generator
            from generate_simple_mesh import generate_cylinder_mesh
            mesh_data_file = generate_cylinder_mesh(re, nx=100, ny=50, output_dir=mesh_dir)

        # Create simulation
        sim = CylinderFlowFEM(
            mesh_data_file=mesh_data_file,
            reynolds_number=re,
            dt=0.001,
            max_velocity=0.1
        )

        # Run simulation
        results = sim.run_simulation(max_steps=1000, save_interval=10)

        # Save results
        sim.save_results(results, f"fem_results_Re{re}.npz")

        # Save solution
        sim.save_solution(f"fem_solution_Re{re}")

        # Print final results
        if results['strouhal']:
            final_strouhal = results['strouhal'][-1]
            print(f"Final Strouhal number: {final_strouhal:.4f}")

        if results['drag']:
            final_drag = results['drag'][-1]
            print(f"Final drag coefficient: {final_drag:.4f}")


if __name__ == "__main__":
    main()
