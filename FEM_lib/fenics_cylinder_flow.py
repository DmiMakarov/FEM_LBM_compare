"""
FEniCS-based cylinder flow simulation.

Wrapper class for cylinder flow simulation with three boundary conditions:
1. Steady flow (Re = 20)
2. Unsteady flow (Re = 100)
3. Oscillating flow (Re = 100)
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
import os

from .fenics_mesh_generator import FenicsMeshGenerator
from .fenics_navier_stokes_solver import FenicsNavierStokesSolver


class FenicsCylinderFlow:
    """
    Cylinder flow simulation using FEniCS library.
    """

    def __init__(self, mesh_density: str = "medium", dt: float = 0.001,
                 initial_condition: str = "steady"):
        """
        Initialize cylinder flow simulation.

        Args:
            mesh_density: Mesh density ("coarse", "medium", "fine")
            dt: Time step size
            initial_condition: Type of initial condition ("steady", "unsteady", "oscillating")
        """
        self.mesh_density = mesh_density
        self.dt = dt
        self.initial_condition = initial_condition

        # Physical parameters
        self.cylinder_diameter = 0.1
        self.cylinder_x = 0.2
        self.cylinder_y = 0.2
        self.domain_length = 2.2
        self.domain_height = 0.41
        self.nu = 1e-3  # Kinematic viscosity
        self.rho = 1.0  # Density

        # Set parameters based on initial condition
        self._set_initial_condition_parameters()

        # Generate mesh
        self._generate_mesh()

        # Initialize solver
        self._initialize_solver()

        # Storage for results
        self.time_history = []
        self.drag_history = []
        self.lift_history = []
        self.pressure_before_history = []
        self.pressure_after_history = []
        self.pressure_drop_history = []
        self.strouhal_history = []

        # Timing information
        self.timing_info = {
            'total_time': 0.0,
            'time_per_step': 0.0,
            'force_calculation_time': 0.0
        }

        print(f"FenicsCylinderFlow initialized:")
        print(f"  Initial condition: {initial_condition}")
        print(f"  Max velocity (U_m): {self.um:.2f} m/s")
        print(f"  Reynolds number: {self.reynolds_number:.2f}")
        print(f"  Time step: {dt}")
        print(f"  Mesh density: {mesh_density}")

    def _set_initial_condition_parameters(self):
        """Set parameters based on initial condition."""
        if self.initial_condition == "steady":
            # Condition 1: U_m = 0.3 m/s, Re = 20
            self.um = 0.3
            self.reynolds_number = self.um * self.cylinder_diameter / self.nu
        elif self.initial_condition == "unsteady":
            # Condition 2: U_m = 1.5 m/s, Re = 100
            self.um = 1.5
            self.reynolds_number = self.um * self.cylinder_diameter / self.nu
        elif self.initial_condition == "oscillating":
            # Condition 3: U_m = 1.5 m/s with time variation, Re = 100
            self.um = 1.5
            self.reynolds_number = self.um * self.cylinder_diameter / self.nu
        else:
            raise ValueError(f"Unknown initial condition: {self.initial_condition}")

    def _generate_mesh(self):
        """Generate mesh for the simulation."""
        print("Generating mesh...")

        self.mesh_generator = FenicsMeshGenerator(
            domain_length=self.domain_length,
            domain_height=self.domain_height,
            cylinder_diameter=self.cylinder_diameter,
            cylinder_x=self.cylinder_x,
            cylinder_y=self.cylinder_y,
            mesh_density=self.mesh_density
        )

        self.mesh = self.mesh_generator.generate_mesh()

        print(f"  Mesh generated: {self.mesh.topology.index_map(0).size_global} vertices, {self.mesh.topology.index_map(2).size_global} cells")

    def _initialize_solver(self):
        """Initialize the Navier-Stokes solver."""
        print("Initializing solver...")

        self.solver = FenicsNavierStokesSolver(
            mesh=self.mesh,
            nu=self.nu,
            rho=self.rho,
            dt=self.dt,
            reynolds_number=self.reynolds_number
        )

        # Set inlet velocity
        oscillating = (self.initial_condition == "oscillating")
        self.solver.set_inlet_velocity(
            um=self.um,
            H=self.domain_height,
            time=0.0,
            oscillating=oscillating
        )

        print("  Solver initialized")

    def run_simulation(self, max_steps: int = 1000, save_interval: int = 10,
                      convergence_threshold: float = 1e-6) -> Dict:
        """
        Run cylinder flow simulation.

        Args:
            max_steps: Maximum number of time steps
            save_interval: Save results every N steps
            convergence_threshold: Convergence threshold for steady state

        Returns:
            Dictionary containing simulation results
        """
        print(f"\nRunning FenicsCylinderFlow simulation...")
        print(f"  Max steps: {max_steps}")
        print(f"  Save interval: {save_interval}")
        print(f"  Convergence threshold: {convergence_threshold}")

        start_time = time.time()

        # Initialize results storage
        results = {
            'time': [],
            'drag': [],
            'lift': [],
            'pressure_before': [],
            'pressure_after': [],
            'pressure_drop': [],
            'strouhal': [],
            'velocity_x_fields': [],
            'velocity_y_fields': [],
            'pressure_fields': [],
            'vorticity_fields': []
        }

        # Timing variables
        solver_times = []
        force_times = []

        # Main simulation loop
        for step in range(max_steps):
            step_start = time.time()

            # Solve one time step
            solver_start = time.time()
            u_new, v_new, p_new = self.solver.solve_time_step()
            solver_end = time.time()
            solver_times.append(solver_end - solver_start)

            # Compute forces every 10 steps
            if step % 10 == 0:
                force_start = time.time()
                drag, lift = self.solver.compute_forces()
                pressure_before, pressure_after, pressure_drop = self.compute_pressure_drop()
                force_end = time.time()
                force_times.append(force_end - force_start)

                self.drag_history.append(drag)
                self.lift_history.append(lift)
                self.pressure_before_history.append(pressure_before)
                self.pressure_after_history.append(pressure_after)
                self.pressure_drop_history.append(pressure_drop)
                self.time_history.append(step)

                # Compute Strouhal number
                strouhal = self._compute_strouhal_number()
                self.strouhal_history.append(strouhal)
            else:
                # Always compute forces for lift history
                drag, lift = self.solver.compute_forces()
                self.lift_history.append(lift)

            # Save results at intervals
            if step % save_interval == 0:
                # Store field data
                ux, uy = self.solver.get_velocity_field()
                pressure = self.solver.get_pressure_field()
                vorticity = self.solver.get_vorticity_field()

                results['velocity_x_fields'].append(ux.copy())
                results['velocity_y_fields'].append(uy.copy())
                results['pressure_fields'].append(pressure.copy())
                results['vorticity_fields'].append(vorticity.copy())

                # Store time series data
                results['time'].append(step)
                results['drag'].append(drag)
                results['lift'].append(lift)
                results['pressure_before'].append(pressure_before)
                results['pressure_after'].append(pressure_after)
                results['pressure_drop'].append(pressure_drop)
                results['strouhal'].append(strouhal)

            # Check for convergence (for steady case)
            if self.initial_condition == "steady" and step > 100:
                if step % 50 == 0:
                    # Check if drag and lift have converged
                    if len(self.drag_history) > 10:
                        drag_std = np.std(self.drag_history[-10:])
                        lift_std = np.std(self.lift_history[-10:])

                        if drag_std < convergence_threshold and lift_std < convergence_threshold:
                            print(f"  Converged at step {step}")
                            break

            step_end = time.time()

        # Compute timing statistics
        elapsed_time = time.time() - start_time
        self.timing_info['total_time'] = elapsed_time
        self.timing_info['time_per_step'] = elapsed_time / max_steps
        if force_times:
            self.timing_info['force_calculation_time'] = np.mean(force_times)

        print(f"\nFenicsCylinderFlow simulation completed:")
        print(f"  Total time: {elapsed_time:.2f} seconds")
        print(f"  Average time per step: {elapsed_time/max_steps:.4f} seconds")
        if force_times:
            print(f"  Average force calculation time per step: {np.mean(force_times):.4f} seconds")

        return results

    def compute_pressure_drop(self) -> Tuple[float, float, float]:
        """Compute pressure drop across the cylinder."""
        # This is a simplified implementation
        # In practice, you'd find inlet and outlet nodes and compute average pressure

        # For now, return approximate values
        inlet_pressure = 1.0
        outlet_pressure = 0.0
        pressure_drop = inlet_pressure - outlet_pressure

        return inlet_pressure, outlet_pressure, pressure_drop

    def _compute_strouhal_number(self) -> float:
        """Compute Strouhal number from lift coefficient history."""
        if len(self.lift_history) < 100:
            return 0.0

        # Use recent history for frequency analysis
        lift_data = self.lift_history[-100:]

        # Compute FFT
        fft = np.fft.fft(lift_data)
        freqs = np.fft.fftfreq(len(lift_data))

        # Find dominant frequency
        power = np.abs(fft)**2
        dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]

        # Convert to physical frequency
        physical_freq = abs(dominant_freq) / self.dt

        # Strouhal number = f * D / U
        strouhal = physical_freq * self.cylinder_diameter / self.um

        return strouhal

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity field."""
        return self.solver.get_velocity_field()

    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field."""
        return self.solver.get_pressure_field()

    def get_vorticity_field(self) -> np.ndarray:
        """Get vorticity field."""
        return self.solver.get_vorticity_field()

    def get_timing_info(self) -> Dict:
        """Get timing information."""
        return self.timing_info

    def visualize_solution(self, save_path: str = None):
        """Visualize the current solution."""
        self.solver.visualize_solution(save_path)

    def save_results(self, results: Dict, filename: str):
        """
        Save simulation results to disk.

        Args:
            results: Dictionary containing simulation results
            filename: Output filename (should be .npz format)
        """
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Get current field data
        ux, uy = self.solver.get_velocity_field()
        pressure = self.solver.get_pressure_field()
        vorticity = self.solver.get_vorticity_field()

        # Get mesh data
        mesh_data = {
            'nodes': self.mesh.geometry.x,
            'elements': None,  # Would need to extract from FEniCS mesh
            'boundary_nodes': {},
        }

        # Get physical parameters
        physical_params = {
            'reynolds_number': self.reynolds_number,
            'um': self.um,
            'dt': self.dt,
            'nu': self.nu,
            'rho': self.rho,
            'cylinder_diameter': self.cylinder_diameter,
            'cylinder_x': self.cylinder_x,
            'cylinder_y': self.cylinder_y,
            'domain_height': self.domain_height,
            'initial_condition': self.initial_condition
        }

        # Get timing information
        timing_info = self.get_timing_info()

        # Prepare data for saving
        save_data = {
            # Field data
            'velocity_x': ux,
            'velocity_y': uy,
            'pressure': pressure,
            'vorticity': vorticity,

            # Time series data
            'time': np.array(results.get('time', [])),
            'drag': np.array(results.get('drag', [])),
            'lift': np.array(results.get('lift', [])),
            'pressure_before': np.array(results.get('pressure_before', [])),
            'pressure_after': np.array(results.get('pressure_after', [])),
            'pressure_drop': np.array(results.get('pressure_drop', [])),
            'strouhal': np.array(results.get('strouhal', [])),

            # Field time series (if available)
            'velocity_x_fields': results.get('velocity_x_fields', []),
            'velocity_y_fields': results.get('velocity_y_fields', []),
            'pressure_fields': results.get('pressure_fields', []),
            'vorticity_fields': results.get('vorticity_fields', []),

            # Mesh data
            'mesh_nodes': mesh_data['nodes'],
            'mesh_elements': mesh_data['elements'],
            'boundary_nodes': mesh_data['boundary_nodes'],

            # Physical parameters
            'reynolds_number': physical_params['reynolds_number'],
            'um': physical_params['um'],
            'dt': physical_params['dt'],
            'nu': physical_params['nu'],
            'rho': physical_params['rho'],
            'cylinder_diameter': physical_params['cylinder_diameter'],
            'cylinder_x': physical_params['cylinder_x'],
            'cylinder_y': physical_params['cylinder_y'],
            'domain_height': physical_params['domain_height'],
            'initial_condition': physical_params['initial_condition'],

            # Timing information
            'total_time': timing_info.get('total_time', 0.0),
            'time_per_step': timing_info.get('time_per_step', 0.0),
            'force_calculation_time': timing_info.get('force_calculation_time', 0.0),

            # Metadata
            'n_nodes': self.mesh.topology.index_map(0).size_global,
            'n_elements': self.mesh.topology.index_map(2).size_global,
            'method': 'FenicsFEM',
            'solver': 'fenics_navier_stokes_solver'
        }

        # Save to compressed NumPy format
        np.savez_compressed(filename, **save_data)

        print(f"  FenicsFEM results saved to: {filename}")
        print(f"  Saved {len(results.get('time', []))} time steps")
        print(f"  Mesh: {self.mesh.topology.index_map(0).size_global} vertices, {self.mesh.topology.index_map(2).size_global} cells")


def main():
    """Test cylinder flow simulation."""
    # Test all three initial conditions
    for condition in ["steady", "unsteady", "oscillating"]:
        print(f"\n=== Testing {condition} flow ===")

        # Create simulation
        simulation = FenicsCylinderFlow(
            mesh_density="medium",
            dt=0.001,
            initial_condition=condition
        )

        # Run short simulation
        results = simulation.run_simulation(max_steps=100, save_interval=20)

        # Save results
        simulation.save_results(results, f"results/fenics/fenics_{condition}_flow.npz")

        # Visualize
        simulation.visualize_solution(f"fenics_{condition}_solution.png")


if __name__ == "__main__":
    main()
