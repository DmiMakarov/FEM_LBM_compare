"""
Proper FEM wrapper for cylinder flow simulation.
Uses the proper FEM solver with element assembly and shape functions.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from proper_fem_solver import ProperFEM_Solver


class ProperCylinderFlowFEM:
    """
    Proper FEM simulation of 2D flow around a circular cylinder.
    Uses proper element assembly with shape functions and Gauss quadrature.
    """

    def __init__(self, mesh_data_file: str = "meshes/small_mesh_20x10_data.npz",
                 dt: float = 0.001, initial_condition: str = "steady"):
        """
        Initialize proper FEM cylinder flow simulation.

        Args:
            mesh_data_file: Path to mesh data file
            dt: Time step size
            initial_condition: Type of initial condition ("steady", "unsteady", "oscillating")
        """
        # Load mesh data
        mesh_data = np.load(mesh_data_file, allow_pickle=True)
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

        # Physical parameters
        self.cylinder_diameter = 0.1
        self.cylinder_x = 0.2
        self.cylinder_y = 0.2
        self.domain_length = 2.2
        self.domain_height = 0.41
        self.um = 0.3  # Maximum velocity

        # Calculate Reynolds number
        self.reynolds_number = self.um * self.cylinder_diameter / 0.001  # nu = 0.001

        # Initialize proper FEM solver
        self.fem = ProperFEM_Solver(
            mesh_dict,
            self.reynolds_number,
            dt,
            0.001,  # nu
            initial_condition=initial_condition,
            um=self.um
        )

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

        print(f"Proper FEM Setup:")
        print(f"  Mesh file: {mesh_data_file}")
        print(f"  Initial condition: {initial_condition}")
        print(f"  Max velocity (U_m): {self.um:.2f} m/s")
        print(f"  Calculated Reynolds number: {self.reynolds_number:.2f}")
        print(f"  Time step: {dt}")

    def run_simulation(self, max_steps: int = 1000, save_interval: int = 10,
                      convergence_threshold: float = 1e-6) -> Dict:
        """
        Run proper FEM simulation.

        Args:
            max_steps: Maximum number of time steps
            save_interval: Save results every N steps
            convergence_threshold: Convergence threshold for steady state

        Returns:
            Dictionary containing simulation results
        """
        print(f"\nRunning proper FEM simulation...")
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
        fem_times = []
        force_times = []

        # Main simulation loop
        for step in range(max_steps):
            step_start = time.time()

            # Solve one time step using proper FEM
            fem_start = time.time()
            u_new, p_new = self.fem.solve_time_step()
            self.fem.u = u_new
            self.fem.p = p_new
            fem_end = time.time()
            fem_times.append(fem_end - fem_start)

            # Compute forces every 10 steps
            if step % 10 == 0:
                force_start = time.time()
                drag, lift = self.fem.compute_forces()
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
                # Always compute forces for lift history, but only store every 10 steps
                drag, lift = self.fem.compute_forces()
                self.lift_history.append(lift)

            # Save results at intervals
            if step % save_interval == 0:
                # Store field data
                ux, uy = self.fem.get_velocity_field()
                pressure = self.fem.get_pressure_field()
                vorticity = self.fem.get_vorticity_field()

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
            if self.fem.initial_condition == "steady" and step > 100:
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

        print(f"\nProper FEM simulation completed:")
        print(f"  Total time: {elapsed_time:.2f} seconds")
        print(f"  Average time per step: {elapsed_time/max_steps:.4f} seconds")
        if force_times:
            print(f"  Average force calculation time per step: {np.mean(force_times):.4f} seconds")

        return results

    def compute_pressure_drop(self) -> Tuple[float, float, float]:
        """Compute pressure drop across the cylinder."""
        if len(self.fem.inlet_nodes) == 0 or len(self.fem.outlet_nodes) == 0:
            return 0.0, 0.0, 0.0

        # Average pressure at inlet
        inlet_pressure = np.mean([self.fem.p[node] for node in self.fem.inlet_nodes])

        # Average pressure at outlet
        outlet_pressure = np.mean([self.fem.p[node] for node in self.fem.outlet_nodes])

        # Pressure drop
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
        dt = 1.0  # Lattice time step
        physical_freq = abs(dominant_freq) / dt

        # Strouhal number = f * D / U
        strouhal = physical_freq * self.cylinder_diameter / self.um

        return strouhal

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity field."""
        return self.fem.get_velocity_field()

    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field."""
        return self.fem.get_pressure_field()

    def get_vorticity_field(self) -> np.ndarray:
        """Get vorticity field."""
        return self.fem.get_vorticity_field()

    def get_timing_info(self) -> Dict:
        """Get timing information."""
        return self.timing_info
