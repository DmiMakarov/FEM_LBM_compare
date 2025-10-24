"""
Simple FEM cylinder flow simulation using the simplified solver.
"""

import numpy as np
import time
import os
from typing import Dict, Tuple
from simple_fem_solver import SimpleFEM_Solver

class SimpleCylinderFlowFEM:
    """
    Simple FEM simulation of 2D flow around a circular cylinder.
    """

    def __init__(self, mesh_data_file: str, dt: float = 0.001,
                 initial_condition: str = "steady"):
        """
        Initialize FEM cylinder flow simulation.
        """
        self.mesh_data_file = mesh_data_file
        self.dt = dt
        self.initial_condition = initial_condition

        # Physical parameters (fixed)
        self.nu = 1e-3  # Fixed kinematic viscosity (m²/s)
        self.rho = 1.0  # Fixed density (kg/m³)
        self.cylinder_diameter = 0.1  # Cylinder diameter (m)

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

        # Load mesh data
        mesh_data = np.load(mesh_data_file)
        mesh_dict = {
            'nodes': mesh_data['nodes'],
            'elements': mesh_data['elements'],
            'boundary_nodes': mesh_data['boundary_nodes'],
            'cylinder_nodes': mesh_data['cylinder_nodes'],
            'inlet_nodes': mesh_data['inlet_nodes'],
            'outlet_nodes': mesh_data['outlet_nodes'],
            'wall_nodes': mesh_data['wall_nodes'],
            'nx': mesh_data.get('nx', 100),
            'ny': mesh_data.get('ny', 25)
        }

        # Initialize simple FEM solver
        self.fem = SimpleFEM_Solver(mesh_dict, self.reynolds_number, dt, self.nu,
                                   initial_condition=initial_condition, um=self.um)

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
            'solver_time': 0.0,
            'force_calculation_time': 0.0
        }

        print(f"Simple FEM Setup:")
        print(f"  Mesh file: {mesh_data_file}")
        print(f"  Initial condition: {initial_condition}")
        print(f"  Max velocity (U_m): {self.um:.2f} m/s")
        print(f"  Calculated Reynolds number: {self.reynolds_number:.2f}")
        print(f"  Time step: {dt}")
        print(f"  Viscosity: {self.nu:.6f}")

    def run_simulation(self, max_steps: int = 1000, save_interval: int = 10,
                      convergence_threshold: float = 1e-6) -> Dict:
        """
        Run the FEM simulation.
        """
        print(f"Starting simple FEM simulation for {max_steps} steps...")
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
                pressure_before, pressure_after, pressure_drop = self.compute_pressure_drop()
                force_end = time.time()
                force_times.append(force_end - force_start)

                self.drag_history.append(drag)
                self.lift_history.append(lift)
                self.pressure_before_history.append(pressure_before)
                self.pressure_after_history.append(pressure_after)
                self.pressure_drop_history.append(pressure_drop)
                self.time_history.append(step * self.dt)

                # Compute Strouhal number
                strouhal = self.compute_strouhal_number()
                self.strouhal_history.append(strouhal)

            # Save results at intervals
            if step % save_interval == 0:
                pressure = self.fem.get_pressure_field()
                ux, uy = self.fem.get_velocity_field()
                vorticity = self.fem.get_vorticity_field()

                results['time'].append(step)
                results['drag'].append(self.drag_history[-1] if self.drag_history else 0)
                results['lift'].append(self.lift_history[-1] if self.lift_history else 0)
                results['strouhal'].append(self.strouhal_history[-1] if self.strouhal_history else 0)
                results['pressure_drop'].append(self.pressure_drop_history[-1] if self.pressure_drop_history else 0)
                results['pressure_field'].append(pressure.copy())
                results['velocity_field'].append((ux.copy(), uy.copy()))
                results['vorticity_field'].append(vorticity.copy())

                print(f"Step {step:6d}: Drag={drag:.4f}, Lift={lift:.4f}, St={strouhal:.4f}")

            # Check convergence
            if len(self.drag_history) > 100:
                recent_drag = np.array(self.drag_history[-100:])
                drag_std = np.std(recent_drag)
                if drag_std < convergence_threshold:
                    print(f"Converged at step {step}")
                    break

        elapsed_time = time.time() - start_time

        # Calculate timing statistics
        self.timing_info['total_time'] = elapsed_time
        self.timing_info['time_per_step'] = elapsed_time / max_steps
        self.timing_info['solver_time'] = np.mean(solver_times) if solver_times else 0.0
        self.timing_info['force_calculation_time'] = np.mean(force_times) if force_times else 0.0

        results['timing'] = self.timing_info

        print(f"Simple FEM simulation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {self.timing_info['time_per_step']:.4f} seconds")
        print(f"Average solver time per step: {self.timing_info['solver_time']:.4f} seconds")
        print(f"Average force calculation time per step: {self.timing_info['force_calculation_time']:.4f} seconds")

        # Save results
        self.save_results(results)

        return results

    def compute_pressure_drop(self) -> Tuple[float, float, float]:
        """Compute pressure drop across the domain."""
        if len(self.fem.inlet_nodes) == 0 or len(self.fem.outlet_nodes) == 0:
            return 0.0, 0.0, 0.0

        # Average pressure at inlet
        inlet_pressure = np.mean(self.fem.p[self.fem.inlet_nodes])

        # Average pressure at outlet
        outlet_pressure = np.mean(self.fem.p[self.fem.outlet_nodes])

        # Pressure drop
        pressure_drop = inlet_pressure - outlet_pressure

        return inlet_pressure, outlet_pressure, pressure_drop

    def compute_strouhal_number(self) -> float:
        """Compute Strouhal number from lift history."""
        if len(self.lift_history) < 10:
            return 0.0

        # Find dominant frequency
        lift_array = np.array(self.lift_history)
        time_array = np.array(self.time_history)

        if len(lift_array) < 2:
            return 0.0

        # Simple frequency analysis
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.001
        freqs = np.fft.fftfreq(len(lift_array), dt)
        fft_lift = np.fft.fft(lift_array)

        # Find dominant frequency
        positive_freqs = freqs[freqs > 0]
        positive_fft = np.abs(fft_lift[freqs > 0])

        if len(positive_freqs) > 0:
            dominant_freq = positive_freqs[np.argmax(positive_fft)]
            strouhal = dominant_freq * self.cylinder_diameter / self.um
            return strouhal

        return 0.0

    def save_results(self, results: Dict):
        """Save simulation results."""
        # Create results directory
        os.makedirs('results/fem', exist_ok=True)

        # Save main results
        results_file = f"results/fem/simple_fem_Re{self.reynolds_number:.1f}_{self.initial_condition}_results.npz"
        np.savez(results_file,
                 reynolds_number=self.reynolds_number,
                 cylinder_diameter=self.cylinder_diameter,
                 dt=self.dt,
                 nu=self.nu,
                 drag=np.array(self.drag_history),
                 lift=np.array(self.lift_history),
                 strouhal=np.array(self.strouhal_history),
                 pressure_drop=np.array(self.pressure_drop_history),
                 time=np.array(self.time_history),
                 timing=self.timing_info)

        # Save field data
        fields_file = f"results/fem/simple_fem_Re{self.reynolds_number:.1f}_{self.initial_condition}_results_fields.npz"
        np.savez(fields_file,
                 pressure_field=results['pressure_field'],
                 velocity_field=results['velocity_field'],
                 vorticity_field=results['vorticity_field'],
                 time=results['time'])

        print(f"Results saved to {results_file} and {fields_file}")
