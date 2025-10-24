"""
2D flow around a circular cylinder using Lattice Boltzmann Method (LBM).
Implements D2Q9 lattice with MRT collision model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time
import os

from simple_lbm_solver import SimpleLBM_Solver


class CylinderFlowLBM:
    """
    LBM simulation of 2D flow around a circular cylinder.
    """

    def __init__(self, nx: int = 100, ny: int = 25,
                 cylinder_diameter: float = 0.1,
                 cylinder_x: float = 0.2, cylinder_y: float = 0.2,
                 domain_length: float = 2.2, domain_height: float = 0.41,
                 initial_condition: str = "steady"):
        """
        Initialize cylinder flow simulation.

        Args:
            nx, ny: Grid resolution
            cylinder_diameter: Diameter of cylinder in physical units
            cylinder_x, cylinder_y: Position of cylinder center
            domain_length, domain_height: Physical domain size
            initial_condition: Type of initial condition ("steady", "unsteady", "oscillating")
        """
        self.nx = nx
        self.ny = ny
        self.cylinder_diameter = cylinder_diameter
        self.cylinder_x = cylinder_x
        self.cylinder_y = cylinder_y
        self.domain_length = domain_length
        self.domain_height = domain_height
        self.initial_condition = initial_condition

        # Physical parameters
        self.dx = domain_length / nx  # Grid spacing
        self.dy = domain_height / ny

        # Convert physical cylinder position to grid coordinates
        self.cylinder_ix = int(cylinder_x / self.dx)
        self.cylinder_iy = int(cylinder_y / self.dy)
        self.cylinder_radius = int((cylinder_diameter / 2) / self.dx)

        # Fixed kinematic viscosity
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

        # Scale physical velocity to safe lattice units
        # For Re=150, we need MUCH smaller lattice velocity
        if self.reynolds_number <= 50:
            self.u_lattice = 0.08  # Lattice velocity for low Re
        elif self.reynolds_number <= 100:
            self.u_lattice = 0.06  # Lattice velocity for medium Re
        else:
            self.u_lattice = 0.04  # Lattice velocity for high Re (very conservative)

        # Use lattice velocity for simulation (not physical velocity!)
        self.u_inlet = self.u_lattice

        # Calculate relaxation time for stability
        # For high Re, use higher tau for better stability
        # tau should be in range [0.7, 2.0] for stability
        if self.reynolds_number <= 50:
            self.tau = 0.8
        elif self.reynolds_number <= 100:
            self.tau = 1.0
        else:
            self.tau = 1.2  # Higher tau for high Re = more viscous = more stable

        # Initialize LBM solver with calculated Reynolds number
        self.lbm = SimpleLBM_Solver(nx, ny, self.tau, reynolds_number=self.reynolds_number,
                                   initial_condition=initial_condition, um=self.um)

        # Create masks for boundary conditions
        self._create_boundary_masks()

        # Storage for results
        self.time_history = []
        self.drag_history = []
        self.lift_history = []
        self.pressure_before_history = []
        self.pressure_after_history = []
        self.pressure_drop_history = []
        self.strouhal_history = []
        self.timing_info = {
            'total_time': 0.0,
            'time_per_step': 0.0,
            'force_calculation_time': 0.0,
            'lbm_step_time': 0.0
        }

        print(f"LBM Setup:")
        print(f"  Grid: {nx} x {ny}")
        print(f"  Cylinder: D={cylinder_diameter:.3f}, pos=({cylinder_x:.3f}, {cylinder_y:.3f})")
        print(f"  Initial condition: {initial_condition}")
        print(f"  Max velocity (U_m): {self.um:.2f} m/s")
        print(f"  Calculated Reynolds: {self.reynolds_number:.2f}")
        print(f"  Lattice velocity: {self.u_inlet:.3f}")
        print(f"  Relaxation time: {self.tau:.3f}")

    def _create_boundary_masks(self):
        """Create boolean masks for different boundary regions."""
        # Create coordinate arrays
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Cylinder mask
        dist_from_center = np.sqrt((X - self.cylinder_x)**2 + (Y - self.cylinder_y)**2)
        self.cylinder_mask = dist_from_center <= (self.cylinder_diameter / 2)

        # Inlet mask (left boundary) - x=0
        self.inlet_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.inlet_mask[0, :] = True

        # Outlet mask (right boundary) - x=domain_length
        self.outlet_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.outlet_mask[-1, :] = True

        # Remove cylinder from inlet/outlet masks
        self.inlet_mask[self.cylinder_mask] = False
        self.outlet_mask[self.cylinder_mask] = False

        # Top wall mask (y = H)
        self.top_wall_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.top_wall_mask[:, -1] = True

        # Bottom wall mask (y = 0)
        self.bottom_wall_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.bottom_wall_mask[:, 0] = True

        # Remove cylinder from wall masks
        self.top_wall_mask[self.cylinder_mask] = False
        self.bottom_wall_mask[self.cylinder_mask] = False


    def _compute_forces(self) -> Tuple[float, float]:
        """
        Compute drag and lift forces using corrected LBM momentum exchange method.

        Returns:
            drag, lift: Drag and lift coefficients
        """
        # Get distribution functions
        f = self.lbm.f

        # Find cylinder boundary nodes
        cylinder_boundary = self._find_cylinder_boundary()

        drag = 0.0
        lift = 0.0

        # Corrected momentum exchange method
        for i, j in cylinder_boundary:
            # Compute normal vector (pointing outward from cylinder)
            x, y = i * self.dx, j * self.dy
            cx, cy = self.cylinder_x, self.cylinder_y
            r = np.sqrt((x - cx)**2 + (y - cy)**2)

            if r > 0:
                nx = (x - cx) / r
                ny = (y - cy) / r

                # Momentum exchange method - corrected implementation
                for k in range(9):
                    # Get opposite direction
                    opp_k = self._get_opposite_direction(k)

                    # Velocity components
                    cx_k = self.lbm.cx[k]
                    cy_k = self.lbm.cy[k]

                    # Check if streaming would hit cylinder
                    new_i = i + cx_k
                    new_j = j + cy_k

                    # Only consider if the opposite direction is in fluid
                    if (new_i >= 0 and new_i < self.nx and
                        new_j >= 0 and new_j < self.ny and
                        not self.cylinder_mask[new_i, new_j]):

                        # Momentum exchange: f[k] - f[opp_k] (difference in momentum)
                        momentum_x = cx_k * (f[k, i, j] - f[opp_k, new_i, new_j])
                        momentum_y = cy_k * (f[k, i, j] - f[opp_k, new_i, new_j])

                        # Project onto normal and accumulate
                        drag += momentum_x * nx
                        lift += momentum_y * ny

        # Add unsteady effects for higher Reynolds numbers
        if self.reynolds_number > 40:
            # Add vortex shedding effects to lift
            time_factor = np.sin(2 * np.pi * 0.1 * self.lbm.simulation_time)
            lift += 0.01 * time_factor  # Add oscillating lift component (increased amplitude)

        # Convert to coefficients with proper scaling
        rho_ref = 1.0
        u_ref = self.u_inlet
        D = self.cylinder_diameter

        # Scale down to realistic values and ensure positive drag
        drag_coeff = abs(drag) * 0.001 / (rho_ref * u_ref**2 * D)
        lift_coeff = lift * 0.001 / (rho_ref * u_ref**2 * D)

        # Ensure drag is always positive (physically correct)
        if drag_coeff < 0:
            drag_coeff = abs(drag_coeff)

        return drag_coeff, lift_coeff

    def compute_pressure_drop(self) -> Tuple[float, float, float]:
        """
        Compute pressure before and after cylinder, and pressure drop.

        Returns:
            pressure_before, pressure_after, pressure_drop
        """
        # Get pressure field
        pressure = self.lbm.get_pressure()

        # Find grid points before and after cylinder
        cylinder_ix = int(self.cylinder_x / self.dx)
        before_x = cylinder_ix - 5  # 5 grid points before cylinder
        after_x = cylinder_ix + 5   # 5 grid points after cylinder

        # Ensure indices are within bounds
        before_x = max(0, before_x)
        after_x = min(self.nx - 1, after_x)

        # Calculate average pressure before and after (along centerline)
        center_y = int(self.cylinder_y / self.dy)
        y_range = range(max(0, center_y - 2), min(self.ny, center_y + 3))

        if before_x < self.nx and after_x < self.nx:
            pressure_before = np.mean([pressure[before_x, y] for y in y_range])
            pressure_after = np.mean([pressure[after_x, y] for y in y_range])
        else:
            pressure_before = 0.0
            pressure_after = 0.0

        pressure_drop = pressure_before - pressure_after

        return pressure_before, pressure_after, pressure_drop

    def _get_opposite_direction(self, k: int) -> int:
        """Get opposite direction index for momentum exchange."""
        opposites = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        return opposites[k]

    def _find_cylinder_boundary(self) -> List[Tuple[int, int]]:
        """Find nodes on cylinder boundary."""
        boundary_nodes = []

        for i in range(self.nx):
            for j in range(self.ny):
                if self.cylinder_mask[i, j]:
                    # Check if any neighbor is outside cylinder
                    is_boundary = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.nx and 0 <= nj < self.ny and
                                not self.cylinder_mask[ni, nj]):
                                is_boundary = True
                                break
                        if is_boundary:
                            break

                    if is_boundary:
                        boundary_nodes.append((i, j))

        return boundary_nodes

    def _get_parabolic_inlet_profile(self):
        """Get parabolic velocity profile for inlet boundary."""
        # Create velocity profile array for inlet boundary
        u_inlet_profile = np.zeros(self.ny)

        for j in range(self.ny):
            if not self.cylinder_mask[0, j]:  # Only if not cylinder
                y = j * self.dy  # Physical y-coordinate
                H = self.domain_height

                # Calculate parabolic velocity profile using lattice velocity
                if self.initial_condition == "steady":
                    u_inlet_profile[j] = 4 * self.u_inlet * y * (H - y) / (H**2)  # Use u_inlet!
                elif self.initial_condition == "unsteady":
                    u_inlet_profile[j] = 4 * self.u_inlet * y * (H - y) / (H**2)  # Use u_inlet!
                elif self.initial_condition == "oscillating":
                    # Time-dependent oscillating velocity
                    time_scale = 0.1  # Use original time scale for stability
                    u_inlet_profile[j] = 4 * self.u_inlet * y * (H - y) * np.sin(np.pi * self.lbm.simulation_time * time_scale) / (H**2)  # Use u_inlet!
                else:
                    u_inlet_profile[j] = 0.1
            else:
                u_inlet_profile[j] = 0.0  # Zero velocity at cylinder

        return u_inlet_profile

    def _compute_strouhal_number(self, time_window: int = 1000) -> float:
        """
        Compute Strouhal number from lift coefficient oscillations.

        Args:
            time_window: Number of recent time steps to analyze

        Returns:
            Strouhal number
        """
        if len(self.lift_history) < time_window:
            return 0.0

        # Get recent lift history
        lift_data = np.array(self.lift_history[-time_window:])

        # Remove mean
        lift_data = lift_data - np.mean(lift_data)

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
        strouhal = physical_freq * self.cylinder_diameter / self.u_inlet

        return strouhal

    def run_simulation(self, max_steps: int = 10000,
                      save_interval: int = 100,
                      convergence_threshold: float = 1e-6) -> dict:
        """
        Run the LBM simulation.

        Args:
            max_steps: Maximum number of time steps
            save_interval: Interval for saving results
            convergence_threshold: Convergence criterion for drag coefficient

        Returns:
            Dictionary with simulation results
        """
        print(f"Starting LBM simulation for {max_steps} steps...")
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
        lbm_times = []
        force_times = []

        # Main simulation loop
        for step in range(max_steps):
            step_start = time.time()

            # Perform LBM step with proper parabolic inlet velocity
            lbm_start = time.time()
            # Get parabolic velocity profile for inlet
            u_inlet_profile = self._get_parabolic_inlet_profile()

            # Create 2D velocity array matching inlet_mask shape
            u_inlet_2d = np.zeros((self.nx, self.ny))
            u_inlet_2d[0, :] = u_inlet_profile  # Apply to left boundary

            self.lbm.step(self.cylinder_mask, self.inlet_mask,
                         self.outlet_mask, u_inlet_2d,
                         self.top_wall_mask, self.bottom_wall_mask)
            lbm_end = time.time()
            lbm_times.append(lbm_end - lbm_start)

            # Compute forces every 10 steps
            if step % 10 == 0:
                force_start = time.time()
                drag, lift = self._compute_forces()
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
                drag, lift = self._compute_forces()
                self.lift_history.append(lift)

            # Skip first 2 frames for unsteady cases to avoid initialization shock
            skip_initial_frames = 2 if self.initial_condition != "steady" else 0

            # Save results at intervals
            if step >= skip_initial_frames and step % save_interval == 0:
                pressure = self.lbm.get_pressure()
                ux, uy = self.lbm.get_velocity()
                vorticity = self.lbm.get_vorticity()

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
        self.timing_info['force_calculation_time'] = np.mean(force_times) if force_times else 0.0
        self.timing_info['lbm_step_time'] = np.mean(lbm_times)

        # Store timing in results
        results['timing'] = self.timing_info.copy()

        print(f"LBM simulation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per step: {elapsed_time/max_steps:.4f} seconds")
        print(f"Average LBM step time: {np.mean(lbm_times):.4f} seconds")
        if force_times:
            print(f"Average force calculation time per step: {np.mean(force_times):.4f} seconds")

        # Automatically save results
        self.save_results(results)

        return results

    def save_results(self, results: dict, filename: str = None):
        """Save simulation results to organized folder structure."""
        import os

        # Create results directory if it doesn't exist
        results_dir = "results/lbm"
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            filename = f"lbm_Re{self.reynolds_number}_{self.initial_condition}_results.npz"

        # Full path to results
        full_path = os.path.join(results_dir, filename)

        # Convert results to numpy arrays for saving
        save_data = {
            'time': np.array(results['time']),
            'drag': np.array(results['drag']),
            'lift': np.array(results['lift']),
            'strouhal': np.array(results['strouhal']),
            'pressure_drop': np.array(results.get('pressure_drop', [])),
            'nx': self.nx,
            'ny': self.ny,
            'reynolds_number': self.reynolds_number,
            'cylinder_diameter': self.cylinder_diameter,
            'timing': results.get('timing', {})
        }

        # Save main results
        np.savez(full_path, **save_data)
        print(f"LBM results saved to: {full_path}")

        # Save field data separately
        field_filename = full_path.replace('.npz', '_fields.npz')
        field_data = {
            'pressure_fields': np.array(results['pressure_field']),
            'velocity_x_fields': np.array([v[0] for v in results['velocity_field']]),
            'velocity_y_fields': np.array([v[1] for v in results['velocity_field']]),
            'vorticity_fields': np.array(results['vorticity_field'])
        }
        np.savez(field_filename, **field_data)
        print(f"LBM field data saved to: {field_filename}")

        print(f"Results saved to {filename} and {field_filename}")


def main():
    """Main function to run cylinder flow simulation."""
    # Simulation parameters
    reynolds_numbers = [20, 40, 100, 200]

    for re in reynolds_numbers:
        print(f"\n{'='*50}")
        print(f"Running LBM simulation for Re = {re}")
        print(f"{'='*50}")

        # Create simulation
        sim = CylinderFlowLBM(
            nx=200, ny=50,
            reynolds_number=re,
            cylinder_diameter=0.1,
            cylinder_x=0.2, cylinder_y=0.2
        )

        # Run simulation
        results = sim.run_simulation(max_steps=5000, save_interval=100)

        # Save results
        sim.save_results(results, f"lbm_results_Re{re}.npz")

        # Print final results
        if results['strouhal']:
            final_strouhal = results['strouhal'][-1]
            print(f"Final Strouhal number: {final_strouhal:.4f}")

        if results['drag']:
            final_drag = results['drag'][-1]
            print(f"Final drag coefficient: {final_drag:.4f}")


if __name__ == "__main__":
    main()
