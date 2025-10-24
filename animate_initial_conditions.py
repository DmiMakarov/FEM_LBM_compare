"""
Animation generation for different initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple
import os
import sys

# Add paths for imports
sys.path.append('FEM')
sys.path.append('LBM')

from FEM.cylinder_flow_fem import CylinderFlowFEM
from LBM.cylinder_flow_lbm import CylinderFlowLBM


class InitialConditionAnimator:
    """
    Create animations for different initial conditions.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize animator.

        Args:
            results_dir: Directory containing simulation results
        """
        self.results_dir = results_dir

    def create_condition_animations(self, condition: str, reynolds_number: int,
                                  field_type: str = "pressure",
                                  duration: float = 10.0, fps: int = 10):
        """
        Create animations for a specific initial condition.

        Args:
            condition: Initial condition type ("steady", "unsteady", "oscillating")
            reynolds_number: Reynolds number
            field_type: Type of field to animate ("pressure", "velocity", "vorticity")
            duration: Animation duration in seconds
            fps: Frames per second
        """
        print(f"Creating animations for {condition} condition (Re={reynolds_number})...")

        # Create output directory
        anim_dir = f"{self.results_dir}/animations/initial_conditions"
        os.makedirs(anim_dir, exist_ok=True)

        # Run simulations if needed
        fem_results = self._run_fem_simulation(condition, reynolds_number)
        lbm_results = self._run_lbm_simulation(condition, reynolds_number)

        # Create individual animations
        fem_anim = self._create_fem_animation(fem_results, condition, reynolds_number,
                                           field_type, duration, fps, anim_dir)
        lbm_anim = self._create_lbm_animation(lbm_results, condition, reynolds_number,
                                           field_type, duration, fps, anim_dir)

        # Create comparison animation
        comp_anim = self._create_comparison_animation(fem_results, lbm_results, condition,
                                                    reynolds_number, field_type, duration, fps, anim_dir)

        return {
            'fem_animation': fem_anim,
            'lbm_animation': lbm_anim,
            'comparison_animation': comp_anim
        }

    def _run_fem_simulation(self, condition: str, reynolds_number: int) -> Dict:
        """Run FEM simulation for animation data."""
        try:
            # Choose appropriate parameters
            if condition == "steady":
                um = 0.3
            else:
                um = 1.5

            # Use appropriate mesh file
            if reynolds_number == 20:
                mesh_file = 'meshes/cylinder_mesh_Re20_data.npz'
            else:
                mesh_file = 'meshes/cylinder_mesh_Re20_data.npz'

            # Initialize and run simulation
            fem_sim = CylinderFlowFEM(
                mesh_file, reynolds_number=reynolds_number, dt=0.001, max_velocity=um,
                initial_condition=condition, um=um
            )

            # Run simulation with more steps for animation
            results = fem_sim.run_simulation(max_steps=100, save_interval=5)

            return results

        except Exception as e:
            print(f"FEM simulation failed: {e}")
            return None

    def _run_lbm_simulation(self, condition: str, reynolds_number: int) -> Dict:
        """Run LBM simulation for animation data."""
        try:
            # Choose appropriate parameters
            if condition == "steady":
                um = 0.3
            else:
                um = 1.5

            # Initialize and run simulation
            lbm_sim = CylinderFlowLBM(
                nx=100, ny=25, reynolds_number=reynolds_number,
                cylinder_diameter=0.1, cylinder_x=0.2, cylinder_y=0.2,
                initial_condition=condition, um=um
            )

            # Run simulation with more steps for animation
            results = lbm_sim.run_simulation(max_steps=100, save_interval=5)

            return results

        except Exception as e:
            print(f"LBM simulation failed: {e}")
            return None

    def _create_fem_animation(self, results: Dict, condition: str, reynolds_number: int,
                            field_type: str, duration: float, fps: int, anim_dir: str) -> str:
        """Create FEM animation for specific condition."""
        if results is None:
            return None

        # Get field data
        if field_type == "pressure":
            fields = results['pressure_field']
            title = f"FEM Pressure - {condition.title()} (Re={reynolds_number})"
            cmap = 'viridis'
        elif field_type == "velocity":
            # Calculate velocity magnitude
            ux_fields = [v[0] for v in results['velocity_field']]
            uy_fields = [v[1] for v in results['velocity_field']]
            fields = [np.sqrt(ux**2 + uy**2) for ux, uy in zip(ux_fields, uy_fields)]
            title = f"FEM Velocity - {condition.title()} (Re={reynolds_number})"
            cmap = 'plasma'
        elif field_type == "vorticity":
            fields = results['vorticity_field']
            title = f"FEM Vorticity - {condition.title()} (Re={reynolds_number})"
            cmap = 'RdBu_r'
        else:
            return None

        # Create animation
        output_file = f"{anim_dir}/fem_{field_type}_{condition}_Re{reynolds_number}_animation.gif"
        self._create_field_animation(fields, title, field_type, duration, fps, output_file)

        return output_file

    def _create_lbm_animation(self, results: Dict, condition: str, reynolds_number: int,
                            field_type: str, duration: float, fps: int, anim_dir: str) -> str:
        """Create LBM animation for specific condition."""
        if results is None:
            return None

        # Get field data
        if field_type == "pressure":
            fields = results['pressure_field']
            title = f"LBM Pressure - {condition.title()} (Re={reynolds_number})"
            cmap = 'viridis'
        elif field_type == "velocity":
            # Calculate velocity magnitude
            ux_fields = [v[0] for v in results['velocity_field']]
            uy_fields = [v[1] for v in results['velocity_field']]
            fields = [np.sqrt(ux**2 + uy**2) for ux, uy in zip(ux_fields, uy_fields)]
            title = f"LBM Velocity - {condition.title()} (Re={reynolds_number})"
            cmap = 'plasma'
        elif field_type == "vorticity":
            fields = results['vorticity_field']
            title = f"LBM Vorticity - {condition.title()} (Re={reynolds_number})"
            cmap = 'RdBu_r'
        else:
            return None

        # Create animation
        output_file = f"{anim_dir}/lbm_{field_type}_{condition}_Re{reynolds_number}_animation.gif"
        self._create_field_animation(fields, title, field_type, duration, fps, output_file)

        return output_file

    def _create_comparison_animation(self, fem_results: Dict, lbm_results: Dict,
                                   condition: str, reynolds_number: int,
                                   field_type: str, duration: float, fps: int, anim_dir: str) -> str:
        """Create side-by-side comparison animation."""
        if fem_results is None or lbm_results is None:
            return None

        # Get FEM field data
        if field_type == "pressure":
            fem_fields = fem_results['pressure_field']
            lbm_fields = lbm_results['pressure_field']
        elif field_type == "velocity":
            # Calculate velocity magnitude for both
            fem_ux = [v[0] for v in fem_results['velocity_field']]
            fem_uy = [v[1] for v in fem_results['velocity_field']]
            lbm_ux = [v[0] for v in lbm_results['velocity_field']]
            lbm_uy = [v[1] for v in lbm_results['velocity_field']]
            fem_fields = [np.sqrt(ux**2 + uy**2) for ux, uy in zip(fem_ux, fem_uy)]
            lbm_fields = [np.sqrt(ux**2 + uy**2) for ux, uy in zip(lbm_ux, lbm_uy)]
        elif field_type == "vorticity":
            fem_fields = fem_results['vorticity_field']
            lbm_fields = lbm_results['vorticity_field']
        else:
            return None

        # Create comparison animation
        output_file = f"{anim_dir}/comparison_{field_type}_{condition}_Re{reynolds_number}_animation.gif"
        self._create_comparison_field_animation(fem_fields, lbm_fields, condition,
                                              reynolds_number, field_type, duration, fps, output_file)

        return output_file

    def _create_field_animation(self, fields: List[np.ndarray], title: str,
                              field_type: str, duration: float, fps: int, output_file: str):
        """Create animation from field data."""
        if not fields:
            return

        # Calculate number of frames
        n_frames = min(len(fields), int(duration * fps))
        frame_indices = np.linspace(0, len(fields) - 1, n_frames, dtype=int)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up the plot
        im = ax.imshow(fields[0].T, origin='lower', cmap='viridis',
                      extent=[0, 1, 0, 0.4], aspect='equal')

        # Add cylinder
        cylinder_x, cylinder_y = 0.2, 0.2
        cylinder_radius = 0.05
        circle = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                          color='black', fill=True)
        ax.add_patch(circle)

        # Set up colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(field_type.title())

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

        # Animation function
        def animate(frame_idx):
            frame = fields[frame_indices[frame_idx]]
            im.set_array(frame.T)
            return [im]

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=n_frames, interval=1000//fps,
            blit=True, repeat=True
        )

        # Save animation
        anim.save(output_file, writer='pillow', fps=fps)
        plt.close(fig)

        print(f"  Animation saved to: {output_file}")

    def _create_comparison_field_animation(self, fem_fields: List[np.ndarray],
                                         lbm_fields: List[np.ndarray], condition: str,
                                         reynolds_number: int, field_type: str,
                                         duration: float, fps: int, output_file: str):
        """Create side-by-side comparison animation."""
        if not fem_fields or not lbm_fields:
            return

        # Calculate number of frames
        n_frames = min(len(fem_fields), len(lbm_fields), int(duration * fps))
        frame_indices = np.linspace(0, min(len(fem_fields), len(lbm_fields)) - 1,
                                   n_frames, dtype=int)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{condition.title()} Condition (Re={reynolds_number}) - {field_type.title()} Comparison',
                    fontsize=14, fontweight='bold')

        # Set up FEM plot
        im1 = ax1.imshow(fem_fields[0].T, origin='lower', cmap='viridis',
                        extent=[0, 1, 0, 0.4], aspect='equal')
        ax1.set_title(f"FEM {field_type.title()}")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # Add cylinder to FEM plot
        circle1 = plt.Circle((0.2, 0.2), 0.05, color='black', fill=True)
        ax1.add_patch(circle1)

        # Set up LBM plot
        im2 = ax2.imshow(lbm_fields[0].T, origin='lower', cmap='viridis',
                        extent=[0, 1, 0, 0.4], aspect='equal')
        ax2.set_title(f"LBM {field_type.title()}")
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        # Add cylinder to LBM plot
        circle2 = plt.Circle((0.2, 0.2), 0.05, color='black', fill=True)
        ax2.add_patch(circle2)

        # Set up colorbars
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar1.set_label(field_type.title())
        cbar2.set_label(field_type.title())

        # Animation function
        def animate(frame_idx):
            fem_frame = fem_fields[frame_indices[frame_idx]]
            lbm_frame = lbm_fields[frame_indices[frame_idx]]

            im1.set_array(fem_frame.T)
            im2.set_array(lbm_frame.T)

            return [im1, im2]

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=n_frames, interval=1000//fps,
            blit=True, repeat=True
        )

        # Save animation
        anim.save(output_file, writer='pillow', fps=fps)
        plt.close(fig)

        print(f"  Comparison animation saved to: {output_file}")

    def create_all_condition_animations(self, field_types: List[str] = ["pressure", "velocity", "vorticity"],
                                      duration: float = 5.0, fps: int = 10):
        """Create animations for all initial conditions."""
        print("Creating animations for all initial conditions...")

        conditions = [
            ("steady", 20),
            ("unsteady", 100),
            ("oscillating", 100)
        ]

        all_animations = {}

        for condition, reynolds_number in conditions:
            print(f"\n{condition.title()} condition (Re={reynolds_number}):")
            condition_animations = {}

            for field_type in field_types:
                print(f"  Creating {field_type} animations...")
                animations = self.create_condition_animations(
                    condition, reynolds_number, field_type, duration, fps
                )
                condition_animations[field_type] = animations

            all_animations[condition] = condition_animations

        print("\nAll animations created!")
        return all_animations


def main():
    """Main function to create animations for all initial conditions."""
    print("Initial Condition Animation Generator")
    print("=" * 50)

    # Create animator
    animator = InitialConditionAnimator()

    # Create animations for all conditions
    all_animations = animator.create_all_condition_animations(
        field_types=["pressure", "velocity", "vorticity"],
        duration=5.0, fps=10
    )

    print("\n" + "=" * 50)
    print("Animation generation completed!")
    print("Check results in: results/animations/initial_conditions/")


if __name__ == "__main__":
    main()
