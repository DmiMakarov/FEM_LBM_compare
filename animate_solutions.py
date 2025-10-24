"""
Animation tools for FEM and LBM flow solutions.
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


class FlowAnimator:
    """
    Create animations of flow solutions from FEM and LBM simulations.
    """

    def __init__(self, results_dir: str = "results"):
        """
        Initialize animator.

        Args:
            results_dir: Directory containing simulation results
        """
        self.results_dir = results_dir

    def create_fem_animation(self, reynolds_number: int,
                           field_type: str = "pressure",
                           duration: float = 10.0,
                           fps: int = 10) -> str:
        """
        Create animation from FEM simulation results.

        Args:
            reynolds_number: Reynolds number for the simulation
            field_type: Type of field to animate ("pressure", "velocity", "vorticity")
            duration: Animation duration in seconds
            fps: Frames per second

        Returns:
            Path to saved animation file
        """
        # Load FEM results
        results_file = f"{self.results_dir}/fem/fem_Re{reynolds_number}_results_fields.npz"
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"FEM results not found: {results_file}")

        data = np.load(results_file)

        # Get field data
        if field_type == "pressure":
            fields = data['pressure_fields']
            title = f"FEM Pressure Field (Re={reynolds_number})"
            cmap = 'viridis'
        elif field_type == "velocity":
            ux_fields = data['velocity_x_fields']
            uy_fields = data['velocity_y_fields']
            fields = np.sqrt(ux_fields**2 + uy_fields**2)
            title = f"FEM Velocity Magnitude (Re={reynolds_number})"
            cmap = 'plasma'
        elif field_type == "vorticity":
            fields = data['vorticity_fields']
            title = f"FEM Vorticity Field (Re={reynolds_number})"
            cmap = 'RdBu_r'
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        # Create animation
        return self._create_field_animation(
            fields, title, field_type, duration, fps, reynolds_number, method="fem"
        )

    def create_lbm_animation(self, reynolds_number: int,
                           field_type: str = "pressure",
                           duration: float = 10.0,
                           fps: int = 10) -> str:
        """
        Create animation from LBM simulation results.

        Args:
            reynolds_number: Reynolds number for the simulation
            field_type: Type of field to animate ("pressure", "velocity", "vorticity")
            duration: Animation duration in seconds
            fps: Frames per second

        Returns:
            Path to saved animation file
        """
        # Load LBM results
        results_file = f"{self.results_dir}/lbm/lbm_Re{reynolds_number}_results_fields.npz"
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"LBM results not found: {results_file}")

        data = np.load(results_file)

        # Get field data
        if field_type == "pressure":
            fields = data['pressure_fields']
            title = f"LBM Pressure Field (Re={reynolds_number})"
            cmap = 'viridis'
        elif field_type == "velocity":
            ux_fields = data['velocity_x_fields']
            uy_fields = data['velocity_y_fields']
            fields = np.sqrt(ux_fields**2 + uy_fields**2)
            title = f"LBM Velocity Magnitude (Re={reynolds_number})"
            cmap = 'plasma'
        elif field_type == "vorticity":
            fields = data['vorticity_fields']
            title = f"LBM Vorticity Field (Re={reynolds_number})"
            cmap = 'RdBu_r'
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        # Create animation
        return self._create_field_animation(
            fields, title, field_type, duration, fps, reynolds_number, method="lbm"
        )

    def _create_field_animation(self, fields: np.ndarray, title: str,
                              field_type: str, duration: float, fps: int,
                              reynolds_number: int, method: str = "fem") -> str:
        """
        Create animation from field data.

        Args:
            fields: Array of field snapshots (time, x, y)
            title: Animation title
            field_type: Type of field
            duration: Animation duration in seconds
            fps: Frames per second
            reynolds_number: Reynolds number

        Returns:
            Path to saved animation file
        """
        # Create output directory
        anim_dir = f"{self.results_dir}/animations"
        os.makedirs(anim_dir, exist_ok=True)

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

        # Save animation with method-specific filename
        output_file = f"{anim_dir}/{method}_{field_type}_Re{reynolds_number}_animation.gif"
        anim.save(output_file, writer='pillow', fps=fps)

        plt.close(fig)

        print(f"Animation saved to: {output_file}")
        return output_file

    def create_comparison_animation(self, reynolds_number: int,
                                 field_type: str = "pressure",
                                 duration: float = 10.0,
                                 fps: int = 10) -> str:
        """
        Create side-by-side comparison animation of FEM and LBM.

        Args:
            reynolds_number: Reynolds number for the simulation
            field_type: Type of field to animate
            duration: Animation duration in seconds
            fps: Frames per second

        Returns:
            Path to saved animation file
        """
        # Load both FEM and LBM results
        fem_file = f"{self.results_dir}/fem/fem_Re{reynolds_number}_results_fields.npz"
        lbm_file = f"{self.results_dir}/lbm/lbm_Re{reynolds_number}_results_fields.npz"

        if not os.path.exists(fem_file) or not os.path.exists(lbm_file):
            raise FileNotFoundError("Both FEM and LBM results required for comparison")

        fem_data = np.load(fem_file)
        lbm_data = np.load(lbm_file)

        # Get field data
        if field_type == "pressure":
            fem_fields = fem_data['pressure_fields']
            lbm_fields = lbm_data['pressure_fields']
        elif field_type == "velocity":
            fem_ux = fem_data['velocity_x_fields']
            fem_uy = fem_data['velocity_y_fields']
            lbm_ux = lbm_data['velocity_x_fields']
            lbm_uy = lbm_data['velocity_y_fields']
            fem_fields = np.sqrt(fem_ux**2 + fem_uy**2)
            lbm_fields = np.sqrt(lbm_ux**2 + lbm_uy**2)
        elif field_type == "vorticity":
            fem_fields = fem_data['vorticity_fields']
            lbm_fields = lbm_data['vorticity_fields']
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        # Create output directory
        anim_dir = f"{self.results_dir}/animations"
        os.makedirs(anim_dir, exist_ok=True)

        # Calculate number of frames
        n_frames = min(len(fem_fields), len(lbm_fields), int(duration * fps))
        frame_indices = np.linspace(0, min(len(fem_fields), len(lbm_fields)) - 1,
                                   n_frames, dtype=int)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Set up FEM plot
        im1 = ax1.imshow(fem_fields[0].T, origin='lower', cmap='viridis',
                        extent=[0, 1, 0, 0.4], aspect='equal')
        ax1.set_title(f"FEM {field_type.title()} (Re={reynolds_number})")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # Add cylinder to FEM plot
        circle1 = plt.Circle((0.2, 0.2), 0.05, color='black', fill=True)
        ax1.add_patch(circle1)

        # Set up LBM plot
        im2 = ax2.imshow(lbm_fields[0].T, origin='lower', cmap='viridis',
                        extent=[0, 1, 0, 0.4], aspect='equal')
        ax2.set_title(f"LBM {field_type.title()} (Re={reynolds_number})")
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
        output_file = f"{anim_dir}/comparison_{field_type}_Re{reynolds_number}_animation.gif"
        anim.save(output_file, writer='pillow', fps=fps)

        plt.close(fig)

        print(f"Comparison animation saved to: {output_file}")
        return output_file


def main():
    """Create animations for all available results."""
    animator = FlowAnimator()

    # Available Reynolds numbers
    reynolds_numbers = [20, 40, 100, 200]
    field_types = ["pressure", "velocity", "vorticity"]

    print("Creating animations for available results...")

    for re in reynolds_numbers:
        for field_type in field_types:
            try:
                # Create FEM animation
                fem_file = f"results/fem/fem_Re{re}_results_fields.npz"
                if os.path.exists(fem_file):
                    fem_anim = animator.create_fem_animation(re, field_type, duration=5.0, fps=10)
                    print(f"  Created FEM animation: {fem_anim}")

                # Create LBM animation
                lbm_file = f"results/lbm/lbm_Re{re}_results_fields.npz"
                if os.path.exists(lbm_file):
                    lbm_anim = animator.create_lbm_animation(re, field_type, duration=5.0, fps=10)
                    print(f"  Created LBM animation: {lbm_anim}")

                # Create comparison animation
                if os.path.exists(fem_file) and os.path.exists(lbm_file):
                    comp_anim = animator.create_comparison_animation(re, field_type, duration=5.0, fps=10)
                    print(f"  Created comparison animation: {comp_anim}")

            except Exception as e:
                print(f"Error creating animation for Re={re}, {field_type}: {e}")

    print("Animation creation completed!")


if __name__ == "__main__":
    main()
