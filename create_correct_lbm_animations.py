"""
Correct LBM animation generator that handles the data dimensions properly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
from typing import Dict, List, Tuple

class CorrectLBMAnimationGenerator:
    """
    Correct LBM animation generator that handles data dimensions properly.
    """

    def __init__(self):
        """Initialize correct LBM animation generator."""
        self.results_dir = "results"
        self.animations_dir = "results/animations"

        # Create animations directory
        os.makedirs(self.animations_dir, exist_ok=True)

        print("Correct LBM Animation Generator initialized")

    def create_all_lbm_animations(self):
        """Create animations for all existing LBM results."""
        print("🎬 Creating correct LBM animations for all existing results...")

        # Find all LBM results
        lbm_files = glob.glob(f"{self.results_dir}/lbm/*_results_fields.npz")
        print(f"Found {len(lbm_files)} LBM result files")

        # Create LBM animations
        for lbm_file in lbm_files:
            self.create_lbm_animation_from_file(lbm_file)

        print("🎉 All correct LBM animations created successfully!")

    def create_lbm_animation_from_file(self, lbm_file: str):
        """Create LBM animation from existing result file."""
        print(f"Creating LBM animation from {lbm_file}")

        try:
            # Load LBM data
            lbm_data = np.load(lbm_file)

            # Extract information from filename
            filename = os.path.basename(lbm_file)
            if "Re30" in filename:
                re = 30.0
                condition = "steady"
            elif "Re150" in filename:
                re = 150.0
                if "unsteady" in filename:
                    condition = "unsteady"
                elif "oscillating" in filename:
                    condition = "oscillating"
                else:
                    condition = "unsteady"  # Default for Re150
            else:
                re = 30.0
                condition = "steady"

            # Create animations for each field type
            for field_type in ["pressure", "velocity", "vorticity"]:
                try:
                    if field_type == "pressure" and "pressure_fields" in lbm_data:
                        self.create_lbm_field_animation_correct(lbm_data, field_type, re, condition)
                    elif field_type == "velocity" and "velocity_x_fields" in lbm_data:
                        self.create_lbm_field_animation_correct(lbm_data, field_type, re, condition)
                    elif field_type == "vorticity" and "vorticity_fields" in lbm_data:
                        self.create_lbm_field_animation_correct(lbm_data, field_type, re, condition)
                    else:
                        print(f"  No {field_type} data found in {lbm_file}")
                except Exception as e:
                    print(f"  Error creating LBM {field_type} animation: {e}")

        except Exception as e:
            print(f"  Error creating LBM animation from {lbm_file}: {e}")

    def create_lbm_field_animation_correct(self, lbm_data: Dict, field_type: str, re: float, condition: str):
        """Create correct LBM field animation with proper dimension handling."""
        print(f"  Creating LBM {field_type} animation")

        try:
            # Get field data - handle numpy arrays correctly
            if field_type == "pressure":
                field_data = lbm_data['pressure_fields']  # Shape: (200, 40, 20)
            elif field_type == "velocity":
                ux_data = lbm_data['velocity_x_fields']  # Shape: (200, 40, 20)
                uy_data = lbm_data['velocity_y_fields']  # Shape: (200, 40, 20)
                # Compute velocity magnitude
                field_data = np.sqrt(ux_data**2 + uy_data**2)  # Shape: (200, 40, 20)
            elif field_type == "vorticity":
                field_data = lbm_data['vorticity_fields']  # Shape: (200, 40, 20)
            else:
                return

            if len(field_data) == 0:
                print(f"  No {field_type} data to animate")
                return

            print(f"    Creating LBM {field_type} animation ({len(field_data)} frames)")
            print(f"    Data shape: {field_data.shape}")

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Domain parameters
            domain_length = 2.2
            domain_height = 0.41
            cylinder_x = 0.2
            cylinder_y = 0.2
            cylinder_radius = 0.05

            # LBM grid parameters - CORRECT: data shape is (frames, ny, nx)
            n_frames, ny_lbm, nx_lbm = field_data.shape

            dx = domain_length / nx_lbm
            dy = domain_height / ny_lbm

            # Create LBM grid - CORRECT dimensions
            x_lbm = np.linspace(dx/2, domain_length - dx/2, nx_lbm)
            y_lbm = np.linspace(dy/2, domain_height - dy/2, ny_lbm)
            X_lbm, Y_lbm = np.meshgrid(x_lbm, y_lbm)

            print(f"    Grid dimensions: nx={nx_lbm}, ny={ny_lbm}")
            print(f"    Grid shapes: X_lbm={X_lbm.shape}, Y_lbm={Y_lbm.shape}")

            # Cylinder in grid coordinates
            cylinder_ix = int(cylinder_x / dx)
            cylinder_iy = int(cylinder_y / dy)
            cylinder_radius_grid = max(1, int(cylinder_radius / dx))

            # Ensure cylinder is within bounds
            cylinder_ix = max(1, min(nx_lbm-2, cylinder_ix))
            cylinder_iy = max(1, min(ny_lbm-2, cylinder_iy))

            cbar = None

            def animate(frame):
                nonlocal cbar
                ax.clear()

                if frame < len(field_data):
                    Z = field_data[frame]  # Shape: (40, 20)
                    Z = Z.T  # Transpose to match X_lbm, Y_lbm grid (20, 40)

                    print(f"    Frame {frame}: Z shape = {Z.shape}, X_lbm shape = {X_lbm.shape}, Y_lbm shape = {Y_lbm.shape}")

                    # Create contour plot - shapes should match now
                    im = ax.contourf(X_lbm, Y_lbm, Z, levels=20, cmap='viridis', extend='both')

                    # Add velocity vectors for velocity field
                    if field_type == "velocity" and "velocity_x_fields" in lbm_data and "velocity_y_fields" in lbm_data:
                        try:
                            ux_data = lbm_data['velocity_x_fields']
                            uy_data = lbm_data['velocity_y_fields']
                            if frame < len(ux_data) and frame < len(uy_data):
                                ux = ux_data[frame].T  # Shape: (40, 20) -> (20, 40)
                                uy = uy_data[frame].T  # Shape: (40, 20) -> (20, 40)
                                # Subsample for cleaner visualization
                                step = max(1, min(nx_lbm//8, ny_lbm//8))
                                ax.quiver(X_lbm[::step, ::step], Y_lbm[::step, ::step],
                                        ux[::step, ::step], uy[::step, ::step],
                                        alpha=0.7, scale=20, width=0.003)
                        except Exception as e:
                            print(f"    Error adding velocity vectors: {e}")
                            pass  # Skip velocity vectors if there's an error

                    # Add cylinder
                    cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                        color='black', zorder=10, alpha=0.9)
                    ax.add_patch(cylinder)
                    cylinder_outline = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                                fill=False, color='white',
                                                linewidth=3, zorder=11)
                    ax.add_patch(cylinder_outline)

                    ax.set_title(f'LBM - {field_type.title()} (Re={re:.0f}, {condition})\nCoarse Mesh ({nx_lbm}x{ny_lbm})',
                                fontsize=14, fontweight='bold')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, domain_length)
                    ax.set_ylim(0, domain_height)

                    # Create colorbar only once
                    if cbar is None:
                        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                        cbar.set_label(f'{field_type.title()}', fontsize=12)

            # Create animation
            max_frames = len(field_data)
            if max_frames > 0:
                anim = FuncAnimation(fig, animate, frames=max_frames, interval=300, repeat=True)
                filename = f"{self.animations_dir}/lbm_{field_type}_{condition}_Re{re:.0f}_correct.gif"
                anim.save(filename, writer='pillow', fps=3)
                print(f"    Created: {filename}")

            plt.close()

        except Exception as e:
            print(f"  Error creating LBM {field_type} animation: {e}")
            plt.close()

def main():
    """Main function to create all correct LBM animations."""
    generator = CorrectLBMAnimationGenerator()
    generator.create_all_lbm_animations()

if __name__ == "__main__":
    main()
