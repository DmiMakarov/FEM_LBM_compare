"""
Create FEM animations from existing results without running new computations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob

class FEMAnimationGenerator:
    """
    Generate FEM animations from existing simulation results.
    """

    def __init__(self):
        """Initialize the animation generator."""
        self.results_dir = "results/fem"
        self.animations_dir = "results/animations"

        # Ensure animations directory exists
        os.makedirs(self.animations_dir, exist_ok=True)

        print("FEM Animation Generator")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Animations directory: {self.animations_dir}")

    def find_fem_results(self):
        """Find all available FEM result files."""
        pattern = os.path.join(self.results_dir, "*_results_fields.npz")
        files = glob.glob(pattern)

        print(f"\nFound {len(files)} FEM result files:")
        for file in files:
            print(f"  {os.path.basename(file)}")

        return files

    def create_fem_animation(self, result_file: str, field_type: str):
        """Create FEM animation for a specific field type."""
        try:
            print(f"\nCreating FEM {field_type} animation from {os.path.basename(result_file)}")

            # Load FEM data
            fem_data = np.load(result_file)

            # Extract field data
            field_key = field_type + '_field'
            if field_key not in fem_data:
                print(f"  Field '{field_key}' not found in {result_file}")
                return False

            field_data = fem_data[field_key]
            print(f"  Field data shape: {field_data.shape}")

            if len(field_data) == 0:
                print(f"  No data available for {field_type}")
                return False

            # Extract simulation parameters from filename
            filename = os.path.basename(result_file)
            if "Re30" in filename:
                re = 30
                condition = "steady"
            elif "Re150" in filename:
                re = 150
                condition = "unsteady"
            else:
                re = 30
                condition = "steady"

            # Create animation
            fig, ax = plt.subplots(figsize=(12, 8))

            # Domain parameters (consistent with FEM setup)
            domain_length = 2.2
            domain_height = 0.41
            cylinder_x = 0.2
            cylinder_y = 0.2
            cylinder_radius = 0.05

            # Create visualization grid
            nx_vis, ny_vis = 20, 10  # Consistent with LBM for comparison
            x_grid = np.linspace(0, domain_length, nx_vis)
            y_grid = np.linspace(0, domain_height, ny_vis)
            X, Y = np.meshgrid(x_grid, y_grid)

            cbar = None

            def animate(frame):
                nonlocal cbar
                ax.clear()

                if frame < len(field_data):
                    data = field_data[frame]

                    # Reshape data for visualization - SWAP X and Y
                    if len(data) >= nx_vis * ny_vis:
                        Z = data[:nx_vis*ny_vis].reshape(ny_vis, nx_vis).T  # Transpose to swap X and Y
                    else:
                        Z = np.zeros((nx_vis, ny_vis))  # Swapped dimensions
                        Z.flat[:len(data)] = data

                    # Create contour plot
                    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', extend='both')

                    # Add cylinder
                    cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                        color='black', zorder=10, alpha=0.9)
                    ax.add_patch(cylinder)
                    cylinder_outline = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                                fill=False, color='white',
                                                linewidth=3, zorder=11)
                    ax.add_patch(cylinder_outline)

                    ax.set_title(f'FEM - {field_type.title()} (Re={re:.0f}, {condition})\nFine Mesh (3191 nodes)',
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
                filename = f"{self.animations_dir}/fem_{field_type}_{condition}_Re{re:.0f}_from_results.gif"
                anim.save(filename, writer='pillow', fps=3)
                print(f"  Created: {filename}")
                return True
            else:
                print(f"  No frames available for {field_type}")
                return False

        except Exception as e:
            print(f"  Failed to create FEM {field_type} animation: {e}")
            return False
        finally:
            plt.close()

    def create_all_animations(self):
        """Create animations for all available FEM results."""
        result_files = self.find_fem_results()

        if not result_files:
            print("No FEM result files found!")
            return

        field_types = ["pressure", "velocity", "vorticity"]
        created_count = 0

        for result_file in result_files:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(result_file)}")

            for field_type in field_types:
                if self.create_fem_animation(result_file, field_type):
                    created_count += 1

        print(f"\n🎉 Created {created_count} FEM animations successfully!")
        print(f"Check {self.animations_dir}/ for the animation files.")

def main():
    """Main function to create FEM animations."""
    generator = FEMAnimationGenerator()
    generator.create_all_animations()

if __name__ == "__main__":
    main()
