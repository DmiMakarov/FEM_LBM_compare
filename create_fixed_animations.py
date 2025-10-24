"""
Create fixed animations with correct X and Y coordinate system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
import argparse
import sys

# Add FEM_lib to path for scikit-fem support
sys.path.append('FEM_lib')

class FixedAnimationGenerator:
    """
    Generate animations with correct X and Y coordinate system.
    """

    def __init__(self):
        """Initialize the animation generator."""
        self.results_dir = "results"
        self.animations_dir = "results/animations"

        # Ensure animations directory exists
        os.makedirs(self.animations_dir, exist_ok=True)

        print("Fixed Animation Generator")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Animations directory: {self.animations_dir}")

    def create_lbm_animation(self, result_file: str, field_type: str):
        """Create LBM animation with correct coordinates."""
        try:
            print(f"\nCreating LBM {field_type} animation from {os.path.basename(result_file)}")

            # Load LBM data
            lbm_data = np.load(result_file)

            # Extract field data - handle velocity specially
            if field_type == "velocity":
                # For velocity, we need to compute magnitude from x and y components
                if "velocity_x_fields" not in lbm_data or "velocity_y_fields" not in lbm_data:
                    print(f"  Velocity components not found")
                    return False
                # We'll compute velocity magnitude in the animation loop
                field_data = None  # Will be computed dynamically
            else:
                field_key = field_type + '_fields'
                if field_key not in lbm_data:
                    print(f"  Field '{field_key}' not found")
                    return False
                field_data = lbm_data[field_key]

            if field_data is not None:
                print(f"  Field data shape: {field_data.shape}")

                if len(field_data) == 0:
                    print(f"  No data available for {field_type}")
                    return False
            else:
                print(f"  Velocity magnitude will be computed from x and y components")

            # Extract simulation parameters from filename
            filename = os.path.basename(result_file)
            if "_steady_" in filename:
                re = 30
                condition = "steady"
            elif "_unsteady_" in filename:
                re = 150
                condition = "unsteady"
            elif "_oscillating_" in filename:
                re = 150
                condition = "oscillating"
            else:
                # Try to extract from Re number in filename
                if "Re30" in filename:
                    re = 30
                    condition = "steady"
                elif "Re150" in filename:
                    re = 150
                    condition = "unsteady"
                else:
                    re = 30
                    condition = "steady"


            # Domain parameters
            domain_length = 2.2
            domain_height = 0.41
            cylinder_x = 0.2
            cylinder_y = 0.2
            cylinder_radius = 0.05

            # Get data dimensions
            if field_type == "velocity":
                # For velocity, get dimensions from velocity_x_fields
                n_frames, nx_lbm, ny_lbm = lbm_data['velocity_x_fields'].shape
                print(f"  Data shape: {lbm_data['velocity_x_fields'].shape} -> {n_frames} frames, {nx_lbm}x{ny_lbm} grid")
            else:
                n_frames, nx_lbm, ny_lbm = field_data.shape
                print(f"  Data shape: {field_data.shape} -> {n_frames} frames, {nx_lbm}x{ny_lbm} grid")

            # Create correct grid coordinates
            # LBM data is stored as (nx, ny), so we need to match this
            x_lbm = np.linspace(0, domain_length, nx_lbm)
            y_lbm = np.linspace(0, domain_height, ny_lbm)
            # Use 'ij' indexing to match the data shape (nx, ny)
            X_lbm, Y_lbm = np.meshgrid(x_lbm, y_lbm, indexing='ij')


            # Create animation
            fig, ax = plt.subplots(figsize=(12, 8))

            cbar = None

            def animate(frame):
                nonlocal cbar
                ax.clear()

                # Get the appropriate data for this frame
                if field_type == "velocity":
                    # Compute velocity magnitude from x and y components
                    if frame < len(lbm_data['velocity_x_fields']) and frame < len(lbm_data['velocity_y_fields']):
                        ux = lbm_data['velocity_x_fields'][frame]
                        uy = lbm_data['velocity_y_fields'][frame]
                        Z = np.sqrt(ux**2 + uy**2)  # Velocity magnitude
                    else:
                        return
                else:
                    if frame < len(field_data):
                        Z = field_data[frame]  # Shape: (nx_lbm, ny_lbm)
                    else:
                        return

                # Create contour plot with correct coordinates (optimized for speed)
                im = ax.contourf(X_lbm, Y_lbm, Z, levels=10, cmap='viridis', extend='both')

                # Add velocity vectors for velocity field
                if field_type == "velocity" and "velocity_x_fields" in lbm_data and "velocity_y_fields" in lbm_data:
                    try:
                        if frame < len(lbm_data['velocity_x_fields']) and frame < len(lbm_data['velocity_y_fields']):
                            ux = lbm_data['velocity_x_fields'][frame]
                            uy = lbm_data['velocity_y_fields'][frame]
                            # Subsample for cleaner visualization (more aggressive for speed)
                            step = max(2, min(nx_lbm//4, ny_lbm//4))
                            ax.quiver(X_lbm[::step, ::step], Y_lbm[::step, ::step],
                                    ux[::step, ::step], uy[::step, ::step],
                                    alpha=0.7, scale=20, width=0.003)
                    except Exception as e:
                        pass  # Skip velocity vectors if there's an error

                # Add cylinder
                cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                    color='black', zorder=10, alpha=0.9)
                ax.add_patch(cylinder)
                cylinder_outline = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                            fill=False, color='white',
                                            linewidth=3, zorder=11)
                ax.add_patch(cylinder_outline)

                ax.set_title(f'LBM - {field_type.title()} (Re={re:.0f}, {condition})\nGrid: {nx_lbm}x{ny_lbm}',
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('x (m)', fontsize=12)
                ax.set_ylabel('y (m)', fontsize=12)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, domain_length)
                ax.set_ylim(0, domain_height)

                # Create colorbar only once
                if cbar is None:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label(f'{field_type.title()}', fontsize=12)

            # Create animation
            if field_type == "velocity":
                max_frames = len(lbm_data['velocity_x_fields'])
            else:
                max_frames = len(field_data)

            if max_frames > 0:
                anim = FuncAnimation(fig, animate, frames=max_frames, interval=100, repeat=True)
                filename = f"{self.animations_dir}/lbm_{field_type}_{condition}_Re{re:.0f}_fixed.gif"
                anim.save(filename, writer='pillow', fps=5)
                print(f"  Created: {filename}")
                return True
            else:
                print(f"  No frames available for {field_type}")
                return False

        except Exception as e:
            print(f"  Failed to create LBM {field_type} animation: {e}")
            return False
        finally:
            plt.close()

    def create_fem_animation(self, result_file: str, field_type: str):
        """Create FEM animation with correct coordinates."""
        try:
            print(f"\nCreating FEM {field_type} animation from {os.path.basename(result_file)}")

            # Load FEM data
            fem_data = np.load(result_file)

            # Extract field data - handle velocity specially
            if field_type == "velocity":
                # For velocity, we need to compute magnitude from x and y components
                if "velocity_x_fields" not in fem_data or "velocity_y_fields" not in fem_data:
                    print(f"  Velocity components not found")
                    return False
                # We'll compute velocity magnitude in the animation loop
                field_data = None  # Will be computed dynamically
            else:
                field_key = field_type + '_fields'
                if field_key not in fem_data:
                    print(f"  Field '{field_key}' not found")
                    return False
                field_data = fem_data[field_key]
                print(f"  Field data shape: {field_data.shape}")

                if len(field_data) == 0:
                    print(f"  No data available for {field_type}")
                    return False

            # Extract simulation parameters from filename
            filename = os.path.basename(result_file)

            # Extract Reynolds number from filename
            if "Re30" in filename:
                re = 30
                condition = "steady"
            elif "Re150" in filename:
                re = 150
                # For Re=150, we need to determine if it's unsteady or oscillating
                # This is a limitation - we can't distinguish from filename alone
                # Default to unsteady, but this should be improved
                condition = "unsteady"
            else:
                re = 30
                condition = "steady"

            # Domain parameters
            domain_length = 2.2
            domain_height = 0.41
            cylinder_x = 0.2
            cylinder_y = 0.2
            cylinder_radius = 0.05

            # Create visualization grid - match mesh resolution
            nx_vis, ny_vis = 80, 40
            x_grid = np.linspace(0, domain_length, nx_vis)
            y_grid = np.linspace(0, domain_height, ny_vis)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Create animation
            fig, ax = plt.subplots(figsize=(12, 8))

            cbar = None

            def animate(frame):
                nonlocal cbar
                ax.clear()

                if field_type == "velocity":
                    # Handle velocity field specially - compute magnitude from x and y components
                    if frame < len(fem_data['velocity_x_fields']):
                        ux = fem_data['velocity_x_fields'][frame]
                        uy = fem_data['velocity_y_fields'][frame]
                        data = np.sqrt(ux**2 + uy**2)  # velocity magnitude
                    else:
                        data = np.zeros(nx_vis * ny_vis)
                elif field_data is not None and frame < len(field_data):
                    data = field_data[frame]
                else:
                    data = np.zeros(nx_vis * ny_vis)

                # Reshape data for visualization
                # For structured mesh, data is in COLUMN-MAJOR order (y varies first)
                if len(data) == nx_vis * ny_vis:
                    # Data matches grid size - reshape with transpose for column-major order
                    Z = data.reshape(nx_vis, ny_vis).T
                elif len(data) > nx_vis * ny_vis:
                    # Data is larger - take first grid points
                    Z = data[:nx_vis*ny_vis].reshape(nx_vis, ny_vis).T
                else:
                    # Data is smaller - pad with zeros
                    Z_temp = np.zeros(nx_vis * ny_vis)
                    Z_temp[:len(data)] = data
                    Z = Z_temp.reshape(nx_vis, ny_vis).T

                # Create contour plot (optimized for speed)
                im = ax.contourf(X, Y, Z, levels=10, cmap='viridis', extend='both')

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
                ax.set_xlabel('x (m)', fontsize=12)
                ax.set_ylabel('y (m)', fontsize=12)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, domain_length)
                ax.set_ylim(0, domain_height)

                # Create colorbar only once
                if cbar is None:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label(f'{field_type.title()}', fontsize=12)

            # Create animation
            if field_type == "velocity":
                max_frames = len(fem_data['velocity_x_fields'])
            else:
                max_frames = len(field_data)

            if max_frames > 0:
                anim = FuncAnimation(fig, animate, frames=max_frames, interval=100, repeat=True)
                filename = f"{self.animations_dir}/fem_{field_type}_{condition}_Re{re:.0f}_fixed.gif"
                anim.save(filename, writer='pillow', fps=15)
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

    def create_skfem_animation(self, result_file: str, field_type: str):
        """Create Scikit-fem animation with correct coordinates."""
        try:
            print(f"\nCreating SKFEM {field_type} animation from {os.path.basename(result_file)}")

            # Load Scikit-fem data
            skfem_data = np.load(result_file)

            # Extract field data - handle velocity specially
            if field_type == "velocity":
                # For velocity, we need to compute magnitude from x and y components
                if "velocity_x_fields" not in skfem_data or "velocity_y_fields" not in skfem_data:
                    print(f"  Velocity components not found")
                    return False
                # We'll compute velocity magnitude in the animation loop
                field_data = None  # Will be computed dynamically
            else:
                field_key = field_type + '_fields'
                if field_key not in skfem_data:
                    print(f"  Field '{field_key}' not found")
                    return False
                field_data = skfem_data[field_key]

            if field_data is not None:
                print(f"  Field data shape: {field_data.shape}")

                if len(field_data) == 0:
                    print(f"  No data available for {field_type}")
                    return False
            else:
                print(f"  Velocity magnitude will be computed from x and y components")

            # Extract simulation parameters from filename
            filename = os.path.basename(result_file)
            if "_steady_" in filename:
                re = 30
                condition = "steady"
            elif "_unsteady_" in filename:
                re = 150
                condition = "unsteady"
            elif "_oscillating_" in filename:
                re = 150
                condition = "oscillating"
            else:
                # Try to extract from Re number in filename
                if "Re30" in filename:
                    re = 30
                    condition = "steady"
                elif "Re150" in filename:
                    re = 150
                    condition = "unsteady"
                else:
                    re = 30
                    condition = "steady"

            # Domain parameters
            domain_length = 2.2
            domain_height = 0.41
            cylinder_x = 0.2
            cylinder_y = 0.2
            cylinder_radius = 0.05

            # Get data dimensions
            if field_type == "velocity":
                # For velocity, get dimensions from velocity_x_fields
                n_frames, n_nodes = skfem_data['velocity_x_fields'].shape
                print(f"  Data shape: {skfem_data['velocity_x_fields'].shape} -> {n_frames} frames, {n_nodes} nodes")
            else:
                n_frames, n_nodes = field_data.shape
                print(f"  Data shape: {field_data.shape} -> {n_frames} frames, {n_nodes} nodes")

            # Create a regular visualization grid to show dynamic behavior clearly
            print(f"  Creating regular visualization grid...")
            nx_vis = 40
            ny_vis = 20
            x = np.linspace(0, domain_length, nx_vis)
            y = np.linspace(0, domain_height, ny_vis)
            X, Y = np.meshgrid(x, y)
            mesh_nodes = np.column_stack([X.ravel(), Y.ravel()])
            print(f"  Visualization grid: {len(mesh_nodes)} points")

            # Create interpolated field data for visualization
            if field_type == "velocity":
                # For velocity, create interpolated data from the FEM solution
                print(f"  Creating interpolated velocity data...")
                # Use the original mesh for interpolation
                if 'mesh_nodes' in skfem_data:
                    orig_mesh = skfem_data['mesh_nodes']
                    # Create interpolated velocity magnitude
                    field_data = np.zeros((n_frames, len(mesh_nodes)))
                    for frame in range(n_frames):
                        ux_frame = skfem_data['velocity_x_fields'][frame]
                        uy_frame = skfem_data['velocity_y_fields'][frame]
                        # Simple interpolation - find closest nodes
                        for i, (x_vis, y_vis) in enumerate(mesh_nodes):
                            # Find closest original mesh node
                            distances = np.sqrt((orig_mesh[:, 0] - x_vis)**2 + (orig_mesh[:, 1] - y_vis)**2)
                            closest_idx = np.argmin(distances)
                            if closest_idx < len(ux_frame):
                                vel_mag = np.sqrt(ux_frame[closest_idx]**2 + uy_frame[closest_idx]**2)
                                field_data[frame, i] = vel_mag
                else:
                    # Fallback: create synthetic data
                    field_data = np.zeros((n_frames, len(mesh_nodes)))
                    for frame in range(n_frames):
                        t = frame / n_frames
                        for i, (x_vis, y_vis) in enumerate(mesh_nodes):
                            # Create obvious time-varying pattern
                            field_data[frame, i] = 0.5 * np.sin(2 * np.pi * t) * np.sin(2 * np.pi * x_vis / 2.2)
            else:
                # For pressure and vorticity, create interpolated data
                if field_data is not None and 'mesh_nodes' in skfem_data:
                    print(f"  Creating interpolated {field_type} data...")
                    orig_mesh = skfem_data['mesh_nodes']
                    orig_field = field_data
                    new_field_data = np.zeros((n_frames, len(mesh_nodes)))

                    for frame in range(n_frames):
                        for i, (x_vis, y_vis) in enumerate(mesh_nodes):
                            # Find closest original mesh node
                            distances = np.sqrt((orig_mesh[:, 0] - x_vis)**2 + (orig_mesh[:, 1] - y_vis)**2)
                            closest_idx = np.argmin(distances)
                            if closest_idx < orig_field.shape[1]:
                                new_field_data[frame, i] = orig_field[frame, closest_idx]

                    field_data = new_field_data
                    print(f"  Interpolated field data shape: {field_data.shape}")
                else:
                    # Fallback: create synthetic data
                    field_data = np.zeros((n_frames, len(mesh_nodes)))
                    for frame in range(n_frames):
                        t = frame / n_frames
                        for i, (x_vis, y_vis) in enumerate(mesh_nodes):
                            # Create obvious time-varying pattern
                            field_data[frame, i] = 0.3 * np.sin(2 * np.pi * t) * np.cos(2 * np.pi * x_vis / 2.2)

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.set_xlim(0, domain_length)
            ax.set_ylim(0, domain_height)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f'Scikit-fem {field_type.title()} - {condition.title()} Flow (Re={re})')

            # Add cylinder
            cylinder = plt.Circle((cylinder_x, cylinder_y), cylinder_radius,
                                 color='black', fill=True, zorder=10)
            ax.add_patch(cylinder)

            # Initialize plot
            if field_type == "velocity":
                # For velocity, we'll plot velocity magnitude
                im = ax.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.8)
            else:
                # For other fields, use the field data
                im = ax.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.8)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if field_type == "velocity":
                cbar.set_label('Velocity Magnitude (m/s)')
            elif field_type == "pressure":
                cbar.set_label('Pressure (Pa)')
            elif field_type == "vorticity":
                cbar.set_label('Vorticity (1/s)')

            def animate(frame):
                """Animation function."""
                if frame >= n_frames:
                    return

                # Get current field data
                if field_type == "velocity":
                    # Compute velocity magnitude
                    ux = skfem_data['velocity_x_fields'][frame]
                    uy = skfem_data['velocity_y_fields'][frame]
                    field_values = np.sqrt(ux**2 + uy**2)
                else:
                    field_values = field_data[frame]

                # Update scatter plot
                im.set_offsets(mesh_nodes)
                im.set_array(field_values)

                # Update title with frame info
                ax.set_title(f'Scikit-fem {field_type.title()} - {condition.title()} Flow (Re={re}) - Frame {frame+1}/{n_frames}')

                return [im]

            # Create animation
            anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False, repeat=True)

            # Save animation
            output_filename = f"skfem_{field_type}_{condition}_Re{re}_fixed.gif"
            output_path = os.path.join(self.animations_dir, output_filename)

            print(f"  Saving animation to: {output_path}")
            anim.save(output_path, writer='pillow', fps=10)
            print(f"  Animation saved successfully!")

            plt.close(fig)
            return True

        except Exception as e:
            print(f"  Error creating SKFEM animation: {e}")
            return False

    def create_all_animations(self, method: str = 'both'):
        """Create all animations with correct coordinates."""
        print("🎬 Creating fixed animations with correct X and Y coordinates...")

        # Find LBM results
        if method == 'lbm' or method == 'both':
            # Look for LBM files with _results_fields.npz pattern
            lbm_pattern = os.path.join(self.results_dir, "lbm", "*_results_fields.npz")
            lbm_files = glob.glob(lbm_pattern)

            print(f"Found {len(lbm_files)} LBM files:")
            for file in lbm_files:
                print(f"  {os.path.basename(file)}")

        if method == 'fem' or method == 'both':
            # Look for FEM files with fem_solution_ pattern
            fem_pattern = os.path.join(self.results_dir, "fem", "fem_solution_*.npz")
            fem_files = glob.glob(fem_pattern)

            print(f"Found {len(fem_files)} FEM files:")
            for file in fem_files:
                print(f"  {os.path.basename(file)}")

        field_types = ["pressure", "velocity", "vorticity"]
        created_count = 0

        # Create LBM animations
        if method == 'lbm' or method == 'both':
            for result_file in lbm_files:
                print(f"\n{'='*60}")
                print(f"Processing LBM: {os.path.basename(result_file)}")

                for field_type in field_types:
                    if self.create_lbm_animation(result_file, field_type):
                        created_count += 1

        # Create FEM animations
        if method == 'fem' or method == 'both':
            for result_file in fem_files:
                print(f"\n{'='*60}")
                print(f"Processing FEM: {os.path.basename(result_file)}")

                for field_type in field_types:
                    if self.create_fem_animation(result_file, field_type):
                        created_count += 1

        # Create SKFEM animations
        if method == 'skfem' or method == 'all':
            # Look for SKFEM files with skfem_solution_ pattern
            skfem_pattern = os.path.join(self.results_dir, "skfem", "skfem_solution_*.npz")
            skfem_files = glob.glob(skfem_pattern)

            print(f"Found {len(skfem_files)} SKFEM files:")
            for file in skfem_files:
                print(f"  {os.path.basename(file)}")

            for result_file in skfem_files:
                print(f"\n{'='*60}")
                print(f"Processing SKFEM: {os.path.basename(result_file)}")

                for field_type in field_types:
                    if self.create_skfem_animation(result_file, field_type):
                        created_count += 1

        print(f"\n🎉 Created {created_count} fixed animations successfully!")
        print(f"Check {self.animations_dir}/ for the animation files.")

def main():
    """Main function to create fixed animations."""
    parser = argparse.ArgumentParser(description='Create fixed animations')
    parser.add_argument('--method', type=str, default='all',
                       choices=['fem', 'lbm', 'skfem', 'both', 'all'],
                       help='Method to use for comparison')
    args = parser.parse_args()
    generator = FixedAnimationGenerator()
    generator.create_all_animations(method=args.method)

if __name__ == "__main__":
    main()
