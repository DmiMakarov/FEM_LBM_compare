"""
Visualization tools for LBM cylinder flow results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from typing import Dict, List, Tuple, Optional


class LBMVisualizer:
    """
    Visualization class for LBM cylinder flow results.
    """

    def __init__(self, results_file: str, fields_file: str = None):
        """
        Initialize visualizer with results.

        Args:
            results_file: Path to results .npz file
            fields_file: Path to fields .npz file (optional)
        """
        self.results = np.load(results_file)
        self.fields = np.load(fields_file) if fields_file else None

        # Extract basic info
        self.nx = int(self.results['nx'])
        self.ny = int(self.results['ny'])
        self.reynolds_number = float(self.results['reynolds_number'])
        self.cylinder_diameter = float(self.results['cylinder_diameter'])

        # Create coordinate arrays
        self.x = np.linspace(0, 2.2, self.nx)
        self.y = np.linspace(0, 0.41, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Cylinder position
        self.cylinder_x = 0.2
        self.cylinder_y = 0.2
        self.cylinder_radius = self.cylinder_diameter / 2

    def plot_pressure_contours(self, time_idx: int = -1,
                              save_path: str = None) -> plt.Figure:
        """
        Plot pressure contours.

        Args:
            time_idx: Time index to plot (-1 for last time)
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        if self.fields is None:
            raise ValueError("Fields data not available")

        pressure = self.fields['pressure_fields'][time_idx]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot pressure contours
        contour = ax.contourf(self.X, self.Y, pressure, levels=20, cmap='RdBu_r')
        ax.contour(self.X, self.Y, pressure, levels=20, colors='black', alpha=0.3, linewidths=0.5)

        # Add cylinder
        cylinder = plt.Circle((self.cylinder_x, self.cylinder_y),
                            self.cylinder_radius, color='black', zorder=10)
        ax.add_patch(cylinder)

        # Formatting
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Pressure Contours (Re={self.reynolds_number:.0f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Pressure')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_velocity_streamlines(self, time_idx: int = -1,
                                 save_path: str = None) -> plt.Figure:
        """
        Plot velocity streamlines.

        Args:
            time_idx: Time index to plot (-1 for last time)
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        if self.fields is None:
            raise ValueError("Fields data not available")

        ux = self.fields['velocity_x_fields'][time_idx]
        uy = self.fields['velocity_y_fields'][time_idx]

        # Compute velocity magnitude
        velocity_magnitude = np.sqrt(ux**2 + uy**2)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot velocity magnitude as background
        im = ax.contourf(self.X, self.Y, velocity_magnitude, levels=20, cmap='viridis')

        # Plot streamlines
        ax.streamplot(self.X, self.Y, ux, uy, density=2, color='white', alpha=0.7, linewidth=0.8)

        # Add cylinder
        cylinder = plt.Circle((self.cylinder_x, self.cylinder_y),
                            self.cylinder_radius, color='black', zorder=10)
        ax.add_patch(cylinder)

        # Formatting
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Velocity Streamlines (Re={self.reynolds_number:.0f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Velocity Magnitude')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_vorticity_field(self, time_idx: int = -1,
                           save_path: str = None) -> plt.Figure:
        """
        Plot vorticity field.

        Args:
            time_idx: Time index to plot (-1 for last time)
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        if self.fields is None:
            raise ValueError("Fields data not available")

        vorticity = self.fields['vorticity_fields'][time_idx]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot vorticity
        contour = ax.contourf(self.X, self.Y, vorticity, levels=30, cmap='RdBu_r')
        ax.contour(self.X, self.Y, vorticity, levels=30, colors='black', alpha=0.3, linewidths=0.5)

        # Add cylinder
        cylinder = plt.Circle((self.cylinder_x, self.cylinder_y),
                            self.cylinder_radius, color='black', zorder=10)
        ax.add_patch(cylinder)

        # Formatting
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Vorticity Field (Re={self.reynolds_number:.0f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Vorticity')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_force_history(self, save_path: str = None) -> plt.Figure:
        """
        Plot drag and lift coefficient history.

        Args:
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        time = self.results['time']
        drag = self.results['drag']
        lift = self.results['lift']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Drag coefficient
        ax1.plot(time, drag, 'b-', linewidth=2, label='Drag Coefficient')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Drag Coefficient')
        ax1.set_title(f'Drag Coefficient History (Re={self.reynolds_number:.0f})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Lift coefficient
        ax2.plot(time, lift, 'r-', linewidth=2, label='Lift Coefficient')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Lift Coefficient')
        ax2.set_title(f'Lift Coefficient History (Re={self.reynolds_number:.0f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_strouhal_analysis(self, save_path: str = None) -> plt.Figure:
        """
        Plot Strouhal number analysis with FFT.

        Args:
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        time = self.results['time']
        lift = self.results['lift']
        strouhal = self.results['strouhal']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Lift coefficient time series
        ax1.plot(time, lift, 'r-', linewidth=2)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Lift Coefficient')
        ax1.set_title('Lift Coefficient Time Series')
        ax1.grid(True, alpha=0.3)

        # Strouhal number evolution
        ax2.plot(time, strouhal, 'b-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Strouhal Number')
        ax2.set_title('Strouhal Number Evolution')
        ax2.grid(True, alpha=0.3)

        # FFT of lift coefficient
        if len(lift) > 100:
            # Remove mean and detrend
            lift_detrended = lift - np.mean(lift)

            # Compute FFT
            fft = np.fft.fft(lift_detrended)
            freqs = np.fft.fftfreq(len(lift_detrended))
            power = np.abs(fft)**2

            # Plot only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power[:len(power)//2]

            ax3.plot(positive_freqs, positive_power, 'g-', linewidth=2)
            ax3.set_xlabel('Frequency')
            ax3.set_ylabel('Power')
            ax3.set_title('FFT of Lift Coefficient')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 0.1)  # Focus on low frequencies

        # Final Strouhal number
        if strouhal[-1] > 0:
            ax4.bar(['Strouhal Number'], [strouhal[-1]], color='orange', alpha=0.7)
            ax4.set_ylabel('Strouhal Number')
            ax4.set_title(f'Final Strouhal Number: {strouhal[-1]:.4f}')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_animation(self, field_type: str = 'pressure',
                        save_path: str = None, fps: int = 10) -> None:
        """
        Create animation of field evolution.

        Args:
            field_type: Type of field ('pressure', 'velocity', 'vorticity')
            save_path: Path to save animation
            fps: Frames per second
        """
        if self.fields is None:
            raise ValueError("Fields data not available")

        if field_type == 'pressure':
            field_data = self.fields['pressure_fields']
            title = 'Pressure Field Evolution'
            cmap = 'RdBu_r'
        elif field_type == 'velocity':
            # Use velocity magnitude
            ux = self.fields['velocity_x_fields']
            uy = self.fields['velocity_y_fields']
            field_data = np.sqrt(ux**2 + uy**2)
            title = 'Velocity Magnitude Evolution'
            cmap = 'viridis'
        elif field_type == 'vorticity':
            field_data = self.fields['vorticity_fields']
            title = 'Vorticity Field Evolution'
            cmap = 'RdBu_r'
        else:
            raise ValueError("field_type must be 'pressure', 'velocity', or 'vorticity'")

        fig, ax = plt.subplots(figsize=(12, 6))

        def animate(frame):
            ax.clear()

            # Plot field
            contour = ax.contourf(self.X, self.Y, field_data[frame],
                                levels=20, cmap=cmap)

            # Add cylinder
            cylinder = plt.Circle((self.cylinder_x, self.cylinder_y),
                                self.cylinder_radius, color='black', zorder=10)
            ax.add_patch(cylinder)

            # Formatting
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{title} (Re={self.reynolds_number:.0f}, Frame {frame})')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            return contour,

        anim = FuncAnimation(fig, animate, frames=len(field_data),
                           interval=1000//fps, blit=False, repeat=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Animation saved to {save_path}")

        return anim

    def generate_all_plots(self, output_dir: str = "lbm_plots") -> None:
        """
        Generate all standard plots.

        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating plots for Re={self.reynolds_number:.0f}...")

        # Pressure contours
        self.plot_pressure_contours(save_path=f"{output_dir}/pressure_Re{self.reynolds_number:.0f}.png")

        # Velocity streamlines
        self.plot_velocity_streamlines(save_path=f"{output_dir}/velocity_Re{self.reynolds_number:.0f}.png")

        # Vorticity field
        self.plot_vorticity_field(save_path=f"{output_dir}/vorticity_Re{self.reynolds_number:.0f}.png")

        # Force history
        self.plot_force_history(save_path=f"{output_dir}/forces_Re{self.reynolds_number:.0f}.png")

        # Strouhal analysis
        self.plot_strouhal_analysis(save_path=f"{output_dir}/strouhal_Re{self.reynolds_number:.0f}.png")

        print(f"Plots saved to {output_dir}/")


def main():
    """Main function to visualize LBM results."""
    import glob

    # Find result files
    result_files = glob.glob("lbm_results_Re*.npz")
    field_files = glob.glob("lbm_results_Re*_fields.npz")

    if not result_files:
        print("No LBM result files found!")
        return

    # Create output directory
    output_dir = "lbm_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Process each result file
    for result_file in result_files:
        # Find corresponding field file
        base_name = result_file.replace('.npz', '')
        field_file = base_name + '_fields.npz'

        if not os.path.exists(field_file):
            print(f"Field file {field_file} not found, skipping...")
            continue

        print(f"Processing {result_file}...")

        # Create visualizer
        visualizer = LBMVisualizer(result_file, field_file)

        # Generate all plots
        visualizer.generate_all_plots(output_dir)

        # Create animation (optional)
        try:
            visualizer.create_animation('pressure',
                                      f"{output_dir}/pressure_animation_Re{visualizer.reynolds_number:.0f}.gif")
        except Exception as e:
            print(f"Animation creation failed: {e}")


if __name__ == "__main__":
    main()
