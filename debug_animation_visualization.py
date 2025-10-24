#!/usr/bin/env python3
"""
Debug animation visualization to see why animations appear static.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def debug_animation_data():
    """Debug the actual data being used in animations."""
    print("Debugging animation visualization...")

    # Load a scikit-fem result file
    result_file = "results/skfem/skfem_solution_Re150_unsteady.npz"

    if not os.path.exists(result_file):
        print(f"  Result file not found: {result_file}")
        return

    print(f"  Loading: {result_file}")
    data = np.load(result_file)

    # Check velocity data
    if 'velocity_x_fields' in data and 'velocity_y_fields' in data:
        ux_data = data['velocity_x_fields']
        uy_data = data['velocity_y_fields']

        print(f"  Velocity data shape: {ux_data.shape}")

        # Check first few time steps
        print("  First 5 time steps velocity ranges:")
        for i in range(min(5, ux_data.shape[0])):
            ux_min, ux_max = np.min(ux_data[i]), np.max(ux_data[i])
            uy_min, uy_max = np.min(uy_data[i]), np.max(uy_data[i])
            print(f"    Step {i}: ux=[{ux_min:.6f}, {ux_max:.6f}], uy=[{uy_min:.6f}, {uy_max:.6f}]")

        # Check if values are actually changing
        ux_diff = np.max(ux_data) - np.min(ux_data)
        uy_diff = np.max(uy_data) - np.min(uy_data)
        print(f"  Total velocity range: ux={ux_diff:.6f}, uy={uy_diff:.6f}")

        # Check specific nodes over time
        print("  Velocity at first 5 nodes over time:")
        for node in range(min(5, ux_data.shape[1])):
            ux_values = ux_data[:, node]
            uy_values = uy_data[:, node]
            print(f"    Node {node}: ux range={np.max(ux_values)-np.min(ux_values):.6f}, uy range={np.max(uy_values)-np.min(uy_values):.6f}")

    # Check pressure data
    if 'pressure_fields' in data:
        p_data = data['pressure_fields']
        print(f"  Pressure data shape: {p_data.shape}")

        # Check first few time steps
        print("  First 5 time steps pressure ranges:")
        for i in range(min(5, p_data.shape[0])):
            p_min, p_max = np.min(p_data[i]), np.max(p_data[i])
            print(f"    Step {i}: p=[{p_min:.6f}, {p_max:.6f}]")

        p_diff = np.max(p_data) - np.min(p_data)
        print(f"  Total pressure range: {p_diff:.6f}")

    # Check mesh data
    if 'mesh_nodes' in data:
        mesh_nodes = data['mesh_nodes']
        print(f"  Mesh nodes shape: {mesh_nodes.shape}")
        print(f"  Mesh x range: [{np.min(mesh_nodes[:, 0]):.3f}, {np.max(mesh_nodes[:, 0]):.3f}]")
        print(f"  Mesh y range: [{np.min(mesh_nodes[:, 1]):.3f}, {np.max(mesh_nodes[:, 1]):.3f}]")

def create_test_animation():
    """Create a simple test animation to verify the animation system works."""
    print("\nCreating test animation...")

    # Create simple test data
    n_frames = 10
    n_nodes = 100

    # Create mesh
    x = np.linspace(0, 2.2, 10)
    y = np.linspace(0, 0.41, 10)
    X, Y = np.meshgrid(x, y)
    mesh_nodes = np.column_stack([X.ravel(), Y.ravel()])

    # Create time-varying data
    time = np.linspace(0, 1, n_frames)
    velocity_x = np.zeros((n_frames, n_nodes))
    velocity_y = np.zeros((n_frames, n_nodes))
    pressure = np.zeros((n_frames, n_nodes))

    for i in range(n_frames):
        for j in range(n_nodes):
            x, y = mesh_nodes[j]
            t = time[i]

            # Create obvious time-varying patterns
            velocity_x[i, j] = 0.5 * np.sin(2 * np.pi * t) * np.sin(2 * np.pi * x / 2.2)
            velocity_y[i, j] = 0.3 * np.cos(2 * np.pi * t) * np.sin(2 * np.pi * y / 0.41)
            pressure[i, j] = 0.2 * np.sin(2 * np.pi * t) * np.cos(2 * np.pi * x / 2.2)

    # Save test data
    test_file = "test_animation_data.npz"
    np.savez(test_file,
             velocity_x_fields=velocity_x,
             velocity_y_fields=velocity_y,
             pressure_fields=pressure,
             mesh_nodes=mesh_nodes,
             time=time)

    print(f"  Test data saved to: {test_file}")

    # Create simple animation
    fig, ax = plt.subplots(figsize=(10, 4))

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 0.41)
        ax.set_title(f'Test Animation - Frame {frame+1}/{n_frames}')

        # Plot velocity magnitude
        vel_mag = np.sqrt(velocity_x[frame]**2 + velocity_y[frame]**2)
        scatter = ax.scatter(mesh_nodes[:, 0], mesh_nodes[:, 1], c=vel_mag, cmap='viridis', s=50)

        return [scatter]

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=200, repeat=True)

    output_file = "test_animation.gif"
    anim.save(output_file, writer='pillow', fps=5)
    print(f"  Test animation saved to: {output_file}")

    plt.close()

def main():
    """Main function."""
    print("=" * 60)
    print("DEBUGGING ANIMATION VISUALIZATION")
    print("=" * 60)

    debug_animation_data()
    create_test_animation()

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    print("Check the test animation to see if the animation system works.")
    print("If the test animation shows movement, the issue is with the data.")
    print("If the test animation is also static, the issue is with the animation system.")

if __name__ == "__main__":
    main()
