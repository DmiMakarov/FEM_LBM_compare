"""
Generate proper FEM mesh for cylinder flow simulation.
Creates structured triangular mesh with adequate cylinder resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os

def generate_fem_mesh(nx=80, ny=40, domain_length=2.2, domain_height=0.41,
                     cylinder_diameter=0.1, cylinder_x=0.2, cylinder_y=0.2,
                     cylinder_resolution=16):
    """
    Generate structured triangular mesh for FEM cylinder flow.

    Args:
        nx, ny: Grid resolution
        domain_length, domain_height: Physical domain size
        cylinder_diameter: Cylinder diameter
        cylinder_x, cylinder_y: Cylinder center position
        cylinder_resolution: Number of nodes around cylinder circumference

    Returns:
        Dictionary with mesh data
    """
    print(f"Generating FEM mesh: {nx}x{ny} grid")
    print(f"Domain: {domain_length}x{domain_height}")
    print(f"Cylinder: D={cylinder_diameter} at ({cylinder_x}, {cylinder_y})")
    print(f"Cylinder resolution: {cylinder_resolution} nodes")

    # Create structured grid
    x = np.linspace(0, domain_length, nx)
    y = np.linspace(0, domain_height, ny)
    X, Y = np.meshgrid(x, y)

    # Flatten to get all grid points
    points = np.column_stack([X.ravel(), Y.ravel()])

    # Identify cylinder region
    cylinder_radius = cylinder_diameter / 2
    cylinder_mask = ((points[:, 0] - cylinder_x)**2 + (points[:, 1] - cylinder_y)**2) <= cylinder_radius**2

    # Remove points inside cylinder
    points = points[~cylinder_mask]

    # Add cylinder boundary points
    cylinder_angles = np.linspace(0, 2*np.pi, cylinder_resolution, endpoint=False)
    cylinder_points = np.column_stack([
        cylinder_x + cylinder_radius * np.cos(cylinder_angles),
        cylinder_y + cylinder_radius * np.sin(cylinder_angles)
    ])

    # Combine all points
    all_points = np.vstack([points, cylinder_points])

    # Create Delaunay triangulation
    tri = Delaunay(all_points)

    # Identify boundary nodes
    boundary_nodes = []
    inlet_nodes = []
    outlet_nodes = []
    wall_nodes = []
    cylinder_nodes = []

    for i, (px, py) in enumerate(all_points):
        # Inlet (left boundary)
        if abs(px) < 1e-6:
            inlet_nodes.append(i)
            boundary_nodes.append(i)

        # Outlet (right boundary)
        elif abs(px - domain_length) < 1e-6:
            outlet_nodes.append(i)
            boundary_nodes.append(i)

        # Walls (top and bottom boundaries)
        elif abs(py) < 1e-6 or abs(py - domain_height) < 1e-6:
            wall_nodes.append(i)
            boundary_nodes.append(i)

        # Cylinder boundary
        elif ((px - cylinder_x)**2 + (py - cylinder_y)**2 - cylinder_radius**2) < 1e-6:
            cylinder_nodes.append(i)
            boundary_nodes.append(i)

    # Create mesh data dictionary
    mesh_data = {
        'nodes': all_points,
        'elements': tri.simplices,
        'boundary_nodes': np.array(boundary_nodes),
        'inlet_nodes': np.array(inlet_nodes),
        'outlet_nodes': np.array(outlet_nodes),
        'wall_nodes': np.array(wall_nodes),
        'cylinder_nodes': np.array(cylinder_nodes),
        'nx': nx,
        'ny': ny,
        'domain_length': domain_length,
        'domain_height': domain_height,
        'cylinder_diameter': cylinder_diameter,
        'cylinder_x': cylinder_x,
        'cylinder_y': cylinder_y,
        'cylinder_resolution': cylinder_resolution
    }

    print(f"Mesh generated:")
    print(f"  Total nodes: {len(all_points)}")
    print(f"  Total elements: {len(tri.simplices)}")
    print(f"  Inlet nodes: {len(inlet_nodes)}")
    print(f"  Outlet nodes: {len(outlet_nodes)}")
    print(f"  Wall nodes: {len(wall_nodes)}")
    print(f"  Cylinder nodes: {len(cylinder_nodes)}")

    return mesh_data

def visualize_mesh(mesh_data, save_path=None):
    """Visualize the generated mesh."""
    nodes = mesh_data['nodes']
    elements = mesh_data['elements']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot elements
    for element in elements:
        triangle = nodes[element]
        triangle = np.vstack([triangle, triangle[0]])  # Close triangle
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.3, linewidth=0.5)

    # Plot boundary nodes
    ax.scatter(nodes[mesh_data['inlet_nodes'], 0],
               nodes[mesh_data['inlet_nodes'], 1],
               c='blue', s=20, label='Inlet', alpha=0.7)

    ax.scatter(nodes[mesh_data['outlet_nodes'], 0],
               nodes[mesh_data['outlet_nodes'], 1],
               c='red', s=20, label='Outlet', alpha=0.7)

    ax.scatter(nodes[mesh_data['wall_nodes'], 0],
               nodes[mesh_data['wall_nodes'], 1],
               c='green', s=20, label='Walls', alpha=0.7)

    ax.scatter(nodes[mesh_data['cylinder_nodes'], 0],
               nodes[mesh_data['cylinder_nodes'], 1],
               c='black', s=30, label='Cylinder', alpha=0.8)

    # Add cylinder circle
    cylinder_circle = plt.Circle((mesh_data['cylinder_x'], mesh_data['cylinder_y']),
                                mesh_data['cylinder_diameter']/2,
                                fill=False, color='black', linewidth=2)
    ax.add_patch(cylinder_circle)

    ax.set_xlim(0, mesh_data['domain_length'])
    ax.set_ylim(0, mesh_data['domain_height'])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'FEM Mesh: {mesh_data["nx"]}x{mesh_data["ny"]} grid')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Mesh visualization saved to: {save_path}")

    plt.show()

def main():
    """Generate different mesh sizes for FEM."""
    mesh_sizes = {
        'coarse': {'nx': 40, 'ny': 20, 'cylinder_resolution': 8},
        'medium': {'nx': 60, 'ny': 30, 'cylinder_resolution': 12},
        'fine': {'nx': 80, 'ny': 40, 'cylinder_resolution': 16}
    }

    # Create meshes directory
    os.makedirs('meshes', exist_ok=True)

    for size_name, params in mesh_sizes.items():
        print(f"\n{'='*50}")
        print(f"Generating {size_name} mesh")
        print(f"{'='*50}")

        # Generate mesh
        mesh_data = generate_fem_mesh(**params)

        # Save mesh data
        mesh_file = f'meshes/fem_{size_name}_mesh_data.npz'
        np.savez(mesh_file, **mesh_data)
        print(f"Mesh saved to: {mesh_file}")

        # Visualize mesh
        viz_file = f'meshes/fem_{size_name}_mesh_visualization.png'
        visualize_mesh(mesh_data, viz_file)

        print(f"Generated {size_name} mesh with {len(mesh_data['nodes'])} nodes")

if __name__ == "__main__":
    main()
