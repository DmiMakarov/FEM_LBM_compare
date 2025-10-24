"""
Simplified mesh generation for cylinder flow without gmsh GUI dependencies.
Creates a structured mesh for the cylinder domain.
"""

import numpy as np
import os
from typing import Dict, List, Tuple


class SimpleMeshGenerator:
    """
    Generate a simple structured mesh for 2D cylinder flow.
    """

    def __init__(self, domain_length: float = 2.2, domain_height: float = 0.41,
                 cylinder_diameter: float = 0.1, cylinder_x: float = 0.2,
                 cylinder_y: float = 0.2, nx: int = 60, ny: int = 30):
        """
        Initialize mesh generator.

        Args:
            domain_length: Length of computational domain
            domain_height: Height of computational domain
            cylinder_diameter: Diameter of cylinder
            cylinder_x, cylinder_y: Position of cylinder center
            nx, ny: Grid resolution
        """
        self.domain_length = domain_length
        self.domain_height = domain_height
        self.cylinder_diameter = cylinder_diameter
        self.cylinder_x = cylinder_x
        self.cylinder_y = cylinder_y
        self.nx = nx
        self.ny = ny
        self.cylinder_radius = cylinder_diameter / 2

    def generate_mesh(self) -> Dict:
        """
        Generate structured mesh.

        Returns:
            Dictionary with mesh data
        """
        print(f"Generating structured mesh: {self.nx} x {self.ny}")

        # Create structured grid
        x = np.linspace(0, self.domain_length, self.nx)
        y = np.linspace(0, self.domain_height, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Flatten to get node coordinates
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        # Create triangular elements
        elements = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                # Get node indices
                n1 = i * self.ny + j
                n2 = (i + 1) * self.ny + j
                n3 = (i + 1) * self.ny + (j + 1)
                n4 = i * self.ny + (j + 1)

                # Create two triangles per quad
                elements.append([n1, n2, n3])
                elements.append([n1, n3, n4])

        # Identify boundary nodes
        boundary_nodes = self._get_boundary_nodes(nodes)
        cylinder_nodes = self._get_cylinder_nodes(nodes)
        inlet_nodes = self._get_inlet_nodes(nodes)
        outlet_nodes = self._get_outlet_nodes(nodes)

        mesh_data = {
            'nodes': nodes,
            'elements': elements,
            'boundary_nodes': boundary_nodes,
            'cylinder_nodes': cylinder_nodes,
            'inlet_nodes': inlet_nodes,
            'outlet_nodes': outlet_nodes
        }

        print(f"Generated mesh:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Elements: {len(elements)}")
        print(f"  Cylinder nodes: {len(cylinder_nodes)}")
        print(f"  Inlet nodes: {len(inlet_nodes)}")
        print(f"  Outlet nodes: {len(outlet_nodes)}")

        return mesh_data

    def _get_boundary_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get all boundary nodes."""
        boundary_nodes = []

        for i, (x, y) in enumerate(nodes):
            # Left boundary
            if abs(x) < 1e-6:
                boundary_nodes.append(i)
            # Right boundary
            elif abs(x - self.domain_length) < 1e-6:
                boundary_nodes.append(i)
            # Bottom boundary
            elif abs(y) < 1e-6:
                boundary_nodes.append(i)
            # Top boundary
            elif abs(y - self.domain_height) < 1e-6:
                boundary_nodes.append(i)

        return boundary_nodes

    def _get_cylinder_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get cylinder boundary nodes."""
        cylinder_nodes = []

        for i, (x, y) in enumerate(nodes):
            # Distance from cylinder center
            dist = np.sqrt((x - self.cylinder_x)**2 + (y - self.cylinder_y)**2)

            # Check if node is inside or on cylinder boundary
            if dist <= self.cylinder_radius + 0.01:  # Include nodes inside cylinder
                cylinder_nodes.append(i)

        return cylinder_nodes

    def _get_inlet_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get inlet boundary nodes."""
        inlet_nodes = []

        for i, (x, y) in enumerate(nodes):
            # Left boundary (inlet)
            if abs(x) < 1e-6:
                inlet_nodes.append(i)

        return inlet_nodes

    def _get_outlet_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get outlet boundary nodes."""
        outlet_nodes = []

        for i, (x, y) in enumerate(nodes):
            # Right boundary (outlet)
            if abs(x - self.domain_length) < 1e-6:
                outlet_nodes.append(i)

        return outlet_nodes

    def save_mesh(self, filename: str):
        """Save mesh to file."""
        mesh_data = self.generate_mesh()

        # Convert elements to numpy array for easier loading
        elements_array = np.array(mesh_data['elements'])

        # Save mesh data
        np.savez(filename,
                nodes=mesh_data['nodes'],
                elements=elements_array,
                boundary_nodes=np.array(mesh_data['boundary_nodes']),
                cylinder_nodes=np.array(mesh_data['cylinder_nodes']),
                inlet_nodes=np.array(mesh_data['inlet_nodes']),
                outlet_nodes=np.array(mesh_data['outlet_nodes']))

        print(f"Mesh saved to {filename}")


def generate_cylinder_mesh(reynolds_number: float = 100,
                          nx: int = 100, ny: int = 50,
                          output_dir: str = "meshes") -> str:
    """
    Generate mesh for cylinder flow simulation.

    Args:
        reynolds_number: Reynolds number (affects mesh resolution)
        nx, ny: Grid resolution
        output_dir: Directory to save mesh files

    Returns:
        Path to generated mesh file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Adjust mesh resolution based on Reynolds number
    if reynolds_number > 200:
        nx, ny = int(nx * 1.5), int(ny * 1.5)
    elif reynolds_number > 100:
        nx, ny = int(nx * 1.2), int(ny * 1.2)

    # Create mesh generator
    generator = SimpleMeshGenerator(nx=nx, ny=ny)

    # Generate mesh
    mesh_data = generator.generate_mesh()

    # Save mesh
    mesh_filename = f"cylinder_mesh_Re{reynolds_number:.0f}_data.npz"
    mesh_path = os.path.join(output_dir, mesh_filename)

    # Convert elements to numpy array for easier loading
    elements_array = np.array(mesh_data['elements'])

    # Save mesh data
    np.savez(mesh_path,
            nodes=mesh_data['nodes'],
            elements=elements_array,
            boundary_nodes=np.array(mesh_data['boundary_nodes']),
            cylinder_nodes=np.array(mesh_data['cylinder_nodes']),
            inlet_nodes=np.array(mesh_data['inlet_nodes']),
            outlet_nodes=np.array(mesh_data['outlet_nodes']))

    print(f"Mesh saved to: {mesh_path}")

    return mesh_path


def main():
    """Main function to generate meshes for different Reynolds numbers."""
    reynolds_numbers = [20, 40, 100, 200]

    for re in reynolds_numbers:
        print(f"\nGenerating mesh for Re = {re}")
        print("=" * 40)

        mesh_file = generate_cylinder_mesh(re, nx=100, ny=50)
        print(f"Mesh saved to: {mesh_file}")


if __name__ == "__main__":
    main()
