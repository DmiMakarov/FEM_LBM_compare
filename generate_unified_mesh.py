"""
Generate unified mesh for both FEM and LBM with the same number of points.
Creates a 100x25 grid (2,500 points) for both methods.
"""

import numpy as np
import os
from typing import Dict, List, Tuple


class UnifiedMeshGenerator:
    """
    Generate unified mesh for both FEM and LBM with the same number of points.
    """

    def __init__(self, domain_length: float = 2.2, domain_height: float = 0.41,
                 cylinder_diameter: float = 0.1, cylinder_x: float = 0.2,
                 cylinder_y: float = 0.2, nx: int = 100, ny: int = 25):
        """
        Initialize unified mesh generator.

        Args:
            domain_length: Length of computational domain
            domain_height: Height of computational domain
            cylinder_diameter: Diameter of cylinder
            cylinder_x, cylinder_y: Position of cylinder center
            nx, ny: Grid resolution (100x25 = 2,500 points)
        """
        self.domain_length = domain_length
        self.domain_height = domain_height
        self.cylinder_diameter = cylinder_diameter
        self.cylinder_x = cylinder_x
        self.cylinder_y = cylinder_y
        self.nx = nx
        self.ny = ny
        self.cylinder_radius = cylinder_diameter / 2

    def generate_unified_mesh(self) -> Dict:
        """
        Generate unified mesh for both FEM and LBM.

        Returns:
            Dictionary with mesh data
        """
        print(f"Generating unified mesh: {self.nx} x {self.ny} = {self.nx * self.ny} points")

        # Create structured grid
        x = np.linspace(0, self.domain_length, self.nx)
        y = np.linspace(0, self.domain_height, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Flatten to get node coordinates
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        # Create triangular elements for FEM
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
            'outlet_nodes': outlet_nodes,
            'nx': self.nx,
            'ny': self.ny,
            'domain_length': self.domain_length,
            'domain_height': self.domain_height,
            'cylinder_diameter': self.cylinder_diameter,
            'cylinder_x': self.cylinder_x,
            'cylinder_y': self.cylinder_y
        }

        print(f"Generated unified mesh:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Elements: {len(elements)}")
        print(f"  Cylinder nodes: {len(cylinder_nodes)}")
        print(f"  Inlet nodes: {len(inlet_nodes)}")
        print(f"  Outlet nodes: {len(outlet_nodes)}")
        print(f"  Grid resolution: {self.nx} x {self.ny}")

        return mesh_data

    def _get_boundary_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get all boundary nodes."""
        boundary_nodes = []
        for i, (x, y) in enumerate(nodes):
            if (x == 0 or x == self.domain_length or
                y == 0 or y == self.domain_height):
                boundary_nodes.append(i)
        return boundary_nodes

    def _get_cylinder_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get nodes on the cylinder surface."""
        cylinder_nodes = []
        for i, (x, y) in enumerate(nodes):
            # Check if node is inside or on the cylinder
            dist = np.sqrt((x - self.cylinder_x)**2 + (y - self.cylinder_y)**2)
            if dist <= self.cylinder_radius * 1.1:  # Slightly larger tolerance
                cylinder_nodes.append(i)
        return cylinder_nodes

    def _get_inlet_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get nodes on the inlet (x=0)."""
        inlet_nodes = []
        for i, (x, y) in enumerate(nodes):
            if abs(x) < 1e-10:  # x = 0
                inlet_nodes.append(i)
        return inlet_nodes

    def _get_outlet_nodes(self, nodes: np.ndarray) -> List[int]:
        """Get nodes on the outlet (x=domain_length)."""
        outlet_nodes = []
        for i, (x, y) in enumerate(nodes):
            if abs(x - self.domain_length) < 1e-10:  # x = domain_length
                outlet_nodes.append(i)
        return outlet_nodes

    def save_mesh(self, filename: str, mesh_data: Dict):
        """Save mesh data to file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **mesh_data)
        print(f"Unified mesh saved to: {filename}")


def generate_unified_meshes():
    """Generate unified meshes for different Reynolds numbers."""
    # Create unified mesh generator
    generator = UnifiedMeshGenerator(nx=100, ny=25)  # 2,500 points

    # Generate mesh
    mesh_data = generator.generate_unified_mesh()

    # Save mesh
    generator.save_mesh("meshes/unified_mesh_data.npz", mesh_data)

    return mesh_data


if __name__ == "__main__":
    generate_unified_meshes()
