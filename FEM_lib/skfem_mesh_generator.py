"""
Mesh generator for cylinder flow using scikit-fem.

Creates a rectangular domain with a circular cylinder using scikit-fem's
mesh generation capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from skfem import MeshTri
from skfem.visuals.matplotlib import draw, plot
import meshio


class SkfemMeshGenerator:
    """
    Generate mesh for cylinder flow simulation using scikit-fem.
    """

    def __init__(self, domain_length: float = 2.2, domain_height: float = 0.41,
                 cylinder_diameter: float = 0.1, cylinder_x: float = 0.2,
                 cylinder_y: float = 0.2, mesh_density: str = "medium"):
        """
        Initialize mesh generator.

        Args:
            domain_length: Length of the domain (m)
            domain_height: Height of the domain (m)
            cylinder_diameter: Diameter of the cylinder (m)
            cylinder_x: X-coordinate of cylinder center (m)
            cylinder_y: Y-coordinate of cylinder center (m)
            mesh_density: Mesh density ("coarse", "medium", "fine")
        """
        self.domain_length = domain_length
        self.domain_height = domain_height
        self.cylinder_diameter = cylinder_diameter
        self.cylinder_radius = cylinder_diameter / 2
        self.cylinder_x = cylinder_x
        self.cylinder_y = cylinder_y
        self.mesh_density = mesh_density

        # Mesh parameters based on density
        self._set_mesh_parameters()

        # Boundary markers
        self.boundary_markers = {
            'inlet': 1,
            'outlet': 2,
            'walls': 3,
            'cylinder': 4
        }

    def _set_mesh_parameters(self):
        """Set mesh parameters based on density."""
        if self.mesh_density == "coarse":
            self.nx = 40
            self.ny = 20
            self.cylinder_resolution = 16
        elif self.mesh_density == "medium":
            self.nx = 80
            self.ny = 40
            self.cylinder_resolution = 32
        elif self.mesh_density == "fine":
            self.nx = 160
            self.ny = 80
            self.cylinder_resolution = 64
        else:
            raise ValueError(f"Unknown mesh density: {self.mesh_density}")

    def generate_mesh(self) -> MeshTri:
        """
        Generate mesh for cylinder flow.

        Returns:
            scikit-fem MeshTri object
        """
        print(f"Generating {self.mesh_density} mesh...")
        print(f"  Domain: {self.domain_length}m × {self.domain_height}m")
        print(f"  Cylinder: diameter={self.cylinder_diameter}m at ({self.cylinder_x}, {self.cylinder_y})")
        print(f"  Grid: {self.nx} × {self.ny}")

        # Create structured mesh first
        mesh = self._create_structured_mesh()

        # Add cylinder hole
        mesh = self._add_cylinder_hole(mesh)

        # Store boundary information separately
        self.boundary_nodes = self._find_boundary_nodes(mesh)

        print(f"  Generated mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} elements")
        print(f"  Boundary nodes: inlet={len(self.boundary_nodes['inlet'])}, outlet={len(self.boundary_nodes['outlet'])}, "
              f"walls={len(self.boundary_nodes['walls'])}, cylinder={len(self.boundary_nodes['cylinder'])}")

        return mesh

    def _create_structured_mesh(self) -> MeshTri:
        """Create initial structured mesh."""
        # Create structured grid
        x = np.linspace(0, self.domain_length, self.nx + 1)
        y = np.linspace(0, self.domain_height, self.ny + 1)
        X, Y = np.meshgrid(x, y)

        # Create points
        points = np.column_stack([X.ravel(), Y.ravel()])

        # Create triangular elements
        elements = []
        for i in range(self.ny):
            for j in range(self.nx):
                # Bottom-left triangle
                n1 = i * (self.nx + 1) + j
                n2 = (i + 1) * (self.nx + 1) + j
                n3 = i * (self.nx + 1) + j + 1
                elements.append([n1, n2, n3])

                # Top-right triangle
                n1 = i * (self.nx + 1) + j + 1
                n2 = (i + 1) * (self.nx + 1) + j
                n3 = (i + 1) * (self.nx + 1) + j + 1
                elements.append([n1, n2, n3])

        elements = np.array(elements).T

        # Create mesh
        mesh = MeshTri(points.T, elements)

        return mesh

    def _add_cylinder_hole(self, mesh: MeshTri) -> MeshTri:
        """Add circular cylinder hole to the mesh."""
        # Find nodes inside cylinder
        cylinder_nodes = []
        for i, (x, y) in enumerate(mesh.p.T):
            dist = np.sqrt((x - self.cylinder_x)**2 + (y - self.cylinder_y)**2)
            if dist <= self.cylinder_radius:
                cylinder_nodes.append(i)

        if not cylinder_nodes:
            print("  Warning: No nodes found inside cylinder - mesh may be too coarse")
            return mesh

        # Create refined boundary around cylinder
        # This is a simplified approach - in practice, you'd want to use
        # a proper mesh refinement tool like gmsh or meshio

        # For now, we'll create a simple approximation
        # by removing elements that are entirely inside the cylinder
        elements_to_remove = []

        for i, element in enumerate(mesh.t.T):
            # Check if all nodes of element are inside cylinder
            nodes_inside = 0
            for node_idx in element:
                x, y = mesh.p[:, node_idx]
                dist = np.sqrt((x - self.cylinder_x)**2 + (y - self.cylinder_y)**2)
                if dist <= self.cylinder_radius:
                    nodes_inside += 1

            # Remove element if all nodes are inside cylinder
            if nodes_inside == 3:
                elements_to_remove.append(i)

        # Remove elements
        if elements_to_remove:
            keep_elements = [i for i in range(mesh.t.shape[1]) if i not in elements_to_remove]
            mesh.t = mesh.t[:, keep_elements]

        print(f"  Removed {len(elements_to_remove)} elements inside cylinder")

        return mesh

    def _mark_boundaries(self, mesh: MeshTri) -> MeshTri:
        """Mark boundary regions."""
        # Find boundary nodes
        boundary_nodes = self._find_boundary_nodes(mesh)

        # Store boundaries as a dictionary attribute using setattr
        setattr(mesh, 'boundaries', {
            'inlet': boundary_nodes['inlet'],
            'outlet': boundary_nodes['outlet'],
            'walls': boundary_nodes['walls'],
            'cylinder': boundary_nodes['cylinder']
        })

        print(f"  Boundary nodes: inlet={len(boundary_nodes['inlet'])}, outlet={len(boundary_nodes['outlet'])}, "
              f"walls={len(boundary_nodes['walls'])}, cylinder={len(boundary_nodes['cylinder'])}")

        return mesh

    def _find_boundary_nodes(self, mesh: MeshTri) -> Dict[str, List[int]]:
        """Find boundary nodes for different regions."""
        boundary_nodes = {
            'inlet': [],
            'outlet': [],
            'walls': [],
            'cylinder': []
        }

        for i, (x, y) in enumerate(mesh.p.T):
            # Inlet (x = 0)
            if abs(x) < 1e-10:
                boundary_nodes['inlet'].append(i)

            # Outlet (x = domain_length)
            elif abs(x - self.domain_length) < 1e-10:
                boundary_nodes['outlet'].append(i)

            # Walls (y = 0 or y = domain_height)
            elif abs(y) < 1e-10 or abs(y - self.domain_height) < 1e-10:
                boundary_nodes['walls'].append(i)

            # Cylinder (on cylinder boundary)
            else:
                dist = np.sqrt((x - self.cylinder_x)**2 + (y - self.cylinder_y)**2)
                if abs(dist - self.cylinder_radius) < 0.01:  # Tolerance for cylinder boundary
                    boundary_nodes['cylinder'].append(i)

        return boundary_nodes

    def visualize_mesh(self, mesh: MeshTri, save_path: str = None):
        """Visualize the generated mesh."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        # Plot mesh
        draw(mesh, ax=ax)

        # Mark boundaries
        if hasattr(mesh, 'boundaries'):
            colors = ['red', 'blue', 'green', 'orange']
            labels = ['Inlet', 'Outlet', 'Walls', 'Cylinder']

            for i, (boundary_name, nodes) in enumerate(mesh.boundaries.items()):
                if nodes:
                    x_coords = mesh.p[0, nodes]
                    y_coords = mesh.p[1, nodes]
                    ax.scatter(x_coords, y_coords, c=colors[i], s=20,
                              label=f'{labels[i]} ({len(nodes)} nodes)', alpha=0.7)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Cylinder Flow Mesh ({self.mesh_density})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Mesh visualization saved to: {save_path}")

        plt.show()

    def get_mesh_info(self, mesh: MeshTri) -> Dict:
        """Get mesh information."""
        info = {
            'n_nodes': mesh.p.shape[1],
            'n_elements': mesh.t.shape[1],
            'domain_length': self.domain_length,
            'domain_height': self.domain_height,
            'cylinder_diameter': self.cylinder_diameter,
            'cylinder_x': self.cylinder_x,
            'cylinder_y': self.cylinder_y,
            'mesh_density': self.mesh_density
        }

        if hasattr(self, 'boundary_nodes'):
            for boundary_name, nodes in self.boundary_nodes.items():
                info[f'{boundary_name}_nodes'] = len(nodes)

        return info

    def save_mesh(self, mesh: MeshTri, filename: str):
        """Save mesh to file."""
        # Convert to meshio format for saving
        points = mesh.p.T
        cells = [("triangle", mesh.t.T)]

        # Add boundary markers if available
        cell_data = {}
        if hasattr(mesh, 'boundaries'):
            for boundary_name, nodes in mesh.boundaries.items():
                if nodes:
                    cell_data[boundary_name] = [nodes]

        meshio_mesh = meshio.Mesh(points, cells, cell_data=cell_data)
        meshio_mesh.write(filename)

        print(f"  Mesh saved to: {filename}")


def main():
    """Test mesh generation."""
    # Test different mesh densities
    for density in ["coarse", "medium", "fine"]:
        print(f"\n=== Testing {density} mesh ===")

        generator = SkfemMeshGenerator(mesh_density=density)
        mesh = generator.generate_mesh()

        # Get mesh info
        info = generator.get_mesh_info(mesh)
        print(f"Mesh info: {info}")

        # Visualize
        generator.visualize_mesh(mesh, f"mesh_{density}.png")

        # Save mesh
        generator.save_mesh(mesh, f"mesh_{density}.msh")


if __name__ == "__main__":
    main()
