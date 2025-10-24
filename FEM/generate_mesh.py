"""
Generate mesh for cylinder flow using gmsh.
Creates a 2D domain with a circular cylinder.
"""

import gmsh
import numpy as np
import os
from typing import Tuple, List


class CylinderMeshGenerator:
    """
    Generate mesh for 2D flow around a circular cylinder.
    """

    def __init__(self, domain_length: float = 2.2, domain_height: float = 0.41,
                 cylinder_diameter: float = 0.1, cylinder_x: float = 0.2,
                 cylinder_y: float = 0.2, mesh_size: float = 0.01):
        """
        Initialize mesh generator.

        Args:
            domain_length: Length of computational domain
            domain_height: Height of computational domain
            cylinder_diameter: Diameter of cylinder
            cylinder_x, cylinder_y: Position of cylinder center
            mesh_size: Characteristic mesh size
        """
        self.domain_length = domain_length
        self.domain_height = domain_height
        self.cylinder_diameter = cylinder_diameter
        self.cylinder_x = cylinder_x
        self.cylinder_y = cylinder_y
        self.mesh_size = mesh_size
        self.cylinder_radius = cylinder_diameter / 2

        # Initialize gmsh
        gmsh.initialize()
        gmsh.model.add("cylinder_flow")

    def create_geometry(self):
        """Create the geometry for the cylinder flow domain."""
        # Domain corners
        x_min, x_max = 0.0, self.domain_length
        y_min, y_max = 0.0, self.domain_height

        # Create points for domain boundary
        p1 = gmsh.model.geo.addPoint(x_min, y_min, 0, self.mesh_size)
        p2 = gmsh.model.geo.addPoint(x_max, y_min, 0, self.mesh_size)
        p3 = gmsh.model.geo.addPoint(x_max, y_max, 0, self.mesh_size)
        p4 = gmsh.model.geo.addPoint(x_min, y_max, 0, self.mesh_size)

        # Create lines for domain boundary
        l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
        l2 = gmsh.model.geo.addLine(p2, p3)  # Right
        l3 = gmsh.model.geo.addLine(p3, p4)  # Top
        l4 = gmsh.model.geo.addLine(p4, p1)  # Left

        # Create domain loop
        domain_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

        # Create cylinder
        # Points for cylinder
        p5 = gmsh.model.geo.addPoint(self.cylinder_x, self.cylinder_y, 0, self.mesh_size/4)
        p6 = gmsh.model.geo.addPoint(self.cylinder_x + self.cylinder_radius, self.cylinder_y, 0, self.mesh_size/4)
        p7 = gmsh.model.geo.addPoint(self.cylinder_x, self.cylinder_y + self.cylinder_radius, 0, self.mesh_size/4)
        p8 = gmsh.model.geo.addPoint(self.cylinder_x - self.cylinder_radius, self.cylinder_y, 0, self.mesh_size/4)
        p9 = gmsh.model.geo.addPoint(self.cylinder_x, self.cylinder_y - self.cylinder_radius, 0, self.mesh_size/4)

        # Create circle for cylinder
        c1 = gmsh.model.geo.addCircleArc(p6, p5, p7)
        c2 = gmsh.model.geo.addCircleArc(p7, p5, p8)
        c3 = gmsh.model.geo.addCircleArc(p8, p5, p9)
        c4 = gmsh.model.geo.addCircleArc(p9, p5, p6)

        # Create cylinder loop
        cylinder_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])

        # Create surface (domain with cylinder hole)
        surface = gmsh.model.geo.addPlaneSurface([domain_loop, cylinder_loop])

        # Synchronize geometry
        gmsh.model.geo.synchronize()

        # Create physical groups for boundaries
        self._create_physical_groups()

    def _create_physical_groups(self):
        """Create physical groups for boundary conditions."""
        # Get all surfaces
        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], name="Fluid")

        # Get all curves (boundaries)
        curves = gmsh.model.getEntities(1)

        # Identify boundaries
        inlet_curves = []
        outlet_curves = []
        wall_curves = []
        cylinder_curves = []

        for curve in curves:
            # Get curve points
            curve_points = gmsh.model.getBoundary([curve])
            points = [gmsh.model.getValue(curve, [curve_points[0][1]])]

            # Check if it's a line (not a circle)
            if len(curve_points) == 2:
                p1 = gmsh.model.getValue(curve, [curve_points[0][1]])
                p2 = gmsh.model.getValue(curve, [curve_points[1][1]])

                # Inlet (left boundary)
                if abs(p1[0]) < 1e-6 and abs(p2[0]) < 1e-6:
                    inlet_curves.append(curve[1])
                # Outlet (right boundary)
                elif abs(p1[0] - self.domain_length) < 1e-6 and abs(p2[0] - self.domain_length) < 1e-6:
                    outlet_curves.append(curve[1])
                # Walls (top and bottom)
                elif (abs(p1[1]) < 1e-6 and abs(p2[1]) < 1e-6) or \
                     (abs(p1[1] - self.domain_height) < 1e-6 and abs(p2[1] - self.domain_height) < 1e-6):
                    wall_curves.append(curve[1])
            else:
                # Circle (cylinder)
                cylinder_curves.append(curve[1])

        # Create physical groups
        if inlet_curves:
            gmsh.model.addPhysicalGroup(1, inlet_curves, name="Inlet")
        if outlet_curves:
            gmsh.model.addPhysicalGroup(1, outlet_curves, name="Outlet")
        if wall_curves:
            gmsh.model.addPhysicalGroup(1, wall_curves, name="Walls")
        if cylinder_curves:
            gmsh.model.addPhysicalGroup(1, cylinder_curves, name="Cylinder")

    def generate_mesh(self, mesh_size_factor: float = 1.0):
        """
        Generate the mesh.

        Args:
            mesh_size_factor: Factor to multiply mesh size by
        """
        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_size_factor)

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Optimize mesh
        gmsh.model.mesh.optimize("Netgen")

    def save_mesh(self, filename: str):
        """
        Save mesh to file.

        Args:
            filename: Base filename (without extension)
        """
        # Save as .msh format
        gmsh.write(f"{filename}.msh")

        # Create mesh data structure for pure scipy FEM
        mesh_data = self._create_mesh_data()

        # Save mesh data
        np.savez(f"{filename}_data.npz", **mesh_data)

        print(f"Mesh saved to {filename}.msh and {filename}_data.npz")

    def _create_mesh_data(self) -> Dict:
        """Create mesh data structure for pure scipy FEM."""
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = node_coords.reshape(-1, 3)[:, :2]  # 2D coordinates

        # Get elements
        element_types, element_tags, element_connectivity = gmsh.model.mesh.getElements()

        # Find triangular elements
        triangle_elements = []
        for i, elem_type in enumerate(element_types):
            if elem_type == 2:  # Triangle
                connectivity = element_connectivity[i]
                n_nodes_per_elem = 3
                for j in range(0, len(connectivity), n_nodes_per_elem):
                    elem_nodes = connectivity[j:j+n_nodes_per_elem] - 1  # Convert to 0-based
                    triangle_elements.append({
                        'nodes': elem_nodes,
                        'type': 'triangle'
                    })

        # Get boundary nodes
        boundary_nodes = self._get_boundary_nodes()
        cylinder_nodes = self._get_cylinder_nodes()
        inlet_nodes = self._get_inlet_nodes()
        outlet_nodes = self._get_outlet_nodes()

        mesh_data = {
            'nodes': nodes,
            'elements': triangle_elements,
            'boundary_nodes': boundary_nodes,
            'cylinder_nodes': cylinder_nodes,
            'inlet_nodes': inlet_nodes,
            'outlet_nodes': outlet_nodes
        }

        return mesh_data

    def _get_boundary_nodes(self) -> List[int]:
        """Get all boundary nodes."""
        boundary_nodes = []

        # Get all boundary entities
        boundary_entities = gmsh.model.getEntities(1)

        for entity in boundary_entities:
            node_tags, _, _ = gmsh.model.mesh.getNodes(1, entity[1])
            boundary_nodes.extend(node_tags - 1)  # Convert to 0-based

        return list(set(boundary_nodes))  # Remove duplicates

    def _get_cylinder_nodes(self) -> List[int]:
        """Get cylinder boundary nodes."""
        cylinder_nodes = []

        # Find cylinder boundary (circle)
        for entity in gmsh.model.getEntities(1):
            # Check if it's a circle (cylinder boundary)
            node_tags, coords, _ = gmsh.model.mesh.getNodes(1, entity[1])
            if len(node_tags) > 0:
                # Check if nodes are on cylinder (distance from center)
                coords = coords.reshape(-1, 3)
                cx, cy = 0.2, 0.2
                r = 0.05

                for i, (x, y, z) in enumerate(coords):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if abs(dist - r) < 1e-6:
                        cylinder_nodes.append(node_tags[i] - 1)  # Convert to 0-based

        return cylinder_nodes

    def _get_inlet_nodes(self) -> List[int]:
        """Get inlet boundary nodes."""
        inlet_nodes = []

        # Find inlet boundary (left side)
        for entity in gmsh.model.getEntities(1):
            node_tags, coords, _ = gmsh.model.mesh.getNodes(1, entity[1])
            if len(node_tags) > 0:
                coords = coords.reshape(-1, 3)

                for i, (x, y, z) in enumerate(coords):
                    if abs(x) < 1e-6:  # Left boundary
                        inlet_nodes.append(node_tags[i] - 1)  # Convert to 0-based

        return inlet_nodes

    def _get_outlet_nodes(self) -> List[int]:
        """Get outlet boundary nodes."""
        outlet_nodes = []

        # Find outlet boundary (right side)
        for entity in gmsh.model.getEntities(1):
            node_tags, coords, _ = gmsh.model.mesh.getNodes(1, entity[1])
            if len(node_tags) > 0:
                coords = coords.reshape(-1, 3)

                for i, (x, y, z) in enumerate(coords):
                    if abs(x - 2.2) < 1e-6:  # Right boundary
                        outlet_nodes.append(node_tags[i] - 1)  # Convert to 0-based

        return outlet_nodes

    def get_mesh_info(self) -> dict:
        """Get information about the generated mesh."""
        # Get mesh statistics
        stats = gmsh.model.mesh.getStats()

        info = {
            'num_vertices': stats[0],
            'num_elements': stats[1],
            'num_triangles': stats[2],
            'num_quadrilaterals': stats[3],
            'num_tetrahedra': stats[4],
            'num_hexahedra': stats[5]
        }

        return info

    def visualize_mesh(self, filename: str = "mesh_visualization.png"):
        """Visualize the mesh."""
        # Set visualization options
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.option.setNumber("Mesh.SurfaceEdges", 1)

        # Launch GUI
        gmsh.fltk.run()

        # Save image
        gmsh.write(filename)

    def cleanup(self):
        """Clean up gmsh."""
        gmsh.finalize()


def generate_cylinder_mesh(reynolds_number: float = 100,
                          mesh_size: float = 0.01,
                          output_dir: str = "meshes") -> str:
    """
    Generate mesh for cylinder flow simulation.

    Args:
        reynolds_number: Reynolds number (affects mesh refinement)
        mesh_size: Base mesh size
        output_dir: Directory to save mesh files

    Returns:
        Path to generated mesh file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Adjust mesh size based on Reynolds number
    if reynolds_number > 200:
        mesh_size *= 0.5  # Finer mesh for high Re
    elif reynolds_number > 100:
        mesh_size *= 0.7

    # Create mesh generator
    generator = CylinderMeshGenerator(mesh_size=mesh_size)

    # Create geometry
    generator.create_geometry()

    # Generate mesh
    generator.generate_mesh()

    # Get mesh info
    info = generator.get_mesh_info()
    print(f"Generated mesh:")
    print(f"  Vertices: {info['num_vertices']}")
    print(f"  Elements: {info['num_elements']}")
    print(f"  Triangles: {info['num_triangles']}")

    # Save mesh
    mesh_filename = f"cylinder_mesh_Re{reynolds_number:.0f}"
    mesh_path = os.path.join(output_dir, mesh_filename)
    generator.save_mesh(mesh_path)

    # Cleanup
    generator.cleanup()

    return f"{mesh_path}.xml"


def main():
    """Main function to generate meshes for different Reynolds numbers."""
    reynolds_numbers = [20, 40, 100, 200]

    for re in reynolds_numbers:
        print(f"\nGenerating mesh for Re = {re}")
        print("=" * 40)

        mesh_file = generate_cylinder_mesh(re, mesh_size=0.01)
        print(f"Mesh saved to: {mesh_file}")


if __name__ == "__main__":
    main()
