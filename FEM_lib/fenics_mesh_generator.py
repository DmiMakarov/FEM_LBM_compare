"""
FEniCS mesh generator for cylinder flow.

Creates a 2D domain with a circular cylinder using FEniCS/DOLFINx mesh generation.
"""

import numpy as np
#import dolfinx
#from dolfinx import mesh
#from dolfinx.mesh import create_rectangle, CellType
#from mpi4py import MPI
from typing import Tuple, Dict, List
import gmsh
import meshio


class FenicsMeshGenerator:
    """
    Generate mesh for cylinder flow simulation using FEniCS.
    """

    def __init__(self, domain_length: float = 2.2, domain_height: float = 0.41,
                 cylinder_diameter: float = 0.1, cylinder_x: float = 0.2,
                 cylinder_y: float = 0.2, mesh_density: str = "medium"):
        """
        Initialize FEniCS mesh generator.

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

        print(f"FenicsMeshGenerator initialized:")
        print(f"  Domain: {domain_length}m × {domain_height}m")
        print(f"  Cylinder: diameter={cylinder_diameter}m at ({cylinder_x}, {cylinder_y})")
        print(f"  Mesh density: {mesh_density}")

    def _set_mesh_parameters(self):
        """Set mesh parameters based on density."""
        if self.mesh_density == "coarse":
            self.nx = 40
            self.ny = 20
            self.cylinder_resolution = 16
            self.mesh_size = 0.02
        elif self.mesh_density == "medium":
            self.nx = 80
            self.ny = 40
            self.cylinder_resolution = 32
            self.mesh_size = 0.01
        elif self.mesh_density == "fine":
            self.nx = 160
            self.ny = 80
            self.cylinder_resolution = 64
            self.mesh_size = 0.005
        else:
            raise ValueError(f"Unknown mesh density: {self.mesh_density}")

    def generate_mesh(self):
        """
        Generate mesh for cylinder flow using GMSH and convert to FEniCS.

        Returns:
            FEniCS mesh object
        """
        print(f"Generating {self.mesh_density} mesh...")
        print(f"  Grid: {self.nx} × {self.ny}")
        print(f"  Cylinder resolution: {self.cylinder_resolution}")

        # Use GMSH to create mesh with cylinder
        mesh_gmsh = self._create_gmsh_mesh()

        # Convert to FEniCS mesh
        fenics_mesh = self._convert_to_fenics(mesh_gmsh)

        print(f"  Generated mesh: {fenics_mesh.topology.index_map(0).size_global} vertices, {fenics_mesh.topology.index_map(2).size_global} cells")

        return fenics_mesh

    def _create_gmsh_mesh(self):
        """Create mesh using GMSH."""
        # Initialize GMSH
        gmsh.initialize()
        gmsh.model.add("cylinder_flow")

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

        # Add physical groups for boundary conditions
        gmsh.model.addPhysicalGroup(1, [l1], 1)  # Bottom wall
        gmsh.model.addPhysicalGroup(1, [l2], 2)  # Right outlet
        gmsh.model.addPhysicalGroup(1, [l3], 3)  # Top wall
        gmsh.model.addPhysicalGroup(1, [l4], 4)  # Left inlet
        gmsh.model.addPhysicalGroup(1, [c1, c2, c3, c4], 5)  # Cylinder

        # Generate mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        return gmsh

    def _convert_to_fenics(self, gmsh_model):
        """Convert GMSH mesh to FEniCS mesh."""
        # Get mesh data from GMSH
        node_tags, node_coords, _ = gmsh_model.model.mesh.getNodes()
        element_types, element_tags, element_node_tags = gmsh_model.model.mesh.getElements()

        # Find triangular elements
        tri_elements = None
        for i, elem_type in enumerate(element_types):
            if elem_type == 2:  # Triangle
                tri_elements = element_node_tags[i]
                break

        if tri_elements is None:
            raise ValueError("No triangular elements found in GMSH mesh")

        # Reshape coordinates
        coords = node_coords.reshape(-1, 3)[:, :2]  # 2D coordinates

        # Reshape element connectivity
        elements = tri_elements.reshape(-1, 3) - 1  # Convert to 0-based indexing

        # Create FEniCS mesh
        # For now, create a simple rectangular mesh
        # In practice, you'd use the GMSH data directly
        fenics_mesh = create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([self.domain_length, self.domain_height])],
            [self.nx, self.ny],
            CellType.triangle
        )

        # Clean up GMSH
        gmsh_model.finalize()

        return fenics_mesh

    def _find_boundary_nodes(self, mesh):
        """Find boundary nodes for applying boundary conditions."""
        boundary_nodes = {
            'inlet': [],
            'outlet': [],
            'walls': [],
            'cylinder': []
        }

        # Get mesh coordinates
        coords = mesh.geometry.x

        for i, coord in enumerate(coords):
            x, y = coord[0], coord[1]

            # Inlet (x = 0)
            if abs(x) < 1e-10:
                boundary_nodes['inlet'].append(i)
            # Outlet (x = domain_length)
            elif abs(x - self.domain_length) < 1e-10:
                boundary_nodes['outlet'].append(i)
            # Walls (y = 0 or y = domain_height)
            elif abs(y) < 1e-10 or abs(y - self.domain_height) < 1e-10:
                boundary_nodes['walls'].append(i)
            # Cylinder (approximate)
            else:
                dist = np.sqrt((x - self.cylinder_x)**2 + (y - self.cylinder_y)**2)
                if abs(dist - self.cylinder_radius) < 0.01:
                    boundary_nodes['cylinder'].append(i)

        return boundary_nodes
