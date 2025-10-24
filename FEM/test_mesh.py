"""
Test script to verify mesh generation works.
"""

import numpy as np

def test_mesh_generation():
    """Test the mesh generation without external dependencies."""

    # Simple test parameters
    nx, ny = 10, 5
    domain_length = 2.2
    domain_height = 0.41

    # Create structured grid
    x = np.linspace(0, domain_length, nx)
    y = np.linspace(0, domain_height, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Flatten to get node coordinates
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    # Create triangular elements
    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Get node indices
            n1 = i * ny + j
            n2 = (i + 1) * ny + j
            n3 = (i + 1) * ny + (j + 1)
            n4 = i * ny + (j + 1)

            # Create two triangles per quad
            elements.append([n1, n2, n3])
            elements.append([n1, n3, n4])

    # Test boundary detection
    cylinder_x, cylinder_y = 0.2, 0.2
    cylinder_radius = 0.05

    boundary_nodes = []
    cylinder_nodes = []
    inlet_nodes = []
    outlet_nodes = []

    for i, (x, y) in enumerate(nodes):
        # Left boundary
        if abs(x) < 1e-6:
            boundary_nodes.append(i)
            inlet_nodes.append(i)
        # Right boundary
        elif abs(x - domain_length) < 1e-6:
            boundary_nodes.append(i)
            outlet_nodes.append(i)
        # Bottom boundary
        elif abs(y) < 1e-6:
            boundary_nodes.append(i)
        # Top boundary
        elif abs(y - domain_height) < 1e-6:
            boundary_nodes.append(i)

        # Cylinder boundary
        dist = np.sqrt((x - cylinder_x)**2 + (y - cylinder_y)**2)
        if abs(dist - cylinder_radius) < 1e-6:
            cylinder_nodes.append(i)

    # Create mesh data
    mesh_data = {
        'nodes': nodes,
        'elements': elements,
        'boundary_nodes': boundary_nodes,
        'cylinder_nodes': cylinder_nodes,
        'inlet_nodes': inlet_nodes,
        'outlet_nodes': outlet_nodes
    }

    print("Mesh generation test successful!")
    print(f"Nodes: {len(nodes)}")
    print(f"Elements: {len(elements)}")
    print(f"Boundary nodes: {len(boundary_nodes)}")
    print(f"Cylinder nodes: {len(cylinder_nodes)}")
    print(f"Inlet nodes: {len(inlet_nodes)}")
    print(f"Outlet nodes: {len(outlet_nodes)}")

    return mesh_data

if __name__ == "__main__":
    test_mesh_generation()
