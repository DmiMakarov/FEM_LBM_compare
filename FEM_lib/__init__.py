"""
FEM Library Implementation using scikit-fem and FEniCS.

This package provides finite element method implementations for 2D incompressible
Navier-Stokes equations using both scikit-fem and FEniCS libraries. It includes
mesh generation, solver implementation, and cylinder flow simulation capabilities.
"""

from .skfem_mesh_generator import SkfemMeshGenerator
from .skfem_navier_stokes_solver import SkfemNavierStokesSolver
from .skfem_cylinder_flow import SkfemCylinderFlow

# FEniCS implementations
#from .fenics_mesh_generator import FenicsMeshGenerator
#from .fenics_navier_stokes_solver import FenicsNavierStokesSolver
#from .fenics_cylinder_flow import FenicsCylinderFlow

# Proper scikit-fem implementations
from .proper_skfem_navier_stokes_solver import ProperSkfemNavierStokesSolver
from .proper_skfem_cylinder_flow import ProperSkfemCylinderFlow

__all__ = [
    # scikit-fem implementations
    'SkfemMeshGenerator',
    'SkfemNavierStokesSolver',
    'SkfemCylinderFlow',
    # FEniCS implementations
    'FenicsMeshGenerator',
    'FenicsNavierStokesSolver',
    'FenicsCylinderFlow',
    # Proper scikit-fem implementations
    'ProperSkfemNavierStokesSolver',
    'ProperSkfemCylinderFlow'
]

__version__ = "1.0.0"
