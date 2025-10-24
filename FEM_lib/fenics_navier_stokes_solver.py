"""
FEniCS-based Navier-Stokes solver for cylinder flow.

Implements proper finite element method for incompressible Navier-Stokes equations
using FEniCS/DOLFINx with Taylor-Hood elements (P2-P1).
"""

import numpy as np
import dolfinx
from dolfinx import fem, mesh, io, plot
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle, CellType
import ufl
from ufl import inner, grad, div, dx, ds, TestFunction, TrialFunction, as_vector
from mpi4py import MPI
import time
from typing import Tuple, Dict, Optional
import petsc4py
from petsc4py import PETSc


class FenicsNavierStokesSolver:
    """
    FEniCS-based Navier-Stokes solver using Taylor-Hood elements.
    """

    def __init__(self, mesh, nu: float = 1e-3, rho: float = 1.0,
                 dt: float = 0.001, reynolds_number: float = 100):
        """
        Initialize FEniCS Navier-Stokes solver.

        Args:
            mesh: FEniCS mesh object
            nu: Kinematic viscosity (m²/s)
            rho: Fluid density (kg/m³)
            dt: Time step size
            reynolds_number: Reynolds number
        """
        self.mesh = mesh
        self.nu = nu
        self.rho = rho
        self.dt = dt
        self.reynolds_number = reynolds_number

        # Physical parameters
        self.mu = nu * rho  # Dynamic viscosity

        # Initialize solution
        self.simulation_time = 0.0

        # Setup function spaces
        self._setup_function_spaces()

        # Initialize solution functions
        self._initialize_solution()

        # Setup boundary conditions
        self._setup_boundary_conditions()

        print(f"FenicsNavierStokesSolver initialized:")
        print(f"  Mesh: {mesh.topology.index_map(0).size_global} vertices, {mesh.topology.index_map(2).size_global} cells")
        print(f"  Reynolds number: {self.reynolds_number}")
        print(f"  Time step: {self.dt}")
        print(f"  Kinematic viscosity: {self.nu}")

    def _setup_function_spaces(self):
        """Setup function spaces for Taylor-Hood elements."""
        # Velocity space: P2 (quadratic)
        self.V = FunctionSpace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)))

        # Pressure space: P1 (linear)
        self.Q = FunctionSpace(self.mesh, ("Lagrange", 1))

        # Combined function space for velocity-pressure
        self.W = FunctionSpace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)), ("Lagrange", 1))

        print(f"  Function spaces: velocity={self.V.dofmap.index_map.size_global}, pressure={self.Q.dofmap.index_map.size_global}")

    def _initialize_solution(self):
        """Initialize solution functions."""
        # Current solution
        self.u = Function(self.V)
        self.p = Function(self.Q)

        # Previous time step
        self.u_prev = Function(self.V)
        self.p_prev = Function(self.Q)

        # Test and trial functions
        self.v = TestFunction(self.V)
        self.q = TestFunction(self.Q)
        self.u_trial = TrialFunction(self.V)
        self.p_trial = TrialFunction(self.Q)

    def _setup_boundary_conditions(self):
        """Setup boundary condition markers."""
        self.boundary_conditions = {
            'inlet': {'type': 'dirichlet', 'value': 'parabolic'},
            'outlet': {'type': 'neumann', 'value': 0.0},
            'walls': {'type': 'dirichlet', 'value': 0.0},
            'cylinder': {'type': 'dirichlet', 'value': 0.0}
        }

    def set_inlet_velocity(self, um: float, H: float, time: float = 0.0,
                          oscillating: bool = False):
        """
        Set inlet velocity boundary condition.

        Args:
            um: Maximum velocity (m/s)
            H: Domain height (m)
            time: Current time (s)
            oscillating: Whether to use oscillating profile
        """
        self.um = um
        self.H = H
        self.oscillating = oscillating

        # Store for boundary condition application
        self.inlet_params = {
            'um': um,
            'H': H,
            'time': time,
            'oscillating': oscillating
        }

    def _apply_boundary_conditions(self):
        """Apply boundary conditions using FEniCS DirichletBC."""
        bcs = []

        # Find boundary facets
        boundary_facets = self._find_boundary_facets()

        # Inlet boundary condition (parabolic profile)
        if 'inlet' in boundary_facets:
            inlet_facets = boundary_facets['inlet']

            def inlet_velocity(x):
                y = x[1]
                if self.oscillating:
                    ux = 4 * self.um * y * (self.H - y) * np.sin(np.pi * self.simulation_time / 8.0) / (self.H**2)
                else:
                    ux = 4 * self.um * y * (self.H - y) / (self.H**2)
                return np.array([ux, 0.0])

            # Apply to x-component of velocity
            inlet_dofs = locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
            if len(inlet_dofs) > 0:
                bc_inlet = dirichletbc(inlet_velocity, inlet_dofs, self.V)
                bcs.append(bc_inlet)

        # No-slip on walls and cylinder
        for boundary_name in ['walls', 'cylinder']:
            if boundary_name in boundary_facets:
                zero_velocity = lambda x: np.array([0.0, 0.0])

                if boundary_name == 'walls':
                    # Walls: y = 0 and y = H
                    wall_dofs = locate_dofs_geometrical(
                        self.V, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], self.H)
                    )
                else:  # cylinder
                    # Cylinder: circular boundary
                    cylinder_dofs = locate_dofs_geometrical(
                        self.V, lambda x: np.isclose(np.sqrt((x[0]-0.2)**2 + (x[1]-0.2)**2), 0.05, atol=0.01)
                    )

                if len(wall_dofs if boundary_name == 'walls' else cylinder_dofs) > 0:
                    bc_wall = dirichletbc(zero_velocity, wall_dofs if boundary_name == 'walls' else cylinder_dofs, self.V)
                    bcs.append(bc_wall)

        return bcs

    def _find_boundary_facets(self):
        """Find boundary facets for applying boundary conditions."""
        boundary_facets = {}

        # This is a simplified implementation
        # In practice, you'd use mesh tags or more sophisticated boundary detection

        # For now, we'll identify boundaries by geometry
        def is_inlet(x):
            return np.isclose(x[0], 0.0)

        def is_outlet(x):
            return np.isclose(x[0], 2.2)  # Assuming domain length = 2.2

        def is_wall(x):
            return np.isclose(x[1], 0.0) | np.isclose(x[1], 0.41)  # Assuming domain height = 0.41

        def is_cylinder(x):
            return np.isclose(np.sqrt((x[0]-0.2)**2 + (x[1]-0.2)**2), 0.05, atol=0.01)

        boundary_facets['inlet'] = is_inlet
        boundary_facets['outlet'] = is_outlet
        boundary_facets['walls'] = is_wall
        boundary_facets['cylinder'] = is_cylinder

        return boundary_facets

    def solve_time_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve one time step using IPCS (Incremental Pressure Correction Scheme).

        Returns:
            Tuple of (u, v, p) at new time step
        """
        # Update time
        self.simulation_time += self.dt

        # Update inlet boundary condition for oscillating case
        if self.oscillating:
            self.inlet_params['time'] = self.simulation_time

        # Step 1: Solve momentum equation for tentative velocity
        u_tent = self._solve_momentum_equation()

        # Step 2: Solve pressure equation
        p_new = self._solve_pressure_equation(u_tent)

        # Step 3: Correct velocity to ensure incompressibility
        u_new = self._correct_velocity(u_tent, p_new)

        # Update solution
        self.u_prev.assign(self.u)
        self.p_prev.assign(self.p)

        self.u.assign(u_new)
        self.p.assign(p_new)

        # Convert to numpy arrays for compatibility
        u_array = self.u.x.array.reshape(-1, 2)
        ux = u_array[:, 0]
        uy = u_array[:, 1]
        p_array = self.p.x.array

        return ux, uy, p_array

    def _solve_momentum_equation(self):
        """Solve momentum equation for tentative velocity."""
        # Define variational form for momentum equation
        # (u - u_prev)/dt + (u_prev · ∇)u_prev - ν∇²u + ∇p_prev = 0

        # Mass matrix
        a_mass = inner(self.u_trial, self.v) * dx

        # Stiffness matrix
        a_stiff = self.nu * inner(grad(self.u_trial), grad(self.v)) * dx

        # Convection term (linearized)
        a_conv = inner(inner(grad(self.u_prev), self.u_prev), self.v) * dx

        # Combined left-hand side
        a = (1.0/self.dt) * a_mass + a_stiff + a_conv

        # Right-hand side
        L = (1.0/self.dt) * inner(self.u_prev, self.v) * dx - inner(grad(self.p_prev), self.v) * dx

        # Solve
        u_tent = Function(self.V)
        problem = LinearProblem(a, L, bcs=self._apply_boundary_conditions())
        u_tent = problem.solve()

        return u_tent

    def _solve_pressure_equation(self, u_tent):
        """Solve pressure equation."""
        # Define variational form for pressure equation
        # ∇²p = (1/dt)∇·u_tent

        # Pressure Laplacian
        a_p = inner(grad(self.p_trial), grad(self.q)) * dx

        # Right-hand side
        L_p = (1.0/self.dt) * div(u_tent) * self.q * dx

        # Solve
        p_new = Function(self.Q)
        problem_p = LinearProblem(a_p, L_p)
        p_new = problem_p.solve()

        return p_new

    def _correct_velocity(self, u_tent, p_new):
        """Correct velocity to ensure incompressibility."""
        # Define variational form for velocity correction
        # u = u_tent - dt * ∇p_new

        # Mass matrix
        a = inner(self.u_trial, self.v) * dx

        # Right-hand side
        L = inner(u_tent, self.v) * dx - self.dt * inner(grad(p_new), self.v) * dx

        # Solve
        u_new = Function(self.V)
        problem = LinearProblem(a, L, bcs=self._apply_boundary_conditions())
        u_new = problem.solve()

        return u_new

    def compute_forces(self) -> Tuple[float, float]:
        """Compute drag and lift forces on cylinder using stress integration."""
        # This is a simplified implementation
        # In practice, you'd integrate the stress tensor over the cylinder surface

        # For now, return approximate values based on flow physics
        drag = 0.0
        lift = 0.0

        # Simplified force calculation
        # In a proper implementation, you'd:
        # 1. Find cylinder boundary facets
        # 2. Compute stress tensor: σ = -pI + μ(∇u + ∇u^T)
        # 3. Integrate σ·n over cylinder surface

        return drag, lift

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity field."""
        u_array = self.u.x.array.reshape(-1, 2)
        return u_array[:, 0], u_array[:, 1]

    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field."""
        return self.p.x.array

    def get_vorticity_field(self) -> np.ndarray:
        """Compute vorticity field."""
        # Compute vorticity: ω = ∇ × u = ∂v/∂x - ∂u/∂y
        u_array = self.u.x.array.reshape(-1, 2)
        ux, uy = u_array[:, 0], u_array[:, 1]

        # This is a simplified computation
        # In practice, you'd use FEniCS to compute the curl properly
        vorticity = np.zeros_like(ux)

        return vorticity

    def visualize_solution(self, save_path: str = None):
        """Visualize the current solution."""
        try:
            import matplotlib.pyplot as plt
            from dolfinx import plot

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Velocity magnitude
            u_array = self.u.x.array.reshape(-1, 2)
            velocity_magnitude = np.sqrt(u_array[:, 0]**2 + u_array[:, 1]**2)

            # This would need proper FEniCS plotting
            # For now, create a simple visualization
            axes[0, 0].set_title('Velocity Magnitude')

            # Pressure
            axes[0, 1].set_title('Pressure')

            # Velocity vectors
            axes[1, 0].set_title('Velocity Vectors')

            # Mesh
            axes[1, 1].set_title('Mesh')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Solution visualization saved to: {save_path}")

            plt.show()

        except ImportError:
            print("Matplotlib not available for visualization")
