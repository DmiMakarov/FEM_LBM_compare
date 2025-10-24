"""
Navier-Stokes solver using scikit-fem.

Implements incompressible Navier-Stokes equations using Taylor-Hood elements
(P2-P1) for velocity-pressure formulation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, gmres
from typing import Tuple, Dict, Optional
import time

from skfem import *
from skfem.helpers import ddot, dd, grad, div, curl, cross, inner, prod, transpose
from skfem.visuals.matplotlib import draw, plot


class SkfemNavierStokesSolver:
    """
    Navier-Stokes solver using scikit-fem with Taylor-Hood elements.
    """

    def __init__(self, mesh, nu: float = 1e-3, rho: float = 1.0,
                 dt: float = 0.001, reynolds_number: float = 100):
        """
        Initialize Navier-Stokes solver.

        Args:
            mesh: scikit-fem mesh object
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

        # Setup finite element spaces
        self._setup_fe_spaces()

        # Initialize solution vectors
        self._initialize_solution()

        # Setup boundary conditions
        self._setup_boundary_conditions()

        print(f"SkfemNavierStokesSolver initialized:")
        print(f"  Nodes: {self.mesh.p.shape[1]}")
        print(f"  Elements: {self.mesh.t.shape[1]}")
        print(f"  Reynolds number: {self.reynolds_number}")
        print(f"  Time step: {self.dt}")
        print(f"  Kinematic viscosity: {self.nu}")

    def _setup_fe_spaces(self):
        """Setup finite element spaces for Taylor-Hood elements."""
        # Velocity space: P2 (quadratic)
        self.velocity_basis = Basis(self.mesh, ElementVector(ElementTriP2()))

        # Pressure space: P1 (linear)
        self.pressure_basis = Basis(self.mesh, ElementTriP1())

        # Combined space for velocity-pressure
        self.basis = {
            'velocity': self.velocity_basis,
            'pressure': self.pressure_basis
        }

        print(f"  FE spaces: velocity={self.velocity_basis.N}, pressure={self.pressure_basis.N}")

    def _initialize_solution(self):
        """Initialize solution vectors."""
        # Velocity: 2 components (u, v) for each velocity node
        self.u = np.zeros(self.velocity_basis.N)
        self.v = np.zeros(self.velocity_basis.N)

        # Pressure: 1 component for each pressure node
        self.p = np.zeros(self.pressure_basis.N)

        # Previous time step values
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        self.p_prev = np.zeros_like(self.p)

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

    def _apply_boundary_conditions(self, A, b, u_rhs, v_rhs, p_rhs):
        """Apply boundary conditions to the system."""
        # Get boundary nodes
        if hasattr(self, 'boundary_nodes'):
            inlet_nodes = self.boundary_nodes.get('inlet', [])
            outlet_nodes = self.boundary_nodes.get('outlet', [])
            wall_nodes = self.boundary_nodes.get('walls', [])
            cylinder_nodes = self.boundary_nodes.get('cylinder', [])
        elif hasattr(self.mesh, 'boundaries'):
            inlet_nodes = self.mesh.boundaries.get('inlet', [])
            outlet_nodes = self.mesh.boundaries.get('outlet', [])
            wall_nodes = self.mesh.boundaries.get('walls', [])
            cylinder_nodes = self.mesh.boundaries.get('cylinder', [])
        else:
            # Fallback: find boundary nodes manually
            inlet_nodes, outlet_nodes, wall_nodes, cylinder_nodes = self._find_boundary_nodes()

        # Apply inlet boundary condition (parabolic profile)
        for node in inlet_nodes:
            if node < len(self.velocity_basis.dofs.u):
                y = self.mesh.p[1, node]

                if self.oscillating:
                    ux = 4 * self.um * y * (self.H - y) * np.sin(np.pi * self.simulation_time / 8.0) / (self.H**2)
                else:
                    ux = 4 * self.um * y * (self.H - y) / (self.H**2)

                # Set u velocity
                u_dof = self.velocity_basis.dofs.u[node]
                A[u_dof, :] = 0
                A[u_dof, u_dof] = 1
                u_rhs[u_dof] = ux

                # Set v velocity to zero
                v_dof = self.velocity_basis.dofs.v[node]
                A[v_dof, :] = 0
                A[v_dof, v_dof] = 1
                v_rhs[v_dof] = 0

        # Apply no-slip on walls and cylinder
        for node in wall_nodes + cylinder_nodes:
            if node < len(self.velocity_basis.dofs.u):
                u_dof = self.velocity_basis.dofs.u[node]
                v_dof = self.velocity_basis.dofs.v[node]

                A[u_dof, :] = 0
                A[u_dof, u_dof] = 1
                u_rhs[u_dof] = 0

                A[v_dof, :] = 0
                A[v_dof, v_dof] = 1
                v_rhs[v_dof] = 0

        # Apply outlet boundary condition (zero pressure)
        for node in outlet_nodes:
            if node < len(self.pressure_basis.dofs):
                p_dof = self.pressure_basis.dofs[node]
                A[p_dof, :] = 0
                A[p_dof, p_dof] = 1
                p_rhs[p_dof] = 0

        return A, b, u_rhs, v_rhs, p_rhs

    def _find_boundary_nodes(self):
        """Find boundary nodes manually if not stored in mesh."""
        inlet_nodes = []
        outlet_nodes = []
        wall_nodes = []
        cylinder_nodes = []

        for i, (x, y) in enumerate(self.mesh.p.T):
            # Inlet (x = 0)
            if abs(x) < 1e-10:
                inlet_nodes.append(i)
            # Outlet (x = domain_length)
            elif abs(x - 2.2) < 1e-10:  # Assuming domain_length = 2.2
                outlet_nodes.append(i)
            # Walls (y = 0 or y = domain_height)
            elif abs(y) < 1e-10 or abs(y - 0.41) < 1e-10:  # Assuming domain_height = 0.41
                wall_nodes.append(i)
            # Cylinder (approximate)
            else:
                dist = np.sqrt((x - 0.2)**2 + (y - 0.2)**2)
                if abs(dist - 0.05) < 0.01:  # Assuming cylinder radius = 0.05
                    cylinder_nodes.append(i)

        return inlet_nodes, outlet_nodes, wall_nodes, cylinder_nodes

    def solve_time_step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve one time step using semi-implicit scheme.

        Returns:
            Tuple of (u, v, p) at new time step
        """
        # Update time
        self.simulation_time += self.dt

        # Update inlet boundary condition for oscillating case
        if self.oscillating:
            self.inlet_params['time'] = self.simulation_time

        # Build system matrices
        A, b = self._build_system_matrices()

        # Solve for velocity and pressure
        u_new, v_new, p_new = self._solve_system(A, b)

        # Update solution
        self.u_prev = self.u.copy()
        self.v_prev = self.v.copy()
        self.p_prev = self.p.copy()

        self.u = u_new
        self.v = v_new
        self.p = p_new

        return u_new, v_new, p_new

    def _build_system_matrices(self):
        """Build system matrices for Navier-Stokes equations."""
        # This is a simplified implementation
        # In practice, you'd use scikit-fem's built-in forms

        # For now, we'll use a simplified approach
        # The full implementation would use scikit-fem's form system

        # Mass matrix for velocity
        @BilinearForm
        def mass(u, v, w):
            return inner(u, v)

        # Stiffness matrix for velocity
        @BilinearForm
        def stiffness(u, v, w):
            return self.nu * ddot(grad(u), grad(v))

        # Gradient matrix
        @BilinearForm
        def gradient(u, p, w):
            return -p * div(u)

        # Divergence matrix
        @BilinearForm
        def divergence(p, u, w):
            return div(u) * p

        # Assemble matrices
        M = mass.assemble(self.velocity_basis)
        K = stiffness.assemble(self.velocity_basis)
        G = gradient.assemble(self.velocity_basis, self.pressure_basis)
        D = divergence.assemble(self.pressure_basis, self.velocity_basis)

        # Build combined system matrix
        n_vel = self.velocity_basis.N
        n_pres = self.pressure_basis.N

        # System matrix: [M/dt + K, G; D, 0]
        A = sp.bmat([
            [M/self.dt + K, G],
            [D, sp.csr_matrix((n_pres, n_pres))]
        ])

        # Right-hand side
        b = np.zeros(n_vel + n_pres)
        b[:n_vel] = M @ np.concatenate([self.u, self.v]) / self.dt

        return A, b

    def _solve_system(self, A, b):
        """Solve the linear system."""
        # Apply boundary conditions
        A, b, u_rhs, v_rhs, p_rhs = self._apply_boundary_conditions(A, b,
                                                                   np.zeros(self.velocity_basis.N),
                                                                   np.zeros(self.velocity_basis.N),
                                                                   np.zeros(self.pressure_basis.N))

        # Solve system
        try:
            x = spsolve(A, b)
        except:
            # Fallback to iterative solver
            x, _ = gmres(A, b, maxiter=1000, tol=1e-6)

        # Extract solution
        n_vel = self.velocity_basis.N
        u_new = x[:n_vel]
        v_new = x[n_vel:2*n_vel]
        p_new = x[2*n_vel:]

        return u_new, v_new, p_new

    def compute_forces(self) -> Tuple[float, float]:
        """Compute drag and lift forces on cylinder."""
        if not hasattr(self.mesh, 'boundaries') or 'cylinder' not in self.mesh.boundaries:
            return 0.0, 0.0

        cylinder_nodes = self.mesh.boundaries['cylinder']
        if not cylinder_nodes:
            return 0.0, 0.0

        drag = 0.0
        lift = 0.0

        # Compute forces using stress integration
        for node in cylinder_nodes:
            if node < len(self.velocity_basis.dofs.u):
                # Get velocity gradients
                grad_u = self._compute_velocity_gradients(node)

                # Compute stress tensor
                sigma_xx = -self.p[node] + 2 * self.mu * grad_u[0, 0]
                sigma_yy = -self.p[node] + 2 * self.mu * grad_u[1, 1]
                sigma_xy = self.mu * (grad_u[0, 1] + grad_u[1, 0])

                # Compute normal vector
                x, y = self.mesh.p[:, node]
                dx = x - 0.2  # cylinder center x
                dy = y - 0.2  # cylinder center y
                dist = np.sqrt(dx**2 + dy**2)

                if dist > 1e-10:
                    nx = dx / dist
                    ny = dy / dist
                else:
                    nx = 1.0
                    ny = 0.0

                # Force components
                fx = (sigma_xx * nx + sigma_xy * ny)
                fy = (sigma_xy * nx + sigma_yy * ny)

                drag += fx
                lift += fy

        # Convert to coefficients
        U_char = self.um
        ref_force = 0.5 * self.rho * U_char**2 * 0.1  # cylinder diameter = 0.1

        if ref_force > 0:
            drag_coeff = drag / ref_force
            lift_coeff = lift / ref_force
        else:
            drag_coeff = 0.0
            lift_coeff = 0.0

        return drag_coeff, lift_coeff

    def _compute_velocity_gradients(self, node):
        """Compute velocity gradients at a node."""
        # Simplified gradient computation
        # In practice, you'd use scikit-fem's gradient computation
        grad_u = np.zeros((2, 2))

        # This is a placeholder - proper implementation would use
        # scikit-fem's gradient computation functions
        return grad_u

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity field."""
        return self.u, self.v

    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field."""
        return self.p

    def get_vorticity_field(self) -> np.ndarray:
        """Compute vorticity field."""
        # Simplified vorticity computation
        # In practice, you'd use scikit-fem's curl computation
        vorticity = np.zeros(self.mesh.p.shape[1])

        # This is a placeholder - proper implementation would use
        # scikit-fem's curl computation functions
        return vorticity

    def visualize_solution(self, save_path: str = None):
        """Visualize the current solution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Velocity magnitude
        velocity_magnitude = np.sqrt(self.u**2 + self.v**2)
        plot(self.velocity_basis, velocity_magnitude, ax=axes[0, 0])
        axes[0, 0].set_title('Velocity Magnitude')

        # Pressure
        plot(self.pressure_basis, self.p, ax=axes[0, 1])
        axes[0, 1].set_title('Pressure')

        # Vorticity
        vorticity = self.get_vorticity_field()
        plot(self.velocity_basis, vorticity, ax=axes[1, 0])
        axes[1, 0].set_title('Vorticity')

        # Streamlines
        axes[1, 1].quiver(self.mesh.p[0, ::10], self.mesh.p[1, ::10],
                         self.u[::10], self.v[::10], alpha=0.7)
        axes[1, 1].set_title('Velocity Vectors')
        axes[1, 1].set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Solution visualization saved to: {save_path}")

        plt.show()
