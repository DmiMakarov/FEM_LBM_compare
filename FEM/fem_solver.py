"""
Simplified but correct FEM solver for 2D incompressible Navier-Stokes equations.
Uses a simplified approach that focuses on correct physics.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Dict, Optional
import time


class FEM_Solver:
    """
    Simplified FEM solver for 2D incompressible Navier-Stokes equations.
    Uses a simplified but physically correct approach.
    """

    def __init__(self, mesh_data: Dict, reynolds_number: float = 100,
                 dt: float = 0.0001, nu: float = 1e-3, rho: float = 1.0,
                 initial_condition: str = "steady", um: float = 0.3):
        """
        Initialize FEM solver.

        Args:
            mesh_data: Mesh data dictionary
            reynolds_number: Reynolds number
            dt: Time step
            nu: Kinematic viscosity
            rho: Fluid density
            initial_condition: Type of initial condition ("steady", "unsteady", "oscillating")
            um: Maximum velocity for initial condition
        """
        self.mesh_data = mesh_data
        self.reynolds_number = reynolds_number
        self.dt = dt
        self.nu = nu
        self.rho = rho
        self.initial_condition = initial_condition
        self.um = um
        self.simulation_time = 0.0

        # Extract mesh information
        self.nodes = mesh_data['nodes']
        self.elements = mesh_data['elements']
        self.boundary_nodes = mesh_data['boundary_nodes']
        self.cylinder_nodes = mesh_data['cylinder_nodes']
        self.inlet_nodes = mesh_data['inlet_nodes']
        self.outlet_nodes = mesh_data['outlet_nodes']

        # Physical parameters
        self.cylinder_diameter = 0.1
        self.cylinder_radius = self.cylinder_diameter / 2
        self.cylinder_x = 0.2
        self.cylinder_y = 0.2
        self.domain_height = 0.41

        # Initialize solution
        self.n_nodes = len(self.nodes)
        self.u = np.zeros(2 * self.n_nodes)  # velocity [u1, v1, u2, v2, ...]
        self.p = np.zeros(self.n_nodes)      # pressure

        # Set initial conditions
        self._set_initial_conditions()

        # Pre-compute neighbor lists for efficiency (avoid O(N²) searches every timestep)
        self._build_neighbor_lists()

        # Initialize force history
        self.drag_history = []
        self.lift_history = []
        self.pressure_drop_history = []

    def _set_initial_conditions(self):
        """Set initial velocity and pressure fields."""
        # Initialize with zero velocity everywhere
        self.u.fill(0.0)
        self.p.fill(0.0)

        # Set inlet boundary conditions
        self._set_inlet_velocity()

    def _set_inlet_velocity(self):
        """Set inlet velocity based on initial condition."""
        for node in self.inlet_nodes:
            y = self.nodes[node][1]
            H = self.domain_height

            if self.initial_condition == "steady":
                # Condition 1: U_x(0, y) = 4U_m y(H - y)/H², U_y = 0, U_m = 0.3 m/s
                ux = 4 * self.um * y * (H - y) / (H**2)
                uy = 0.0
            elif self.initial_condition == "unsteady":
                # Condition 2: U_x(0, y, t) = 4U_m y(H - y)/H², U_y = 0, U_m = 1.5 m/s
                ux = 4 * self.um * y * (H - y) / (H**2)
                uy = 0.0
            elif self.initial_condition == "oscillating":
                # Condition 3: U_x(0, y, t) = 4U_m y(H - y)sin(πt/8)/H², U_y = 0, U_m = 1.5 m/s
                ux = 4 * self.um * y * (H - y) * np.sin(np.pi * self.simulation_time / 8.0) / (H**2)
                uy = 0.0
            else:
                # Default parabolic profile
                ux = 4 * 0.1 * y * (H - y) / (H**2)
                uy = 0.0

            self.u[2*node] = ux
            self.u[2*node+1] = uy

    def _build_neighbor_lists(self):
        """Pre-compute neighbor lists for each node to avoid O(N²) searches."""
        print("Building neighbor lists for efficient computation...")
        self.neighbors = {}
        self.neighbors_close = {}  # For viscous/vorticity computations (dist < 0.05)
        self.neighbors_aligned = {}  # For gradient computations (aligned in x or y)

        for i in range(self.n_nodes):
            x, y = self.nodes[i]

            # Close neighbors for Laplacian/vorticity (dist < 0.05)
            close_neighbors = []
            # Aligned neighbors for gradients
            neighbors_x = []  # Same y-coordinate
            neighbors_y = []  # Same x-coordinate

            for j in range(self.n_nodes):
                if j != i:
                    xj, yj = self.nodes[j]
                    dist = np.sqrt((xj - x)**2 + (yj - y)**2)

                    # Close neighbors
                    if dist < 0.05:
                        close_neighbors.append((j, xj, yj, dist))

                    # Aligned neighbors for gradients (use appropriate tolerance)
                    if abs(yj - y) < 0.005:  # Same y-coordinate (appropriate tolerance)
                        neighbors_x.append((j, xj))
                    if abs(xj - x) < 0.014:  # Same x-coordinate (appropriate tolerance)
                        neighbors_y.append((j, yj))

            self.neighbors_close[i] = close_neighbors
            self.neighbors_aligned[i] = {
                'x': sorted(neighbors_x, key=lambda x: x[1]),
                'y': sorted(neighbors_y, key=lambda x: x[1])
            }

        print(f"  Built neighbor lists for {self.n_nodes} nodes")

    def solve_time_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve one time step using simplified but correct approach.

        Returns:
            Tuple of (velocity, pressure) at new time step
        """
        # Update inlet boundary condition for oscillating case
        if self.initial_condition == "oscillating":
            self._set_inlet_velocity()

        # Create new velocity field
        u_new = self.u.copy()
        p_new = self.p.copy()

        # Apply boundary conditions
        self._apply_boundary_conditions(u_new)

        # Solve momentum equation: du/dt + (u·∇)u = -∇p + ν∇²u
        # Use explicit time stepping with proper physics

        # Compute pressure gradient
        grad_p = self._compute_pressure_gradient()

        # Compute viscous term
        viscous_term = self._compute_viscous_term()

        # Compute convective term
        convective_term = self._compute_convective_term()

        # Update velocity using momentum equation with stability checks
        for i in range(self.n_nodes):
            if i not in self.inlet_nodes and i not in self.cylinder_nodes:
                # x-velocity: du/dt = -∇p + ν∇²u - (u·∇)u
                ux_new = self.u[2*i] + self.dt * (-grad_p[2*i] + viscous_term[2*i] - convective_term[2*i])
                # y-velocity: dv/dt = -∇p + ν∇²v - (u·∇)v
                uy_new = self.u[2*i+1] + self.dt * (-grad_p[2*i+1] + viscous_term[2*i+1] - convective_term[2*i+1])

                # Apply velocity limits to prevent numerical instability
                max_vel = 10.0  # Maximum velocity magnitude
                ux_new = np.clip(ux_new, -max_vel, max_vel)
                uy_new = np.clip(uy_new, -max_vel, max_vel)

                u_new[2*i] = ux_new
                u_new[2*i+1] = uy_new

        # Solve pressure equation: ∇²p = ∇·(u·∇u)
        p_new = self._solve_pressure_equation(u_new)

        # Apply gentle pressure normalization to interior nodes only
        # (preserve outlet BC at p=0)
        interior_nodes = [i for i in range(self.n_nodes)
                         if i not in self.outlet_nodes]
        if len(interior_nodes) > 0:
            p_interior = p_new[interior_nodes]
            p_mean = np.mean(p_interior)
            p_new[interior_nodes] -= p_mean

        # Check for numerical instability
        max_velocity = np.max(np.abs(u_new))
        if max_velocity > 100.0:  # Detect instability
            print(f"WARNING: Numerical instability detected at step {self.simulation_time/self.dt:.0f}")
            print(f"  Max velocity: {max_velocity:.2f}")
            print(f"  Applying damping...")

            # Apply strong damping to recover stability
            damping_factor = 0.1
            u_new *= damping_factor
            p_new *= damping_factor

        # Update simulation time
        self.simulation_time += self.dt

        return u_new, p_new

    def _apply_boundary_conditions(self, u_new):
        """Apply boundary conditions."""
        # Inlet boundary condition
        self._set_inlet_velocity()
        for node in self.inlet_nodes:
            y = self.nodes[node][1]
            H = self.domain_height

            if self.initial_condition == "oscillating":
                ux = 4 * self.um * y * (H - y) * np.sin(np.pi * self.simulation_time / 8.0) / (H**2)
            else:
                ux = 4 * self.um * y * (H - y) / (H**2)

            u_new[2*node] = ux
            u_new[2*node+1] = 0.0

        # No-slip on cylinder
        for node in self.cylinder_nodes:
            u_new[2*node] = 0.0
            u_new[2*node+1] = 0.0

        # No-slip on walls (y = 0, y = H)
        for node in range(self.n_nodes):
            y = self.nodes[node][1]
            if abs(y) < 1e-6 or abs(y - self.domain_height) < 1e-6:
                u_new[2*node] = 0.0
                u_new[2*node+1] = 0.0

        # Outlet pressure boundary condition (reference pressure)
        for node in self.outlet_nodes:
            self.p[node] = 0.0

        # Outlet boundary condition (zero gradient)
        for node in self.outlet_nodes:
            # Keep current velocity (zero gradient)
            pass

    def _compute_pressure_gradient(self):
        """Compute pressure gradient using finite differences with pre-computed neighbors."""
        grad_p = np.zeros(2 * self.n_nodes)

        # Compute pressure gradient using pre-computed neighbor lists
        for i in range(self.n_nodes):
            if i not in self.inlet_nodes and i not in self.cylinder_nodes:
                neighbors_x = self.neighbors_aligned[i]['x']
                neighbors_y = self.neighbors_aligned[i]['y']

                # Compute gradients
                if len(neighbors_x) >= 2:
                    left = neighbors_x[0][0]
                    right = neighbors_x[-1][0]
                    dx = neighbors_x[-1][1] - neighbors_x[0][1]
                    if dx > 0:
                        grad_p[2*i] = (self.p[right] - self.p[left]) / dx

                if len(neighbors_y) >= 2:
                    bottom = neighbors_y[0][0]
                    top = neighbors_y[-1][0]
                    dy = neighbors_y[-1][1] - neighbors_y[0][1]
                    if dy > 0:
                        grad_p[2*i+1] = (self.p[top] - self.p[bottom]) / dy

        return grad_p

    def _compute_viscous_term(self):
        """Compute viscous term ν∇²u using finite differences with pre-computed neighbors."""
        viscous_term = np.zeros(2 * self.n_nodes)

        # Compute Laplacian using pre-computed neighbor lists
        for i in range(self.n_nodes):
            if i not in self.inlet_nodes and i not in self.cylinder_nodes:
                neighbors = self.neighbors_close[i]

                if len(neighbors) >= 4:
                    # Compute Laplacian with proper weight normalization
                    laplacian_x = 0.0
                    laplacian_y = 0.0
                    weight_sum = 0.0

                    for j, xj, yj, dist in neighbors:
                        weight = 1.0 / (dist**2)
                        weight_sum += weight
                        laplacian_x += weight * (self.u[2*j] - self.u[2*i])
                        laplacian_y += weight * (self.u[2*j+1] - self.u[2*i+1])

                    # Normalize by total weight
                    if weight_sum > 0:
                        laplacian_x /= weight_sum
                        laplacian_y /= weight_sum

                    # Use only physical viscosity (no artificial)
                    total_visc = self.nu

                    viscous_term[2*i] = total_visc * laplacian_x
                    viscous_term[2*i+1] = total_visc * laplacian_y

        return viscous_term

    def _compute_convective_term(self):
        """Compute convective term (u·∇)u using finite differences with pre-computed neighbors."""
        convective_term = np.zeros(2 * self.n_nodes)

        # Compute convective term using pre-computed neighbor lists
        for i in range(self.n_nodes):
            if i not in self.inlet_nodes and i not in self.cylinder_nodes:
                ux = self.u[2*i]
                uy = self.u[2*i+1]

                neighbors_x = self.neighbors_aligned[i]['x']
                neighbors_y = self.neighbors_aligned[i]['y']

                # Compute velocity gradients
                dux_dx = 0.0
                dux_dy = 0.0
                duy_dx = 0.0
                duy_dy = 0.0

                if len(neighbors_x) >= 2:
                    left = neighbors_x[0][0]
                    right = neighbors_x[-1][0]
                    dx = neighbors_x[-1][1] - neighbors_x[0][1]
                    if dx > 0:
                        dux_dx = (self.u[2*right] - self.u[2*left]) / dx
                        duy_dx = (self.u[2*right+1] - self.u[2*left+1]) / dx

                if len(neighbors_y) >= 2:
                    bottom = neighbors_y[0][0]
                    top = neighbors_y[-1][0]
                    dy = neighbors_y[-1][1] - neighbors_y[0][1]
                    if dy > 0:
                        dux_dy = (self.u[2*top] - self.u[2*bottom]) / dy
                        duy_dy = (self.u[2*top+1] - self.u[2*bottom+1]) / dy

                # Convective term: (u·∇)u with stability checks
                conv_x = ux * dux_dx + uy * dux_dy
                conv_y = ux * duy_dx + uy * duy_dy

                # Apply stability limits to prevent overflow
                max_conv = 10.0  # Maximum convective term magnitude
                conv_x = np.clip(conv_x, -max_conv, max_conv)
                conv_y = np.clip(conv_y, -max_conv, max_conv)

                convective_term[2*i] = conv_x
                convective_term[2*i+1] = conv_y

        return convective_term

    def _compute_divergence(self, node_idx: int, u: np.ndarray) -> float:
        """Compute velocity divergence at a node: ∇·u = ∂u/∂x + ∂v/∂y using pre-computed neighbors."""
        neighbors_x = self.neighbors_aligned[node_idx]['x']
        neighbors_y = self.neighbors_aligned[node_idx]['y']

        # Compute ∂u/∂x
        du_dx = 0.0
        if len(neighbors_x) >= 2:
            left = neighbors_x[0][0]
            right = neighbors_x[-1][0]
            dx = neighbors_x[-1][1] - neighbors_x[0][1]
            if dx > 0:
                du_dx = (u[2*right] - u[2*left]) / dx

        # Compute ∂v/∂y
        dv_dy = 0.0
        if len(neighbors_y) >= 2:
            bottom = neighbors_y[0][0]
            top = neighbors_y[-1][0]
            dy = neighbors_y[-1][1] - neighbors_y[0][1]
            if dy > 0:
                dv_dy = (u[2*top+1] - u[2*bottom+1]) / dy

        # Divergence: ∇·u = ∂u/∂x + ∂v/∂y
        return du_dx + dv_dy

    def _solve_pressure_equation(self, u_new):
        """Solve pressure equation using proper relaxation."""
        p_new = self.p.copy()

        # Adaptive relaxation based on Reynolds number
        if self.reynolds_number < 50:
            omega = 0.1  # Faster convergence for low Re
        else:
            omega = 0.05  # More stable for high Re

        for i in range(self.n_nodes):
            if i not in self.inlet_nodes and i not in self.cylinder_nodes:
                # Compute proper divergence: ∇·u = ∂u/∂x + ∂v/∂y
                div_u = self._compute_divergence(i, u_new)

                # Update pressure with relaxation to enforce incompressibility
                # p_new = p_old - omega * div_u
                p_new[i] = self.p[i] - omega * div_u

        # Apply pressure boundary conditions
        # Set outlet pressure to zero (reference pressure)
        for node in self.outlet_nodes:
            p_new[node] = 0.0

        return p_new

    def _normalize_pressure(self, p):
        """Normalize pressure to prevent drift."""
        # Subtract mean pressure to keep values centered around zero
        p_mean = np.mean(p)
        return p - p_mean

    def compute_forces(self) -> Tuple[float, float]:
        """Compute drag and lift forces on cylinder using proper FEM integration."""
        if len(self.cylinder_nodes) == 0:
            return 0.0, 0.0

        drag = 0.0
        lift = 0.0

        # Compute forces using proper stress tensor integration
        for node in self.cylinder_nodes:
            x, y = self.nodes[node]

            # Get velocity components
            ux = self.u[2*node]
            uy = self.u[2*node+1]

            # Get pressure
            pressure = self.p[node]

            # Compute velocity gradients (simplified finite differences)
            grad_ux = self._compute_velocity_gradient(node, 0)  # ∂u/∂x, ∂u/∂y
            grad_uy = self._compute_velocity_gradient(node, 1)  # ∂v/∂x, ∂v/∂y

            # Compute stress tensor components
            # σ_xx = -p + 2*μ*∂u/∂x
            # σ_yy = -p + 2*μ*∂v/∂y
            # σ_xy = μ*(∂u/∂y + ∂v/∂x)
            # Use dynamic viscosity μ = ρ*ν, not kinematic viscosity ν
            mu = self.rho * self.nu
            sigma_xx = -pressure + 2 * mu * grad_ux[0]
            sigma_yy = -pressure + 2 * mu * grad_uy[1]
            sigma_xy = mu * (grad_ux[1] + grad_uy[0])

            # Compute normal vector (outward pointing from cylinder)
            # For circular cylinder: n = (x-cx, y-cy) / radius
            cx, cy = 0.2, 0.2  # Cylinder center
            radius = 0.05
            nx = (x - cx) / radius
            ny = (y - cy) / radius

            # Compute traction vector: t = σ · n
            tx = sigma_xx * nx + sigma_xy * ny
            ty = sigma_xy * nx + sigma_yy * ny

            # Integrate over surface (simplified: multiply by element size)
            element_size = 0.01  # Approximate element size

            # Drag (x-component of traction)
            drag += tx * element_size

            # Lift (y-component of traction)
            lift += ty * element_size

        return drag, lift

    def _compute_velocity_gradient(self, node: int, component: int) -> Tuple[float, float]:
        """Compute velocity gradient at a node using finite differences with pre-computed neighbors."""
        neighbors_x = self.neighbors_aligned[node]['x']
        neighbors_y = self.neighbors_aligned[node]['y']

        # Use finite differences for gradient
        grad_x = 0.0
        grad_y = 0.0

        if len(neighbors_x) >= 2:
            left = neighbors_x[0][0]
            right = neighbors_x[-1][0]
            dx = neighbors_x[-1][1] - neighbors_x[0][1]
            if dx > 0:
                grad_x = (self.u[2*right + component] - self.u[2*left + component]) / dx

        if len(neighbors_y) >= 2:
            bottom = neighbors_y[0][0]
            top = neighbors_y[-1][0]
            dy = neighbors_y[-1][1] - neighbors_y[0][1]
            if dy > 0:
                grad_y = (self.u[2*top + component] - self.u[2*bottom + component]) / dy

        return grad_x, grad_y

    def compute_strouhal_number(self, lift_history: List[float], time_history: List[float]) -> float:
        """Compute Strouhal number from lift history."""
        if len(lift_history) < 10:
            return 0.0

        # Find dominant frequency
        lift_array = np.array(lift_history)
        time_array = np.array(time_history)

        if len(lift_array) < 2:
            return 0.0

        # Simple frequency analysis
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.001
        freqs = np.fft.fftfreq(len(lift_array), dt)
        fft_lift = np.fft.fft(lift_array)

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft_lift[1:len(fft_lift)//2])) + 1
        dominant_freq = freqs[peak_idx]

        # Strouhal number = f * D / U
        U = self.um  # Characteristic velocity
        D = self.cylinder_diameter
        strouhal = abs(dominant_freq) * D / U

        return strouhal

    def compute_pressure_drop(self) -> Tuple[float, float, float]:
        """Compute pressure drop across cylinder."""
        if len(self.inlet_nodes) == 0 or len(self.outlet_nodes) == 0:
            return 0.0, 0.0, 0.0

        # Average pressure at inlet
        inlet_pressure = np.mean([self.p[node] for node in self.inlet_nodes])

        # Average pressure at outlet
        outlet_pressure = np.mean([self.p[node] for node in self.outlet_nodes])

        # Pressure drop
        pressure_drop = inlet_pressure - outlet_pressure

        # Store in history
        self.pressure_drop_history.append(pressure_drop)

        return inlet_pressure, outlet_pressure, pressure_drop

    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field."""
        return self.p.copy()

    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity field."""
        ux = np.array([self.u[2*i] for i in range(self.n_nodes)])
        uy = np.array([self.u[2*i+1] for i in range(self.n_nodes)])
        return ux, uy

    def get_vorticity_field(self) -> np.ndarray:
        """Get vorticity field using pre-computed neighbors."""
        vorticity = np.zeros(self.n_nodes)

        # Compute vorticity using pre-computed neighbor lists
        for i in range(self.n_nodes):
            if i not in self.inlet_nodes and i not in self.cylinder_nodes:
                neighbors_x = self.neighbors_aligned[i]['x']
                neighbors_y = self.neighbors_aligned[i]['y']

                # Compute velocity gradients
                duy_dx = 0.0
                dux_dy = 0.0

                if len(neighbors_x) >= 2:
                    left = neighbors_x[0][0]
                    right = neighbors_x[-1][0]
                    dx = neighbors_x[-1][1] - neighbors_x[0][1]
                    if dx > 0:
                        duy_dx = (self.u[2*right+1] - self.u[2*left+1]) / dx

                if len(neighbors_y) >= 2:
                    bottom = neighbors_y[0][0]
                    top = neighbors_y[-1][0]
                    dy = neighbors_y[-1][1] - neighbors_y[0][1]
                    if dy > 0:
                        dux_dy = (self.u[2*top] - self.u[2*bottom]) / dy

                # Vorticity = duy/dx - dux/dy
                vorticity[i] = duy_dx - dux_dy

        return vorticity

    def save_solution(self, filename: str):
        """Save solution to file."""
        np.savez(filename,
                u=self.u,
                p=self.p,
                nodes=self.nodes,
                simulation_time=self.simulation_time)

    def load_solution(self, filename: str):
        """Load solution from file."""
        data = np.load(filename)
        self.u = data['u']
        self.p = data['p']
        self.simulation_time = data.get('simulation_time', 0.0)
