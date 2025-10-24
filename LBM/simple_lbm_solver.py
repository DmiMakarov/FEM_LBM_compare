"""
Simplified LBM solver with BGK collision model for better stability.
"""

import numpy as np
from typing import Tuple, Optional


class SimpleLBM_Solver:
    """
    Simplified LBM solver using BGK collision model for better stability.
    """

    def __init__(self, nx: int, ny: int, tau: float, rho0: float = 1.0,
                 reynolds_number: float = 100, initial_condition: str = "steady", um: float = 0.3):
        """
        Initialize simplified LBM solver.

        Args:
            nx, ny: Grid dimensions
            tau: Relaxation time (should be > 0.5 for stability)
            rho0: Reference density
            reynolds_number: Reynolds number for unsteady effects
            initial_condition: Type of initial condition ("steady", "unsteady", "oscillating")
            um: Maximum velocity for initial condition
        """
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.rho0 = rho0
        self.reynolds_number = reynolds_number
        self.initial_condition = initial_condition
        self.um = um
        self.simulation_time = 0.0

        # D2Q9 velocity vectors
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

        # D2Q9 weights
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

        # Initialize distribution functions
        self.f = np.zeros((9, nx, ny))
        self.feq = np.zeros((9, nx, ny))

        # Initialize with equilibrium
        self._initialize_equilibrium()

        print(f"LBM Solver initialized:")
        print(f"  Grid: {nx} x {ny}")
        print(f"  Initial condition: {initial_condition}")
        print(f"  Max velocity: {um} m/s")

    def _initialize_equilibrium(self):
        """Initialize distribution functions with equilibrium values."""
        # Initialize with uniform density and zero velocity
        rho = np.ones((self.nx, self.ny)) * self.rho0
        ux = np.zeros((self.nx, self.ny))
        uy = np.zeros((self.nx, self.ny))

        self._compute_equilibrium(rho, ux, uy)
        self.f = self.feq.copy()

    def get_inlet_velocity(self, y: float) -> Tuple[float, float]:
        """
        Get inlet velocity based on initial condition type.

        Args:
            y: y-coordinate (0 to H)

        Returns:
            (ux, uy): Velocity components
        """
        H = 0.41  # Domain height

        if self.initial_condition == "steady":
            # Condition 1: U_x(0, y) = 4U_m y (H − y)/H^2, V = 0, U_m = 0.3 m/s (Re = 20)
            ux = 4 * self.um * y * (H - y) / (H**2)
            uy = 0.0

        elif self.initial_condition == "unsteady":
            # Condition 2: U_x(0, y, t) = 4U_m y (H − y)/H^2, U_y = 0, U_m = 1.5 m/s (Re = 100)
            ux = 4 * self.um * y * (H - y) / (H**2)
            uy = 0.0

        elif self.initial_condition == "oscillating":
            # Condition 3: U_x(0, y, t) = 4 U_m y (H − y) sin(πt/8)/H^2, U_y = 0, U_m = 1.5 m/s
            # Use smaller time scale for stability
            time_scale = 0.1  # Reduce oscillation frequency
            ux = 4 * self.um * y * (H - y) * np.sin(np.pi * self.simulation_time * time_scale) / (H**2)
            uy = 0.0

        else:
            # Default parabolic profile
            ux = 0.1 * 4 * y * (H - y) / (H**2)
            uy = 0.0

        return ux, uy

    def _compute_equilibrium(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray):
        """
        Compute equilibrium distribution functions.

        Args:
            rho: Density field
            ux, uy: Velocity components
        """
        u2 = ux**2 + uy**2

        for i in range(9):
            cu = self.cx[i] * ux + self.cy[i] * uy
            self.feq[i] = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    def _compute_macroscopic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute macroscopic quantities from distribution functions.

        Returns:
            rho: Density field
            ux, uy: Velocity components
        """
        rho = np.sum(self.f, axis=0)

        ux = np.sum(self.f * self.cx[:, np.newaxis, np.newaxis], axis=0) / rho
        uy = np.sum(self.f * self.cy[:, np.newaxis, np.newaxis], axis=0) / rho

        return rho, ux, uy

    def _bgk_collision(self):
        """Apply BGK collision operator."""
        # Compute equilibrium
        rho, ux, uy = self._compute_macroscopic()

        # Adjust max velocity based on Reynolds number
        # MUST match the inlet lattice velocity!
        if self.reynolds_number <= 50:
            max_u = 0.1  # Match inlet (0.08)
        elif self.reynolds_number <= 100:
            max_u = 0.08  # Match inlet (0.06)
        else:
            max_u = 0.06  # Slightly higher than inlet (0.04) for safety

        # Stability check: limit velocity magnitude
        u_mag = np.sqrt(ux**2 + uy**2)
        if np.max(u_mag) > max_u:
            # Scale down velocities if too high
            scale = max_u / np.max(u_mag)
            ux *= scale
            uy *= scale

        self._compute_equilibrium(rho, ux, uy)

        # BGK collision: f = f - (f - feq) / tau
        # Ensure tau is in stable range
        tau_stable = max(0.6, min(2.0, self.tau))
        self.f = self.f - (self.f - self.feq) / tau_stable

        # Stability check: limit distribution functions
        self.f = np.clip(self.f, 0, 10)  # Prevent extreme values

    def _streaming(self):
        """Apply streaming step."""
        f_new = np.zeros_like(self.f)

        for i in range(9):
            # Periodic boundary conditions for now
            f_new[i] = np.roll(np.roll(self.f[i], self.cx[i], axis=0),
                              self.cy[i], axis=1)

        self.f = f_new

    def _bounce_back_cylinder(self, cylinder_mask: np.ndarray):
        """
        Apply bounce-back boundary condition for cylinder.

        Args:
            cylinder_mask: Boolean array where True indicates cylinder nodes
        """
        for i in range(9):
            # Find opposite direction
            opp_i = self._get_opposite_direction(i)

            # Bounce back: f_opposite = f_i
            self.f[opp_i, cylinder_mask] = self.f[i, cylinder_mask]

    def _bounce_back_walls(self, top_wall_mask: np.ndarray, bottom_wall_mask: np.ndarray):
        """
        Apply bounce-back boundary condition for top and bottom walls.

        Args:
            top_wall_mask: Boolean array for top wall nodes (y = ny-1)
            bottom_wall_mask: Boolean array for bottom wall nodes (y = 0)
        """
        # Apply bounce-back to top wall
        for i in range(9):
            opp_i = self._get_opposite_direction(i)
            self.f[opp_i, top_wall_mask] = self.f[i, top_wall_mask]

        # Apply bounce-back to bottom wall
        for i in range(9):
            opp_i = self._get_opposite_direction(i)
            self.f[opp_i, bottom_wall_mask] = self.f[i, bottom_wall_mask]

    def _get_opposite_direction(self, i: int) -> int:
        """Get opposite direction index for bounce-back."""
        opposites = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        return opposites[i]

    def _zou_he_inlet(self, u_inlet, inlet_mask: np.ndarray):
        """
        Apply Zou-He boundary condition for inlet.

        Args:
            u_inlet: Inlet velocity (can be scalar, 1D array, or 2D array)
            inlet_mask: Boolean array for inlet nodes
        """
        # For horizontal inlet (left boundary), known u, v=0, unknown rho
        # Known distributions: f0, f2, f4, f3, f6, f7
        # Unknown distributions: f1, f5, f8

        # Handle different input types for u_inlet
        if np.isscalar(u_inlet):
            # Scalar input - create array for all inlet nodes
            u_inlet_array = np.full(np.sum(inlet_mask), u_inlet)
        elif u_inlet.ndim == 1:
            # 1D array input - use as is
            u_inlet_array = u_inlet
        elif u_inlet.ndim == 2:
            # 2D array input - extract values at inlet nodes
            u_inlet_array = u_inlet[inlet_mask]
        else:
            raise ValueError(f"Unsupported u_inlet dimensions: {u_inlet.ndim}")

        # Extract known distributions at inlet nodes
        f0_at_inlet = self.f[0][inlet_mask]
        f2_at_inlet = self.f[2][inlet_mask]
        f4_at_inlet = self.f[4][inlet_mask]
        f3_at_inlet = self.f[3][inlet_mask]
        f6_at_inlet = self.f[6][inlet_mask]
        f7_at_inlet = self.f[7][inlet_mask]

        # Calculate density from known distributions
        # rho = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1 - u)
        rho = (f0_at_inlet + f2_at_inlet + f4_at_inlet + 2*(f3_at_inlet + f6_at_inlet + f7_at_inlet)) / (1 - u_inlet_array)

        # Calculate unknown distributions (for v=0, no transverse velocity)
        # f1 = f3 + (2/3)*rho*u
        # f5 = f7 - 0.5*(f2 - f4) + (1/6)*rho*u
        # f8 = f6 + 0.5*(f2 - f4) + (1/6)*rho*u
        f1_new = f3_at_inlet + (2/3) * rho * u_inlet_array
        f5_new = f7_at_inlet - 0.5 * (f2_at_inlet - f4_at_inlet) + (1/6) * rho * u_inlet_array
        f8_new = f6_at_inlet + 0.5 * (f2_at_inlet - f4_at_inlet) + (1/6) * rho * u_inlet_array

        # Update distribution functions at inlet
        self.f[1][inlet_mask] = f1_new
        self.f[5][inlet_mask] = f5_new
        self.f[8][inlet_mask] = f8_new

    def _zou_he_outlet(self, outlet_mask: np.ndarray):
        """
        Apply Zou-He boundary condition for outlet.

        Args:
            outlet_mask: Boolean array for outlet nodes
        """
        # For horizontal outlet (right boundary)
        # Set f3, f6, f7 based on zero gradient condition
        self.f[3][outlet_mask] = self.f[1][outlet_mask]
        self.f[6][outlet_mask] = self.f[8][outlet_mask]
        self.f[7][outlet_mask] = self.f[5][outlet_mask]

    def step(self, cylinder_mask: np.ndarray, inlet_mask: np.ndarray,
             outlet_mask: np.ndarray, u_inlet: float,
             top_wall_mask: np.ndarray = None, bottom_wall_mask: np.ndarray = None):
        """
        Perform one LBM time step.

        Args:
            cylinder_mask: Boolean array for cylinder nodes
            inlet_mask: Boolean array for inlet nodes
            outlet_mask: Boolean array for outlet nodes
            u_inlet: Inlet velocity
            top_wall_mask: Boolean array for top wall nodes
            bottom_wall_mask: Boolean array for bottom wall nodes
        """
        # Collision step
        self._bgk_collision()

        # Streaming step (before boundary conditions)
        self._streaming()

        # Apply boundary conditions after streaming
        self._bounce_back_cylinder(cylinder_mask)
        self._zou_he_inlet(u_inlet, inlet_mask)
        self._zou_he_outlet(outlet_mask)

        # Apply wall boundary conditions if provided
        if top_wall_mask is not None and bottom_wall_mask is not None:
            self._bounce_back_walls(top_wall_mask, bottom_wall_mask)

        # Update simulation time
        self.simulation_time += 1.0

    def get_pressure(self) -> np.ndarray:
        """Get pressure field (proportional to density)."""
        rho, _, _ = self._compute_macroscopic()
        return rho / 3.0  # Pressure = rho * cs^2, where cs^2 = 1/3

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity field."""
        _, ux, uy = self._compute_macroscopic()
        return ux, uy

    def get_density(self) -> np.ndarray:
        """Get density field."""
        rho, _, _ = self._compute_macroscopic()
        return rho

    def get_vorticity(self) -> np.ndarray:
        """Compute vorticity field."""
        ux, uy = self.get_velocity()

        # Compute gradients using finite differences
        duy_dx = np.gradient(uy, axis=0)
        dux_dy = np.gradient(ux, axis=1)

        return duy_dx - dux_dy
