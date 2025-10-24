"""
Lattice Boltzmann Method (LBM) solver with MRT collision model for 2D flows.
Implements D2Q9 lattice structure with Multiple Relaxation Time collision operator.
"""

import numpy as np
from typing import Tuple, Optional


class LBM_MRT_Solver:
    """
    LBM solver using D2Q9 lattice with MRT collision model.

    The D2Q9 lattice has 9 velocity directions:
    - 0: rest particle (0, 0)
    - 1-4: cardinal directions (±1, 0), (0, ±1)
    - 5-8: diagonal directions (±1, ±1)
    """

    def __init__(self, nx: int, ny: int, tau: float, rho0: float = 1.0):
        """
        Initialize LBM solver.

        Args:
            nx, ny: Grid dimensions
            tau: Relaxation time (related to viscosity)
            rho0: Reference density
        """
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.rho0 = rho0

        # D2Q9 velocity vectors
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

        # D2Q9 weights
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

        # Initialize distribution functions
        self.f = np.zeros((9, nx, ny))
        self.feq = np.zeros((9, nx, ny))

        # MRT transformation matrix (9x9)
        self._setup_mrt_matrix()

        # Initialize with equilibrium
        self._initialize_equilibrium()

    def _setup_mrt_matrix(self):
        """Setup MRT transformation matrix for D2Q9."""
        # MRT transformation matrix M (9x9)
        # This transforms from velocity space to moment space
        self.M = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],                    # density
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],              # energy
            [4, -2, -2, -2, -2, 1, 1, 1, 1],               # energy squared
            [0, 1, 0, -1, 0, 1, -1, -1, 1],                # momentum x
            [0, -2, 0, 2, 0, 1, -1, -1, 1],                # momentum x squared
            [0, 0, 1, 0, -1, 1, 1, -1, -1],                # momentum y
            [0, 0, -2, 0, 2, 1, 1, -1, -1],                # momentum y squared
            [0, 1, -1, 1, -1, 0, 0, 0, 0],                 # stress tensor xx-yy
            [0, 0, 0, 0, 0, 1, -1, 1, -1]                  # stress tensor xy
        ])

        # Inverse transformation matrix
        self.Minv = np.linalg.inv(self.M)

        # Relaxation times for different moments
        # s0=s3=s5=0 (conserved moments), s1=s2=s4=s6=s7=s8=1/tau
        self.s = np.array([0, 1/self.tau, 1/self.tau, 0, 1/self.tau,
                          0, 1/self.tau, 1/self.tau, 1/self.tau])

    def _initialize_equilibrium(self):
        """Initialize distribution functions with equilibrium values."""
        # Initialize with uniform density and zero velocity
        rho = np.ones((self.nx, self.ny)) * self.rho0
        ux = np.zeros((self.nx, self.ny))
        uy = np.zeros((self.nx, self.ny))

        self._compute_equilibrium(rho, ux, uy)
        self.f = self.feq.copy()

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

    def _mrt_collision(self):
        """Apply MRT collision operator."""
        # Transform to moment space
        m = np.zeros((9, self.nx, self.ny))
        for i in range(9):
            for j in range(9):
                m[i] += self.M[i, j] * self.f[j]

        # Compute equilibrium moments
        rho, ux, uy = self._compute_macroscopic()
        meq = np.zeros((9, self.nx, self.ny))

        # Equilibrium moments
        meq[0] = rho                                    # density
        meq[1] = -2*rho + 3*rho*(ux**2 + uy**2)       # energy
        meq[2] = rho - 3*rho*(ux**2 + uy**2)          # energy squared
        meq[3] = rho*ux                                # momentum x
        meq[4] = -rho*ux                              # momentum x squared
        meq[5] = rho*uy                                # momentum y
        meq[6] = -rho*uy                              # momentum y squared
        meq[7] = rho*(ux**2 - uy**2)                  # stress tensor xx-yy
        meq[8] = rho*ux*uy                            # stress tensor xy

        # Relaxation in moment space
        for i in range(9):
            m[i] = m[i] - self.s[i] * (m[i] - meq[i])

        # Transform back to velocity space
        for i in range(9):
            self.f[i] = 0
            for j in range(9):
                self.f[i] += self.Minv[i, j] * m[j]

    def _streaming(self):
        """Apply streaming step."""
        f_new = np.zeros_like(self.f)

        for i in range(9):
            # Periodic boundary conditions for now
            # Will be overridden by specific boundary conditions
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

    def _get_opposite_direction(self, i: int) -> int:
        """Get opposite direction index for bounce-back."""
        opposites = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        return opposites[i]

    def _zou_he_inlet(self, u_inlet: float, inlet_mask: np.ndarray):
        """
        Apply Zou-He boundary condition for inlet.

        Args:
            u_inlet: Inlet velocity
            inlet_mask: Boolean array for inlet nodes
        """
        # For horizontal inlet (left boundary)
        # Set f1, f5, f8 based on known velocity
        rho = np.sum(self.f[:, inlet_mask], axis=0)

        # Zou-He relations for horizontal inlet
        self.f[1, inlet_mask] = self.f[3, inlet_mask] + (2/3) * rho * u_inlet
        self.f[5, inlet_mask] = self.f[7, inlet_mask] + (1/6) * rho * u_inlet
        self.f[8, inlet_mask] = self.f[6, inlet_mask] + (1/6) * rho * u_inlet

    def _zou_he_outlet(self, outlet_mask: np.ndarray):
        """
        Apply Zou-He boundary condition for outlet.

        Args:
            outlet_mask: Boolean array for outlet nodes
        """
        # For horizontal outlet (right boundary)
        # Set f3, f6, f7 based on zero gradient condition
        rho = np.sum(self.f[:, outlet_mask], axis=0)

        # Zou-He relations for horizontal outlet
        self.f[3, outlet_mask] = self.f[1, outlet_mask]
        self.f[6, outlet_mask] = self.f[8, outlet_mask]
        self.f[7, outlet_mask] = self.f[5, outlet_mask]

    def step(self, cylinder_mask: np.ndarray, inlet_mask: np.ndarray,
             outlet_mask: np.ndarray, u_inlet: float):
        """
        Perform one LBM time step.

        Args:
            cylinder_mask: Boolean array for cylinder nodes
            inlet_mask: Boolean array for inlet nodes
            outlet_mask: Boolean array for outlet nodes
            u_inlet: Inlet velocity
        """
        # Collision step
        self._mrt_collision()

        # Apply boundary conditions before streaming
        self._bounce_back_cylinder(cylinder_mask)
        self._zou_he_inlet(u_inlet, inlet_mask)
        self._zou_he_outlet(outlet_mask)

        # Streaming step
        self._streaming()

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
