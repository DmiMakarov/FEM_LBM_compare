"""
Jacobian transformations for finite elements.
Implements coordinate transformations from reference to physical elements.
"""

import numpy as np
from typing import Tuple


class JacobianTransform:
    """Jacobian transformation for finite elements."""

    @staticmethod
    def compute_jacobian(coords: np.ndarray, dN_dxi: np.ndarray,
                        dN_deta: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix J = [∂x/∂ξ, ∂x/∂η; ∂y/∂ξ, ∂y/∂η].

        J[0,0] = ∂x/∂ξ = Σ x_i ∂N_i/∂ξ
        J[0,1] = ∂y/∂ξ = Σ y_i ∂N_i/∂ξ
        J[1,0] = ∂x/∂η = Σ x_i ∂N_i/∂η
        J[1,1] = ∂y/∂η = Σ y_i ∂N_i/∂η

        Args:
            coords: Physical coordinates of element nodes (n_nodes, 2)
            dN_dxi: Shape function derivatives w.r.t. ξ (n_nodes,)
            dN_deta: Shape function derivatives w.r.t. η (n_nodes,)

        Returns:
            Jacobian matrix (2, 2)
        """
        J = np.zeros((2, 2))
        for i in range(len(coords)):
            J[0, 0] += coords[i, 0] * dN_dxi[i]
            J[0, 1] += coords[i, 1] * dN_dxi[i]
            J[1, 0] += coords[i, 0] * dN_deta[i]
            J[1, 1] += coords[i, 1] * dN_deta[i]
        return J

    @staticmethod
    def physical_derivatives(dN_dxi: np.ndarray, dN_deta: np.ndarray,
                            J_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform derivatives to physical coordinates.
        [∂N/∂x, ∂N/∂y]^T = J^{-1} [∂N/∂ξ, ∂N/∂η]^T

        Args:
            dN_dxi: Shape function derivatives w.r.t. ξ (n_nodes,)
            dN_deta: Shape function derivatives w.r.t. η (n_nodes,)
            J_inv: Inverse Jacobian matrix (2, 2)

        Returns:
            Tuple of (dN_dx, dN_dy) - derivatives w.r.t. physical coordinates
        """
        dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
        dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
        return dN_dx, dN_dy

    @staticmethod
    def test_jacobian():
        """Test Jacobian computation for a simple triangle."""
        print("Testing Jacobian computation...")

        # Simple triangle with vertices at (0,0), (1,0), (0,1)
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Shape function derivatives (constant for linear elements)
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])

        # Compute Jacobian
        J = JacobianTransform.compute_jacobian(coords, dN_dxi, dN_deta)

        # Expected Jacobian for this triangle
        J_expected = np.array([[1.0, 0.0], [0.0, 1.0]])

        error = np.max(np.abs(J - J_expected))
        print(f"  Jacobian error: {error:.2e}")
        assert error < 1e-10, f"Jacobian test failed: error = {error}"

        # Test determinant (should be 1.0 for unit triangle)
        det_J = np.linalg.det(J)
        print(f"  Jacobian determinant: {det_J:.6f}")
        assert abs(det_J - 1.0) < 1e-10, f"Jacobian determinant test failed: {det_J}"

        # Test inverse
        J_inv = np.linalg.inv(J)
        J_inv_expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        inv_error = np.max(np.abs(J_inv - J_inv_expected))
        print(f"  Inverse Jacobian error: {inv_error:.2e}")
        assert inv_error < 1e-10, f"Inverse Jacobian test failed: error = {inv_error}"

        print("✓ Jacobian computation test passed")

    @staticmethod
    def test_derivative_transformation():
        """Test transformation of derivatives to physical coordinates."""
        print("Testing derivative transformation...")

        # Simple triangle
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])

        J = JacobianTransform.compute_jacobian(coords, dN_dxi, dN_deta)
        J_inv = np.linalg.inv(J)

        # Transform derivatives
        dN_dx, dN_dy = JacobianTransform.physical_derivatives(dN_dxi, dN_deta, J_inv)

        # For this simple triangle, derivatives should be unchanged
        dN_dx_expected = dN_dxi
        dN_dy_expected = dN_deta

        x_error = np.max(np.abs(dN_dx - dN_dx_expected))
        y_error = np.max(np.abs(dN_dy - dN_dy_expected))

        print(f"  ∂N/∂x error: {x_error:.2e}")
        print(f"  ∂N/∂y error: {y_error:.2e}")

        assert x_error < 1e-10, f"∂N/∂x transformation failed: error = {x_error}"
        assert y_error < 1e-10, f"∂N/∂y transformation failed: error = {y_error}"

        print("✓ Derivative transformation test passed")


if __name__ == "__main__":
    # Test Jacobian transformations
    print("Testing Jacobian transformations...")
    JacobianTransform.test_jacobian()
    JacobianTransform.test_derivative_transformation()
    print("All Jacobian transformation tests passed!")
