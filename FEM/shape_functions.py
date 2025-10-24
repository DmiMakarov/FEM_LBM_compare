"""
Shape functions for finite element analysis.
Implements proper shape functions for linear triangular elements.
"""

import numpy as np
from typing import Tuple


class LinearTriangleShapeFunctions:
    """Shape functions for 3-node linear triangular elements."""

    @staticmethod
    def evaluate(xi: float, eta: float) -> np.ndarray:
        """
        Evaluate shape functions at (ξ, η).
        N_1 = 1 - ξ - η  (node 1)
        N_2 = ξ          (node 2)
        N_3 = η          (node 3)

        Args:
            xi: Reference coordinate ξ
            eta: Reference coordinate η

        Returns:
            Array of shape function values [N_1, N_2, N_3]
        """
        return np.array([1 - xi - eta, xi, eta])

    @staticmethod
    def derivatives_reference() -> Tuple[np.ndarray, np.ndarray]:
        """
        Derivatives with respect to reference coordinates.

        Returns:
            Tuple of (dN_dxi, dN_deta) for each shape function
        """
        # ∂N/∂ξ
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        # ∂N/∂η
        dN_deta = np.array([-1.0, 0.0, 1.0])
        return dN_dxi, dN_deta

    @staticmethod
    def test_partition_of_unity():
        """Test that shape functions sum to 1 (partition of unity)."""
        xi, eta = 0.2, 0.3
        N = LinearTriangleShapeFunctions.evaluate(xi, eta)
        assert abs(np.sum(N) - 1.0) < 1e-10, f"Partition of unity failed: sum = {np.sum(N)}"
        print("✓ Partition of unity test passed")

    @staticmethod
    def test_kronecker_delta():
        """Test that shape functions satisfy Kronecker delta property at nodes."""
        # At node 1 (ξ=0, η=0): N_1=1, N_2=0, N_3=0
        N1 = LinearTriangleShapeFunctions.evaluate(0.0, 0.0)
        assert abs(N1[0] - 1.0) < 1e-10 and abs(N1[1]) < 1e-10 and abs(N1[2]) < 1e-10

        # At node 2 (ξ=1, η=0): N_1=0, N_2=1, N_3=0
        N2 = LinearTriangleShapeFunctions.evaluate(1.0, 0.0)
        assert abs(N2[0]) < 1e-10 and abs(N2[1] - 1.0) < 1e-10 and abs(N2[2]) < 1e-10

        # At node 3 (ξ=0, η=1): N_1=0, N_2=0, N_3=1
        N3 = LinearTriangleShapeFunctions.evaluate(0.0, 1.0)
        assert abs(N3[0]) < 1e-10 and abs(N3[1]) < 1e-10 and abs(N3[2] - 1.0) < 1e-10

        print("✓ Kronecker delta test passed")


if __name__ == "__main__":
    # Test shape functions
    print("Testing linear triangle shape functions...")
    LinearTriangleShapeFunctions.test_partition_of_unity()
    LinearTriangleShapeFunctions.test_kronecker_delta()
    print("All shape function tests passed!")
