"""
Gauss quadrature for triangular elements.
Implements proper quadrature points and weights for numerical integration.
"""

import numpy as np
from typing import Tuple


class GaussQuadratureTriangle:
    """Gauss quadrature for triangular elements."""

    @staticmethod
    def get_points_weights(order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Gauss points and weights for triangular elements.

        Args:
            order: Quadrature order (1, 2, or 3)

        Returns:
            Tuple of (points, weights) where:
            - points: array of shape (n_points, 2) with (ξ, η) coordinates
            - weights: array of shape (n_points,) with weights

        Order 1: 1-point rule (centroid)
        Order 2: 3-point rule (midpoints)
        Order 3: 4-point rule (higher order)
        """
        if order == 1:
            # 1-point rule: centroid
            points = np.array([[1/3, 1/3]])
            weights = np.array([0.5])
        elif order == 2:
            # 3-point rule: midpoints of edges
            points = np.array([
                [1/6, 1/6],
                [2/3, 1/6],
                [1/6, 2/3]
            ])
            weights = np.array([1/6, 1/6, 1/6])
        elif order == 3:
            # 4-point rule: higher order
            points = np.array([
                [1/3, 1/3],
                [1/5, 1/5],
                [3/5, 1/5],
                [1/5, 3/5]
            ])
            weights = np.array([-9/32, 25/96, 25/96, 25/96])
        else:
            raise ValueError(f"Order {order} not implemented. Available: 1, 2, 3")

        return points, weights

    @staticmethod
    def test_integration():
        """Test that quadrature rules can integrate simple functions exactly."""
        print("Testing Gauss quadrature integration...")

        # Test integration of constant function f(ξ,η) = 1
        # Should give area of reference triangle = 0.5
        for order in [1, 2, 3]:
            points, weights = GaussQuadratureTriangle.get_points_weights(order)
            integral = np.sum(weights)
            expected = 0.5
            error = abs(integral - expected)
            print(f"  Order {order}: integral = {integral:.6f}, error = {error:.2e}")
            assert error < 1e-10, f"Integration test failed for order {order}"

        # Test integration of linear function f(ξ,η) = ξ + η
        # Should give 1/3
        for order in [1, 2, 3]:
            points, weights = GaussQuadratureTriangle.get_points_weights(order)
            integral = np.sum(weights * (points[:, 0] + points[:, 1]))
            expected = 1/3
            error = abs(integral - expected)
            print(f"  Order {order}: linear integral = {integral:.6f}, error = {error:.2e}")
            assert error < 1e-10, f"Linear integration test failed for order {order}"

        print("✓ All quadrature integration tests passed")

    @staticmethod
    def get_quadrature_info():
        """Get information about available quadrature rules."""
        print("Available Gauss quadrature rules for triangles:")
        print("  Order 1: 1 point (centroid)")
        print("  Order 2: 3 points (midpoints)")
        print("  Order 3: 4 points (higher order)")
        print("  All rules are exact for polynomials of degree ≤ order")


if __name__ == "__main__":
    # Test Gauss quadrature
    print("Testing Gauss quadrature for triangles...")
    GaussQuadratureTriangle.test_integration()
    GaussQuadratureTriangle.get_quadrature_info()
    print("All Gauss quadrature tests passed!")
