"""
Element matrix assembly for finite elements.
Implements proper element matrix computation using FEM theory with Gauss quadrature.
"""

import numpy as np
from typing import Tuple
from shape_functions import LinearTriangleShapeFunctions
from gauss_quadrature import GaussQuadratureTriangle
from jacobian import JacobianTransform


def compute_element_mass_matrix(coords: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute element mass matrix: M_ij = ∫_Ω N_i N_j dΩ

    Args:
        coords: Physical coordinates of element nodes (n_nodes, 2)
        order: Gauss quadrature order

    Returns:
        Element mass matrix (n_nodes, n_nodes)
    """
    n_nodes = len(coords)
    Me = np.zeros((n_nodes, n_nodes))

    # Get Gauss points and weights
    gauss_points, gauss_weights = GaussQuadratureTriangle.get_points_weights(order)

    for gp, weight in zip(gauss_points, gauss_weights):
        # Evaluate shape functions
        N = LinearTriangleShapeFunctions.evaluate(gp[0], gp[1])

        # Compute Jacobian
        dN_dxi, dN_deta = LinearTriangleShapeFunctions.derivatives_reference()
        J = JacobianTransform.compute_jacobian(coords, dN_dxi, dN_deta)
        det_J = np.linalg.det(J)

        # Assemble mass matrix: M_ij = Σ N_i(ξ_gp) N_j(ξ_gp) |J| w_gp
        for i in range(n_nodes):
            for j in range(n_nodes):
                Me[i, j] += N[i] * N[j] * det_J * weight

    return Me


def compute_element_stiffness_matrix(coords: np.ndarray, nu: float,
                                    order: int = 2) -> np.ndarray:
    """
    Compute element stiffness matrix: K_ij = ν ∫_Ω ∇N_i · ∇N_j dΩ

    Args:
        coords: Physical coordinates of element nodes (n_nodes, 2)
        nu: Kinematic viscosity
        order: Gauss quadrature order

    Returns:
        Element stiffness matrix (n_nodes, n_nodes)
    """
    n_nodes = len(coords)
    Ke = np.zeros((n_nodes, n_nodes))

    gauss_points, gauss_weights = GaussQuadratureTriangle.get_points_weights(order)

    for gp, weight in zip(gauss_points, gauss_weights):
        # Compute Jacobian and inverse
        dN_dxi, dN_deta = LinearTriangleShapeFunctions.derivatives_reference()
        J = JacobianTransform.compute_jacobian(coords, dN_dxi, dN_deta)
        det_J = np.linalg.det(J)
        J_inv = np.linalg.inv(J)

        # Transform derivatives to physical coordinates
        dN_dx, dN_dy = JacobianTransform.physical_derivatives(dN_dxi, dN_deta, J_inv)

        # Assemble stiffness matrix: K_ij = ν Σ (∇N_i · ∇N_j) |J| w_gp
        for i in range(n_nodes):
            for j in range(n_nodes):
                grad_dot = dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]
                Ke[i, j] += nu * grad_dot * det_J * weight

    return Ke


def compute_element_gradient_matrix(coords: np.ndarray, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element gradient matrix: G_ij = ∫_Ω N_i ∂N_j/∂x_k dΩ
    Returns separate matrices for x and y components.

    Args:
        coords: Physical coordinates of element nodes (n_nodes, 2)
        order: Gauss quadrature order

    Returns:
        Tuple of (Gx, Gy) - gradient matrices for x and y components
    """
    n_nodes = len(coords)
    Gx = np.zeros((n_nodes, n_nodes))
    Gy = np.zeros((n_nodes, n_nodes))

    gauss_points, gauss_weights = GaussQuadratureTriangle.get_points_weights(order)

    for gp, weight in zip(gauss_points, gauss_weights):
        N = LinearTriangleShapeFunctions.evaluate(gp[0], gp[1])
        dN_dxi, dN_deta = LinearTriangleShapeFunctions.derivatives_reference()
        J = JacobianTransform.compute_jacobian(coords, dN_dxi, dN_deta)
        det_J = np.linalg.det(J)
        J_inv = np.linalg.inv(J)
        dN_dx, dN_dy = JacobianTransform.physical_derivatives(dN_dxi, dN_deta, J_inv)

        for i in range(n_nodes):
            for j in range(n_nodes):
                Gx[i, j] += N[i] * dN_dx[j] * det_J * weight
                Gy[i, j] += N[i] * dN_dy[j] * det_J * weight

    return Gx, Gy


def compute_element_divergence_matrix(coords: np.ndarray, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element divergence matrix: D_ij = ∫_Ω N_i ∇·N_j dΩ

    Args:
        coords: Physical coordinates of element nodes (n_nodes, 2)
        order: Gauss quadrature order

    Returns:
        Tuple of (Dx, Dy) - divergence matrices for x and y components
    """
    n_nodes = len(coords)
    Dx = np.zeros((n_nodes, n_nodes))
    Dy = np.zeros((n_nodes, n_nodes))

    gauss_points, gauss_weights = GaussQuadratureTriangle.get_points_weights(order)

    for gp, weight in zip(gauss_points, gauss_weights):
        N = LinearTriangleShapeFunctions.evaluate(gp[0], gp[1])
        dN_dxi, dN_deta = LinearTriangleShapeFunctions.derivatives_reference()
        J = JacobianTransform.compute_jacobian(coords, dN_dxi, dN_deta)
        det_J = np.linalg.det(J)
        J_inv = np.linalg.inv(J)
        dN_dx, dN_dy = JacobianTransform.physical_derivatives(dN_dxi, dN_deta, J_inv)

        for i in range(n_nodes):
            for j in range(n_nodes):
                Dx[i, j] += N[i] * dN_dx[j] * det_J * weight
                Dy[i, j] += N[i] * dN_dy[j] * det_J * weight

    return Dx, Dy


def test_element_matrices():
    """Test element matrix computation for a simple triangle."""
    print("Testing element matrix assembly...")

    # Simple unit triangle
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    nu = 0.001

    # Test mass matrix
    Me = compute_element_mass_matrix(coords, order=2)
    print(f"  Mass matrix shape: {Me.shape}")
    print(f"  Mass matrix sum: {np.sum(Me):.6f}")

    # For unit triangle, mass matrix should be symmetric and positive definite
    assert np.allclose(Me, Me.T), "Mass matrix should be symmetric"
    assert np.all(np.linalg.eigvals(Me) > 0), "Mass matrix should be positive definite"

    # Test stiffness matrix
    Ke = compute_element_stiffness_matrix(coords, nu, order=2)
    print(f"  Stiffness matrix shape: {Ke.shape}")
    print(f"  Stiffness matrix sum: {np.sum(Ke):.6f}")

    # Stiffness matrix should be symmetric and positive semi-definite
    assert np.allclose(Ke, Ke.T), "Stiffness matrix should be symmetric"
    eigvals = np.linalg.eigvals(Ke)
    print(f"  Stiffness eigenvalues: {eigvals}")
    assert np.all(eigvals >= -1e-10), "Stiffness matrix should be positive semi-definite"

    # Test gradient matrices
    Gx, Gy = compute_element_gradient_matrix(coords, order=2)
    print(f"  Gradient matrices shape: {Gx.shape}, {Gy.shape}")

    # Test divergence matrices
    Dx, Dy = compute_element_divergence_matrix(coords, order=2)
    print(f"  Divergence matrices shape: {Dx.shape}, {Dy.shape}")

    print("✓ Element matrix assembly tests passed")


if __name__ == "__main__":
    # Test element assembly
    print("Testing element matrix assembly...")
    test_element_matrices()
    print("All element assembly tests passed!")
