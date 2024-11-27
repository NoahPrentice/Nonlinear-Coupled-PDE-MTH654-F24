import math
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt


def rhs_fun(u: float) -> float:
    return u - math.pow(u, 3)


def rhs_fun_for_vectors(x: np.ndarray) -> np.ndarray:
    # Column vector
    if len(x.shape) == 2:
        assert x.shape[1] == 1
        for i in range(x.shape[0]):
            x[i, 0] = rhs_fun(x[i, 0])
        return x
    # Row vector
    for i in range(x.size):
        x[i] = rhs_fun(x[i])
        return x


def build_diffusion_matrix(dimension: int) -> sparse.lil_array:
    """Builds the discrete Laplacian matrix A for Neumann boundary conditions.

    Parameters:
        dimension: int

    Returns:
        The s.p.d. discrete Laplacian matrix with first and final rows as required for
        Neumann boundary conditions.
    """
    diffusion_matrix = sparse.lil_array((dimension, dimension))

    # For the interior nodes, the matrix follows the pattern of the discrete Laplacian:
    # since we compute 2*u_prev[j] - u_prev[j-1] - u_prev[j+1], the matrix is tridiagonal
    # with -1s on the off-diagonals and 2s on the main diagonal.
    for j in range(1, dimension - 1):
        diffusion_matrix[j, j] = 2
        diffusion_matrix[j, j - 1] = -1
        diffusion_matrix[j, j + 1] = -1

    # For the boundary nodes, Neumann boundary conditions reduce the diffusion term to
    # u_prev[1] - u_prev[0] (at the left boundary) or u_prev[M-1] - u_prev[M-2] (at the
    # right boundary, where M = dimension).
    diffusion_matrix[0, 0] = 1
    diffusion_matrix[0, 1] = -1
    diffusion_matrix[dimension - 1, dimension - 1] = 1
    diffusion_matrix[dimension - 1, dimension - 2] = -1

    return diffusion_matrix

def get_cell_centers_and_h_values_from_grid(grid: list[float]) -> tuple[np.ndarray]:
    grid_size = len(grid)
    cell_boundaries = np.array(grid)[None].T
    h_values = np.diff(cell_boundaries, axis=0)
    cell_centers = cell_boundaries[:grid_size] + 0.5 * h_values
    return (cell_centers, h_values)

def build_rhs_vector(grid: list[float], tau: float) -> np.ndarray:
    cell_centers, h_values = get_cell_centers_and_h_values_from_grid(grid)
    rhs_vector = rhs_fun_for_vectors(cell_centers)
    rhs_vector *= tau
    rhs_vector *= np.square(h_values)
    return rhs_vector


def build_lhs_matrix(grid: list[float], tau: float, eps: float) -> sparse.lil_array:
    return


def find_solution_at_next_timestep(
    previous_solution: np.ndarray, grid: list[float], tau: float, eps: float
) -> np.ndarray:
    return
