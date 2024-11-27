import math
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tau = 0.01
h = 0.01
end_time = 0.1
current_time = 0
k = 3


def reaction_function(u: float) -> float:
    """The nonlinear reaction function f(u) = u - u^3."""
    return u * (1 - u)


def reaction_function_for_vectors(x: np.ndarray) -> np.ndarray:
    """Takes a vector of inputs and computes reaction_function for each input."""
    # Column vector
    if len(x.shape) == 2:
        assert x.shape[1] == 1
        y = x.copy()
        for i in range(x.shape[0]):
            y[i, 0] = reaction_function(x[i, 0])
        return y
    # Row vector
    y = x.copy()
    for i in range(x.size):
        y[i] = reaction_function(x[i])
    return y


def u_init(x: float) -> float:
    return math.sin(4 * math.pi * (x + math.sin(x))) + math.pi * math.pow(x, 4)


def u_init_for_vectors(x: np.ndarray) -> np.ndarray:
    # Column vector
    if len(x.shape) == 2:
        assert x.shape[1] == 1
        y = x.copy()
        for i in range(x.shape[0]):
            y[i, 0] = u_init(x[i, 0])
        return y
    # Row vector
    y = x.copy()
    for i in range(x.size):
        y[i] = u_init(x[i])
    return y


def build_uniform_grid_boundaries(step_size: float) -> np.ndarray:
    """Takes a desired number of elements in the spatial discretization of [0, 1] and
    returns a length number_of_cells + 1 "column vector" (np.ndarray with 1 column) of
    uniformly distributed x values within that range."""
    return np.arange(0, 1, step_size)[None].T


def get_cell_centers_and_h_values_from_cell_boundaries(
    cell_boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes an array of cell boundaries and returns the cell centers and
    h-values (width of cells).

    Parameters:
        grid: "column vector" (2d np.ndarray with 1 column) of cell boundaries.

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


def build_RHS_vector(
    h_values: np.ndarray, tau: float, previous_solution: np.ndarray
) -> np.ndarray:
    """Builds the column vector F that results from putting the fully-discrete equations
    in matrix-vector form AU = F.
    """
    RHS_vector = reaction_function_for_vectors(previous_solution)  # f(u) in Pbm. 1
    RHS_vector *= tau
    RHS_vector += previous_solution
    RHS_vector *= h_values
    return RHS_vector


def build_lhs_matrix(grid: list[float], tau: float, eps: float) -> sparse.lil_array:
    return


def find_solution_at_next_timestep(
    previous_solution: np.ndarray, h_values: np.ndarray
) -> np.ndarray:
    LHS_matrix = sparse.csr_matrix(build_LHS_matrix(h_values))
    RHS_vector = build_RHS_vector(h_values, tau, previous_solution)
    return spsolve(LHS_matrix, RHS_vector)[None].T


def update_tau():
    return


def plot_solution(cell_centers: np.ndarray, solution: np.ndarray) -> None:
    plt.figure()
    plt.xticks(np.arange(0, 1, 0.2)[1:])
    plt.yticks(np.arange(0, 2, 0.5))
    plt.ylim((0, 2))
    plt.grid(True)
    plt.plot(cell_centers.T[0], solution.T[0])
    plt.title(
        "Solution at t="
        + str(round(current_time, 2))
        + r" $\tau$="
        + str(tau)
        + " h="
        + str(h)
        + " k="
        + str(k)
    )
    plt.show()


cell_boundaries = build_uniform_grid_boundaries(h)
cell_centers, h_values = get_cell_centers_and_h_values_from_cell_boundaries(
    cell_boundaries
)
previous_solution = u_init_for_vectors(cell_centers)

# --- Time Stepping ---
while current_time < end_time:
    update_tau()
    if current_time + tau > end_time:
        break

    current_time += tau
    previous_solution = find_solution_at_next_timestep(previous_solution, h_values)
plot_solution(cell_centers, previous_solution)
