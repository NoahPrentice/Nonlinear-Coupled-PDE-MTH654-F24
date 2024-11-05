import math
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


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


def build_uniform_grid_boundaries(number_of_cells: int) -> np.ndarray:
    """Takes a desired number of elements in the spatial discretization of [0, 1] and
    returns a length number_of_cells + 1 "column vector" (np.ndarray with 1 column) of
    uniformly distributed x values within that range."""
    return np.linspace(0, 1, num=number_of_cells)[None].T


def get_cell_centers_and_h_values_from_cell_boundaries(
    cell_boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Takes an array of cell boundaries and returns the cell centers and
    h-values (width of cells).

    Parameters:
        grid: "column vector" (2d np.ndarray with 1 column) of cell boundaries.

    Returns:
        A pair. The first element is a "column vector" of cell-centers, and the second
        element is a "column vector" of cell lengths. Both vectors have 1 fewer entry
        than "grid."
    """
    grid_size = cell_boundaries.size
    h_values = np.diff(cell_boundaries, axis=0)
    cell_centers = cell_boundaries[: grid_size - 1] + 0.5 * h_values
    return (cell_centers, h_values)


def build_k_vector(dimension: int) -> np.ndarray:
    """Builds the vector of conductivity values (called k or epsilon in HW). Here we are
    taking these to be 1 uniformly."""
    return np.ones((dimension, 1))


def build_transmissibility_vector(h_values: np.ndarray) -> np.ndarray:
    number_of_cells = h_values.size
    transmissibility_vector = np.zeros((number_of_cells + 1, 1))
    k_vector = build_k_vector(number_of_cells)
    for j in range(1, number_of_cells):
        transmissibility_vector[j][0] = 2 / (
            h_values[j - 1][0] / k_vector[j - 1][0] + h_values[j][0] / k_vector[j][0]
        )
    return transmissibility_vector


def build_LHS_matrix(h_values: np.ndarray, tau: float):
    transmissibility_vector = build_transmissibility_vector(h_values)
    number_of_cells = h_values.size
    LHS_matrix = sparse.lil_array((number_of_cells, number_of_cells))

    # Interior cells
    for j in range(1, number_of_cells - 1):
        LHS_matrix[j, j - 1] = -tau * transmissibility_vector[j - 1][0]
        LHS_matrix[j, j] = tau * (
            transmissibility_vector[j - 1][0] + transmissibility_vector[j][0]
        )
        LHS_matrix[j, j + 1] = -tau * transmissibility_vector[j][0]

    # Boundary cells
    LHS_matrix[0, 0] = h_values[0] + tau * transmissibility_vector[1]
    LHS_matrix[0, 1] = -tau * transmissibility_vector[1]

    last_cell_index = number_of_cells - 1
    LHS_matrix[last_cell_index, last_cell_index - 1] = (
        -tau * transmissibility_vector[last_cell_index - 1]
    )
    LHS_matrix[last_cell_index, last_cell_index] = (
        h_values[last_cell_index] + tau * transmissibility_vector[last_cell_index - 1]
    )
    return LHS_matrix


def build_RHS_vector(
    h_values: np.ndarray, tau: float, previous_solution: np.ndarray
) -> np.ndarray:
    RHS_vector = reaction_function_for_vectors(previous_solution)
    RHS_vector *= tau
    RHS_vector += previous_solution
    RHS_vector *= h_values
    return RHS_vector


def find_solution_at_next_timestep(
    previous_solution: np.ndarray,
    h_values: np.ndarray,
    tau: float,
) -> np.ndarray:
    LHS_matrix = build_LHS_matrix(h_values, tau)
    RHS_vector = build_RHS_vector(h_values, tau, previous_solution)
    return spsolve(LHS_matrix, RHS_vector)[None].T


def main():
    cell_boundaries = build_uniform_grid_boundaries(40)
    cell_centers, h_values = get_cell_centers_and_h_values_from_cell_boundaries(
        cell_boundaries
    )
    previous_solution = u_init_for_vectors(cell_centers)
    plt.figure()
    plt.plot(cell_centers, previous_solution)
    plt.show()
    tau = 0.01

    def update_tau():
        return

    end_time = 0.1
    current_time = 0
    while current_time < end_time:
        update_tau()
        if current_time + tau > end_time:
            break

        current_time += tau
        previous_solution = find_solution_at_next_timestep(
            previous_solution, h_values, tau
        )
        print(previous_solution)
        plt.figure()
        plt.plot(cell_centers.T[0], previous_solution.T[0])
        plt.show()

# x_values = np.linspace(0, 1, 100)
# y_values = u_init_for_vectors(x_values)
# print(y_values)
# plt.figure()
# plt.plot(x_values, y_values)
# plt.show()

main()
