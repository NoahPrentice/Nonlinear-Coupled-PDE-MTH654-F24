import numpy as np
import matplotlib.pyplot as plt


def exact_velocity(x: float, y: float) -> tuple[float, float]:
    """Returns q(x, y) as defined by the problem."""
    return (-1, -1)


def build_1d_uniform_grid(
    starting_value: float, ending_value: float, number_of_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a uniform grid from starting_value to ending_value (inclusive) with the
    given number of cells.

    Parameters:
        starting_value: value to start at.
        ending_value: value to stop at (inclusive).
        number_of_cells: desired number of cells.

    Returns:
        A triple. The first element is a 1d np.ndarray of boundary values, including
        starting_value and ending_value. Next is a 1d np.ndarray of the cell centers.
        Last is a 1d np.ndarray of the cell widths.

        Note that, if x_i is a cell center, then it has boundaries given by
        x_(i-1/2) = boundary_values[i] and x_(i+1/2) = boundary_values[i+1].
    """
    boundary_values = np.linspace(starting_value, ending_value, number_of_cells + 1)
    cell_widths = np.diff(boundary_values, axis=0)
    cell_centers = boundary_values[:number_of_cells] + 0.5 * cell_widths
    return (boundary_values, cell_centers, cell_widths)


def build_initial_numerical_solution(x_centers: np.ndarray, y_centers) -> np.ndarray:
    nx = x_centers.size
    ny = y_centers.size
    midpointx = np.floor([nx * 2 / 3, nx * 5 / 6]).astype(int)
    midpointy = np.floor([ny * 2 / 3, ny * 5 / 6]).astype(int)

    uinit = np.zeros((nx, ny))
    uinit[midpointx[0] : midpointx[1], midpointy[0] : midpointy[1]] = 1
    return uinit


def build_left_right_velocities(x_boundaries: np.ndarray, y_boundaries: np.ndarray):
    left_right_velocities = np.ndarray((x_boundaries.size, y_boundaries.size - 1))
    for i in range(x_boundaries.size):
        for j in range(y_boundaries.size - 1):
            left_right_velocities[i, j] = exact_velocity(
                x_boundaries[i], y_boundaries[j]
            )[0]
    return left_right_velocities


def build_up_down_velocities(x_boundaries: np.ndarray, y_boundaries: np.ndarray):
    up_down_velocities = np.ndarray((x_boundaries.size - 1, y_boundaries.size))
    for i in range(x_boundaries.size - 1):
        for j in range(y_boundaries.size):
            up_down_velocities[i, j] = exact_velocity(x_boundaries[i], y_boundaries[j])[
                1
            ]
    return up_down_velocities


def build_left_right_fluxes(
    x_boundaries: np.ndarray, y_boundaries: np.ndarray, previous_solution: np.ndarray
) -> np.ndarray:
    left_right_fluxes = build_left_right_velocities(x_boundaries, y_boundaries)
    for i in range(left_right_fluxes.shape[0] - 1):
        for j in range(left_right_fluxes.shape[1]):
            if left_right_fluxes[i, j] >= 0 and i - 1 >= 0:
                left_right_fluxes[i, j] *= previous_solution[i - 1, j]
            elif left_right_fluxes[i, j] <= 0:
                left_right_fluxes[i, j] *= previous_solution[i, j]
            else:  # Inflow boundary
                left_right_fluxes[i, j] = 0
    left_right_fluxes[-1, :] = 0
    return left_right_fluxes


def build_up_down_fluxes(
    x_boundaries: np.ndarray, y_boundaries: np.ndarray, previous_solution: np.ndarray
) -> np.ndarray:
    up_down_fluxes = build_up_down_velocities(x_boundaries, y_boundaries)
    for i in range(up_down_fluxes.shape[0]):
        for j in range(up_down_fluxes.shape[1] - 1):
            if up_down_fluxes[i, j] >= 0 and j - 1 >= 0:
                up_down_fluxes[i, j] *= previous_solution[i, j - 1]
            elif up_down_fluxes[i, j] <= 0:
                up_down_fluxes[i, j] *= previous_solution[i, j]
            else:  # Inflow boundary
                up_down_fluxes[i, j] = 0
    up_down_fluxes[:, -1] = 0
    return up_down_fluxes
