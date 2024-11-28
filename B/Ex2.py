import math
import numpy as np
import matplotlib.pyplot as plt


def u_init(x: float, y: float) -> float:
    return


def initial_numerical_solution(x_centers: np.ndarray, y_centers) -> np.ndarray:
    initial_numerical_solution = np.zeros((x_centers.size, y_centers.size))
    for i in range(x_centers.size):
        for j in range(y_centers.size):
            initial_numerical_solution[i, j] = u_init(x_centers[i], y_centers[j])
    return initial_numerical_solution


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


def build_left_right_velocities(x_boundaries: np.ndarray, y_boundaries: np.ndarray):
    left_right_velocities = np.ndarray((x_boundaries.size + 1, y_boundaries.size))
    for i in range(x_boundaries.size + 1):
        for j in range(y_boundaries.size):
            left_right_velocities[i, j] = exact_velocity(
                x_boundaries[i], y_boundaries[j]
            )
    return left_right_velocities


def build_up_down_velocities(x_boundaries: np.ndarray, y_boundaries: np.ndarray):
    up_down_velocities = np.ndarray((x_boundaries.size, y_boundaries.size + 1))
    for i in range(x_boundaries.size):
        for j in range(y_boundaries.size + 1):
            up_down_velocities[i, j] = exact_velocity(x_boundaries[i], y_boundaries[j])
    return up_down_velocities


def build_left_right_fluxes(
    x_boundaries: np.ndarray, y_boundaries: np.ndarray, previous_solution: np.ndarray
) -> np.ndarray:
    left_right_fluxes = build_left_right_velocities(x_boundaries, y_boundaries)
    for i in range(left_right_fluxes.shape[0]):
        for j in range(left_right_fluxes.shape[1]):
            if left_right_fluxes[i, j] >= 0 and i - 1 >= 0:
                left_right_fluxes[i, j] *= previous_solution[i - 1, j]
            elif left_right_fluxes[i, j] <= 0:
                left_right_fluxes[i, j] *= previous_solution[i, j]
            else:  # Inflow boundary
                left_right_fluxes[i, j] = 0
    return left_right_fluxes


def build_up_down_fluxes(
    x_boundaries: np.ndarray, y_boundaries: np.ndarray, previous_solution: np.ndarray
) -> np.ndarray:
    up_down_fluxes = build_up_down_velocities(x_boundaries, y_boundaries)
    for i in range(up_down_fluxes.shape[0]):
        for j in range(up_down_fluxes.shape[1]):
            if up_down_fluxes[i, j] >= 0 and j - 1 >= 0:
                up_down_fluxes[i, j] *= previous_solution[i, j - 1]
            elif up_down_fluxes[i, j] <= 0:
                up_down_fluxes[i, j] *= previous_solution[i, j]
            else:  # Inflow boundary
                up_down_fluxes[i, j] = 0
    return up_down_fluxes


def calculate_tau_from_CFL():
    return 0.1


def plot_solution(solution: np.ndarray):
    return


def find_solution_at_next_timestep(
    previous_solution: np.ndarray,
    x_boundaries: np.ndarray,
    y_boundaries: np.ndarray,
    x_widths: np.ndarray,
    y_widths: np.ndarray,
    tau: float,
) -> np.ndarray:
    left_right_fluxes = build_left_right_fluxes(
        x_boundaries, y_boundaries, previous_solution
    )
    up_down_fluxes = build_up_down_fluxes(x_boundaries, y_boundaries, previous_solution)

    for i in range(previous_solution.shape[0]):
        for j in range(previous_solution.shape[1]):
            previous_solution[i, j] -= (tau / x_widths[i]) * (
                left_right_fluxes[i + 1, j] - left_right_fluxes[i, j]
            )
            previous_solution[i, j] -= (tau / y_widths[i]) * (
                up_down_fluxes[i, j + 1] - up_down_fluxes[i, j]
            )
    return previous_solution


def transport2d(
    initial_tau: float, final_time: float, nx: int, ny: int, show_plot: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically solves the 2D transport problem: Given Omega = (-1, 1)^2, find
    u(x, y, t) such that u_t - div(qu) = 0 on Omega x [0, final_time], u(x, y, 0) =
    u_init(x, y), q(x, y) = exact_velocity(x, y), and u = 0 on the inflow boundary of
    Omega."""

    x_boundaries, x_centers, x_widths = build_1d_uniform_grid(-1, 1, nx)
    y_boundaries, y_centers, y_widths = build_1d_uniform_grid(-1, 1, ny)

    times = np.zeros(1)
    tau = calculate_tau_from_CFL()
    n = 0
    current_time = 0
    previous_solution = initial_numerical_solution(x_centers, y_centers)
    while True:
        previous_time = current_time
        current_time += tau
        if current_time >= final_time:
            current_time = final_time
            tau = current_time - previous_time
        np.append(times, current_time)

        previous_solution = find_solution_at_next_timestep(
            previous_solution, x_boundaries, y_boundaries, x_widths, y_widths, tau
        )

        if show_plot > 0 and show_plot % n == 0:
            plot_solution(previous_solution)

        if current_time >= final_time:
            break
        n += 1
