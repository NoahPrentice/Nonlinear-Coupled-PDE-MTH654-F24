import numpy as np
import matplotlib.pyplot as plt
from Ex2_processing import *
from Ex2_constructors import *


def find_tau_from_CFL(
    initial_tau: float,
    x_boundaries: np.ndarray,
    y_boundaries: np.ndarray,
    x_widths: np.ndarray,
    y_widths: np.ndarray,
) -> float:
    left_right_velocities = build_left_right_velocities(x_boundaries, y_boundaries)
    up_down_velocities = build_up_down_velocities(x_boundaries, y_boundaries)

    nx = up_down_velocities.shape[0]
    ny = left_right_velocities.shape[1]
    left_right_velocities_at_centers = np.zeros((nx, ny))
    up_down_velocities_at_centers = np.zeros((nx, ny))

    for i in range(nx):
        left_right_velocities_at_centers[i, :] = 0.5 * (
            left_right_velocities[i, :] + left_right_velocities[i + 1, :]
        )
    for j in range(ny):
        up_down_velocities_at_centers[:, j] = 0.5 * (
            up_down_velocities[:, j] + up_down_velocities[:, j + 1]
        )
    speeds_at_centers = np.sqrt(
        np.square(left_right_velocities_at_centers)
        + np.square(up_down_velocities_at_centers)
    )

    vccof = np.ones((nx, ny))
    for i in range(nx):
        for j in range(ny):
            vccof[i, j] *= x_widths[i] * y_widths[j]
    datcfl = np.divide(vccof, speeds_at_centers)
    dtcfl = np.min(datcfl)

    if initial_tau > 0:
        print("Time step is chosen by CFL as " + str(min(0.5 * dtcfl, initial_tau)))
        return min(0.5 * dtcfl, initial_tau)
    print("Time step is chosen by CFL as " + str(0.5 * dtcfl))
    return 0.5 * dtcfl


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
            previous_solution[i, j] -= (tau / y_widths[j]) * (
                up_down_fluxes[i, j + 1] - up_down_fluxes[i, j]
            )
    return previous_solution


def transport2d(
    initial_tau: float,
    final_time: float,
    nx: int,
    ny: int,
    plot_frequency: int = 0,
    compute_quantity_loss: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically solves the 2D transport problem: Given Omega = (-1, 1)^2, find
    u(x, y, t) such that u_t - div(qu) = 0 on Omega x [0, final_time], u(x, y, 0) =
    u_init(x, y), q(x, y) = exact_velocity(x, y), and u = 0 on the inflow boundary of
    Omega."""

    x_boundaries, x_centers, x_widths = build_1d_uniform_grid(-1, 1, nx)
    y_boundaries, y_centers, y_widths = build_1d_uniform_grid(-1, 1, ny)

    times = np.zeros(1)
    tau = find_tau_from_CFL(initial_tau, x_boundaries, y_boundaries, x_widths, y_widths)
    iteration = 0
    current_time = 0
    previous_solution = build_initial_numerical_solution(x_centers, y_centers)

    if compute_quantity_loss:
        initial_quantity = calculate_total_quantity(
            previous_solution, x_widths, y_widths
        )
    if plot_frequency > 0:
        fig, ax = setup_plotting(previous_solution)

    while current_time < final_time:
        previous_time = current_time
        current_time += tau
        if current_time >= final_time:
            current_time = final_time
            tau = current_time - previous_time
        np.append(times, current_time)

        previous_solution = find_solution_at_next_timestep(
            previous_solution, x_boundaries, y_boundaries, x_widths, y_widths, tau
        )

        if compute_quantity_loss:
            current_quantity = calculate_total_quantity(
                previous_solution, x_widths, y_widths
            )
            compute_quantity_loss = check_quantity_loss(
                initial_quantity, current_quantity, current_time, iteration
            )
        if plot_frequency > 0 and iteration % plot_frequency == 0:
            plot_next_solution(previous_solution, current_time, fig, ax)

        if current_time >= final_time:
            break
        iteration += 1

    if plot_frequency > 0:
        plt.ioff()
        plt.show()
    return (times, previous_solution)
