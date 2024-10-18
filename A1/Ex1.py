import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def energy(u: float) -> float:
    return 0.25 * pow(1 - u * u, 2)


def plot_energy(t_values: list[float], u_values: list[float], scheme_name="") -> None:
    energy_values = [energy(u_value) for u_value in u_values]
    plt.plot(t_values, energy_values)
    if scheme_name != "":
        scheme_name += " "
    plt.title(scheme_name + "Energy")
    plt.ylabel("Energy")
    plt.xlabel("t")
    plt.show()


def plot_solution(
    t_values: list[float], u_values: list[float], scheme_name: str
) -> None:
    plt.plot(t_values, u_values)
    plt.title(
        scheme_name
        + " solution to u' = u - u^3, u_init = "
        + str(u_values[0])
        + ", tau_init = "
        + str(t_values[1] - t_values[0])
    )
    plt.xlabel("t")
    plt.ylabel("U")
    plt.show()


def forward_euler(
    u_init: float, tau_init: float, T: float = 5, should_plot_energy=False
) -> tuple[list[float], list[float]]:
    """Numerically solves the ODE u' = u - u^3, u(0) = u_init on (0, T). Uses the fully
    explicit Forward Euler (FE) method and plots the numerical solution.

    Parameters:
        u_init: initial value of u, i.e., u(0).
        tau_init: initial time step size, i.e., t_1 - t_0.
        T: end time of solution. 5 by default.

    Returns:
        A pair. The first element is a list of t values, while the second element is a
        list of the numerical approximations at those t values.
    """
    assert tau_init > 0
    assert T > tau_init

    t = 0
    tau = tau_init
    u_prev = u_init
    t_values = [t]
    u_values = [u_prev]

    def update_tau():
        return

    while True:
        update_tau()
        t += tau
        if t >= T:
            break
        t_values.append(t)

        # Solve (u - u_prev)/tau = u_prev - (u_prev)^3 for u.
        u = u_prev + tau * u_prev - tau * math.pow(u_prev, 3)
        u_values.append(u)
        u_prev = u

    plot_solution(t_values, u_values, "FE")
    if should_plot_energy:
        plot_energy(t_values, u_values, "FE")
    return (t_values, u_values)


def backward_euler(
    u_init: float, tau_init: float, T: float = 5, should_plot_energy=False
) -> tuple[list[float], list[float]]:
    """Numerically solves the ODE u' = u - u^3, u(0) = u_init on (0, T). Uses the fully
    implicit Backward Euler (BE) method and plots the numerical solution.

    Parameters:
        u_init: initial value of u, i.e., u(0).
        tau_init: initial time step size, i.e., t_1 - t_0.
        T: end time of solution.

    Returns:
        A pair. The first element is a list of t values, while the second element is a
        list of the numerical approximations at those t values.
    """
    assert tau_init > 0
    assert T > tau_init

    t = 0
    tau = tau_init
    u_prev = u_init
    t_values = [t]
    u_values = [u_prev]

    def update_tau():
        return

    def F(u: float) -> float:
        """The nonlinear function we attempt to solve at each time step. The scheme for
        BE is (u - u_prev)/tau = u - u^3. So we want solutions to the equation
        F(u) := u - u_prev - tau * u + tau * u^3 = 0. This function implements F.

        Parameters:
            u: float input of F.

        Returns:
            F(u).
        """
        return u - u_prev - tau * u + tau * pow(u, 3)

    while True:
        update_tau()
        t += tau
        if t >= T:
            break
        t_values.append(t)

        # Solve F(u) = 0 for u.
        u = fsolve(F, u_prev)[0]
        u_values.append(u)
        u_prev = u

    plot_solution(t_values, u_values, "BE")
    if should_plot_energy:
        plot_energy(t_values, u_values, "BE")
    return (t_values, u_values)


def convexity_splitting(
    u_init: float, tau_init: float, T: float = 5, should_plot_energy=False
) -> tuple[list[float], list[float]]:
    """Numerically solves the ODE u' = u - u^3, u(0) = u_init on (0, T). Uses the semi-
    implicit Convexity Splitting (CS) method and plots the numerical solution.

    Parameters:
        u_init: initial value of u, i.e., u(0).
        tau_init: initial time step size, i.e., t_1 - t_0.
        T: end time of solution.

    Returns:
        A pair. The first element is a list of t values, while the second element is a
        list of the numerical approximations at those t values.
    """
    assert tau_init > 0
    assert T > tau_init

    t = 0
    tau = tau_init
    u_prev = u_init
    t_values = [t]
    u_values = [u_prev]

    def update_tau():
        return

    def F(u: float) -> float:
        """The nonlinear function we attempt to solve at each time step. The scheme for
        CS is (u - u_prev)/tau = u_prev - u^3. So we want solutions to the equation
        F(u) := u - u_prev - tau * u_prev + tau * u^3 = 0. This function implements F.

        Parameters:
            u: float input of F.

        Returns:
            F(u).
        """
        return u - u_prev - tau * u_prev + tau * pow(u, 3)

    while True:
        update_tau()
        t += tau
        if t >= T:
            break
        t_values.append(t)

        # Solve F(u) = 0 for u.
        u = fsolve(F, u_prev)[0]
        u_values.append(u)
        u_prev = u

    plot_solution(t_values, u_values, "CS")
    if should_plot_energy:
        plot_energy(t_values, u_values, "CS")
    return (t_values, u_values)
