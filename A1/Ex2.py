import math
import matplotlib.pyplot as plt

M = 150
u_star = 1
number_of_iterations = 15
convergence_tolerance = 0.001
order_tolerance = 0.5


def F(u):
    return u * u * u - u


def G(u):
    return math.pow(u, 1 / 3)


def F_prime(u):
    return 3 * u * u - 1


def Newton_iteration(u):
    if F_prime(u) != 0:
        return u - F(u) / F_prime(u)
    print("Hit critical point at u = " + str(u))
    return u


def fixed_point_iteration(u):
    return G(u)


def get_history_of_point_from_iterations(
    iterations: list[list[float]], index: int
) -> list[float]:
    assert index < len(iterations[0])
    history = []
    for iteration in iterations:
        history.append(iteration[index])
    return history


def order_of_convergence_sequence(history: list[float], u_star: float) -> list[float]:
    error_sequence = [abs(u - u_star) for u in history]
    order_of_convergence_sequence = []
    for i in range(1, len(error_sequence) - 1):
        if (
            error_sequence[i] == error_sequence[i - 1]
            or error_sequence[i + 1] == 0
            or error_sequence[i] == 0
        ):
            break
        numerator = math.log(error_sequence[i + 1]) - math.log(error_sequence[i])
        denominator = math.log(error_sequence[i]) - math.log(error_sequence[i - 1])
        order_of_convergence_sequence.append(numerator / denominator)
    return order_of_convergence_sequence

initial_values = [u / M for u in range(-2 * M, 2 * M)]
previous_iteration = initial_values

for i in range(number_of_iterations):
    next_iteration = [Newton_iteration(u) for u in previous_iteration]
    previous_iteration = next_iteration

plt.scatter(initial_values, previous_iteration, marker=".")
plt.title("Limit of Newton's method by initial value")
plt.yticks([-1, 0, 1])
plt.xlabel(r"$u^{(0)}$")
plt.ylabel("Limit")
plt.show()
