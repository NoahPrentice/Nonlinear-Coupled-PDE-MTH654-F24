import math
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns

tau = 0.025
a = 1
c = 10
end_time = 1
iteration_depth = 10

current_time = 0
u_prev = 1
v_prev = 0.1

times = [current_time]
u_values = [u_prev]
v_values = [v_prev]


def g(t: float) -> float:
    return -math.sin(4 * t)


def F(x: float) -> float:
    return (
        x
        - u_prev
        + a * tau * math.pow(x, 3)
        + c * tau * (x - (v_prev + c * tau * x) / (1 + c * tau))
        - tau * g(current_time)
    )


def F_prime(x: float) -> float:
    return (
        1
        + 3 * a * tau * math.pow(x, 2)
        + c * tau
        - (math.pow(c * tau, 2) / (1 + c * tau))
    )


def find_v_from_u(u: float) -> float:
    return (v_prev + c * tau * u) / (1 + c * tau)


def one_newton_iteration(last_iterate: float) -> float:
    correction = -F(last_iterate) / F_prime(last_iterate)
    return last_iterate + correction


def plot_u_and_v() -> None:
    plt.figure()
    plt.plot(times, u_values, color="r", label="u")
    plt.plot(times, v_values, color="b", linestyle="--", label="v")
    plt.legend()
    plt.title("Fully implicit solutions")
    plt.show()

start = perf_counter_ns()
# --- Time Stepping ---
while current_time < end_time:
    if current_time + tau > end_time:
        break
    current_time += tau

    last_iterate = u_prev
    for i in range(iteration_depth):
        last_iterate = one_newton_iteration(last_iterate)
    u_prev = last_iterate
    v_prev = find_v_from_u(u_prev)

    times.append(current_time)
    u_values.append(u_prev)
    v_values.append(v_prev)
end = perf_counter_ns()
print(end - start)
plot_u_and_v()