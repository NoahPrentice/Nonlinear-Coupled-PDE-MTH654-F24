import math
import numpy as np
import matplotlib.pyplot as plt

M = 10
number_of_iterations = 5
convergence_tolerance = 0.01
u_star = np.array([[1], [1]])

def F(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    assert x.shape[0] == 2 and x.shape[1] == 1

    x1 = x[0][0]
    x2 = x[1][0]

    y1 = math.pow(x1, 2) + math.pow(x2, 2) - 2
    y2 = math.exp(x1 - 1) + math.pow(x2, 3) - 2

    return np.array([[y1], [y2]])


def Jacobian_evaluated_at_point(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    assert x.shape[0] == 2 and x.shape[1] == 1

    x1 = x[0][0]
    x2 = x[1][0]

    return np.array([[2 * x1, 2 * x2], [math.exp(x1 - 1), 3 * math.pow(x2, 2)]])


def Newton_Iteration(x: np.ndarray) -> np.ndarray:
    return x - np.linalg.inv(Jacobian_evaluated_at_point(x)) @ F(x)


initial_x = [i / M for i in range(0 * M, 2 * M)]
initial_y = [i / M for i in range(0 * M, 2 * M)]
initial_points = [
    np.array([[initial_x[i]], [initial_y[i]]]) for i in range(len(initial_x))
]
previous_points = initial_points
for i in range(number_of_iterations):
    next_points = [Newton_Iteration(u) for u in previous_points]
    previous_points = next_points

converged_points = []
non_converged_points = []
for u in previous_points:
    if np.linalg.norm(u - u_star, 2) < convergence_tolerance:
        converged_points.append(u)
    else:
        non_converged_points.append(u)
converged_x = [u[0] for u in converged_points]
