import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math

n = 100
h = 1 / n
end_time = 1
tau = 0.01
regularization_lambda = 0.01
desired_average_temperature = 1


def finite_dimensional_objective(y: np.ndarray) -> float:
    average_as_vector = desired_average_temperature * np.ones(y.shape)
    objective = 0.5 * h * np.sum(np.square(y - average_as_vector))
    return objective + 0.5 * regularization_lambda * math.pow(y[0], 2)


def build_constraint_matrix() -> sparse.lil_array:
    constraint_matrix = sparse.lil_array((n, n))
    constraint_matrix[-1, -1] = 1
    for i in range(1, n - 1):
        constraint_matrix[i, i - 1] = -1
        constraint_matrix[i, i] = 2
        constraint_matrix[i, i + 1] = -1
    return sparse.csr_matrix(constraint_matrix)


def build_upper_constraint_vector() -> np.ndarray:
    ub = np.zeros(n)
    ub[0] = np.inf
    return ub


def build_lower_constraint_vector() -> np.ndarray:
    lb = np.zeros(n)
    return lb


constraint = opt.LinearConstraint(
    build_constraint_matrix(),
    build_lower_constraint_vector(),
    build_upper_constraint_vector(),
)

result = opt.minimize(
    finite_dimensional_objective,
    np.ones(n),
    method="trust-constr",
    constraints=constraint,
)
print(result)

cell_centers = np.array([i * h for i in range(1, n + 1)])

numerical_solution = result.x

plt.plot(cell_centers, numerical_solution, "g.")
plt.plot(cell_centers, np.ones(n), "-r")
plt.show()
