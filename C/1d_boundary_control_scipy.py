import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math

m_values = [8, 16, 32, 64, 128, 256, 512]
h_values = []
errors = []

for M in m_values:
    h = 1 / M
    h_values.append(h)
    regularization_lambda = 0.01


    def finite_dimensional_objective(y: np.ndarray) -> float:
        average_as_vector = np.ones(y.shape)
        objective = 0.5 * h * np.sum(
            np.square(y - average_as_vector)
        ) + 0.5 * regularization_lambda * math.pow(y[0], 2)
        return objective


    def build_constraint_matrix() -> sparse.lil_array:
        constraint_matrix = sparse.lil_array((M, M))
        constraint_matrix[-1, -1] = 1
        for i in range(1, M - 1):
            constraint_matrix[i, i - 1] = -1
            constraint_matrix[i, i] = 2
            constraint_matrix[i, i + 1] = -1
        return sparse.csr_matrix(constraint_matrix)


    def build_constraint_vector() -> np.ndarray:
        return np.zeros(M)

    def plot_numerical_solution(result: opt.OptimizeResult):
        cell_centers = np.array([i * h for i in range(1, M + 1)])
        numerical_solution = result.x
        optimal_solution = np.array([1.5 * (1 - x) for x in cell_centers])

        plt.plot(cell_centers, numerical_solution, "g.", label="Numerical solution")
        plt.plot(cell_centers, optimal_solution, "-r", label="Analytic solution")
        plt.plot(cell_centers, np.ones(M), "--b", label="Desired average")
        plt.title("Result of scipy.optimize")
        plt.legend()
        plt.show()
    
    def calculate_grid_norm_error(result: opt.OptimizeResult):
        cell_centers = np.array([i * h for i in range(1, M + 1)])
        numerical_solution = result.x
        optimal_solution = np.array([1.5 * (1 - x) for x in cell_centers])

        return h * np.sum(np.square(numerical_solution - optimal_solution))


    constraint = opt.LinearConstraint(
        build_constraint_matrix(),
        build_constraint_vector(),
        build_constraint_vector(),
    )

    result = opt.minimize(
        finite_dimensional_objective,
        np.ones(M),
        method="trust-constr",
        constraints=constraint,
    )
    plot_numerical_solution(result)
    errors.append(calculate_grid_norm_error(result))

print(errors)
plt.loglog(h_values, errors, "g*", label="Grid norm error", base=2)
plt.loglog(h_values, h_values, "--k", label="Linear", base=2)
plt.loglog(h_values, [math.pow(hi, 2) for hi in h_values], "--r", label="Quadratic", base=2)
plt.loglog(h_values, [math.pow(hi, 3) for hi in h_values], "--b", label="Cubic", base=2)
plt.legend()
plt.title("Error for M = 16, 32, ..., 512")
plt.xlabel("h")
plt.ylabel("L2 grid norm error")
plt.show()
