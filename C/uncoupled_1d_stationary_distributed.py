import numpy as np
from scipy.integrate import quad
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
from auxiliary_files import *


regularization_lambda = 0.01
M = 30
a = np.zeros(M)
b = 10 * np.ones(M)
k = 20

cell_boundaries, cell_centers, cell_widths = build_1d_uniform_grid(0, 1, M)


def e_k(k: int, x: float) -> float:
    if cell_boundaries[k] <= x <= cell_boundaries[k + 1]:
        return 1
    return 0


def rhsfun(x: float) -> float:
    assert isinstance(x, float)
    return e_k(20, x)


def rhsfun_for_vectors(vector: np.ndarray) -> np.ndarray:
    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 2
    assert vector.shape[1] == 1

    output_vector = []
    for row_number in range(vector.shape[0]):
        row = vector[row_number]
        entry = row[0]
        output = rhsfun(entry)
        output_vector.append([output])
    return np.array(output_vector)


def analytic_v_k(x: float, k: int = 0) -> float:
    C = -cell_widths[k] * (1 - cell_centers[k])
    if x <= cell_boundaries[k]:
        return -C * x
    elif cell_boundaries[k] <= x <= cell_boundaries[k + 1]:
        return -(
            0.5 * math.pow(x, 2)
            - x * cell_boundaries[k]
            + 0.5 * math.pow(cell_boundaries[k], 2)
            + C * x
        )
    return -(cell_widths[k] * (x - cell_centers[k]) + C * x)


def integral_of_v_k(k: int) -> float:
    return quad(analytic_v_k, 0, 1, args=(k))[0]


integrals_of_v_k = np.array([integral_of_v_k(i) for i in range(M)])

initial_u = np.ones(M)
initial_p = np.ones(M)
p, u = initial_p, initial_u

pos_prev, neg_prev, ina_prev = set(), set(), set()
pos, neg, ina = set(), set(), set()
n = 0
while n < 1000:
    n += 1
    print("iteration " + str(n))

    pos_prev, neg_prev, ina_prev = pos, neg, ina
    pos, neg, ina = set(), set(), set()
    for i in range(M):
        if -p[i] / regularization_lambda > b[i]:
            pos.add(i)
        elif -p[i] / regularization_lambda < a[i]:
            neg.add(i)
        else:
            ina.add(i)
    p = np.zeros(M)
    for i in range(M):
        for j in range(M):
            p[i] += u[j] * integrals_of_v_k[j]
        p[i] -= integrals_of_v_k[i]

    for i in pos:
        u[i] = a[i]
    for i in neg:
        u[i] = b[i]
    for i in ina:
        u[i] = -p[i] / regularization_lambda

    if pos == pos_prev and neg == neg_prev:
        print("Found u: " + str(u))
        break
print("No convergence")


# yvalues = np.array([analytic_v_k(x, k) for x in cell_centers])
# centers, nsol = ELLIPTIC1d(list(cell_boundaries), [0, 0, 0, 0], 0, 0, 0)
# plt.plot(cell_centers, yvalues, "g.")
# plt.plot(centers, nsol, "r.")
# plt.show()
