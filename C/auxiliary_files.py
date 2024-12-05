import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import math


regularization_lambda = 0.01
a = 0
b = 10
M = 30
k = 20


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


cell_boundaries, cell_centers, cell_widths = build_1d_uniform_grid(0, 1, M)


def e_k(k: int, x: float) -> float:
    if cell_boundaries[k] <= x <= cell_boundaries[k + 1]:
        return 1
    return 0


def rhsfun(x: float) -> float:
    assert isinstance(x, float)
    return e_k(k, x)


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


def exfun(x: float) -> tuple[float, float]:
    assert isinstance(x, float)
    output = math.exp(10 * math.pow(x, 2)) / math.exp(10)
    derivative = 20 * x * math.exp(10 * math.pow(x, 2)) / math.exp(10)
    return (output, derivative)


def exfun_for_vectors(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 2
    assert vector.shape[1] == 1

    output_vector = []
    derivative_vector = []
    for row_number in range(vector.shape[0]):
        row = vector[row_number]
        entry = row[0]
        output, derivative = exfun(entry)
        output_vector.append([output])
        derivative_vector.append([derivative])
    return (np.array(output_vector), np.array(derivative_vector))


def step_size_from_x_value(x: float, reduction_factor: float = 1):
    derivative = exfun(x)[1]
    if derivative == 0:
        return 1 / 8
    return min(
        [1 / (reduction_factor * 4), 1 / (reduction_factor * 16 * abs(derivative))]
    )


def build_grid(reduction_factor: float = 1) -> list[float]:
    grid = [0.0]
    while grid[-1] < 1:
        x = grid[-1]
        h = step_size_from_x_value(x, reduction_factor)
        grid.append(x + h)
    return grid


def ELLIPTIC1d(
    nxdx, bcond, ifexact, ifplot=1, ifdemo=0
) -> tuple[list[float], list[float]]:
    """1d finite volume solution to linear diffusion equation -(ku_x)_x = f(x) on (a, b).

    Parameters:
        nxdx: either nx (number of cells) or x position of grid nodes.
        bcond: vector of 4 numbers, (bflag_left, bval_left, bflag_right, bval_right). If
            a flag value is 0, that boundary has Dirichlet boundary conditions. Otherwise
            the boundary uses a Neumann condition. The corresponding value is either the
            Dirichlet value or -ku'.
        ifexact: 0 if no exact solution is known. If nonzero, then boundary values are
            ignored, the exact solution is plotted next to the numerical solution, and
            L2 grid norm error is computed.
        ifplot: 1 if solution should be plotted.
        ifdemo: 1 if some values should be output.

    Returns:
        A pair. The first element is a list of locations of the cell centers, while the
        second element is the numerical solution at these cell centers.
    """

    if isinstance(nxdx, list):
        nx = len(nxdx) - 1
        x = np.array(nxdx)[None].T
        dx = np.diff(x, axis=0)
    else:
        nx = nxdx
        dx = 1 / nx * np.ones((nx, 1))
        a = 0
        b = 1
        x = np.zeros((nx + 1, 1))
        x[0][0] = a
        for j in range(1, nx + 1):
            x[j][0] = x[j - 1][0] + dx[j - 1][0]
    xc = x[:nx] + 0.5 * dx
    h = np.amax(dx)
    kcof = np.ones((nx, 1))

    bflag_left = bcond[0]
    bval_left = bcond[1]
    bflag_right = bcond[2]
    bval_right = bcond[3]

    tx = np.zeros((nx + 1, 1))
    for j in range(1, nx):
        tx[j][0] = 2 / (dx[j - 1][0] / kcof[j - 1][0] + dx[j][0] / kcof[j][0])
    if bflag_left == 0:
        j = 0
        tx[j][0] = 2 / (dx[j][0] / kcof[j][0])
    if bflag_right == 0:
        j = nx
        tx[j][0] = 2 / (dx[j - 1][0] / kcof[j - 1][0])
    if ifdemo:
        print("Transmissibilities\n")
        print(tx.T)

    diffmat = sparse.lil_array((nx, nx))
    for j in range(1, nx):
        gl = j - 1
        gr = j
        diffmat[gl, gl] = diffmat[gl, gl] + tx[j, 0]
        diffmat[gl, gr] = diffmat[gl, gr] - tx[j, 0]

        diffmat[gr, gl] = diffmat[gr, gl] - tx[j, 0]
        diffmat[gr, gr] = diffmat[gr, gr] + tx[j, 0]

    if bflag_left == 0:
        j = 0
        gr = 0
        diffmat[gr, gr] = diffmat[gr, gr] + tx[j, 0]
    if bflag_right == 0:
        j = nx
        gl = nx - 1
        diffmat[gl, gl] = diffmat[gl, gl] + tx[j, 0]

    if ifdemo:
        print(diffmat)
        wait = input("Press Enter to continue.\n")
    allmat = diffmat
    if ifdemo:
        print(allmat)
    rhs = dx * rhsfun_for_vectors(xc)

    if bflag_left == 0:
        if ifexact:
            dval_left, tilde = exfun(x[0][0])
        else:
            dval_left = bval_left
        qleft = tx[0][0] * dval_left
    else:
        if ifexact:
            tilde, nval_left = exfun(x[0][0])
            qleft = -kcof[0][0] * nval_left
        else:
            qleft = bval_left

    if bflag_right == 0:
        if ifexact:
            dval_right, tilde = exfun(x[nx][0])
        else:
            dval_right = bval_right
        qright = -tx[nx][0] * dval_right
    else:
        if ifexact:
            tilde, nval_right = exfun(x[nx][0])
            qright = -kcof[nx - 1][0] * nval_right
        else:
            qright = bval_right
    if ifdemo:
        print("qleft:")
        print(qleft)
        print("qright:")
        print(qright)

    rhs[0][0] = rhs[0][0] + qleft
    rhs[nx - 1][0] = rhs[nx - 1][0] - qright
    nsol = spsolve(sparse.csr_matrix(allmat), rhs.T[0])
    if ifexact and ifplot:
        exsol = exfun_for_vectors(xc)[0]
        error_vector = exsol.T[0] - nsol
        grid_norm_error = math.sqrt(np.sum(np.square(error_vector) * dx.T[0]))
        print(grid_norm_error)
        plt.plot(xc, exsol, marker="o", color="k", label="exact solution")
        plt.plot(xc, nsol, marker="*", color="r", label="numerical solution")
        plt.legend()
        plt.show()
    if ifexact:
        exsol = exfun_for_vectors(xc)[0]
        error_vector = exsol.T[0] - nsol
        grid_norm_error = math.sqrt(np.sum(np.square(error_vector) * dx.T[0]))
        print(grid_norm_error)
    elif ifplot:
        plt.plot(xc, nsol, marker="*", color="b")
        plt.title("Solution")
        plt.show()
    return (xc, nsol)
