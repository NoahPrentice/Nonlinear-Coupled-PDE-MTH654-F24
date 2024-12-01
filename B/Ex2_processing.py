import numpy as np
import matplotlib.pyplot as plt


def calculate_total_quantity(
    solution: np.ndarray, x_widths: np.ndarray, y_widths: np.ndarray
) -> float:
    quantity = 0
    for i in range(x_widths.size):
        for j in range(y_widths.size):
            quantity += solution[i, j] * x_widths[i] * y_widths[j]
    return quantity


def check_quantity_loss(
    initial_quantity: float, current_quantity: np.ndarray, time: float, iteration: int
) -> bool:
    if abs(current_quantity - initial_quantity) < 0.0001:
        return True

    print(
        "The loss in the quantity surpassed 10^-4 at time "
        + str(time)
        + ", on iteration #"
        + str(iteration + 1)
    )
    return False


def setup_plotting(solution: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    plt.ion()
    fig, ax = plt.subplots()

    im = ax.imshow(
        solution.T,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        extent=(-1, 1, -1, 1),
    )
    ax.set_title("Time = 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Solution Value")
    return (fig, ax)


def plot_next_solution(
    solution: np.ndarray, current_time: float, fig: plt.Figure, ax: plt.Axes
):
    ax.clear()
    im = ax.imshow(
        solution.T,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        extent=(-1, 1, -1, 1),
    )
    ax.set_title(f"Time = {current_time:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.draw()
    plt.pause(0.1)
    return
