import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math

regularization_lambda = 0.01
a = 0
b = 10


def get_next_p_from_previous_u_D(u_D: float) -> float:
    return u_D / 2 - u_D / 3 - 1 / 2


def active_set_algorithm(
    initial_u_D: float, initial_p: float, regularization_lambda: float
) -> float:
    u_D, p = initial_u_D, initial_p
    (
        positive_active_set,
        negative_active_set,
        inactive_set,
        previous_positive_active_set,
        previous_negative_active_set,
        previous_inactive_set,
    ) = (set(), set(), set(), set(), set(), set())

    n = 0
    while n < 1000:
        n += 1
        print("Iteration " + str(n))

        if (-p / regularization_lambda) > b:
            positive_active_set.add(1)
        if (-p / regularization_lambda) < a:
            negative_active_set.add(1)
        inactive_set = {1}.difference(positive_active_set.union(negative_active_set))
        print("pos " + str(positive_active_set))
        print("neg " + str(negative_active_set))
        print("inactive " + str(inactive_set))

        p = get_next_p_from_previous_u_D(u_D)
        if 1 in negative_active_set:
            u_D = a
        elif 1 in positive_active_set:
            u_D = b
        else:
            u_D = -p / regularization_lambda
        print("p " + str(p))
        print("u_D " + str(u_D))
        if (
            previous_positive_active_set == positive_active_set
            and previous_negative_active_set == negative_active_set
        ):
            return u_D

        (
            previous_positive_active_set,
            previous_negative_active_set,
            previous_inactive_set,
        ) = (
            positive_active_set,
            negative_active_set,
            inactive_set,
        )
        (positive_active_set, negative_active_set, inactive_set) = (set(), set(), set())

    print("No convergence")
    return


print(active_set_algorithm(1.5, 2, regularization_lambda))
