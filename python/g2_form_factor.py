"""
Calculate the G_2 matrix element for the Lieb-Liniger model,
as presented by Piroli and Calabrese.
"""

import numpy as np
import copy

import lieb_liniger_state as lls

def matrix_element(state_a, state_b):
    """
    Since the expression is quite involved, we solve it in parts.
    We set lambda_s = lambda_p = 0.
    """
    print(state_a.lambdas)
    print(state_b.lambdas)
    J = compute_J(state_a, state_b)
    matrix_element = (-1)**state_a.N * J / (6 * state_a.c) * np.prod(llstate_a.lambdas[:,np.newaxis] - llstate_b.lambdas) * 1 / np.prod(llstate_a.lambdas[:,np.newaxis] - llstate_b.lambdas)
    print("me1", matrix_element)

    
    det_delta_plus_U = construct_U_matrix(state_a, state_b)

    matrix_element *= det_delta_plus_U

    product = 1
    for j in range(state_a.N):
        V_plus =  V_j_plus(state_a, state_b, state_b.lambdas[j])
        V_minus = np.conj(V_j_plus(state_a, state_b, state_b.lambdas[j]))
        product *= (V_plus - V_minus)

    matrix_element *= product

    print("me2", matrix_element)


def construct_U_matrix(state_a, state_b):
    identity_plus_U = np.identity(state_a.N, dtype=np.complex_)
    for j in range(state_a.N):
        for k in range(state_a.N):

            V_plus = V_j_plus(state_a, state_b, state_b.lambdas[j])
            V_minus = np.conj(V_j_plus(state_a, state_b, state_b.lambdas[j]))

            numerator_product = 1
            denominator_product = 1
            for m in range(state_a.N):
                numerator_product *= (state_b.lambdas[m] - state_a.lambdas[j])
                if m != j:
                    denominator_product *= (state_a.lambdas[m] - state_a.lambdas[j])

            fraction = numerator_product / denominator_product

            kernel_part = lls.lieb_liniger_state.kernel(state_a.lambdas[j] - state_a.lambdas[k], state_a.c) - lls.lieb_liniger_state.kernel(state_a.lambdas[k], state_a.c) * lls.lieb_liniger_state.kernel(state_a.lambdas[j], state_a.c)
            print(kernel_part)

            identity_plus_U[j, k] = 1j / (V_plus - V_minus) * fraction * kernel_part

    print(identity_plus_U)
    print("det", np.linalg.det(identity_plus_U))

    return np.linalg.det(identity_plus_U)


def compute_J(state_a, state_b):
    Q_a_minus_Q_b = np.sum(state_a.lambdas**3 - state_b.lambdas**3)
    return (state_a.momentum - state_b.momentum)**4 - 4 * (state_a.momentum - state_b.momentum) * Q_a_minus_Q_b + 3 * (state_a.energy - state_b.energy)**2


def V_j_plus(state_a, state_b, lambda_j):
    """
    V_j_minus is the conjugate.
    """
    product = 1
    for n in range(state_a.N):
        product *= (state_b.lambdas[n] - lambda_j + state_a.c * 1j) / (state_a.lambdas[n] - lambda_j + state_a.c * 1j)
    return product


if __name__ == "__main__":
    llstate_a = lls.lieb_liniger_state(1, 100, 4)
    bethe_numbers_state_b = copy.copy(llstate_a.Is)
    bethe_numbers_state_b[0] -= 1
    llstate_b = lls.lieb_liniger_state(1, 100, 4, bethe_numbers_state_b)
    print(llstate_a.Is)
    print(llstate_b.Is)
    llstate_a.calculate_all()
    llstate_b.calculate_all()
    print(llstate_a.lambdas)
    print(llstate_b.lambdas)
    matrix_element(llstate_a, llstate_b)


    llstate_c = lls.lieb_liniger_state(1, 100, 4, lls.generate_bethe_numbers(4))
    llstate_d = lls.lieb_liniger_state(1, 100, 4, lls.generate_bethe_numbers(4))
    print(llstate_c.Is)
    print(llstate_d.Is)
    llstate_c.calculate_all()
    llstate_d.calculate_all()
    print(llstate_c.lambdas)
    print(llstate_d.lambdas)
    matrix_element(llstate_c, llstate_d)
