"""
Calculate the \rho(x=0) matrix element for the Lieb-Liniger model,
as presented by Piroli and Calabrese.
"""

import numpy as np
import copy

import lieb_liniger_state as lls
kernel = lls.lieb_liniger_state.kernel


def psi_form_factor(mu, lambda_):
    momentum_sum = np.sum(mu.lambdas - lambda_.lambdas)
    # print("Kout", momentum_sum)
    denominator = 1 / (V_plus(mu, lambda_, lambda_.lambdas[0]) - V_minus(mu, lambda_, lambda_.lambdas[0]))
    # print("denom", 1/(V_plus(mu, lambda_, lambda_.lambdas[0]) - V_minus(mu, lambda_, lambda_.lambdas[0])))
    detpart = np.linalg.det(np.identity(mu.N) + construct_U(mu, lambda_))
    # print("det(1 + U)", np.linalg.det(np.identity(mu.N) + construct_U(mu, lambda_)))
    # print("log(det(1 + U))", np.log(np.linalg.det(np.identity(mu.N) + construct_U(mu, lambda_))))

    # print("Vplus[0]", V_plus(mu, lambda_, lambda_.lambdas[0]))
    # print("det_frac", detpart * denominator)

    prod1 = 1
    for j in range(mu.N):
        prod1 *= (V_plus(mu, lambda_, lambda_.lambdas[j]) - V_minus(mu, lambda_, lambda_.lambdas[j]))
    # print("prod1", prod1)
    prod2 = 1
    for j in range(mu.N):
        for k in range(mu.N):
            prod2 *= (lambda_.lambdas[j] - lambda_.lambdas[k] + 1j * mu.c) / (mu.lambdas[j] - lambda_.lambdas[k])
    # print("prod2", prod2)
    ff = momentum_sum * prod1 * prod2 * detpart * denominator
    return ff * 1j


def V_plus(mu, lambda_, lambda_j):
    product = 1 + 0 * 1j
    for m in range(mu.N):
        product *= (mu.lambdas[m] - lambda_j + 1j * mu.c) / (lambda_.lambdas[m] - lambda_j + 1j * mu.c)
    return product


def V_minus(mu, lambda_, lambda_j):
    return np.conj(V_plus(mu, lambda_, lambda_j))


def construct_U(mu, lambda_):
    U = np.zeros((mu.N, mu.N), dtype=np.complex_)
    products = []
    for j in range(mu.N):
        product = 1
        for m in range(mu.N):
            if m != j:
                product *= (mu.lambdas[m] - lambda_.lambdas[j]) / (lambda_.lambdas[m] - lambda_.lambdas[j])
        products.append(product)

        for k in range(mu.N):
            # print("part1", j, k, 1j * (mu.lambdas[j] - lambda_.lambdas[j]) / (V_plus(mu, lambda_, lambda_.lambdas[j]) - V_minus(mu, lambda_, lambda_.lambdas[j])))
            U[j, k] = 1j * (mu.lambdas[j] - lambda_.lambdas[j]) / (V_plus(mu, lambda_, lambda_.lambdas[j]) - V_minus(mu, lambda_, lambda_.lambdas[j]))
            kernel_part = (kernel(lambda_.lambdas[j] - lambda_.lambdas[k], mu.c) - kernel(lambda_.lambdas[0] - lambda_.lambdas[k], mu.c))
            # print("kernel", j, k, kernel_part)
            U[j, k] *= products[j]
            U[j, k] *= kernel_part
            # print(U[j, k])
    # print("prods", products)

    return U


def calculate_normalized_form_factor(mu, lambda_):
    unnormalized_ff = psi_form_factor(mu, lambda_)
    return unnormalized_ff / np.sqrt(mu.norm * lambda_.norm)


if __name__ == "__main__":
    rstate = lls.lieb_liniger_state(1, 10, 3)
    bethe_numbers = copy.copy(rstate.Is)
    bethe_numbers[0] -= 100
    lstate = lls.lieb_liniger_state(1, 10, 3, bethe_numbers)
    rstate.calculate_all()
    lstate.calculate_all()
    print(lstate.norm)
    print(rstate.norm)

    print("<l|rho|r>", psi_form_factor(lstate, rstate))
    print(calculate_normalized_form_factor(lstate, rstate))
    print("<r|rho|l>", psi_form_factor(rstate, lstate))
    print(calculate_normalized_form_factor(rstate, lstate))

    print("\n")

    # Random states
    lstate = lls.lieb_liniger_state(1, 10, 10)
    rstate = lls.lieb_liniger_state(1, 10, 10, lls.generate_bethe_numbers(10))
    print(rstate.Is)
    lstate.calculate_all()
    rstate.calculate_all()

    print("<l|rho|r>", psi_form_factor(lstate, rstate))
    print(np.log(calculate_normalized_form_factor(lstate, rstate)))
    print("<r|rho|l>", psi_form_factor(rstate, lstate))
    print(calculate_normalized_form_factor(rstate, lstate))
    print(np.log(calculate_normalized_form_factor(rstate, lstate)))
