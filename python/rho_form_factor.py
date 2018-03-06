"""
Calculate the \rho(x=0) matrix element for the Lieb-Liniger model,
as presented by Piroli and Calabrese.
"""

import pickle

import numpy as np

import lieb_liniger_state as lls
kernel = lls.lieb_liniger_state.kernel

np.seterr(all='raise')


def psi_form_factor(mu, lambda_):
    if list(mu.Is) == list(lambda_.Is):
        return mu.N / mu. L
    elif mu.integer_momentum == lambda_.integer_momentum:
        return 0
    momentum_sum = np.sum(mu.lambdas - lambda_.lambdas)
    denominator = -0.5 * 1j / np.imag(V_plus(mu, lambda_, lambda_.lambdas[0]))
    U = construct_U(mu, lambda_)
    detpart = np.linalg.det(np.identity(mu.N) + U)

    prod1 = 1
    for j in range(mu.N):
        prod1 *= 2 * 1j * np.imag(V_plus(mu, lambda_, lambda_.lambdas[j]))
    prod2 = 1
    for j in range(mu.N):
        for k in range(mu.N):
            prod2 *= (lambda_.lambdas[j] - lambda_.lambdas[k] + 1j * mu.c) / (mu.lambdas[j] - lambda_.lambdas[k])
    ff = momentum_sum * prod1 * prod2 * detpart * denominator
    return ff * 1j


def V_plus(mu, lambda_, lambda_j):
    return np.prod((mu.lambdas - lambda_j + 1j * mu.c) / (lambda_.lambdas - lambda_j + 1j * mu.c))


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
            U[j, k] = 0.5 * (mu.lambdas[j] - lambda_.lambdas[j]) / np.imag(V_plus(mu, lambda_, lambda_.lambdas[j]))
            kernel_part = (kernel(lambda_.lambdas[j] - lambda_.lambdas[k], mu.c) - kernel(lambda_.lambdas[0] - lambda_.lambdas[k], mu.c))
            U[j, k] *= products[j]
            U[j, k] *= kernel_part

    return U


def calculate_normalized_form_factor(mu, lambda_):
    unnormalized_ff = psi_form_factor(mu, lambda_)
    return unnormalized_ff / np.sqrt(mu.norm * lambda_.norm)


if __name__ == "__main__":
    N = 10
    data = []
    rstate = lls.lieb_liniger_state(1, N, N)
    rstate.calculate_all()
    for k in range(10000):
        bethe_numbers = lls.generate_bethe_numbers(N, list(rstate.Is))
        lstate = lls.lieb_liniger_state(1, N, N, bethe_numbers)
        lstate.calculate_all()
        ff = calculate_normalized_form_factor(lstate, rstate)
        try:
            data.append({"I": lstate.Is, "ff": np.abs(np.log(np.real(ff**2)))})
        except FloatingPointError:
            print(np.real(ff**2))

    pickle.dump(data, open("data.p", "wb+"))





    
    # rstate = lls.lieb_liniger_state(1, 10, 3)
    # lstate = lls.lieb_liniger_state(1, 10, 3, [-6, 0, 6])
    # rstate.calculate_all()
    # lstate.calculate_all()
    # print(lstate.norm)
    # print(rstate.norm)

    # print("<l|rho|r>", psi_form_factor(lstate, rstate))
    # print(calculate_normalized_form_factor(lstate, rstate))
    # print("<r|rho|l>", psi_form_factor(rstate, lstate))
    # print(calculate_normalized_form_factor(rstate, lstate))

    # print("\n")

    # # Random states
    # lstate = lls.lieb_liniger_state(1, 10, 10)
    # rstate = lls.lieb_liniger_state(1, 10, 10, lls.generate_bethe_numbers(10))
    # print(rstate.Is)
    # lstate.calculate_all()
    # rstate.calculate_all()

    # print("<l|rho|r>", psi_form_factor(lstate, rstate))
    # print(np.log(calculate_normalized_form_factor(lstate, rstate)))
    # print("<r|rho|l>", psi_form_factor(rstate, lstate))
    # print(calculate_normalized_form_factor(rstate, lstate))
    # print(np.log(calculate_normalized_form_factor(rstate, lstate)))
