"""
Calculate the \rho(x=0) matrix element for the Lieb-Liniger model,
as presented by Piroli and Calabrese.
"""

import numpy as np
import copy

import lieb_liniger_state as lls
kernel = lls.lieb_liniger_state.kernel

p = 10

def psi_form_factor(mu, lambda_):
    ff = np.sum(mu.lambdas - lambda_.lambdas)
    ff /= (V_plus(mu, lambda_, p) - V_minus(mu, lambda_, p))
    ff *= np.linalg.det(np.identity(mu.N) + construct_U(mu, lambda_))
    # print("det(1 + U)", np.linalg.det(np.identity(mu.N) + construct_U(mu, lambda_)))
    # print("log(det(1 + U))", np.log(np.linalg.det(np.identity(mu.N) + construct_U(mu, lambda_))))

    for j in range(mu.N):
        ff *= (V_plus(mu, lambda_, lambda_.lambdas[j]) - V_minus(mu, lambda_, lambda_.lambdas[j]))
    for j in range(mu.N):
        for k in range(mu.N):
            ff *= (lambda_.lambdas[j] - lambda_.lambdas[k] + 1j * mu.c) / (mu.lambdas[j] - lambda_.lambdas[k])
    return ff


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
    print("prods", products)

    return U


# def calculate_normalized_form_factor(mu, lambda_):
#     unnormalized_ff = psi_form_factor(mu, lambda_)
#     return unnormalized_ff / np.sqrt(mu_state.norm * lambda_state.norm)
    
if __name__ == "__main__":
    rstate = lls.lieb_liniger_state(1, 10, 3)
    bethe_numbers = copy.copy(rstate.Is)
    bethe_numbers[0] -= 1
    lstate = lls.lieb_liniger_state(1, 10, 3, bethe_numbers)
    rstate.calculate_all()
    lstate.calculate_all()
    print(lstate.lambdas)

    
    print("norm of rstate", rstate.norm)
    print("norm of lstate", lstate.norm)
    # print(mu_state.Is, 2 * np.pi / 10 * np.sum(mu_state.Is))
    # print(lambda_state.Is, 2 * np.pi / 10 * np.sum(lambda_state.Is))
    # print(mu_state.lambdas, np.sum(mu_state.lambdas))
    # print(lambda_state.lambdas, np.sum(lambda_state.lambdas))

    print(np.identity(rstate.N) + construct_U(lstate, rstate))
    print("det", np.linalg.det(np.identity(rstate.N) + construct_U(lstate, rstate)))
    
    print("<l|rho|r>", psi_form_factor(lstate, rstate))
    print("<r|rho|l>", psi_form_factor(rstate, lstate))

    # print("\n")

    
    # Random states
    # mu_state = lls.lieb_liniger_state(1, 10, 10, lls.generate_bethe_numbers(10))
    # lambda_state = lls.lieb_liniger_state(1, 10, 10, lls.generate_bethe_numbers(10))
    # mu_state.calculate_all()
    # lambda_state.calculate_all()
    # print("norm of mu", mu_state.norm)
    # print("norm of lambda", lambda_state.norm)
    # print(mu_state.Is, 2 * np.pi / 10 * np.sum(mu_state.Is))
    # print(lambda_state.Is, 2 * np.pi / 10 * np.sum(lambda_state.Is))
    # print(mu_state.lambdas, np.sum(mu_state.lambdas))
    # print(lambda_state.lambdas, np.sum(lambda_state.lambdas))

    # print("<mu|rho|lambda>", psi_form_factor(mu_state, lambda_state))
    # print("<lambda|rho|mu>", psi_form_factor(lambda_state, mu_state))

    # mu_state = lls.lieb_liniger_state(1, 10, 10, [-13.5, -12.5, -5.5, -2.5,  -1.5,  -0.5, 0.5, 6.5, 7.5, 18.5])
    # lambda_state = lls.lieb_liniger_state(1, 10, 10, [-13.5, -10.5, -5.5, -2.5, 4.5, 6.5, 11.5, 15.5, 16.5, 18.5])
    # print(mu_state.Is)
    # print(lambda_state.Is)
    # mu_state.calculate_all()
    # lambda_state.calculate_all()
    # print(mu_state.lambdas)
    # print(lambda_state.lambdas)
    # print(psi_form_factor(mu_state, lambda_state))
    # print("ff", np.abs(calculate_normalized_form_factor(mu_state, lambda_state)))

    # print(construct_U(mu_state, lambda_state))
