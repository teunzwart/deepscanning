import numpy as np


class lieb_liniger_state:
    def __init__(self, c, L, N, bethe_numbers=None):
        self.c = c
        self.L = L
        self.N = N
        if bethe_numbers is not None:
            self.Is = bethe_numbers
        else:
            self.Is = self.generate_gs_bethe_numbers()
        self.lambdas = np.zeros(N)
        self.gaudin_matrix = np.matrix([[0.0 for _ in range(self.N)] for _ in range(self.N)])

    def generate_gs_bethe_numbers(self):
        """Generate ground state Bethe numbers for the Lieb-Linger model."""
        if self.N % 2 == 0:
            return -0.5 - self.N/2 + np.linspace(1, self.N, self.N)
        else:
            return np.linspace(-(self.N-1)/2, (self.N-1)/2, self.N)

    def kernel(self, k, c):
        return 2 * c / (c**2 + k**2)

    def calculate_gaudin_matrix(self):
        """Calculate the Gaudin matrix."""
        for j in range(self.N):
            for k in range(self.N):
                if j == k:
                    kernel_sum = self.L
                    for kp in range(self.N):
                        kernel_sum += self.kernel(0, self.c)
                    self.gaudin_matrix[j, k] = kernel_sum
                else:
                    self.gaudin_matrix[j, k] = -self.kernel(self.lambdas[j] - self.lambdas[k], self.c)

    def calculate_rapidities_newton(self, printing=False):
        """Calculate the rapidities using a multidimensional Newton method."""
        for no_of_iterations in range(20):
            rhs_bethe_equations = np.zeros(self.N)
            for j in range(self.N):
                for k in range(self.N):
                    rhs_bethe_equations[j] += 2 * np.arctan((self.lambdas[j] - self.lambdas[k]) / self.c)
                rhs_bethe_equations[j] += self.L * self.lambdas[j] - 2 * np.pi * self.Is[j]

            self.calculate_gaudin_matrix()
            delta_lambda = np.linalg.solve(self.gaudin_matrix, -rhs_bethe_equations)

            diff_square = 0
            for i in range(self.N):
                diff_square += delta_lambda[i]**2 / (self.lambdas[i]**2 + 10**-6)
            diff_square /= self.N
            if printing:
                print(diff_square)

            for i in range(self.N):
                self.lambdas[i] += delta_lambda[i]

            if diff_square < 10**-14:
                return no_of_iterations
            # if printing:
            #     print(self.lambdas)

        return no_of_iterations


def generate_gs_bethe_numbers(N):
    bethe_numbers = np.full(N, 10.**7, dtype=np.float)
    no_of_unique_entries = 0
    while no_of_unique_entries < N:
        random_number = np.round(np.random.normal(0, N))
        if N % 2 == 0:
            random_number += np.sign(np.random.uniform(-1, 1)) * 0.5

        if (np.abs(bethe_numbers - random_number) < 10e-4).any():
            continue
        else:
            bethe_numbers[no_of_unique_entries] = random_number
            no_of_unique_entries += 1

    return np.sort(bethe_numbers)


if __name__ == "__main__":
    llstate = lieb_liniger_state(1, 100, 5)
    llstate.lambdas = 2 * np.pi / llstate.L * llstate.Is
    llstate.calculate_rapidities_newton()
    print(llstate.lambdas)
    print(generate_gs_bethe_numbers(6))
