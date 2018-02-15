import numpy as np


class lieb_liniger_state:
    def __init__(self, c, L, N, bethe_numbers=None):
        self.c = c
        self.L = L
        self.N = N
        self.energy = 0
        self.momentum = 0
        self.integer_momentum = 0
        self.norm = np.inf
        self.ff = 0
        if bethe_numbers is not None:
            self.Is = bethe_numbers
        else:
            self.Is = self.generate_gs_bethe_numbers()
        self.lambdas = np.zeros(N)
        self.gaudin_matrix = np.zeros((self.N, self.N))

    def generate_gs_bethe_numbers(self):
        """Generate ground state Bethe numbers for the Lieb-Linger model."""
        if self.N % 2 == 0:
            return -0.5 - self.N/2 + np.linspace(1, self.N, self.N)
        else:
            return np.linspace(-(self.N-1)/2, (self.N-1)/2, self.N)

    @staticmethod
    def kernel(k, c):
        return 2 * c / (c**2 + k**2)

    def calculate_gaudin_matrix(self):
        # """Calculate the Gaudin matrix."""
        for j in range(self.N):
            for k in range(self.N):
                if j == k:
                    kernel_sum = self.L
                    for kp in range(self.N):
                        kernel_sum += self.kernel(self.lambdas[j] - self.lambdas[kp], self.c)
                    self.gaudin_matrix[j, k] = kernel_sum - self.kernel(0, self.c)
                else:
                    self.gaudin_matrix[j, k] = -self.kernel(self.lambdas[j] - self.lambdas[k], self.c)

    def calculate_rapidities_newton(self, printing=False):
        """Calculate the rapidities using a multidimensional Newton method."""
        # TODO: Implement relaxation as done by Caux.
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

        return no_of_iterations

    def calculate_rapidities(self):
        self.lambdas = 2 * np.pi / self.L * np.array(self.Is)
        self.calculate_rapidities_newton()

    def calculate_energy(self):
        self.energy = np.sum(self.lambdas**2)

    def calculate_momentum(self):
        self.momentum = np.sum(self.lambdas)

    def calculate_integer_momentum(self):
        self.integer_momentum = np.sum(self.Is)

    def calculate_norm(self):
        self.norm = self.c**self.N * np.linalg.det(self.gaudin_matrix)
        for k in range(self.N):
            for j in range(k+1, self.N):
                self.norm *= ((self.lambdas[j] - self.lambdas[k])**2 + self.c**2) / (self.lambdas[j] - self.lambdas[k])**2

    def calculate_all(self):
        self.calculate_rapidities()
        self.calculate_energy()
        self.calculate_momentum()
        self.calculate_integer_momentum()
        self.calculate_norm()


def generate_bethe_numbers(N, ref_state, max_I = np.inf):
    """Generate Bethe numbers for excited states."""
    bethe_numbers = np.full(N, 10.**7, dtype=np.float)
    no_of_unique_entries = 0
    while no_of_unique_entries < N:
        random_number = np.round(np.random.normal(0, N))
        if N % 2 == 0:
            random_number += np.sign(np.random.uniform(-1, 1)) * 0.5

        if (np.abs(bethe_numbers - random_number) < 10e-4).any():
            continue
        # Implement a UV cutoff in Bethe number space.
        elif abs(random_number) > max_I:
            continue
        else:
            bethe_numbers[no_of_unique_entries] = random_number
            no_of_unique_entries += 1

    # Make sure we do not generate the reference state.
    if list(np.sort(bethe_numbers)) == ref_state:
        return generate_bethe_numbers(N, ref_state)
    else:
        return np.sort(bethe_numbers)


def mutate_bethe_numbers(bethe_numbers):
    """Mutate a given state of Bethe numbers to have an excited particle."""
    while True:
        rand_index = np.random.randint(len(bethe_numbers))
        new_bethe_numbers = list(bethe_numbers[:rand_index]) + list(bethe_numbers[rand_index+1:])
        new_bethe_numbers += [bethe_numbers[rand_index] + np.sign(np.random.uniform(-1, 1))]
        if len(np.unique(new_bethe_numbers)) == len(bethe_numbers):
            return sorted(new_bethe_numbers)


if __name__ == "__main__":
    with open("lieblinigerc1L100N10random.txt", "a+") as file:
        for i in range(10000):
            bethe_numbers = generate_bethe_numbers(10)
            llstate = lieb_liniger_state(1, 100, 10, bethe_numbers)
            llstate.calculate_all()
            file.write(str(repr(list(llstate.Is))) + "\n" + str(llstate.energy) + "\n")
