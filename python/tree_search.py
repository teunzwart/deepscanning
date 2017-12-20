import operator
import numpy as np
import matplotlib.pyplot as plt


import lieb_liniger_state as lls

L = 100


def all_unique(state):
    """Check if all entries of a state are unique."""
    if len(state) > len(set(state)):
        return False
    else:
        return True


def generate_all_legal_new_states(current_state, previous_states):
    """
    Generate all new legal states for the Lieb-Liniger model.

    Legal states are all those states derived from the current state
    that mutate one of the Bethe numbers by +/- 1, have only unique
    entries, and have not previously been visited.
    """
    legal_new_states = []
    for i, bethe_number in enumerate(current_state):
        for p in [-1, 1]:
            new_state = sorted(list(current_state[:i]) + list(current_state[i+1:]) + list([current_state[i] + p]))
            if all_unique(new_state) and new_state not in previous_states:
                legal_new_states.append(new_state)

    return legal_new_states


def recurse_through_states(state, previous_states, depth):
    if depth == 0:
        return previous_states
    new_legal_states = generate_all_legal_new_states(state, previous_states)
    for a in new_legal_states:
        previous_states.append({"parent": state, "child": a})
        previous_states = recurse_through_states(a, previous_states, depth-1)
    return previous_states


if __name__ == "__main__":
    state = lls.lieb_liniger_state(1, L, 2)
    previous_states = [{"parent": [0, 0], "child": list(state.Is)}]
    b = recurse_through_states(list(state.Is), previous_states, 3)
    print(b)
    energies = []
    for z in b[1:]:
        print(z)
        statea = lls.lieb_liniger_state(1, L, 2, z["child"])
        statea.calculate_all()
        stateb = lls.lieb_liniger_state(1, L, 2, z["parent"])
        stateb.calculate_all()
        print(statea.energy - stateb.energy)
        energies.append([z, 1/(1 + abs(statea.energy - stateb.energy))])
    print(energies[0][1])
    print(sorted(energies, key=operator.itemgetter(1)))
    for a in energies:
        print(np.array(a[0]["child"]) * 2, a[1])

    x = y = np.linspace(-8, 8)

    X, Y = np.meshgrid(x,y)
    Z = np.zeros((50,50))

    for k in energies:
        i, j = np.array(a[0]["child"]) * 2
        print(i, j)
        Z[int(i), int(j)] = k[1]

    plt.pcolor(X, Y, Z)
    plt.show()
