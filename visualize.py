import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sum_rule as sr

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def visualize_dsf(dsf_data, fermi_momentum):
    x_list = []
    y_list = []
    z_list = []
    for k, states in sorted(dsf_data.items()):
        if k > 0:
            for state in states:
                x_list.append(k / fermi_momentum)
                y_list.append(state.energy / fermi_momentum**2)
                z_list.append(np.abs(state.ff)**2)

    x = np.array(x_list)
    y = np.array(y_list)
    z = np.array(z_list)
    f, ax = plt.subplots()

    contourplot = ax.tricontourf(x, y, z, 15) 
    for c in contourplot.collections:
        c.set_edgecolor("face")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 15)
    ax.set_xlabel(r"$k/k_F$")
    ax.set_ylabel(r"$\omega/k_F^2$")
    sns.despine()
    plt.show()
    

def visualize_sum_rule_momentum_distribution(dsf_data, reference_state):
    data = []
    for k, states in sorted(dsf_data.items()):
        if k > 0:
            sum_rule_value = sr.left_side(states, reference_state.energy) / sr.right_side(k, reference_state.N, reference_state.L)
            print(sum_rule_value)
            data.append((k, sum_rule_value))
    plt.semilogy(np.array([d[0] for d in data]) / (2 * np.pi * np.max(reference_state.Is)/ reference_state.L), [d[1] for d in data] / (2 * np.pi * np.max(reference_state.Is)/ reference_state.L)**2)
    plt.show()


def visualize_no_of_states_per_slice(dsf_data):
    plt.bar(sorted(dsf_data.keys()), [len(dsf_data[x]) for x in sorted(dsf_data.keys())])
    sns.despine()
    plt.show()


def visualize_sumrule_per_contributing_state(data, ref_energy, L, N):
    momenta = []
    dsf_per_state = []
    dsf_per_slice = []
    # Catch empty dictionaries.
    if not data:
        return 0
    for momentum, states in sorted(data.items()):
        if momentum != 0:
            momenta.append(momentum)
            dsf_per_slice.append(sr.left_side(states, ref_energy) / sr.right_side(momentum, L, N))
            dsf_per_state.append(sr.left_side(states, ref_energy) / sr.right_side(momentum, L, N) / len(states))

    plt.bar(momenta, dsf_per_slice, color="b")
    plt.ylabel("Sumrule saturation")
    plt.show()

    plt.bar(momenta, dsf_per_state, color="r")
    plt.ylabel("Sumrule saturation per state")
    plt.show()
