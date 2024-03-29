import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sum_rule as sr

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

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
    z = np.log(np.array(z_list) + 1)
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
            data.append((k, sum_rule_value))
    plt.semilogy(np.array([d[0] for d in data]) / (2 * np.pi * np.max(reference_state.Is)/ reference_state.L), [d[1] for d in data] / (2 * np.pi * np.max(reference_state.Is)/ reference_state.L)**2)
    sns.despine()
    plt.show()


def visualize_no_of_states_per_slice(dsf_data, save=True):
    plt.bar(sorted(dsf_data.keys()), [len(dsf_data[x]) for x in sorted(dsf_data.keys())])
    plt.xlabel("Total state momentum")
    plt.ylabel("Number of states")
    sns.despine()
    if save:
        plt.savefig("no_of_states.pdf", bbox='tight')
    plt.show()


def visualize_sumrule_per_contributing_state(data, ref_energy, L, N, xlim, save=False):
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

    plt.bar(momenta, dsf_per_slice)
    plt.xlabel("Integer momentum")
    plt.ylabel("Sum rule saturation")
    plt.xlim(xlim)
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig("saturations_over_momenta.pdf", bbox='tight')
    plt.show()

    plt.bar(momenta, dsf_per_state, color="r")
    plt.xlabel("Integer momentum")
    plt.ylabel("Mean sum rule saturation per state")
    plt.xlim(xlim)
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig("saturations_over_states.pdf", bbox='tight')
    plt.show()

def visualize_form_factor_sizes(form_factors, include_ordered=True, include_trend=True, save=True, filename=""):
    """Input should be in 'raw' form, i.e. the complex form factors. ABACUS outputs real(FF)^2."""
    plt.semilogy(np.abs(form_factors)**2, 'ro', markersize=1)
    if include_ordered:
        plt.semilogy(sorted(np.abs(form_factors)**2)[::-1], label="Ordered form factors")
    if include_trend:
        with np.errstate(divide='ignore'):
            abs_ff = np.log(np.abs(form_factors)**2)
            indices = np.where(np.isfinite(abs_ff))
            polynomial = np.polyfit(indices[0], abs_ff[indices], 1)
            p = np.poly1d(polynomial)
            print("mid", np.exp(p(int(len(form_factors)/2))), "coeff", np.exp(polynomial[1]))
            plt.semilogy(np.exp(p(range(len(form_factors)))), label="Trend line", color="black")
    sns.despine()
    plt.legend()
    plt.xlim(xmin=0)
    plt.xlabel("Order in computation")
    plt.ylabel("Square form factor")
    if save:
        plt.savefig(f"{filename}.pdf")
    plt.show()

    return np.exp(p(int(len(form_factors)/2))), np.exp(polynomial[1])


def visualize_q_function(q_matrix, overlay=None, save=True):
    fig, ax = plt.subplots()
    ax.imshow(q_matrix)
    try:
        ax.imshow(overlay)
    except TypeError:
        pass
    ax.set_xlabel("Location to place particle")
    ax.set_ylabel("Location to remove particle from")
    if save:
        fig.savefig("q_function.pdf", bbox_inches='tight')
    plt.show()

def visualize_state(state, save=True):
    plt.imshow(np.vstack((state, state)))
    plt.yticks([])
    if save:
        plt.savefig("state.pdf", bbox_inches='tight')
    plt.show()

def generate_overlay(prediction, overlay_type, number, N_world):
    """
    overlay_type: highest or lowest N predictions to highlight
    number: number of predictions to highlight"""
    if overlay_type == "highest":
        poss = np.argpartition(prediction.reshape(1,-1), N_world**2 - number)[0][-number:]
    elif overlay_type == "lowest":
        poss = np.argpartition(prediction.reshape(1,-1), number)[0][:number]

    m = np.array(range(N_world **2))
    for ty in m:
        if ty in poss:
            m[ty] = True
        else:
            m[ty] = False

    overlay = np.zeros((N_world, N_world, 4))
    overlay[...,0] = 1.
    overlay[...,3] = m.reshape(N_world , N_world)

    return overlay

def visualize_saturation_history(saturations, save=False, filename=""):
    for hist in saturations:
        plt.plot(hist)
    sns.despine()
    plt.xlim(xmin=0)
    plt.xlabel("Order in computation")
    plt.ylabel("Total sum rule saturation")
    if save:
        plt.savefig(f"{filename}.pdf")
    plt.show()
