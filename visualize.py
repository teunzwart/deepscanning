import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sum_rule as sr

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
                plt.scatter(x_list, y_list)
    plt.show()
    from scipy.interpolate import interp2d

    # # f will be a function with two arguments (x and y coordinates),
    # # but those can be array_like structures too, in which case the
    # # result will be a matrix representing the values in the grid 
    # # specified by those arguments
    # f = interp2d(x_list,y_list,z_list,kind="linear")

    # x_coords = np.arange(min(x_list),max(x_list)+1)
    # y_coords = np.arange(min(y_list),max(y_list)+1)
    # Z = f(x_coords,y_coords)
    # print(Z)
    
    # fig = plt.imshow(Z,
    #                  extent=[min(x_list),max(x_list),min(y_list),max(y_list)],
    #                  origin="lower")

    # # Show the positions of # TODO: he sample points, just to have some reference
    # fig.axes.set_autoscale_on(False)
    # plt.scatter(x_list,y_list,400,facecolors='none')
    

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
