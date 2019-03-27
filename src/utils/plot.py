import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_gate(x1, x2, gates, gate_names, filename=None, normalized=True):
    """

    :param x1:
    :param x2:
    :param gates: a list of 2d gates, features in different gates should match
    :param gate_names: a list of gate names.
    :param filename:
    :param normalized:
    :return:
    """
    fig, ax = plt.subplots(1)
    # Add scatter plot
    ax.scatter(x1, x2)
    n_gates = len(gates)
    for i in range(n_gates):
        dim1 = gates[i].gate_dim1
        dim2 = gates[i].gate_dim2
        gate_low1 = gates[i].gate_low1
        gate_low2 = gates[i].gate_low1
        gate_upp1 = gates[i].gate_upp1
        gate_upp2 = gates[i].gate_upp1
        # Create a Rectangle patch
        rect = patches.Rectangle((gate_low1, gate_low2), gate_upp1 - gate_low1, gate_upp2 - gate_low2, linewidth=1,
                                 edgecolor='r', facecolor='none', label=gate_names[i])
        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.set_xlabel(dim1)
    ax.set_ylabel(dim2)
    if normalized == True:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    if filename:
        fig.savefig(filename, dpi=90, bbox_inches='tight')
    return ax
