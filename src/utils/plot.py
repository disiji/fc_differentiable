import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_gate(x1, x2, dim1, dim2, gate_low1, gate_upp1, gate_low2, gate_upp2, filename=None, normalized=True):
    """

    :param x1: a numpy array of data
    :param x2: a numpy array of data
    :param dim1: a string of feature name
    :param dim2: a string of feautre name
    :param gate_low1: lower boundary on dim1
    :param gate_upp1: upper boundary on dim1
    :param gate_low2: lower boundary on dim2
    :param gate_upp2: upper boundary on dim2
    :return:
    """
    fig, ax = plt.subplots(1)
    # Add scatter plot
    ax.scatter(x1, x2)
    # Create a Rectangle patch
    rect = patches.Rectangle((gate_low1, gate_low2), gate_upp1 - gate_low1, gate_upp2 - gate_low2, linewidth=1,
                             edgecolor='r', facecolor='none')
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
