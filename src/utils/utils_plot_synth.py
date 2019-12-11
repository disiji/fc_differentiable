import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from utils.DataAndGatesPlotter import DataAndGatesPlotter
from utils.input import SynthInput

matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.linewidth'] = 5
DEFAULT_PLOT_CONFIG = \
    {
        'figsize': (20, 10),
        'ncols': 7,
        'hspace': None,
        'wspace': 0,
        'size': 1,
        'right': .7,
        'top': .7,
        'title': 'Synth Gates',
        'savepath_fmt_str': '../output/%s/%s'
    }


def plot_synth_data_with_gates(models, concatenated_data, hparams, plot_config={}):
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
    DEFAULT_PLOT_CONFIG.update(plot_config)
    plot_config = DEFAULT_PLOT_CONFIG
    features = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    if plot_config['ncols'] < 7:
        raise ValueError('Need at least seven columns to display tree')
    input = SynthInput(hparams)

    fig = plt.figure(figsize=plot_config['figsize'])
    plt.tight_layout()
    grid = gridspec.GridSpec(nrows=3, ncols=plot_config['ncols'], figure=fig)
    grid.update(hspace=plot_config['hspace'], wspace=plot_config['wspace'])
    axes = []
    axes.append(fig.add_subplot(grid[0, plot_config['ncols'] // 2]))
    axes.append(fig.add_subplot(grid[1, plot_config['ncols'] // 2 - 1]))
    axes.append(fig.add_subplot(grid[1, plot_config['ncols'] // 2 + 1]))
    axes.append(fig.add_subplot(grid[2, plot_config['ncols'] // 2 + 2]))
    fig.suptitle(plot_config['title'], x=.53, y=0, fontsize=25)

    for i in range(len(axes)):
        axes[i].set_xlim(right=plot_config['right'])
        axes[i].set_ylim(top=plot_config['top'])
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
    axes[0].set_ylabel('M2')
    axes[0].set_xlabel('M1')
    con = ConnectionPatch(xyA=(0., 0.), xyB=(.5, 1.), coordsA="axes fraction", \
                          coordsB="axes fraction", axesA=axes[0], axesB=axes[1], color='k', arrowstyle='-|>')

    axes[0].add_artist(con)

    axes[1].set_ylabel('M4')
    axes[1].set_xlabel('M3')
    con = ConnectionPatch(xyA=(1., 0.), xyB=(.5, 1.), coordsA="axes fraction", \
                          coordsB="axes fraction", axesA=axes[0], axesB=axes[2], color='k', arrowstyle='-|>')

    axes[0].add_artist(con)

    axes[2].set_ylabel('M6')
    axes[2].set_xlabel('M5')
    con = ConnectionPatch(xyA=(1., 0.), xyB=(.5, 1.), coordsA="axes fraction", \
                          coordsB="axes fraction", axesA=axes[2], axesB=axes[3], color='k', arrowstyle='-|>')

    axes[2].add_artist(con)

    axes[3].set_ylabel('M8')
    axes[3].set_xlabel('M7')

    flat_axes = np.array([axes[0], axes[2], axes[3], axes[1]])
    for axis in flat_axes:
        axis.set_ylim(0, 1)
        axis.set_xlim(0, 1)
    init_gate_plotter = DataAndGatesPlotter(models['init'], concatenated_data)
    init_gate_plotter.plot_on_axes(flat_axes, hparams)
    final_gates_and_data_plotter = DataAndGatesPlotter(models['final'], concatenated_data, color='r')
    final_gates_and_data_plotter.plot_on_axes(flat_axes, hparams)




def plot_synth_depth3_tree_with_data(pickled_model, sample, figsize=(20, 12), ncols=7, hspace=None, wspace=0,
                                     init_gates='large_middle', size=1, saveas='testing_synth_tree.png', right=.7,
                                     top=.7, title='NAME ME'):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    features = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
    feature2id = dict((features[i], i) for i in range(len(features)))
    if ncols < 7:
        raise ValueError('Need at least seven columns to display tree')
    sample, offset, scale = dh.normalize_x_list([sample])
    sample = sample[0]  # unpack list of one element
    print(offset, scale)

    if type(init_gates) == list:
        init_gates = np.array(init_gates)
    if init_gates == 'large_middle':
        init_gates = get_large_middle_init_nested_list(offset, scale, feature2id)
    elif init_gates == 'easy':
        init_gates = get_easy_init_nested_list(offset, scale, feature2id)
    elif init_gates == 'close_to_zero':
        init_gates = get_close_to_zero_init_list(offset, scale, feature2id)
    else:
        raise ValueError('Init gate type not found')

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    grid = gridspec.GridSpec(nrows=3, ncols=ncols, figure=fig)
    grid.update(hspace=hspace, wspace=wspace)
    axes = []
    axes.append(fig.add_subplot(grid[0, ncols // 2]))
    axes.append(fig.add_subplot(grid[1, ncols // 2 - 1]))
    axes.append(fig.add_subplot(grid[1, ncols // 2 + 1]))
    axes.append(fig.add_subplot(grid[2, ncols // 2 + 2]))
    fig.suptitle(title, x=.53, y=0, fontsize=10)

    for i in range(len(axes)):
        axes[i].set_xlim(right=right)
        axes[i].set_ylim(top=top)
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
    axes[0].set_ylabel('M2')
    axes[0].set_xlabel('M1')
    con = ConnectionPatch(xyA=(0., 0.), xyB=(.5, 1.), coordsA="axes fraction", \
                          coordsB="axes fraction", axesA=axes[0], axesB=axes[1], color='k', arrowstyle='->')

    axes[0].add_artist(con)

    axes[1].set_ylabel('M4')
    axes[1].set_xlabel('M3')
    con = ConnectionPatch(xyA=(1., 0.), xyB=(.5, 1.), coordsA="axes fraction", \
                          coordsB="axes fraction", axesA=axes[0], axesB=axes[2], color='k', arrowstyle='->')

    axes[0].add_artist(con)

    axes[2].set_ylabel('M6')
    axes[2].set_xlabel('M5')
    con = ConnectionPatch(xyA=(1., 0.), xyB=(.5, 1.), coordsA="axes fraction", \
                          coordsB="axes fraction", axesA=axes[2], axesB=axes[3], color='k', arrowstyle='->')

    axes[2].add_artist(con)

    axes[3].set_ylabel('M8')
    axes[3].set_xlabel('M7')

    keys = [key for key in pickled_model.children_dict.keys()]
    node0 = pickled_model.root
    node1 = pickled_model.children_dict[keys[3]][0]
    node2 = pickled_model.children_dict[keys[3]][1]
    node3 = pickled_model.children_dict[keys[2]][0]

    gate0 = get_gate(node0, scale[0:2], offset[0:2])
    gate1 = get_gate(node1, scale[2:4], offset[2:4])
    gate2 = get_gate(node2, scale[4:6], offset[4:6])
    gate3 = get_gate(node3, scale[6:], offset[6:])

    plot_initial_final_gate(axes[0], sample[:, [0, 1]], init_gates[0], gate0, size=size, color='r')
    gate0_filtered = filter_2d(sample, gate0, 0, 1)
    plot_initial_final_gate(axes[1], gate0_filtered[:, [2, 3]], init_gates[1], gate1, size=4 * size, color='r')
    plot_initial_final_gate(axes[2], gate0_filtered[:, [4, 5]], init_gates[2], gate2, size=4 * size, color='r')
    plot_initial_final_gate(axes[3], filter_2d(gate0_filtered, gate2, 4, 5)[:, [6, 7]], init_gates[3], gate3,
                            size=10 * size, color='r')

    fig.savefig('../output/' + saveas, dpi=300, bbox_inches='tight')
