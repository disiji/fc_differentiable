import pickle
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'

from utils.utils_plot import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch.nn.functional as F
import torch
from math import *
import utils.utils_load_data as dh
from utils.utils_plot import run_leaf_gate_plots
from utils.utils_plot import run_gate_motion_from_saved_results
import yaml

def make_dev_data_plots():
    model_path = '../output/single_two_phase_gs=10/model.pkl'
    cell_sz = .1
    with open('../data/cll/x_dev_4d_1p.pkl', 'rb') as f:
        x_dev_list = pickle.load(f)

    with open('../data/cll/y_dev_4d_1p.pkl', 'rb') as f:
        labels= pickle.load(f)

    feature_names = ['CD5', 'CD19', 'CD10', 'CD79b']
    feature2id = dict((feature_names[i], i) for i in range(len(feature_names)))
    x_dev_list, offset, scale = dh.normalize_x_list(x_dev_list)

#    get_dafi_gates(offset, scale, feature2id)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    DAFI_GATES = get_dafi_gates(offset, scale, feature2id)

    plot_samples_and_gates_cll_4d_dev(x_dev_list, labels, model, DAFI_GATES, cell_sz=cell_sz)



if __name__ == '__main__':
    #experiment_yaml_file = '../configs/testing_corner_init.yaml'
    experiment_yaml_file = '../configs/testing_overlaps.yaml'
    #run_gate_motion_from_saved_results(experiment_yaml_file)
    run_leaf_gate_plots(experiment_yaml_file)
