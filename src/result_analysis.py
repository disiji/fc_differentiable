import matplotlib
import yaml

matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'
import pandas as pd
from scipy import stats
import os
from utils.ParameterParser import ParameterParser
from utils.CellOverlaps import CellOverlaps
from utils.bayes_gate import ModelTree
from utils.input import *
import utils.utils_load_data as dh
import torch.nn as nn

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 10000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.0,
    'negative_proportion_default': 0.0001,
    'positive_box_penalty': 0.0,
    'corner_penalty': .0,
    'feature_diff_penalty': 0.,
    'gate_size_penalty': .0,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 100,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.05,
    'batch_size': 10,
    'n_epoch': 1000,
    'seven_epochs_for_gate_motion_plot': [0, 50, 100, 200, 300, 400, 500],
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 2,
    'init_type': 'random_corner',
    'corner_init_deterministic_size': .75,
    'train_alternate': True,
    'run_logistic_to_convergence': False,
    'output': {
        'type': 'full'
    },
    'annealing': {
        'anneal_logistic_k': False,
        'final_k': 1000,
        'init_k': 1
    },
    'two_phase_training': {
        'turn_on': False,
        'num_only_log_loss_epochs': 50
    },
    'plot_params': {
        'figsize': [10, 10],
        'marker_size': .01,
    },
    'use_out_of_sample_eval_data': False,
    'dictionary_is_broken': True
}

column_names = ['random_state',
                'train_accuracy',
                'eval_accuracy',
                'overall_accuracy',
                'train_accuracy_dafi',
                'eval_accuracy_dafi',
                'overall_accuracy_dafi',
                'train_tracker.acc_opt'
                'eval_tracker.acc_opt',
                'train_logloss',
                'eval_logloss',
                'overall_logloss',
                'train_logloss_dafi',
                'eval_logloss_dafi',
                'overall_logloss_dafi',
                'train_auc',
                'eval_auc',
                'overall_auc',
                'train_auc_dafi',
                'eval_auc_dafi',
                'overall_auc_dafi',
                'train_brier_score',
                'eval_brier_score',
                'overall_brier_score',
                'train_brier_score_dafi',
                'eval_brier_score_dafi',
                'overall_brier_score_dafi',
                'run_time']

#  generate scatter plots of a method and dafi gates
metric_dict = ['accuracy', 'logloss', 'auc', 'brier_score']
# model_dict = ['default', 'dafi_init', 'dafi_regularization', 'default_non_alternate', 'emp_regularization_off',
#               'gate_size_regularization_off']
model_dict = ['default']


# load from csv
def scatter_vs_dafi_feature(dataname, method_name, metric_name, ax):
    filename = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name)
    if metric_name not in metric_dict:
        raise ValueError('%s is not in metric_dict.' % metric_name)
    df = pd.read_csv(filename, header=None, names=column_names)
    # stats test
    print(stats.ttest_rel(df['eval_%s' % metric_name], df['eval_%s_dafi' % metric_name]))
    print(stats.ks_2samp(df['eval_%s' % metric_name], df['eval_%s_dafi' % metric_name]))

    ax.scatter(df['eval_%s' % metric_name], df['eval_%s_dafi' % metric_name], s=5)
    ax.set_xlim(min(min(df['eval_%s' % metric_name]), min(df['eval_%s_dafi' % metric_name])), \
                max(max(df['eval_%s' % metric_name]), max(df['eval_%s_dafi' % metric_name])))
    ax.set_ylim(min(min(df['eval_%s' % metric_name]), min(df['eval_%s_dafi' % metric_name])), \
                max(max(df['eval_%s' % metric_name]), max(df['eval_%s_dafi' % metric_name])))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    return ax


def scatter_methods(dataname, method_name_1, method_name_2, metric_name):
    # todo: need to make df of different methods to plot of same length
    """

    :param dataname:
    :param method_name_1:
    :param method_name_2:
    :param metric_name:
    :return:
    """
    filename_1 = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name_1)
    filename_2 = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name_2)
    if metric_name not in metric_dict:
        raise ValueError('%s is not in metric_dict.' % metric_name)
    df_1 = pd.read_csv(filename_1, header=None, names=column_names)
    df_2 = pd.read_csv(filename_2, header=None, names=column_names)

    figname_train = '../fig/%s/%s_vs_%s_%s_train.png' % (dataname, method_name_1, method_name_2, metric_name)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_1['train_%s' % metric_name], df_2['train_%s' % metric_name], s=5)
    ax.set_xlabel(method_name_1)
    ax.set_ylabel(method_name_2)
    if metric_name in ['accuracy', 'auc', 'brier_score']:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    elif metric_name == 'logloss':
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 5.0)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    fig.tight_layout()
    plt.savefig(figname_train)

    figname_test = '../fig/%s/%s_vs_%s_%s_test.png' % (dataname, method_name_1, method_name_2, metric_name)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_1['eval_%s' % metric_name], df_2['eval_%s' % metric_name], s=5)
    ax.set_xlabel(method_name_1)
    ax.set_ylabel(method_name_2)
    if metric_name in ['accuracy', 'auc', 'brier_score']:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.plot()
    elif metric_name == 'logloss':
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 5.0)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    fig.tight_layout()
    plt.savefig(figname_test)


def combine_results_into_one_csv(directory_prefix, output_dir, y_data_path, corner_reg_grid=[0., .1, .2, .3, .4, .5],
                                 gate_size_reg_grid=[0., .1, .2, .3, .4, .5], decimal_points_in_dir_name=2):
    # Looks like the data is not shuffled assuming the test_size hparam is set to 0
    # Other wise this wont be matched up properly
    # I have two sets of results: one with corner reg 0->.5 and the other with corner reg
    # taking values in [.001, .050]
    str_identifiers = []
    features_list = []
    avg_results_dicts = []
    weights_list = []
    for c, corner_reg in enumerate(corner_reg_grid):
        for gate_size_reg in gate_size_reg_grid:
            if directory_prefix == '../output/logreg_to_conv_grid_search':
                if decimal_points_in_dir_name == 2:
                    str_identifier = '_corner=%.2f_gate_size=%.2f' % (corner_reg, gate_size_reg)
                else:
                    str_identifier = '_corner=%.3f_gate_size=%.3f' % (corner_reg, gate_size_reg)
            elif directory_prefix == '../output/two_phase_logreg_to_conv_grid_search_gate_size=':
                str_identifier = '%.2f' % gate_size_reg
            directory = directory_prefix + str_identifier
            # include Dafi results as the first column
            if c == 0:
                #                dafi_features = get_features(os.path.join(directory, 'features_dafi.csv'))
                #                str_identifiers.append('DAFI')
                #                features_list.append(dafi_features)
                # get DAFI outputs here
                #                avg_results_dicts.append(None)
                #                weights_list.append(None)
                pass

            avg_results_dict = avg_results(os.path.join(directory, 'results_cll_4D.csv'))
            features = get_features(os.path.join(directory, 'features_model.csv'))
            weights = get_weights(os.path.join(directory, 'model_classifier_weights.csv'))

            avg_results_dicts.append(avg_results_dict)
            features_list.append(features)
            weights_list.append(weights)

            str_identifiers.append(str_identifier)
    with open(y_data_path, 'rb') as f:
        labels = pickle.load(f)
    write_concatenated_results(output_dir, avg_results_dicts, features_list, weights_list, str_identifiers, labels)
    fig, axes = plt.subplots(1, 2, sharey=True)
    feats_noreg = np.array(features_list[0])
    labels = np.array(labels)
    features_pos = feats_noreg[labels == 0]
    features_neg = feats_noreg[labels == 1]
    axes[0].boxplot(features_pos)
    axes[1].boxplot(features_neg)
    plt.savefig(os.path.join(output_dir, 'feats_box_plot_noreg.png'))


def write_concatenated_results(output_dir, avg_results_dicts, features_list, weights_list, str_identifiers, labels):
    with open(os.path.join(output_dir, 'concatenated_results.csv'), 'w') as f:
        # get the column labels which correspond to different settings
        col_labels = ''
        for s, string in enumerate(str_identifiers):
            if s == len(str_identifiers) - 1:
                col_labels += string + ', label' + '\n'
                continue
            col_labels += string + ','
        f.write(col_labels)
        num_samples = len(features_list[0])  # each feature is one sample in this case
        for row in range(num_samples):
            for column in range(len(features_list)):
                feature = features_list[column][row]
                if column == len(features_list) - 1:
                    f.write(str(feature) + ',%s' % (labels[row]) + '\n')  # assumes labels are matched up
                    continue
                f.write(str(feature) + ',')


def get_weights(weights_path):
    with open(weights_path, 'r') as f:
        last_run_weights = f.readlines()[-1][0:-2]  # get rid of newline char
        last_run_weights = [float(last_run_weights.split(',')[0]), float(last_run_weights.split(',')[1])]
    return last_run_weights


def get_features(features_path):
    with open(features_path, 'r') as f:
        lines = f.readlines()
        # only giving Padhraic the last run for now
        # have 35 features/samples, and the last line is blank
        lines_last_run = lines[-36:-1]
        feats = [float(feat_str[0:-1]) for feat_str in lines_last_run]

    return feats


def avg_results(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
        # only want odd lines which have actual integers
        lines_with_results = [line.split(',') for l, line in enumerate(lines) if l % 2 == 1]
        accs = [float(split_line[1]) for split_line in lines_with_results]
        log_losses = [float(split_line[2]) for split_line in lines_with_results]

    return {'avg_acc': sum(accs) / len(accs),
            'avg_log_loss': sum(log_losses) / len(log_losses)}


def make_and_write_concatenated_8d_data_with_dafi_gate_flags_and_ids(path_to_hparams, savepath=None):
    concatenated_data = make_data_with_dafi_gate_flags_and_ids(path_to_hparams)
    if savepath:
        savepath = savepath
    else:
        savepath = '../data/concatenated_8d_data_with_dafi_filtering_indicators.csv'
    write_concatenated_8d_data_with_dafi_gate_flags_and_ids(concatenated_data, savepath)


def make_and_write_catted_data(path_to_x_list, path_to_y_list, savepath):
    catted_data = make_catted_data(path_to_x_list, path_to_y_list)
    write_catted_data(catted_data, savepath)
    return catted_data


def write_catted_data(catted_data, savepath):
    COL_NAMES = (
        'FSC-A',
        'SSC-H',
        'CD45',
        'SSC-A',
        'CD5',
        'CD19',
        'CD10',
        'CD79b',
        'sample_ids',
        'labels'
    )

    with open(savepath, 'w') as f:
        f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % COL_NAMES)
        for cell_row in catted_data:
            f.write('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' % tuple(cell_row))


def make_catted_data(path_x_list, path_y_list):
    with open(path_x_list, 'rb') as f:
        x_list = pickle.load(f)
    with open(path_y_list, 'rb') as f:
        y_list = pickle.load(f)

    for sample_id, (x, y) in enumerate(zip(x_list, y_list)):
        x_with_sample_id = np.hstack([x, sample_id * np.ones([x.shape[0], 1])])
        x_with_sample_id_and_label = np.hstack([x_with_sample_id, y * np.ones([x.shape[0], 1])])
        if sample_id == 0.:
            catted_data = x_with_sample_id_and_label
        else:
            catted_data = np.concatenate([catted_data, x_with_sample_id_and_label])
    return catted_data


def write_concatenated_8d_data_with_dafi_gate_flags_and_ids(concatenated_data, savepath):
    COL_NAMES = (
        'FSC-A',
        'SSC-H',
        'CD45',
        'SSC-A',
        'CD5',
        'CD19',
        'CD10',
        'CD79b',
        'sample_ids',
        'labels',
        'cell_ids',
        'In Dafi Gate1',
        'In Dafi Gate2',
        'In Dafi Gate3',
        'In Dafi Gate4',
    )
    with open(savepath, 'w') as f:
        f.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' % COL_NAMES)
        for cell_row in concatenated_data:
            for i in range(4):
                if cell_row[-(i + 1)] == True:
                    cell_row[-(i + 1)] = 1
                else:
                    cell_row[-(i + 1)] = 0
            f.write(
                '%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %d, %d\n' % tuple(cell_row))


def parse_hparams(path_to_hparams):
    hparams = default_hparams
    with open(path_to_hparams, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    if hparams['train_alternate']:
        hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
                hparams['n_mini_batch_update_gates'] - 1)
    else:
        hparams['n_epoch_dafi'] = hparams['n_epoch']

    print(hparams)
    return hparams


def load_output(path_to_hparams):
    output = {}
    hparams = ParameterParser(path_to_hparams).parse_params()
    output['hparams'] = hparams
    exp_name = hparams['experiment_name']
    model_checkpoint_path = '../output/%s/model_checkpoints.pkl' \
                            % hparams['experiment_name']

    with open(model_checkpoint_path, 'rb') as f:
        model_checkpoint_dict = pickle.load(f)
    # note that the initial cuts stored in this input
    # object are not the cuts that this function uses
    # this input object is only used here because the dafi gates
    # are saved inside it
    output['cll_1p_full_input'] = Cll8d1pInput(hparams)

    output['dafi_tree'] = ModelTree(output['cll_1p_full_input'].reference_tree,
                                    logistic_k=hparams['logistic_k_dafi'],
                                    negative_box_penalty=hparams['negative_box_penalty'],
                                    positive_box_penalty=hparams['positive_box_penalty'],
                                    corner_penalty=hparams['corner_penalty'],
                                    gate_size_penalty=hparams['gate_size_penalty'],
                                    init_tree=None,
                                    loss_type=hparams['loss_type'],
                                    gate_size_default=hparams['gate_size_default'])

    output['models_per_iteration'] = [
        model_checkpoint_dict[iteration]
        for iteration in
        hparams['seven_epochs_for_gate_motion_plot']
    ]
    # Checkpoint dictionary is messed up when saving
    # since str(id(node)) for each node is changed
    # (pickling is out of place and makes a new object with a
    # new id. This only works with a chain graph to make the ids match
    # the saved object ids
    fixed_models_per_iteration = []
    for model in output['models_per_iteration']:
        cur_node = model.root
        fixed_children_dict = {}
        num_nodes = len(model.children_dict.keys())
        for key, item in model.children_dict.items():
            fixed_children_dict[str(id(cur_node))] = nn.ModuleList(item)
            if not len(model.children_dict[key]) == 0:
                cur_node = model.children_dict[key][0]
        model.children_dict = nn.ModuleDict(fixed_children_dict)

    print('root id is: ', str(id(output['models_per_iteration'][0].root)))
    keys = [key for key in output['models_per_iteration'][0].children_dict.keys()]
    print('keys are: ', output['models_per_iteration'][0].children_dict.keys())
    print('id of root in new dict is: ', str(id(output['models_per_iteration'][0].children_dict[keys[0]])))
    print('init model is: ', output['models_per_iteration'][0])
    # call split on input here if theres a bug
    return output


def write_ranked_features_model_dafi(path_to_hparams):
    output = load_output(path_to_hparams)
    hparams = output['hparams']
    model = output['models_per_iteration'][-1]
    dafi = output['dafi_tree']
    x_list = output['cll_1p_full_input'].x_train
    labels = output['cll_1p_full_input'].y_train
    features_model = []
    feature_dafi = []
    for idx, x in enumerate(x_list):
        print(x.shape)
        model_results = model(x)
        dafi_results = dafi(x)
        features_model.append([model_results['leaf_probs'][0], idx, labels[idx]])
        features_dafi.append([dafi_results['leaf_probs'][0], idx, labels[idx]])
    features_model = np.array(features_model).sort(axis=0)
    features_dafi = np.array(features_dafi).sort(axis=0)
    savepath = '../output/%s/ranked_features_table' % hparams['experiment_name']
    with open(savepath, 'w') as f:
        f.write(
            'ranked features from model, sample id, matching label, ranked features from dafi, sample id, matching label\n')
        for idx, label in enumerate(labels):
            row = (
                features_model[idx][0], features_model[idx][1],
                features_model[idx][2],
                features_dafi[idx][0], features_dafi[idx][1],
                features_dafi[idx][2]

            )
            f.write('%.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n' % row)


def load_from_pickle(path_to_file):
    with open(path_to_file, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object


def make_data_with_dafi_gate_flags_and_ids(path_to_hparams):
    hparams = ParameterParser(path_to_hparams).parse_params()

    cll_1p_full_input = Cll8d1pInput(hparams)
    dafi_tree = make_dafi_tree(hparams, cll_1p_full_input)
    x_list = cll_1p_full_input.unnormalized_x_list_of_numpy
    y_list = cll_1p_full_input.y_numpy
    for sample_id, (x, y) in enumerate(zip(x_list, y_list)):
        x_with_sample_id = np.hstack([x, sample_id * np.ones([x.shape[0], 1])])
        x_with_sample_id_and_label = np.hstack([x_with_sample_id, y * np.ones([x.shape[0], 1])])
        # wrote a function to get unique cell ids in this class already

        x_with_sample_ids_cell_ids_and_labels = \
            CellOverlaps(dafi_tree, dafi_tree, [x_with_sample_id_and_label]).data_list_with_ids[0]

        # filtered_data = dafi_tree.filter_data(x_with_sample_ids_cell_ids_and_labels)
        flat_gates = [
            [102., 921., 2048., 3891.],
            [921., 2150., 102., 921.],
            [1638., 3891., 2150., 3891.],
            [0, 1228., 0., 1843.]
        ]

        flat_ids = dafi_tree.get_flat_ids()
        filtered_data = filter_data(
            x_with_sample_ids_cell_ids_and_labels,
            flat_gates,
            flat_ids
        )
        print(x_with_sample_ids_cell_ids_and_labels[:, -1])
        print(y)
        print(filtered_data[0].shape, x.shape)
        print(filtered_data[1].shape)
        print(filtered_data[2].shape)
        print(filtered_data[3].shape)
        print(filtered_data[4].shape)
        gate_1_flags = np.isin(x_with_sample_ids_cell_ids_and_labels[:, -1], filtered_data[1][:, -1])
        gate_2_flags = np.isin(x_with_sample_ids_cell_ids_and_labels[:, -1], filtered_data[2][:, -1])
        gate_3_flags = np.isin(x_with_sample_ids_cell_ids_and_labels[:, -1], filtered_data[3][:, -1])
        gate_4_flags = np.isin(x_with_sample_ids_cell_ids_and_labels[:, -1], filtered_data[4][:, -1])

        x_all_cols = np.hstack(
            [
                x_with_sample_ids_cell_ids_and_labels,
                gate_1_flags[:, np.newaxis], gate_2_flags[:, np.newaxis], gate_3_flags[:, np.newaxis],
                gate_4_flags[:, np.newaxis]
            ]
        )

        if sample_id == 0:
            catted_data = x_all_cols
        else:
            catted_data = np.concatenate([catted_data, x_all_cols])

    return catted_data


def filter_single_flat_gate(data, gate, ids):
    print(ids)
    filtered_data = dh.filter_rectangle(
        data, ids[0],
        ids[1], gate[0], gate[1],
        gate[2], gate[3]
    )
    return filtered_data


def filter_data(data, flat_gates, flat_ids):
    filtered_data = [data]
    for gate, ids in zip(flat_gates, flat_ids):
        filtered_data.append(
            filter_single_flat_gate(filtered_data[-1], gate, ids)
        )
    return filtered_data


def make_dafi_tree(hparams, cll_1p_full_input):
    dafi_tree = ModelTree(cll_1p_full_input.get_unnormalized_reference_tree(),
                          logistic_k=hparams['logistic_k_dafi'],
                          negative_box_penalty=hparams['negative_box_penalty'],
                          positive_box_penalty=hparams['positive_box_penalty'],
                          corner_penalty=hparams['corner_penalty'],
                          gate_size_penalty=hparams['gate_size_penalty'],
                          init_tree=None,
                          loss_type=hparams['loss_type'],
                          gate_size_default=hparams['gate_size_default'])
    return dafi_tree


# def test_filtered_data_matches_overlap_results():
#    filtered_path = '../output/'
#    overlap_path = ''
#    with open(filtered_path, 'r') as f:
#        cell_rows = f.readlines()
#    with open(overlap_path, 'r') as f:
#        pass


def save_feats_panel1_in_sample(OOS_hparams, model_path, dafi_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(dafi_path, 'rb') as f:
        dafi = pickle.load(f)
    if torch.cuda.is_available():
        model.cuda(OOS_hparams['device'])
        dafi.cuda(OOS_hparams['device'])

    input = Cll8d1pInput(OOS_hparams)
    # y_train and x_train are all the in sample data for this hparams setting
    feats_model_soft = model.forward_4chain(input.x_train, input.y_train)['leaf_probs'].detach().cpu().numpy()
    feats_model_hard = model.forward_4chain(input.x_train, input.y_train, use_hard_proportions=True)[
        'leaf_probs'].detach().cpu().numpy()
    feats_dafi = dafi.forward_4chain(input.x_train, input.y_train, use_hard_proportions=True)[
        'leaf_probs'].detach().cpu().numpy()

    save_array_feats = np.concatenate(
        [input.y_train.detach().cpu().numpy()[:, np.newaxis], feats_model_soft, feats_model_hard, feats_dafi], axis=1)

    header = 'Label,Soft Features Model, Hard Features Model, Hard Features DAFI'

    np.savetxt('../output/%s/feats_model_hard_and_soft_IS.csv' % OOS_hparams['experiment_name'], save_array_feats,
               delimiter=',', header=header)


def save_feats_panel1_OOS(OOS_hparams, model_path, dafi_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(dafi_path, 'rb') as f:
        dafi = pickle.load(f)
    if torch.cuda.is_available():
        model.cuda(OOS_hparams['device'])
        dafi.cuda(OOS_hparams['device'])

    input = Cll8d1pInput(OOS_hparams)
    # y_train and x_train are all the in sample data for this hparams setting
    feats_model_soft = model.forward_4chain(input.x_eval, input.y_eval)['leaf_probs'].detach().cpu().numpy()
    feats_model_hard = model.forward_4chain(input.x_eval, input.y_eval, use_hard_proportions=True)[
        'leaf_probs'].detach().cpu().numpy()
    feats_dafi = dafi.forward_4chain(input.x_eval, input.y_eval, use_hard_proportions=True)[
        'leaf_probs'].detach().cpu().numpy()

    save_array_feats = np.concatenate(
        [input.y_eval.detach().cpu().numpy()[:, np.newaxis], feats_model_soft, feats_model_hard, feats_dafi], axis=1)

    header = 'Label,Soft Features Model, Hard Features Model, Hard Features DAFI'

    np.savetxt('../output/%s/feats_model_hard_and_soft_OOS.csv' % OOS_hparams['experiment_name'], save_array_feats,
               delimiter=',', header=header)


def save_feats_both_all_data(OOS_hparams, model_both_checkpoints_path, dafi_both_path):
    with open(model_both_checkpoints_path, 'rb') as f:
        model_checkpoints_both = pickle.load(f)

    model_both = model_checkpoints_both[hparams['n_epoch']]

    with open(dafi_both_path, 'rb') as f:
        dafi_both = pickle.load(f)
    if torch.cuda.is_available():
        model_both.cuda(OOS_hparams['device'])
        dafi_both.cuda(OOS_hparams['device'])

    input = CllBothPanelsInput(OOS_hparams)
    # y_train and x_train are all the in sample data for this hparams setting
    feats_model = model_both(input.x_train, input.y_train, device=OOS_hparams['device'])[
        'leaf_probs'].detach().cpu().numpy()
    print(model_both.linear.weight, model_both.linear.bias)
    print('dafi weights for logistic regressor')
    print(dafi_both.linear.weight, dafi_both.linear.bias)
    feats_dafi = dafi_both(input.x_train, input.y_train, use_hard_proportions=True, device=OOS_hparams['device'])[
        'leaf_probs'].detach().cpu().numpy()

    feats_model_with_labels = np.concatenate([input.y_train.detach().cpu().numpy()[:, np.newaxis], feats_model], axis=1)
    feats_dafi_with_labels = np.concatenate(
        [input.y_train.detach().cpu().numpy()[:, np.newaxis], np.squeeze(feats_dafi)], axis=1)
    np.savetxt('../output/%s/feats_model.csv' % OOS_hparams['experiment_name'], feats_model_with_labels, delimiter=',',
               header='Label, Panel1, Panel2, Panel2')
    np.savetxt('../output/%s/feats_dafi.csv' % OOS_hparams['experiment_name'], feats_dafi_with_labels, delimiter=',',
               header='Label, Panel1, Panel2, Panel2')


def save_correct_dafi_hard_thresh_accs_regular_CV(CV_hparams):
    SEEDS = np.concatenate([np.arange(73, 74), np.arange(29) + 1, np.arange(51, 72)], axis=0)
    RESULTS_DIR_PREFIX = '../output/CV_neg=0.001_diff=0.001'
    # will save seed, and accuracy for model/dafi
    te_accs_per_seeds = -1 * np.ones([SEEDS.shape[0], 2])
    for s, seed in enumerate(SEEDS):
        input = Cll8d1pInput(hparams, random_state=seed)
        seed_results_dir = RESULTS_DIR_PREFIX + '_seed%d' % seed
        # path_to_model_chekpoints = seed_results_dir + '/model_checkpoints.pkl'
        # path_to_dafi_model = seed_results_dir + '/dafi_model.pkl'
        dafi_path = seed_results_dir + '/dafi_model.pkl'
        with open(dafi_path, 'rb') as f:
            dafi_model = pickle.load(f)
        dafi_model.cuda(0)
        feats_dafi = dafi_model.forward_4chain(input.x_eval, input.y_eval, use_hard_proportions=True, device=0)[
            'leaf_probs'].detach().cpu().numpy()
        feats_dafi_p1 = feats_dafi
        preds = feats_dafi_p1 > 0.0001
        acc_updated_thresh_dafi = np.sum(
            np.array([preds[i] == y.cpu().detach().numpy() for i, y in enumerate(input.y_eval)])) / preds.shape[0]

        te_accs_per_seeds[s] = [seed, acc_updated_thresh_dafi]

    header = 'seed, dafi te acc threshold at 0.0001'
    savepath = RESULTS_DIR_PREFIX + '/accs_updated_thresh=0.0001_dafi.csv'
    np.savetxt(savepath, te_accs_per_seeds, header=header, fmt='%.4f', delimiter=',')


def avg_both_CV_te_results():
    SEEDS = np.concatenate([np.arange(73, 74), np.arange(29) + 1, np.arange(51, 72)], axis=0)
    RESULTS_DIR_PREFIX = '../output/Both_Panels_CV_neg=0.001_diff=0.001'
    # will save seed, and accuracy for model/dafi
    te_accs_per_seeds = -1 * np.ones([SEEDS.shape[0], 3])
    for s, seed in enumerate(SEEDS):
        seed_results_dir = RESULTS_DIR_PREFIX + '_seed%d' % seed
        # path_to_model_chekpoints = seed_results_dir + '/model_checkpoints.pkl'
        # path_to_dafi_model = seed_results_dir + '/dafi_model.pkl'
        eval_tracker_m_path = seed_results_dir + '/tracker_eval_m.pkl'
        eval_tracker_d_path = seed_results_dir + '/tracker_eval_d.pkl'
        with open(eval_tracker_m_path, 'rb') as f:
            eval_tracker_m = pickle.load(f)

        with open(eval_tracker_d_path, 'rb') as f:
            eval_tracker_d = pickle.load(f)

        te_accs_per_seeds[s] = [seed, eval_tracker_m.acc[-1], eval_tracker_d.acc[-1]]

    variances = (np.var(te_accs_per_seeds, axis=0) ** (1 / 2))
    print('variance: model %.4f, dafi %.4f' % (variances[1], variances[2]))
    print('Avg model both te acc: %.4f' % (np.mean(te_accs_per_seeds, axis=0)[1]))
    print('Avg Dafi both te acc: %.4f' % (np.mean(te_accs_per_seeds, axis=0)[2]))
    header = 'seed, model te acc, dafi te acc logreg'
    savepath = RESULTS_DIR_PREFIX + '/accs_both.csv'
    np.savetxt(savepath, te_accs_per_seeds, header=header, fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    # yaml_filename = '../configs/OOS_both_panels.yaml' #for both panels
    yaml_filename = '../configs/OOS_Final_Model.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)

    ## For single panel feat generation trained on all 102
    # Note I accidentally mislabelled the save results with CV in the front
    model_path = '../output/CV_neg=0.001_diff=0.001_FINAL_OOS_seed0/model.pkl'
    dafi_path = '../output/CV_neg=0.001_diff=0.001_FINAL_OOS_seed0/dafi_model.pkl'
    save_feats_panel1_OOS(hparams, model_path, dafi_path)
    save_feats_panel1_in_sample(hparams, model_path, dafi_path)

    ## for both panels feat generation
    # OOS_model_both_checkpoint_path = '../output/OOS_both_panels_seed0/model_checkpoints.pkl'
    # OOS_dafi_both_path = '../output/OOS_both_panels_seed0/dafi_model.pkl'
    # save_feats_both_all_data(hparams, OOS_model_both_checkpoint_path, OOS_dafi_both_path)

    # combine_results_into_one_csv('../output/logreg_to_conv_grid_search', '../output/agg_results_logreg_to_conv_gs1', '../data/cll/y_dev_4d_1p.pkl')
    # combine_results_into_one_csv('../output/logreg_to_conv_grid_search', '../output/agg_results_logreg_to_conv_gs2', '../data/cll/y_dev_4d_1p.pkl', corner_reg_grid=[0.001, 0.050], gate_size_reg_grid=[0.25, 0.5], decimal_points_in_dir_name=3)
    # combine_results_into_one_csv('../output/two_phase_logreg_to_conv_grid_search_gate_size=', '../output/agg_results_two_phase', '../data/cll/y_dev_4d_1p.pkl', corner_reg_grid=[0.00], gate_size_reg_grid= [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.])
    path_to_hparams = '../configs/baseline_plot.yaml'
    savepath = '../data/cll/8d_FINAL/x_all.csv'
    # make_and_write_concatenated_8d_data_with_dafi_gate_flags_and_ids(path_to_hparams, savepath)
    # write_ranked_features_model_dafi(path_to_hparams)

    # avg_both_CV_te_results()

    # Get dafi accs with updated threshold
##    yaml_filename = '../configs/CV_runs.yaml'
#    hparams = default_hparams
#    with open(yaml_filename, "r") as f_in:
#        yaml_params = yaml.safe_load(f_in)
#    hparams.update(yaml_params)
#    save_correct_dafi_hard_thresh_accs_regular_CV(hparams)


# dataname = 'cll_4d_1p'
# for method_name in model_dict:
#    figname = '../output/%s/comparison.pdf' % (dataname + '_' + method_name)
#    f, axarr = plt.subplots(1, len(metric_dict), figsize=(10, 2))
#    for i, metric_name in enumerate(metric_dict):
#        axarr[i] = scatter_vs_dafi_feature(dataname, method_name, metric_name, axarr[i])
#        axarr[i].set_xlabel('Model gates')
#    axarr[0].set_ylabel('Expert gates')
#    f.savefig(figname, bbox_inches='tight')


# for i in range(len(model_dict)):
#     for j in range(i + 1, len(model_dict)):
#         for metric_name in metric_dict:
#             scatter_methods(dataname, model_dict[i], model_dict[j], metric_name)
