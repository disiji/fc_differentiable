import warnings

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'

from utils.utils_plot import *
from utils.utils_plot_synth import *
import yaml

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
}


def make_dev_data_plots():
    model_path = '../output/single_two_phase_gs=10/model.pkl'
    cell_sz = .1
    with open('../data/cll/x_dev_4d_1p.pkl', 'rb') as f:
        x_dev_list = pickle.load(f)

    with open('../data/cll/y_dev_4d_1p.pkl', 'rb') as f:
        labels = pickle.load(f)

    feature_names = ['CD5', 'CD19', 'CD10', 'CD79b']
    feature2id = dict((feature_names[i], i) for i in range(len(feature_names)))
    x_dev_list, offset, scale = dh.normalize_x_list(x_dev_list)

    #    get_dafi_gates(offset, scale, feature2id)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    DAFI_GATES = get_dafi_gates(offset, scale, feature2id)

    plot_samples_and_gates_cll_4d_dev(x_dev_list, labels, model, DAFI_GATES, cell_sz=cell_sz)


def make_accs_and_losses_final_model():
    savefolder = '../output/CV_neg=0.001_diff=0.001_FINAL_OOS_seed0/'
    tracker_train_path = savefolder + 'tracker_train_m.pkl'
    tracker_eval_path = savefolder + 'tracker_eval_m.pkl'

    plot_accs_and_losses(tracker_train_path, tracker_eval_path, savefolder=savefolder)


def make_synth_plot(hparams, model_paths, cells_to_plot=100000, device=1):
    with open(model_paths['init'], 'rb') as f:
        model_init = pickle.load(f).cuda(device)
    if hparams['dictionary_is_broken']:
        model_init.fix_children_dict_synth()

    with open(model_paths['final'], 'rb') as f:
        model_final = pickle.load(f).cuda(device)
    if hparams['dictionary_is_broken']:
        model_final.fix_children_dict_synth()

    models = {'init': model_init, 'final': model_final}

    synth_input = SynthInput(hparams, device=device)
    data = [x.cpu().detach().numpy() for x in synth_input.x_train]
    data_pos = [x for x, y in zip(data, synth_input.y_train) if y == 1.]
    data_neg = [x for x, y in zip(data, synth_input.y_train) if y == 0.]
    catted_data_pos = np.concatenate(data_pos)
    shuffled_idxs = np.random.permutation(len(catted_data_pos))
    print('max is', np.max(catted_data_pos[shuffled_idxs][0:cells_to_plot]))
    plot_synth_data_with_gates(models, catted_data_pos[shuffled_idxs][0:cells_to_plot], hparams,
                               {'title': 'Class 1 Results'})
    savepath = '../output/%s/synth_gates_pos.png' % hparams['experiment_name']
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.clf()

    catted_data_neg = np.concatenate(data_neg)
    shuffled_idxs = np.random.permutation(len(catted_data_neg))
    plot_synth_data_with_gates(models, catted_data_neg[shuffled_idxs][0:cells_to_plot], hparams,
                               {'title': 'Class 2 Results'})
    savepath = '../output/%s/synth_gates_neg.png' % hparams['experiment_name']
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.clf()


def make_model_plot(hparams, path_to_model, device):
    # make a run that takes in just a model rather than a model chekcpoint dict
    run_model_single_iter_pos_and_neg_gates(hparams, path_to_model, device_data=device)


def make_model_plots_both_panels(hparams, path_to_model_checkpoints, num_iters=120):
    with open(path_to_model_checkpoints, 'rb') as f:
        model_checkpoints = pickle.load(f)
    model_init = model_checkpoints[0]
    model_final = model_checkpoints[num_iters]
    run_both_panels_pos_and_neg_gates(model_init, hparams, savename='pos_and_neg_plots_both_init.png')
    run_both_panels_pos_and_neg_gates(model_final, hparams, savename='pos_and_neg_plots_both_final.png')


def make_dafi_plot(hparams):
    run_dafi_single_iter_pos_and_neg_gates(hparams, device_data=0)


def make_model_loss_plots(output_dir, figsize=(9, 3)):
    with open(os.path.join(output_dir, 'tracker_train_m.pkl'), 'rb') as f:
        tracker_train = pickle.load(f)
    with open(os.path.join(output_dir, 'tracker_eval_m.pkl'), 'rb') as f:
        tracker_eval = pickle.load(f)
    log_loss_train = tracker_train.log_loss
    log_loss_eval = tracker_eval.log_loss

    acc_train = tracker_train.acc
    acc_eval = tracker_eval.acc

    reg_train = [n + f for n, f in zip(tracker_train.neg_prop_loss, tracker_train.feature_diff_loss)]
    reg_eval = [n + f for n, f in zip(tracker_eval.neg_prop_loss, tracker_eval.feature_diff_loss)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x_ticks = np.arange(len(log_loss_train))
    axes[0].plot(x_ticks, acc_train, color='b', label='Train')
    axes[1].plot(x_ticks, log_loss_train, color='b', label='Train')
    axes[2].plot(x_ticks, reg_train, color='b', label='Train')

    axes[0].plot(x_ticks, acc_eval, color='tab:orange', label='Test')
    axes[1].plot(x_ticks, log_loss_eval, color='tab:orange', label='Test')
    axes[2].plot(x_ticks, reg_eval, color='tab:orange', label='Test')
    savepath = os.path.join(output_dir, 'diagnostics.png')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # experiment_yaml_file = '../configs/testing_corner_init.yaml'
    # experiment_yaml_file = '../configs/testing_overlaps.yaml'
    # experiment_yaml_file = '../configs/testing_my_heuristic_init.yaml'

    # for both panels plots
    # path_to_model_checkpoints = '../output/Both_Panels_CV_neg=0.001_diff=0.001_seed1/model_checkpoints.pkl'
    # yaml_filename = '../configs/both_panels.yaml'

    # for dafi/model plots
    # path_to_saved_model = '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/init_model.pkl'
    # yaml_filename = '../configs/OOS_Final_Model.yaml'
    model_paths_synth = \
        {
            'init': '../output/Synth_same_reg_as_alg_seed0/model_init_seed0.pkl',
            'final': '../output/Synth_same_reg_as_alg_seed0/model_final_seed0.pkl'
        }

    # make_accs_and_losses_final_model()

    # for synth plots
    yaml_filename = '../configs/synth_plot.yaml'

    # for dafi/model plots
    # path_to_saved_model = '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/init_model.pkl'
    # path_to_saved_model = '../output/Middle_neg=0.001_diff=0.001_FINAL_OOS_seed0/model.pkl'
    # yaml_filename = '../configs/OOS_Final_Model.yaml'
    # yaml_filename = '../configs/FINAL_MODEL_middle_init.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    # make_dafi_plot(hparams)
    make_synth_plot(hparams, model_paths_synth)
    # make_model_plot(hparams, path_to_saved_model, 0)
    # make_model_loss_plots('../output/CV_neg=0.001_diff=0.001_FINAL_OOS_seed0')
    # make_model_plots_both_panels(hparams, path_to_model_checkpoints)

    # run_gate_motion_from_saved_results(experiment_yaml_file)
#    run_leaf_gate_plots(experiment_yaml_file)
# run_single_iter_pos_and_neg_gates_plot(experiment_yaml_file)
# make_dev_data_plots(hparams)
