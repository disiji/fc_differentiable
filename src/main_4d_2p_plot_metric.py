import csv

from train import *
from utils.bayes_gate import ModelForest
from utils.utils_plot import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.1,
    'positive_box_penalty': 0,
    'corner_penalty': 1.0,
    'gate_size_penalty': 0.5,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 20,
    'n_mini_batch_update_gates': 20,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.05,
    'batch_size': 10,
    'n_epoch': 500,
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 50,
    'train_alternate': True,
    'plot_figures': False
}


def run_multiple_panel(yaml_filename, random_state_start=0, model_checkpoint=True):
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
            hparams['n_mini_batch_update_gates'] - 1)
    print(hparams)

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])

    cll_4d_2p_input = Cll4d2pInput(hparams)

    for random_state in range(random_state_start, hparams['n_run']):
        hparams['random_state'] = random_state
        cll_4d_2p_input.split(random_state)

        model_forest = ModelForest(cll_4d_2p_input.reference_tree,
                                   logistic_k=hparams['logistic_k'],
                                   regularisation_penalty=hparams['regularization_penalty'],
                                   negative_box_penalty=hparams['negative_box_penalty'],
                                   positive_box_penalty=hparams['positive_box_penalty'],
                                   corner_penalty=hparams['corner_penalty'],
                                   gate_size_penalty=hparams['gate_size_penalty'],
                                   init_tree_list=cll_4d_2p_input.init_tree,
                                   loss_type=hparams['loss_type'],
                                   gate_size_default=hparams['gate_size_default'])

        model_forest, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
            run_train_model(model_forest, hparams, cll_4d_2p_input, model_checkpoint=model_checkpoint)

        with open('../output/%s/model_forest_%d.pkl' % (hparams['experiment_name'], random_state), 'wb') as pfile:
            pickle.dump(model_forest, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        print("regression parameters:", model_forest.linear.weight.detach(), model_forest.linear.bias.detach())

        with open('../output/%s/train_tracker_%d.pkl' % (hparams['experiment_name'], random_state), 'wb') as f:
            pickle.dump(train_tracker, f, pickle.HIGHEST_PROTOCOL)
        with open('../output/%s/eval_tracker_%d.pkl' % (hparams['experiment_name'], random_state), 'wb') as f:
            pickle.dump(eval_tracker, f, pickle.HIGHEST_PROTOCOL)

    return hparams


if __name__ == '__main__':

    yaml_filename = "../configs/cll_4d_2p_metric_plot.yaml"
    run_multiple_panel(yaml_filename, 10, True)

    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)

    n_runs = 50
    train_trackers = []
    eval_trackers = []

    for random_state in range(n_runs):
        with open('../output/%s/train_tracker_%d.pkl' % (hparams['experiment_name'], random_state), 'rb') as f:
            train_trackers.append(pickle.load(f))
        with open('../output/%s/eval_tracker_%d.pkl' % (hparams['experiment_name'], random_state), 'rb') as f:
            eval_trackers.append(pickle.load(f))

    x_range = [i * hparams['n_epoch_eval'] for i in range(hparams['n_epoch'] // hparams['n_epoch_eval'])]
    filename_acc = "../output/%s/acc_err.pdf" % (hparams['experiment_name'])
    plot_accuracy_error_bar(x_range, train_trackers, eval_trackers, filename_acc)
    filename_auc = "../output/%s/auc_err.pdf" % (hparams['experiment_name'])
    plot_auc_error_bar(x_range, train_trackers, eval_trackers, filename_auc)
    filename_logloss = "../output/%s/logloss_err.pdf" % (hparams['experiment_name'])
    plot_logloss_error_bar(x_range, train_trackers, eval_trackers, filename_logloss)
    filename_reg = "../output/%s/reg_err.pdf" % (hparams['experiment_name'])
    plot_reg_error_bar(x_range, train_trackers, eval_trackers, filename_reg)
