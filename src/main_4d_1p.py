import csv
import os

import yaml

from train import *
from utils.bayes_gate import ModelTree
from utils.input import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.1,
    'positive_box_penalty': 0.0,
    'corner_penalty': 1.0,
    'gate_size_penalty': 1,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 20,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.05,
    'batch_size': 10,
    'n_epoch': 500,
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 10,
    'train_alternate': True
}


def run_single_panel(yaml_filename, random_state_start=0, model_checkpoint=True):
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    if hparams['train_alternate']:
        hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
                hparams['n_mini_batch_update_gates'] - 1)
    else:
        hparams['n_epoch_dafi'] = hparams['n_epoch']
    print(hparams)

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])

    cll_4d_input = Cll4dInput(hparams)

    for random_state in range(random_state_start, hparams['n_run']):
        hparams['random_state'] = random_state
        cll_4d_input.split(random_state)

        model_tree = ModelTree(cll_4d_input.reference_tree,
                               logistic_k=hparams['logistic_k'],
                               regularisation_penalty=hparams['regularization_penalty'],
                               negative_box_penalty=hparams['negative_box_penalty'],
                               positive_box_penalty=hparams['positive_box_penalty'],
                               corner_penalty=hparams['corner_penalty'],
                               gate_size_penalty=hparams['gate_size_penalty'],
                               init_tree=cll_4d_input.init_tree,
                               loss_type=hparams['loss_type'],
                               gate_size_default=hparams['gate_size_default'])

        dafi_tree = ModelTree(cll_4d_input.reference_tree,
                              logistic_k=hparams['logistic_k_dafi'],
                              negative_box_penalty=hparams['negative_box_penalty'],
                              positive_box_penalty=hparams['positive_box_penalty'],
                              corner_penalty=hparams['corner_penalty'],
                              gate_size_penalty=hparams['gate_size_penalty'],
                              init_tree=None,
                              loss_type=hparams['loss_type'],
                              gate_size_default=hparams['gate_size_default'])

        # dafi_tree = run_train_dafi(dafi_tree, hparams, cll_4d_input)
        model_tree, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
            run_train_model(model_tree, hparams, cll_4d_input, model_checkpoint=model_checkpoint)
        output_metric_dict = run_output(
            model_tree, dafi_tree, hparams, cll_4d_input, train_tracker, eval_tracker, run_time)

        # only plot once
        # if not os.path.isfile('../output/%s/metrics.png' % hparams['experiment_name']) and plot_and_write_output:
        run_plot_metric(hparams, train_tracker, eval_tracker, dafi_tree, cll_4d_input, output_metric_dict)
        run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, cll_4d_input)
        run_write_prediction(model_tree, dafi_tree, cll_4d_input, hparams)
        # run_gate_motion(hparams, cll_4d_input, model_checkpoint_dict, train_tracker)
        print(model_checkpoint)
        print(model_checkpoint_dict)
        run_gate_motion_in_one_figure(hparams, cll_4d_input, model_checkpoint_dict, train_tracker)
        # model_checkpoint = False


if __name__ == '__main__':
    # run_single_panel(sys.argv[1], int(sys.argv[2]), True)
    run_single_panel("../configs/test.yaml", 0, True)
