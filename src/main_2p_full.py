import csv
import os
import pickle

import yaml

from train import *
from utils.bayes_gate import ModelForest
from utils.input import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.1,
    'positive_box_penalty': 0,
    'corner_penalty': 1.0,
    'gate_size_penalty': 0.5,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': False,
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
    'n_run': 100,
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

    cll_2p_full_input = Cll2pFullInput(hparams)

    for random_state in range(random_state_start, hparams['n_run']):
        hparams['random_state'] = random_state
        cll_2p_full_input.split(random_state)

        model_forest = ModelForest(cll_2p_full_input.reference_tree,
                                   logistic_k=hparams['logistic_k'],
                                   regularisation_penalty=hparams['regularization_penalty'],
                                   negative_box_penalty=hparams['negative_box_penalty'],
                                   positive_box_penalty=hparams['positive_box_penalty'],
                                   corner_penalty=hparams['corner_penalty'],
                                   gate_size_penalty=hparams['gate_size_penalty'],
                                   init_tree_list=cll_2p_full_input.init_tree,
                                   loss_type=hparams['loss_type'],
                                   gate_size_default=hparams['gate_size_default'])

        dafi_forest = ModelForest(cll_2p_full_input.reference_tree,
                                  logistic_k=hparams['logistic_k_dafi'],
                                  regularisation_penalty=hparams['regularization_penalty'],
                                  negative_box_penalty=hparams['negative_box_penalty'],
                                  positive_box_penalty=hparams['positive_box_penalty'],
                                  corner_penalty=hparams['corner_penalty'],
                                  gate_size_penalty=hparams['gate_size_penalty'],
                                  init_tree_list=[None] * 2,
                                  loss_type=hparams['loss_type'],
                                  gate_size_default=hparams['gate_size_default'])

        # dafi_forest = run_train_dafi(dafi_forest, hparams, cll_2p_full_input)
        model_forest, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
            run_train_model(model_forest, hparams, cll_2p_full_input, model_checkpoint=model_checkpoint)
        output_metric_dict = run_output(
            model_forest, dafi_forest, hparams, cll_2p_full_input, train_tracker, eval_tracker, run_time)

        with open('../output/%s/model_forest_%d.pkl' % (hparams['experiment_name'], random_state), 'wb') as pfile:
            pickle.dump(model_forest, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open('../output/%s/dafi_forest_%d.pkl' % (hparams['experiment_name'], random_state), 'wb') as pfile:
            pickle.dump(dafi_forest, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        print("regression parameters:", model_forest.linear.weight.detach(), model_forest.linear.bias.detach())
        print("regression parameters:", dafi_forest.linear.weight.detach(), dafi_forest.linear.bias.detach())
        run_write_prediction(model_forest, dafi_forest, cll_2p_full_input, hparams)
        if hparams['plot_figures']:
            run_gate_motion_2p(hparams, cll_2p_full_input, model_checkpoint_dict)
            run_plot_metric(hparams, train_tracker, eval_tracker, dafi_forest, cll_2p_full_input, output_metric_dict)


#
if __name__ == '__main__':
# run(sys.argv[1], int(sys.argv[2]))
# run_multiple_panel("../configs/cll_4d_2p_default.yaml", 0, True)
    # run_multiple_panel("../configs/test.yaml", 0, True)
    # run_multiple_panel("../configs/cll_4d_2p_default.yaml", 50, True)
    run_multiple_panel("../configs/test.yaml", 0, True)
