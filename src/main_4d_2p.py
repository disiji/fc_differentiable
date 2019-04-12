import csv
import os
import yaml
from train import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'emptyness_penalty': 1,
    'gate_size_penalty': 1,
    'gate_size_default': 1. / 4,
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 20,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.1,
    'batch_size': 85,
    'n_epoch': 500,
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 100,
}


def run_multiple_panel(yaml_filename, random_state_start=0):
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
                               emptyness_penalty=hparams['emptyness_penalty'],
                               gate_size_penalty=hparams['gate_size_penalty'],
                               init_tree=cll_4d_2p_input.init_tree,
                               loss_type=hparams['loss_type'],
                               gate_size_default=hparams['gate_size_default'])

        dafi_forest = ModelForest(cll_4d_2p_input.reference_tree,
                              logistic_k=hparams['logistic_k_dafi'],
                              regularisation_penalty=hparams['regularization_penalty'],
                              emptyness_penalty=hparams['emptyness_penalty'],
                              gate_size_penalty=hparams['gate_size_penalty'],
                              init_tree=None,
                              loss_type=hparams['loss_type'],
                              gate_size_default=hparams['gate_size_default'])

        dafi_forest = run_train_dafi(dafi_forest, hparams, cll_4d_2p_input)
        model_forest, train_tracker, eval_tracker, run_time = run_train_model(model_forest, hparams, cll_4d_2p_input)
        output_metric_dict = run_output(
            model_forest, dafi_forest, hparams, cll_4d_2p_input, train_tracker, eval_tracker, run_time)

        # only plot once
        if not os.path.isfile('../output/%s/metrics.png' % hparams['experiment_name']):
            run_plot_metric(hparams, train_tracker, eval_tracker, dafi_forest, cll_4d_2p_input, output_metric_dict)
            # run_plot_gates(hparams, train_tracker, eval_tracker, model_forest, dafi_forest, cll_4d_2p_input)


if __name__ == '__main__':
    #run(sys.argv[1], int(sys.argv[2]))
    run_multiple_panel("../configs/4d_2p_default.yaml", 0)
