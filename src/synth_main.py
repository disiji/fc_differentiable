import warnings
import csv
import os

import yaml

from train import *
from utils.bayes_gate import ModelTree
from utils.input import *
#from utils.utils_plot import plot_pos_and_neg_gate_motion
from utils.utils_plot import *
from full_output_for_CV import *
import time
import matplotlib.pyplot as plt

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
    'plot_params':{
        'figsize': [10, 10],
        'marker_size': .01,
    },
    'use_out_of_sample_eval_data': False,
    'dictionary_is_broken': True
}

DEV_DATA_PATHS = {
    'X': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',  
    'Y': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
    }


def run_single_panel(hparams, random_state_start=0, model_checkpoint=True):
    torch.cuda.set_device(hparams['device'])
    warnings.filterwarnings("ignore")
    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])
    default_exp_name = hparams['experiment_name']
    last_y_label = []
    for random_state in range(random_state_start, hparams['n_run']):
        start_time = time.time()
        hparams['random_state'] = random_state
        hparams['experiment_name'] = default_exp_name + '_seed%d' %random_state
        if not os.path.exists('../output/%s' % hparams['experiment_name']):
            os.makedirs('../output/%s' % hparams['experiment_name'])
        savedir = '../output/%s/' %hparams['experiment_name']
        if not(os.path.exists(savedir)):
            os.mkdir(savedir)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        synth_input = SynthInput(hparams, random_state=random_state)

       
        model_tree = ModelTree(synth_input.reference_tree,
                               logistic_k=hparams['logistic_k'],
                               regularisation_penalty=hparams['regularization_penalty'],
                               negative_box_penalty=hparams['negative_box_penalty'],
                               positive_box_penalty=hparams['positive_box_penalty'],
                               corner_penalty=hparams['corner_penalty'],
                               init_reg_penalty=hparams['init_reg_penalty'],
                               feature_diff_penalty=hparams['feature_diff_penalty'],
                               gate_size_penalty=hparams['gate_size_penalty'],
                               init_tree=synth_input.init_tree,
                               loss_type=hparams['loss_type'],
                               gate_size_default=hparams['gate_size_default'],
                               neg_proportion_default = hparams['neg_proportion_default'],
                               node_type = hparams['node_type']
                               )
        model_init_savepath = '../output/%s/model_init_seed%d.pkl' %(hparams['experiment_name'], random_state)
        with open(model_init_savepath, 'wb') as f:
            pickle.dump(model_tree, f)

        if torch.cuda.is_available():
            model_tree.cuda()

        print('Training model with just the tr split of the first 1000 samples')
        model_tree, train_tracker_m, eval_tracker_m, run_time, model_checkpoint_dict = \
            run_train_full_batch_logreg_to_conv(hparams, synth_input, model_tree, model_checkpoint=model_checkpoint)


        model_final_savepath = '../output/%s/model_final_seed%d.pkl' %(hparams['experiment_name'], random_state)
        with open(model_final_savepath, 'wb') as f:
            pickle.dump(model_tree, f)

        print('The full loop for random_state %d took %d seconds' %(random_state, time.time() - start_time))


if __name__ == '__main__':
    #yaml_filename = '../configs/Final_Model.yaml'
    yaml_filename = '../configs/default_synth.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    print(hparams)
    run_single_panel(hparams, 0, True)
