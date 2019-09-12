import warnings
import copy
import csv
import os

import yaml

from train import *
from utils.bayes_gate import ModelTreeBothPanels
from utils.input import *
#from utils.utils_plot import plot_pos_and_neg_gate_motion
from utils.utils_plot import *
from full_output_for_CV import *
import time
import matplotlib.pyplot as plt

#SEEDS = np.concatenate([np.arange(67, 72), np.arange(29) + 1, np.arange(51, 67)], axis=0) 
SEEDS = np.concatenate([np.arange(73, 74), np.arange(29) + 1, np.arange(51, 72)], axis=0)
#SEEDS = np.concatenate([np.arange(51, 72), np.arange(29) + 1], axis=0)
default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 10000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.0,
    'init_reg_penalty': 0.0,
    'neg_proportion_default': 0.0001,
    'positive_box_penalty': 0.0,
    'corner_penalty': .0,
    'feature_diff_penalty': 0.,
    'gate_size_penalty': .0,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'node_type': 'square',
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
    'init_type': 'heuristic_init',
    'corner_init_deterministic_size': .75,
    'train_alternate': False,
    'run_logistic_to_convergence': True,
    'augment_training_with_dev_data': False,
    'filter_uncertain_samples': False,
    'use_model_CV_seeds': False,
    'device': 0, 
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
    'heuristic_init': {
        'num_gridcells_per_axis': 4,
        'use_greedy_filtering': False,
        'consider_all_gates': False
    },
    'use_out_of_sample_eval_data': False,
    'dictionary_is_broken': False
}


DEV_DATA_PATHS = {
    'X': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',  
    'Y': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
    }

#OUT_OF_SAMPLE_TEST_DATA_PATHS = {
#        'X': '../data/cll/8d_FINAL/x_test_1p.pkl',
#        'Y': '../data/cll/8d_FINAL/y_test_1p.pkl'
#    }

def run_both_panels(hparams, random_state_start=0, model_checkpoint=True):
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(hparams['device'])
    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])
    default_exp_name = hparams['experiment_name']
    last_y_label = []
    iterate_over = range(random_state_start, hparams['n_run']) if not hparams['use_model_CV_seeds'] else SEEDS
    print(iterate_over)
    for random_state in iterate_over:
        print('Seed %d' %random_state)
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
        cll_both_panels_input = CllBothPanelsInput(hparams, random_state=random_state)
        model_tree = ModelTreeBothPanels(hparams, cll_both_panels_input.features, cll_both_panels_input.reference_tree, init_tree=cll_both_panels_input.init_tree)
        dafi_tree = ModelTreeBothPanels(hparams, cll_both_panels_input.features, cll_both_panels_input.reference_tree, init_tree=None)

        if torch.cuda.is_available():
            model_tree.cuda()
            dafi_tree.cuda()

        print('Training Dafi')
        dafi_tree, train_tracker_d, eval_tracker_d = run_train_dafi_logreg_to_conv(dafi_tree, hparams, cll_both_panels_input)

        print('Training model with just the tr split of validation data')
        model_tree, train_tracker_m, eval_tracker_m, run_time, model_checkpoint_dict = \
            run_train_full_batch_logreg_to_conv(hparams, cll_both_panels_input, model_tree, model_checkpoint=model_checkpoint)


        experiment_name = hparams['experiment_name']
        with open('../output/%s/model_checkpoints.pkl' %(experiment_name), 'wb') as f:
                pickle.dump(model_checkpoint_dict, f)
        with open('../output/%s/dafi_model.pkl' %(experiment_name), 'wb') as f:
                pickle.dump(dafi_tree, f)
        with open('../output/%s/tracker_train_m.pkl' %(experiment_name), 'wb') as f:
                pickle.dump(train_tracker_m, f)
        with open('../output/%s/tracker_eval_m.pkl' %(experiment_name), 'wb') as f:
                pickle.dump(eval_tracker_m, f)
        with open('../output/%s/tracker_train_d.pkl' %(experiment_name), 'wb') as f:
                pickle.dump(train_tracker_d, f)
        with open('../output/%s/tracker_eval_d.pkl' %(experiment_name), 'wb') as f:
                pickle.dump(eval_tracker_d, f)

        print(model_tree)
        print('Testing Accuracy Model: %.4f' %eval_tracker_m.acc[-1])
        print('Testing Accuracy Dafi:', eval_tracker_d.acc[-1])
        #write_model_diagnostics(model_tree, dafi_tree, cll_both_panels_input, train_tracker_m, eval_tracker_m, train_tracker_d, eval_tracker_d, experiment_name)
        print('The full loop for random_state %d took %d seconds' %(random_state, time.time() - start_time))
        

if __name__ == '__main__':
    yaml_filename = '../configs/default_2p.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    print(hparams)
    run_both_panels(hparams, 0, True)
