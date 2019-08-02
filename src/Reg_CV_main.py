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
from sklearn.model_selection import KFold

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
}

DEV_DATA_PATHS = {
    'X': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',  
    'Y': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
    }

def run_single_panel(hparams, random_state_start=0, model_checkpoint=True):
    torch.cuda.set_device(1)
    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])
    default_exp_name = hparams['experiment_name']
    random_state = random_state_start
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    folds = KFold(n_splits=hparams['n_folds_for_Reg_CV'])
    
    # doesn't split at all here since hparams[test_size] = 0 in matching yaml file
    for init_reg_penalty, negative_box_penalty in zip(hparams['init_reg_grid'], hparams['negative_box_grid']):
        print('Init reg: %.4f, Neg reg: %.4f' %(init_reg_penalty, negative_box_penalty))
        hparams['negative_box_penalty'] = negative_box_penalty
        hparams['init_reg_penalty'] = init_reg_penalty
        default_exp_name_for_outer_loop = default_exp_name + '_neg_box_=%.4f_init_reg=%.4f' %(negative_box_penalty, init_reg_penalty)
        input_no_split = Cll8d1pInput(hparams)
        for fold_idx, (tr_idxs, te_idxs) in enumerate(folds.split(input_no_split.x_list)): 
            start_time = time.time()
            cll_1p_full_input = Cll8d1pInput(hparams, split_fold_idxs=[tr_idxs, te_idxs])
            hparams['experiment_name'] = default_exp_name_for_outer_loop + '_fold%d' %fold_idx
            if not os.path.exists('../output/%s' % hparams['experiment_name']):
                os.makedirs('../output/%s' % hparams['experiment_name'])
            savedir = '../output/%s/' %hparams['experiment_name']
            if not(os.path.exists(savedir)):
                os.mkdir(savedir)
            

            dafi_tree = ModelTree(cll_1p_full_input.reference_tree,
                                  logistic_k=hparams['logistic_k_dafi'],
                                  negative_box_penalty=hparams['negative_box_penalty'],
                                  positive_box_penalty=hparams['positive_box_penalty'],
                                  corner_penalty=hparams['corner_penalty'],
                                  gate_size_penalty=hparams['gate_size_penalty'],
                                  init_tree=None,
                                  loss_type=hparams['loss_type'],
                                  gate_size_default=hparams['gate_size_default'])

            model_tree = ModelTree(cll_1p_full_input.reference_tree,
                                   logistic_k=hparams['logistic_k'],
                                   regularisation_penalty=hparams['regularization_penalty'],
                                   negative_box_penalty=hparams['negative_box_penalty'],
                                   positive_box_penalty=hparams['positive_box_penalty'],
                                   corner_penalty=hparams['corner_penalty'],
                                   init_reg_penalty=hparams['init_reg_penalty'],
                                   feature_diff_penalty=hparams['feature_diff_penalty'],
                                   gate_size_penalty=hparams['gate_size_penalty'],
                                   init_tree=cll_1p_full_input.init_tree,
                                   loss_type=hparams['loss_type'],
                                   gate_size_default=hparams['gate_size_default'],
                                   neg_proportion_default = hparams['neg_proportion_default'],
                                   node_type = hparams['node_type']
                                   )

            if torch.cuda.is_available():
                model_tree.cuda()
                dafi_tree.cuda()

            model_tree, train_tracker_m, eval_tracker_m, run_time, model_checkpoint_dict = \
                run_train_full_batch_logreg_to_conv(hparams, cll_1p_full_input, model_tree, model_checkpoint=model_checkpoint)
            dafi_tree, train_tracker_d, eval_tracker_d = run_train_dafi_logreg_to_conv(dafi_tree, hparams, cll_1p_full_input)



            trackers_dict = {
                    'tracker_train_m': train_tracker_m,
                    'tracker_eval_m': eval_tracker_m,
                    'tracker_train_d': train_tracker_d,
                    'tracker_eval_d': eval_tracker_d
            }
            run_write_full_output_for_CV(
                model_tree,
                dafi_tree,
                cll_1p_full_input,
                trackers_dict,
                hparams,
                model_checkpoint_dict
            )





if __name__ == '__main__':
    yaml_filename = '../configs/Reg_CV.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    print(hparams)
    run_single_panel(hparams, 1, True)
