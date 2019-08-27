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
}

DEV_DATA_PATHS = {
    'X': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',  
    'Y': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
    }

#OUT_OF_SAMPLE_TEST_DATA_PATHS = {
#        'X': '../data/cll/8d_FINAL/x_test_1p.pkl',
#        'Y': '../data/cll/8d_FINAL/y_test_1p.pkl'
#    }

def run_single_panel(hparams, random_state_start=0, model_checkpoint=True):
    torch.cuda.set_device(hparams['device'])
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
        cll_1p_full_input = Cll8d1pInput(hparams, random_state=random_state)
        if hparams['augment_training_with_dev_data']:
            cll_1p_full_input_augmented = Cll8d1pInput(hparams, random_state=random_state, augment_data_paths=DEV_DATA_PATHS) 
        #some_eval_data = [data.detach().cpu().numpy() for data in cll_1p_full_input.x_eval[10:]]
        #some_eval_data = np.concatenate(some_eval_data)[0:100000]
        #plt.scatter(some_eval_data[:, 0], some_eval_data[:, 3], s=.1)
        #plt.savefig('meow.png')
        #plt.clf()
        #plt.scatter(some_eval_data[:, 1], some_eval_data[:, 2], s=.1)
        #plt.savefig('meow2.png')
        #print(cll_1p_full_input.y_train)
        #print(last_y_label)
        #last_y_label = cll_1p_full_input.y_train
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
        if hparams['augment_training_with_dev_data']:
            model_tree_aug = ModelTree(cll_1p_full_input_augmented.reference_tree,
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
        
        dafi_tree = ModelTree(cll_1p_full_input.reference_tree,
                              logistic_k=hparams['logistic_k_dafi'],
                              negative_box_penalty=hparams['negative_box_penalty'],
                              positive_box_penalty=hparams['positive_box_penalty'],
                              corner_penalty=hparams['corner_penalty'],
                              gate_size_penalty=hparams['gate_size_penalty'],
                              init_tree=None,
                              loss_type=hparams['loss_type'],
                              gate_size_default=hparams['gate_size_default'])

        # dafi_tree = run_train_dafi(dafi_tree, hparams, cll_1p_full_input)

        if torch.cuda.is_available():
            model_tree.cuda()
            if hparams['augment_training_with_dev_data']:
                model_tree_aug.cuda()
            dafi_tree.cuda()


        print(cll_1p_full_input.x_eval)
        model_tree = run_train_only_logistic_regression(model_tree, cll_1p_full_input.x_train, cll_1p_full_input.y_train, hparams['learning_rate_classifier'], verbose=False)
        model_out = model_tree(cll_1p_full_input.x_eval, cll_1p_full_input.y_eval)
        print(model_out)
        print(cll_1p_full_input.y_eval)
        predictions = model_out['y_pred'].cpu().detach().numpy() >= 0.5 
        print('Acc is %.4f' %(np.sum(predictions == cll_1p_full_input.y_eval.cpu().detach().numpy())/predictions.shape[0]))
        


        print('The full loop for random_state %d took %d seconds' %(random_state, time.time() - start_time))


if __name__ == '__main__':
    #yaml_filename = '../configs/Final_Model.yaml'
    #yaml_filename = '../configs/OOS_Final_Model.yaml'
    yaml_filename  = '../configs/heuristic_baseline.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    print(hparams)
    run_single_panel(hparams, 0, True)
