import numpy as np
from sklearn.metrics import roc_auc_score
import os
import yaml
import torch
from utils.input import *
from recreate_feats_and_preds_from_saved_model import get_probs_and_feats_single_model
from scipy.stats import kendalltau
OUTPUT_DIR = '../output'
CV_EXP_PREFIX = 'CV_neg=0.001_diff=0.001'
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
    'out_of_sample_eval_data': None
}

SEEDS = np.concatenate([np.arange(51, 72) + 1, np.arange(29) + 1], axis=0)

DEV_DATA_PATHS = {
    'X': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',  
    'Y': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
    }
def run_write_save_accuracy_diagnostics_one_file(cv_exp_prefix, seeds):
    acc_results = -1 * np.ones([seeds.shape[0], 9])
    for s, seed in enumerate(seeds):
        m_acc, m_e_acc, d_acc, d_e_acc = get_accs(os.path.join(OUTPUT_DIR, cv_exp_prefix), seed)
        m_aug_acc, m_aug_e_acc, d_aug_acc, d_aug_e_acc = get_accs(os.path.join(OUTPUT_DIR, cv_exp_prefix), seed, augmented=True)

        acc_results[s] = [seed, m_acc, m_e_acc, d_acc, d_e_acc, m_aug_acc, m_aug_e_acc, d_aug_acc, d_aug_e_acc]
    header = 'Seed, Model, Model_Expert_Thresh, Dafi, Dafi_Expert_Thresh, Model_aug, Model_aug_Expert_Thresh, Dafi_aug, Dafi_aug_Expert_Thresh'
    np.savetxt(os.path.join(OUTPUT_DIR, cv_exp_prefix, 'te_acc_per_seed.csv'), acc_results, header=header, delimiter=',', fmt='%.4f')

def run_write_kendalls_tau_and_auc(cv_exp_prefix, seeds, hparams):
    results = -1 * np.ones([seeds.shape[0], 7])
    for s, seed in enumerate(seeds):
        input = Cll8d1pInput(hparams, random_state=seed)
        data_dict = {
            'x_list': input.x_eval,
            'y_list': input.y_eval
        }
        print(input.x_eval[0].get_device())
        path = os.path.join(OUTPUT_DIR, cv_exp_prefix)
        path = path + '_seed%d' %seed
        path_augment = path + '_augmented_with_dev'

        model = load_model(path)
        device = model.root.center1_param.get_device()
        data_dict['x_list'] = [x.cuda(device) for x in data_dict['x_list']]
        data_dict['y_list'] = input.y_eval.cuda(device)
        model_probs, model_feats = get_probs_and_feats_single_model(model, data_dict, is_Dafi=False, device=device)

        model_aug = load_model(path_augment)
        model_probs_aug,  model_feats_aug = get_probs_and_feats_single_model(model_aug, data_dict, is_Dafi=False, device=device)
        
        dafi = load_dafi(path)
        dafi_probs, dafi_feats = get_probs_and_feats_single_model(dafi, data_dict, is_Dafi=True, device=device)
        dafi_aug = load_dafi(path_augment)
        dafi_probs_aug, dafi_feats_aug = get_probs_and_feats_single_model(dafi_aug, data_dict, is_Dafi=True, device=device)



        ktau = get_kendalls_tau(dafi_probs, model_probs)
        ktau_aug = get_kendalls_tau(dafi_probs_aug, model_probs_aug)

        auc_model = get_auc(model_probs, input.y_eval.cpu().detach().numpy())
        auc_model_aug = get_auc(model_probs_aug, input.y_eval.cpu().detach().numpy())

        auc_dafi = get_auc(dafi_probs, input.y_eval.cpu().detach().numpy())
        auc_dafi_aug = get_auc(dafi_probs_aug, input.y_eval.cpu().detach().numpy())

        results[s] = [seed, ktau, ktau_aug, auc_model, auc_model_aug, auc_dafi, auc_dafi_aug]
    header = 'seed, ktau, ktau_aug, auc_model, auc_model_aug, auc_dafi, auc_dafi_aug'
    savepath = os.path.join(OUTPUT_DIR, cv_exp_prefix, 'ktau_and_auc_per_seed.csv')
    np.savetxt(savepath, results, fmt='%.4f', header=header, delimiter=',')

def get_accs(path_prefix, seed, augmented=False):
    results_model, results_dafi = get_results_model_dafi(path_prefix, seed, augmented=augmented)

    return results_model[-1], results_model[-3], results_dafi[-1], results_dafi[-3]

def get_kendalls_tau(dafi_probs, model_probs):
    rankings_dafi = np.argsort(-dafi_probs)
    rankings_model = np.argsort(-model_probs)
    print(rankings_dafi, rankings_model)
    tau, p_value = kendalltau(rankings_dafi, rankings_model)

    print('p value: %.4f' %p_value)
    return p_value

def get_auc(model_preds, true_labels):
    auc = roc_auc_score(true_labels, model_preds)
    print('auc, %.4f' %auc)
    return auc



def get_results_model_dafi(path_prefix, seed, augmented=False):
    path_to_diag_dir = path_prefix + '_seed%d' %seed
    if augmented:
        path_to_diag_dir += '_augmented_with_dev'
    path_to_diag_dir = os.path.join(path_to_diag_dir, 'Diagnostics')
    path_to_model_diag = os.path.join(path_to_diag_dir, 'te_diagnostics_model.csv')
    path_to_dafi_diag = os.path.join(path_to_diag_dir, 'te_diagnostics_dafi.csv')

    results_model = np.genfromtxt(path_to_model_diag, delimiter=',')
    results_dafi = np.genfromtxt(path_to_dafi_diag, delimiter=',')
    return results_model, results_dafi

def load_model(path):
    with open(os.path.join(path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

def load_dafi(path):
    with open(os.path.join(path, 'dafi_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model
if __name__ == '__main__':
    yaml_filename = '../configs/CV_runs.yaml'
    #yaml_filename = '../configs/testing_overlaps.yaml'
    #yaml_filename = '../configs/CV_prefiltering.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    
    run_write_save_accuracy_diagnostics_one_file(CV_EXP_PREFIX, SEEDS)
    run_write_kendalls_tau_and_auc(CV_EXP_PREFIX, SEEDS, hparams)




