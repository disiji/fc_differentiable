import torch
import numpy
import pickle
import yaml
from utils.input import *

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

DEVICE = 1
PATHS_TO_DEV= \
        {
            'x_list': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',
            'y_list': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
        }
PATHS_TO_VAL = \
        {
            'x_list': '../data/cll/8d_FINAL/x_val_8d_1p.pkl',
            'y_list': '../data/cll/8d_FINAL/y_val_8d_1p.pkl'
        }
MODEL_PATH = '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/model.pkl' 
DAFI_PATH = '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/dafi_model.pkl'

PATHS_TO_OOS_TEST_DATA = \
        {
            'x_list': '../data/cll/8d_FINAL/x_test_1p.pkl',
            'y_list': '../data/cll/8d_FINAL/y_test_1p.pkl'
        }


def run_save_diagnostics_oos(model_path, dafi_path, data_dict, savepath, hparams):
    torch.device(DEVICE)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        print(model.root.center1_param.get_device())
    with open(dafi_path, 'rb') as f:
        dafi = pickle.load(f)
    loaded_data_dict = {}
    input = Cll8d1pInput(hparams, random_state=0) #only one run for oos data
    loaded_data_dict['x_list'] = [x.cuda(DEVICE) for x in input.x_eval]
    loaded_data_dict['y_list'] = input.y_eval.cuda(DEVICE)
#    with open(data_dict['x_list'], 'rb') as f:
#        loaded_data_dict['x_list'] = [torch.tensor(x, dtype=torch.float32).cuda(DEVICE) for x in pickle.load(f)]
#        print(loaded_data_dict['x_list'][0].get_device())
#    with open(data_dict['y_list'], 'rb') as f:
#        loaded_data_dict['y_list'] = torch.tensor(pickle.load(f), dtype=torch.double).cuda(DEVICE)

    save_diagnostics(model, dafi, loaded_data_dict, savepath, device=hparams['device'])

def save_diagnostics(model, dafi, data_dict, savepath, device=1):

    torch.device(device)
    dafi_probs_logreg, dafi_feats = get_probs_and_feats_single_model(dafi, data_dict, is_Dafi=True, device=device)
    model_probs_logreg, model_feats = get_probs_and_feats_single_model(model, data_dict, is_Dafi=False, device=device)


    thresh_labels_model = model_feats >= 0.00015 
    thresh_labels_dafi = dafi_feats >= 0.00015

    labels_logreg_model = model_probs_logreg >= 0.5
    labels_logreg_dafi = dafi_probs_logreg >= 0.5

    sample_ids = np.arange(len(data_dict['x_list']))
    labels = data_dict['y_list'].cpu().detach().numpy()

    header = \
        'sample_ids, labels,\
        model_probs_logreg, thresh_labels_model, labels_logreg_model,\
        dafi_probs_logreg, thresh_labels_dafi, labels_logreg_dafi,\
        model_feats, dafi_feats'
    
    results = \
    [
            sample_ids, labels,
            model_probs_logreg, thresh_labels_model, labels_logreg_model,
            dafi_probs_logreg, thresh_labels_dafi, labels_logreg_dafi,
            model_feats, dafi_feats
    ]
    print([result.shape for result in results])
    results_per_sample = \
        np.concatenate([
            sample_ids[:, np.newaxis], labels[:, np.newaxis],
            model_probs_logreg[:, np.newaxis], thresh_labels_model, labels_logreg_model[:, np.newaxis],
            dafi_probs_logreg[:, np.newaxis], thresh_labels_dafi, labels_logreg_dafi[:, np.newaxis],
            model_feats, dafi_feats

        ], axis=1)

    np.savetxt(savepath, results_per_sample, header=header, fmt='%.6f')
    return results_per_sample


    


def get_all_feats(model, dafi, dev_data_dict, val_data_dict):
    model_feats = get_feats_single_model(model, dev_data_dict['x_list'], val_data_dict['x_list'], device=device)
    dafi_feats = get_feats_single_model(dafi, dev_data_dict['x_list'], val_data_dict['x_list'], hard_feats=True, device=device)

    return model_feats, dafi_feats

def get_feats_single_model(model, dev_x_list, val_x_list, hard_feats=False, device=1):
    if hard_feats:
        dev_feats = model.get_hard_proportions_4chain(dev_x_list, device=device)
        val_feats = model.get_hard_proportions_4chain(val_x_list, device=device)
    else:
        dev_feats = model.forward_4chain(dev_x_list, device=device)['leaf_probs'].cpu().detach().numpy()
        val_feats = model.forward_4chain(val_x_list, device=device)['leaf_probs'].cpu().detach().numpy()
    return np.concatenate([dev_feats, val_feats])

def get_probs_and_feats_single_model(model, data_dict, is_Dafi=False, device=1):
    torch.device(device)
    if is_Dafi:
        out = model.forward_4chain(data_dict['x_list'], data_dict['y_list'], use_hard_proportions=True, device=device)
    else:
        out = model.forward_4chain(data_dict['x_list'], data_dict['y_list'], device=device)


    return out['y_pred'].cpu().detach().numpy(),\
           out['leaf_probs'].cpu().detach().numpy()

def run_write_all_feats(path_to_model, path_to_dafi_model, hparams, random_state=3):
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)
    with open(path_to_dafi_model, 'rb') as f:
        dafi = pickle.load(f)

    input_val = Cll8d1pInput(hparams, random_state=random_state)
    hparams['data']['features_path'] = PATHS_TO_DEV['x_list']
    hparams['data']['labels_path'] = PATHS_TO_DEV['y_list']
    input_dev = Cll8d1pInput(hparams, random_state=random_state)
    dev_data_dict = {'x_list': input_dev.x_list, 'y_list': input_dev.y_list}
    val_data_dict = {'x_list': input_val.x_list, 'y_list': input_val.y_list}

    model_feats, dafi_feats = get_all_feats(model, dafi, dev_data_dict, val_data_dict, device=hparams['device'])
    labels = np.concatenate([input_dev.y_list.cpu().detach().numpy(), 
                        input_val.y_list.cpu().detach().numpy()])
    model_feats = model_feats.reshape([-1, 1])
    dafi_feats = dafi_feats.reshape([-1, 1])
    labels = labels.reshape([-1, 1])

    all_feats_and_labels = np.concatenate([model_feats, dafi_feats, labels], axis=1)

    header = 'model_feats, dafi_feats, labels'

    np.savetxt('./all_feats.csv', all_feats_and_labels, header=header, delimiter=',')


if __name__ == '__main__':
    
    yaml_filename = '../configs/OOS_Final_Model.yaml'
    #yaml_filename = '../configs/testing_overlaps.yaml'
    #yaml_filename = '../configs/CV_prefiltering.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    torch.device(hparams['device'])
    print(hparams)
    #run_write_all_feats(MODEL_PATH, DAFI_PATH, hparams, random_state=3)

    run_save_diagnostics_oos(MODEL_PATH, DAFI_PATH, PATHS_TO_OOS_TEST_DATA, '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/testing_diagnostics.csv', hparams) 
