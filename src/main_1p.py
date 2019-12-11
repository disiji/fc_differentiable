import csv
import warnings
import yaml

from full_output_for_CV import *
from train import *
from utils.utils_plot import *
from utils.load_and_split_data import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 10000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.001,
    'init_reg_penalty': 0.0,
    'neg_proportion_default': 0.0001,
    'positive_box_penalty': 0.0,
    'corner_penalty': .0,
    'feature_diff_penalty': 0.001,
    'gate_size_penalty': .0,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'node_type': 'square',
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 40,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.05,
    'batch_size': 10,
    'n_epoch': 200,
    'seven_epochs_for_gate_motion_plot': [0, 40, 80, 120, 140, 160, 200],
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 1,
    'init_type': 'heuristic_init',
    'corner_init_deterministic_size': .75,
    'train_alternate': False,
    'run_logistic_to_convergence': True,
    'augment_training_with_dev_data': False,
    'filter_uncertain_samples': False,
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
    'plot_params': {
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


def run_single_panel(hparams, random_state_start=0, model_checkpoint=True):
    warnings.filterwarnings("ignore")
    if torch.cuda.is_available():
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
        hparams['experiment_name'] = default_exp_name + '_seed%d' % random_state
        if not os.path.exists('../output/%s' % hparams['experiment_name']):
            os.makedirs('../output/%s' % hparams['experiment_name'])
        savedir = '../output/%s/' % hparams['experiment_name']
        if not (os.path.exists(savedir)):
            os.mkdir(savedir)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        cll_1p_full_input = Cll8d1pInput(hparams, random_state=random_state)
        if hparams['augment_training_with_dev_data']:
            cll_1p_full_input_augmented = Cll8d1pInput(hparams, random_state=random_state,
                                                       augment_data_paths=DEV_DATA_PATHS)
        if hparams['filter_uncertain_samples']:
            model_tree_filter = ModelTree(cll_1p_full_input.reference_tree,
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
                                          neg_proportion_default=hparams['neg_proportion_default'],
                                          node_type=hparams['node_type']
                                          )

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
                               neg_proportion_default=hparams['neg_proportion_default'],
                               node_type=hparams['node_type']
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
                                       neg_proportion_default=hparams['neg_proportion_default'],
                                       node_type=hparams['node_type']
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

        if torch.cuda.is_available():
            model_tree.cuda()
            if hparams['augment_training_with_dev_data']:
                model_tree_aug.cuda()
            dafi_tree.cuda()
        dafi_tree, train_tracker_d, eval_tracker_d = run_train_dafi_logreg_to_conv(dafi_tree, hparams,
                                                                                   cll_1p_full_input)
        if hparams['filter_uncertain_samples']:
            model_tree_filter, _, _, _, _ = \
                run_train_full_batch_logreg_to_conv(hparams, cll_1p_full_input, model_tree_filter,
                                                    model_checkpoint=model_checkpoint)
            cll_1p_full_input.filter_samples_with_large_uncertainty(model_tree_filter)

        model_tree, train_tracker_m, eval_tracker_m, run_time, model_checkpoint_dict = \
            run_train_full_batch_logreg_to_conv(hparams, cll_1p_full_input, model_tree,
                                                model_checkpoint=model_checkpoint)
        if hparams['augment_training_with_dev_data']:
            print('Training model with tr split of validation data, and the dev data')
            model_tree_aug, train_tracker_maug, eval_tracker_maug, run_time, model_checkpoint_dict_aug = \
                run_train_full_batch_logreg_to_conv(hparams, cll_1p_full_input_augmented, model_tree_aug,
                                                    model_checkpoint=model_checkpoint)

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
            model_checkpoint_dict,
            device_data=hparams['device']
        )

        if hparams['augment_training_with_dev_data']:
            hparams['experiment_name'] = hparams['experiment_name'] + '_augmented_with_dev'
            if not os.path.exists('../output/%s' % hparams['experiment_name']):
                os.makedirs('../output/%s' % hparams['experiment_name'])
            trackers_dict_aug = {
                'tracker_train_m': train_tracker_maug,
                'tracker_eval_m': eval_tracker_maug,
                'tracker_train_d': train_tracker_d,
                'tracker_eval_d': eval_tracker_d
            }
            run_write_full_output_for_CV(
                model_tree_aug,
                dafi_tree,
                cll_1p_full_input_augmented,
                trackers_dict_aug,
                hparams,
                model_checkpoint_dict_aug,
                device_data=hparams['device']
            )

        print('The full loop for random_state %d took %d seconds' % (random_state, time.time() - start_time))


if __name__ == '__main__':
    # Load yaml files
    yaml_filename = '../configs/default_1p.yaml'
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    print(hparams)
    
    # Now load and split data if it isn't already
    # FILL IN WITH CORRECT PATHS
    panel1_data_dir = '../data/cll/PB1'
    panel2_data_dir = '../data/cll/PB2'

    # Save paths for the preprocessed/split data
    # do NOT modify these paths.
    save_path = '../data/cll/panel1/'
    save_path_both_panels = '../data/cll/both_panels/'
    # If data hasn't already been made, then make it
    if not os.path.exists(os.path.join(save_path, 'x_dev_8d_1p.pkl')):
        save_and_preprocess_data_fcs(panel1_data_dir, panel2_data_dir, save_path, save_path_both_panels)

    # Now run the main
    run_single_panel(hparams, 0, True)
