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

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
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

    for random_state in range(random_state_start, hparams['n_run']):
        start_time = time.time()
        hparams['random_state'] = random_state
        hparams['experiment_name'] = hparams['experiment_name'] + '_seed%d' %random_state
        savedir = '../output/%s/' %hparams['experiment_name']
        if not(os.path.exists(savedir)):
            os.mkdir(savedir)
        #np.random.seed(random_state)
        #torch.manual_seed(random_state)
        cll_1p_full_input = Cll8d1pInput(hparams, random_state=random_state)

        model_tree = ModelTree(cll_1p_full_input.reference_tree,
                               logistic_k=hparams['logistic_k'],
                               regularisation_penalty=hparams['regularization_penalty'],
                               negative_box_penalty=hparams['negative_box_penalty'],
                               positive_box_penalty=hparams['positive_box_penalty'],
                               corner_penalty=hparams['corner_penalty'],
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
            dafi_tree.cuda()

        if hparams['two_phase_training']['turn_on']: 
            model_tree, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
                run_train_model_two_phase(hparams, cll_1p_full_input, model_checkpoint=model_checkpoint)
        elif hparams['testing_new_single_phase_training']:

            cll_1p_full_input = Cll8d1pInput(hparams, random_state=random_state, augment_data_paths=DEV_DATA_PATHS)
            model_tree, train_tracker_m, eval_tracker_m, run_time, model_checkpoint_dict = \
                run_train_full_batch_logreg_to_conv(hparams, cll_1p_full_input, model_tree, model_checkpoint=model_checkpoint)

            dafi_tree, train_tracker_d, eval_tracker_d = run_train_dafi_logreg_to_conv(dafi_tree, hparams, cll_1p_full_input)
        else:
            model_tree, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
                run_train_model(model_tree, hparams, cll_1p_full_input, model_checkpoint=model_checkpoint)

        if hparams['output']['type'] == 'full':
            output_metric_dict = run_output(
                model_tree, dafi_tree, hparams, cll_1p_full_input, train_tracker, eval_tracker, run_time)
        #elif hparams['output']['type'] == 'lightweight':
        #    output_metric_dict = run_lightweight_output_no_split_no_dafi(
        #        model_tree, dafi_tree, hparams, cll_1p_full_input, train_tracker, eval_tracker, run_time, model_checkpoint_dict)
        elif hparams['testing_new_single_phase_training']:
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

        else:
            raise ValueError('Output type not recognized')

        
        # only plot once
        # # if not os.path.isfile('../output/%s/metrics.png' % hparams['experiment_name']) and plot_and_write_output:
        # run_plot_metric(hparams, train_tracker, eval_tracker, dafi_tree, cll_1p_full_input, output_metric_dict)
        # run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, cll_1p_full_input)
        #run_write_prediction(model_tree, dafi_tree, cll_1p_full_input, hparams)
        
        print('The full loop for random_state %d took %d seconds' %(random_state, time.time() - start_time))

        #models_per_iteration = [model_checkpoint_dict[iteration] 
        #        for iteration in 
        #        hparams['seven_epochs_for_gate_motion_plot']
        #]
        #print(models_per_iteration)
        #detached_data_x_tr = [x.cpu().detach().numpy() for x in cll_1p_full_input.x_train]
#        plot_pos_and_neg_gate_motion(
#                models_per_iteration, 
#                dafi_tree,
#                detached_data_x_tr,
#                hparams,
#                cll_1p_full_input.y_train
#        )
        #run_write_overlap_diagnostics(model_tree, dafi_tree, detached_data_x_tr, cll_1p_full_input.y_train, hparams)
        # model_checkpoint = False


if __name__ == '__main__':
    #yaml_filename = '../configs/testing_full_1p.yaml'
    #yaml_filename = '../configs/testing_full_panel_plots.yaml'
    #yaml_filename = '../configs/full_panel_plots_gs=5.yaml'
    #yaml_filename = '../configs/testing_corner_init.yaml'
    #yaml_filename = '../configs/testing_gs_hard_constraint.yaml'
    #yaml_filename = '../configs/testing_overlaps.yaml'
    yaml_filename = '../configs/test_run_with_CV_params.yaml'
   # yaml_filename = '../configs/testing_my_heuristic_init.yaml'
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
    # run_single_panel(sys.argv[1], int(sys.argv[2]), True)
    run_single_panel(hparams, 1, True)
    #run_single_iter_pos_and_neg_gates_plot(yaml_filename)
    #run_leaf_gate_plots(yaml_filename)
