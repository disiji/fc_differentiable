import pickle
import yaml
DEFAULT_HPARAMS =  {
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

class ParameterParser:

    def __init__(self, path_to_params_file):
        self.path_to_params_file = path_to_params_file
        self.hparams = DEFAULT_HPARAMS

    def parse_params(self):
        with open(self.path_to_params_file, 'rb') as f:
            params_from_file = yaml.safe_load(f)
        self.hparams.update(params_from_file)
        self.hparams['init_method'] = \
            "dafi_init" if self.hparams['dafi_init'] \
            else "random_init"

        if self.hparams['train_alternate']:
            self.hparams['n_epoch_dafi'] = self.hparams['n_epoch'] // self.hparams['n_mini_batch_update_gates'] * (
                    self.hparams['n_mini_batch_update_gates'] - 1)
        else:
            self.hparams['n_epoch_dafi'] = self.hparams['n_epoch']

        return self.hparams


