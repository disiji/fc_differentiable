import pickle
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import utils.utils_load_data as dh
from utils.HeuristicInitializer import HeuristicInitializer
from utils.bayes_gate import ReferenceTree
from utils.bayes_gate import ReferenceTreeBoth


class CLLInputBase:
    def __init__(self):
        self.x_list = None,
        self.y_list = None,
        self.x = None,
        self.y = None,
        self.x_train = None
        self.x_eval = None
        self.y_train = None
        self.y_eval = None
        self.reference_nested_list = None,
        self.reference_tree = None,
        self.init_nested_list = None
        self.init_tree = None

    def _load_data_(self):
        pass

    def _get_reference_nested_list_(self):
        pass

    def _get_init_nested_list_(self):
        pass

    def _normalize_(self):
        pass

    def _construct_(self):
        pass

    def split(self, random_state=123):
        pass


class Cll4d1pInput(CLLInputBase):
    def __init__(self, hparams):
        self.hparams = hparams
        self.features = ['CD5', 'CD19', 'CD10', 'CD79b']
        self.features_full = ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3']
        self.feature2id = dict((self.features[i], i) for i in range(len(self.features)))

        self._load_data_(hparams)
        self._get_reference_nested_list_()
        self._get_init_nested_list_(hparams)
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)
        on_cuda_list_x_all = []
        if torch.cuda.is_available():
            for i in range(len(self.x)):
                on_cuda_list_x_all.append(self.x[i].cuda())
            self.x = on_cuda_list_x_all
            self.y = self.y.cuda()

    def _load_data_(self, hparams):
        X_DATA_PATH = hparams['data']['features_path']
        Y_DATA_PATH = hparams['data']['labels_path']
        DATA_DIR = '../data/cll'
        CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'

        # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
        # y: a list of labels; 1 is CLL, 0 is healthy
        if self.hparams['load_from_pickle']:
            with open(X_DATA_PATH, 'rb') as f:
                self.x_list = pickle.load(f)
                print('Number of samples: %d' % (len(self.x_list)))
            with open(Y_DATA_PATH, 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data_1p(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, self.features_full)
            x_4d = dh.filter_cll_4d_pb1(x)
            with open(X_DATA_PATH, 'wb') as f:
                pickle.dump(x_4d, f)
            with open(Y_DATA_PATH, 'wb') as f:
                pickle.dump(y, f)
            self.x_list, self.y_list = x_4d, y

    def _get_reference_nested_list_(self):
        self.reference_nested_list = \
            [
                [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
                [
                    [
                        [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                        []
                    ]
                ]
            ]

    def _get_init_nested_list_(self, hparams):
        if hparams['init_type'] == 'random':
            size_mean = 2000 * 2000
            size_var = 600 * 600  # about a third of the mean
            cut_var = 400  # about one tenth of the range
            min_size = 500 * 500
            self.init_nested_list = self._get_random_init_nested_list_(size_mean, size_var, cut_var, min_size=3)
        else:
            self.init_nested_list = self._get_middle_plots_init_nested_list_()

    def _get_random_init_nested_list_(self, size_mean, size_var, cut_var, min_size=.1, max_cut=5000):
        middle_gates = self._get_middle_plots_flattened_list_()
        random_gate_flat_init = []
        for gate in middle_gates:
            size = min_size
            while size <= min_size:
                size = np.random.normal(size_mean, size_var)
            last_cut_to_sample = np.random.randint(0, len(gate))
            num_iters_in_while = 0
            cuts_in_order = False
            while not cuts_in_order:
                cur_cuts = -1 * np.ones(4)
                for i in range(len(gate)):
                    if i == last_cut_to_sample:
                        continue
                    while cur_cuts[i] < 0:
                        cur_cuts[i] = np.random.normal(gate[i], cut_var)
                cuts_in_order = (cur_cuts[1] > cur_cuts[0]) if (
                            last_cut_to_sample == 3 or last_cut_to_sample == 2) else (cur_cuts[3] > cur_cuts[2])
                num_iters_in_while += 1
                if num_iters_in_while > 10:
                    raise ValueError('The cut variance is way too large, try lowering it.')

            # now do the four cases from scratch work
            if last_cut_to_sample == 0:  # lower boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[1] - size / (cur_cuts[3] - cur_cuts[2])
            elif last_cut_to_sample == 1:  # upper boundary
                cur_cuts[last_cut_to_sample] = size / (cur_cuts[3] - cur_cuts[2]) + cur_cuts[0]
            elif last_cut_to_sample == 2:  # lower boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[3] - size / (cur_cuts[1] - cur_cuts[0])
            elif last_cut_to_sample == 3:  # upper boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[2] + size / (cur_cuts[1] - cur_cuts[0])
            assert (np.abs((cur_cuts[1] - cur_cuts[0]) * (
                        cur_cuts[3] - cur_cuts[2]) - size) < 1e-3)  # make sure it has the correct size
            random_gate_flat_init.append(cur_cuts)
            if cur_cuts[last_cut_to_sample] < 0 or cur_cuts[last_cut_to_sample] > max_cut:
                print('meow')
                return self._get_random_init_nested_list_(size_mean, size_var, cut_var, min_size=min_size)
        print(random_gate_flat_init)
        return (self._convert_flattened_list_to_nested_(random_gate_flat_init))

    def _get_middle_plots_flattened_list_(self):
        return [[1019, 3056, 979, 2937], [1024., 3071., 992., 2975.]]

    def _convert_flattened_list_to_nested_(self, flat_list):
        converted_list = \
            [
                [[u'CD5', flat_list[0][0], flat_list[0][1]], [u'CD19', flat_list[0][2], flat_list[0][3]]],
                [
                    [
                        [[u'CD10', flat_list[1][0], flat_list[1][1]], [u'CD79b', flat_list[1][2], flat_list[1][3]]],
                        []
                    ]
                ]
            ]
        return converted_list

    def _get_middle_plots_init_nested_list_(self):
        self.init_nested_list = \
            [
                [[u'CD5', 1019., 3056.], [u'CD19', 979., 2937.]],
                [
                    [
                        [[u'CD10', 1024., 3071.], [u'CD79b', 992., 2975.]],
                        []
                    ]
                ]
            ]
        return self.init_nested_list

    def _normalize_x_list_all(self):
        self.x_list, offset, scale = dh.normalize_x_list(self.x_list, offset=self.offset, scale=self.scale)
        self.x_list = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y_list = torch.tensor(self.y_list, dtype=torch.float32)
        if torch.cuda.is_available():
            on_cuda_list_x = []
            for i in range(len(self.x_list)):
                on_cuda_list_x.append(self.x_list[i].cuda())
            self.x_list = on_cuda_list_x
            self.y_list = self.y_list.cuda()

    def _normalize_data_tr_and_nested_list(self):
        self.x_train, offset, scale = dh.normalize_x_list(self.x_train)
        self.reference_nested_list = dh.normalize_nested_tree(self.reference_nested_list, offset, scale,
                                                              self.feature2id)
        if not (self.hparams['init_type'] == 'random_corner' or self.hparams['init_type'] == 'same_corners_as_DAFI' or
                self.hparams['init_type'] == 'padhraics_init' or self.hparams['init_type'] == 'heuristic_init'):
            self.init_nested_list = dh.normalize_nested_tree(self.init_nested_list, offset, scale, self.feature2id)
        self.offset = offset
        self.scale = scale

    def _normalize_data_eval(self):
        self.x_eval, offset, scale = dh.normalize_x_list(self.x_eval, offset=self.offset, scale=self.scale)

    def _normalize_(self):
        self._normalize_data_tr_and_nested_list()
        if self.hparams['test_size'] == 0 and (
                (self.split_fold_idxs is None) and (not self.hparams['use_out_of_sample_eval_data'])):
            self.x_train = [torch.tensor(_, dtype=torch.float32) for _ in self.x_train]
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        else:
            self._normalize_data_eval()
            self.x_train = [torch.tensor(_, dtype=torch.float32) for _ in self.x_train]
            self.x_eval = [torch.tensor(_, dtype=torch.float32) for _ in self.x_eval]
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
            self.y_eval = torch.tensor(self.y_eval, dtype=torch.float32)
        if torch.cuda.is_available():
            on_cuda_list_x_train = []
            on_cuda_list_x_eval = []
            for i in range(len(self.x_train)):
                on_cuda_list_x_train.append(self.x_train[i].cuda(self.hparams['device']))
            if not (self.x_eval is None):
                for i in range(len(self.x_eval)):
                    on_cuda_list_x_eval.append(self.x_eval[i].cuda(self.hparams['device']))
            self.x_train = on_cuda_list_x_train
            if not (self.x_eval is None):
                self.x_eval = on_cuda_list_x_eval
                self.y_eval = self.y_eval.cuda(self.hparams['device'])
            self.y_train = self.y_train.cuda(self.hparams['device'])
        self._normalize_x_list_all()

    def _construct_(self):
        self.reference_tree = ReferenceTree(self.reference_nested_list, self.feature2id)
        self.init_tree = ReferenceTree(self.init_nested_list, self.feature2id)
        if self.hparams['dafi_init']:
            self.init_tree = None

    def split(self, random_state=123):
        if (self.hparams['test_size'] == 0.) and (
                (self.split_fold_idxs is None) and (not self.hparams['use_out_of_sample_eval_data'])):
            self.x_train = self.x_list
            self.y_train = self.y_list
            self.x_eval = None
            self.y_eval = None

        elif self.hparams['use_out_of_sample_eval_data']:
            with open(self.hparams['out_of_sample_eval_data'], 'rb') as f:
                self.x_eval = pickle.load(f)
            with open(self.hparams['out_of_sample_eval_labels'], 'rb') as f:
                self.y_eval = pickle.load(f)
            self.idxs_train = np.arange(len(self.x_list))
            self.idxs_eval = np.arange(len(self.x_list), len(self.x_list) + len(self.x_eval))
            self.sample_ids = np.arange(len(self.x_list) + len(self.x_eval))

            self.x_list.extend(self.x_eval)
            self.y_list.extend(self.y_eval)
            self.x_train = [self.x_list[idx] for idx in self.idxs_train]
            self.y_train = [self.y_list[idx] for idx in self.idxs_train]
            print('%d samples in training data and %d samples in eval' % (len(self.x_train), len(self.x_eval)))

        elif self.split_fold_idxs is None:
            idxs = np.arange(len(self.x_list))
            self.sample_ids = idxs
            self.x_train, self.x_eval, self.y_train, self.y_eval, self.idxs_train, self.idxs_eval = train_test_split(
                self.x_list, self.y_list, idxs,
                test_size=self.hparams['test_size'],
                random_state=random_state
            )
            if not (self.augment_data_paths is None):
                with open(self.augment_data_paths['X'], 'rb') as f:
                    augment_x_list = pickle.load(f)
                with open(self.augment_data_paths['Y'], 'rb') as f:
                    augment_y_list = pickle.load(f)
                augment_ids = np.arange(len(augment_x_list))
                self.idxs_train = np.concatenate([self.idxs_train, len(self.sample_ids) + augment_ids])
                self.x_train.extend(augment_x_list)
                self.y_train.extend(augment_y_list)
        else:
            idxs = np.arange(len(self.x_list))
            self.sample_ids = idxs
            self.idxs_train = self.split_fold_idxs[0]
            self.idxs_eval = self.split_fold_idxs[1]

            self.x_train = [self.x_list[idx] for idx in self.split_fold_idxs[0]]
            self.y_train = [self.y_list[idx] for idx in self.split_fold_idxs[0]]

            self.x_eval = [self.x_list[idx] for idx in self.split_fold_idxs[1]]
            self.y_eval = [self.y_list[idx] for idx in self.split_fold_idxs[1]]
            if not (self.augment_data_paths is None):
                with open(self.augment_data_paths['X'], 'rb') as f:
                    augment_x_list = pickle.load(f)
                with open(self.augment_data_paths['Y'], 'rb') as f:
                    augment_y_list = pickle.load(f)
                augment_ids = np.arange(len(augment_x_list))
                self.idxs_train = np.concatenate([self.idxs_train, len(self.sample_ids) + augment_ids])
                self.x_train.extend(augment_x_list)
                self.y_train.extend(augment_y_list)


class Cll8d1pInput(Cll4d1pInput):
    """
    apply FSC-A and FSC-H prefiltering gate and learn other gate locations
    """

    def __init__(self, hparams, random_state=0, augment_data_paths=None, split_fold_idxs=None):
        self.hparams = hparams
        self.augment_data_paths = augment_data_paths
        self.random_state = random_state
        self.features = ['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b']
        self.features_full = ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3']
        self.feature2id = dict((self.features[i], i) for i in range(len(self.features)))
        self.split_fold_idxs = split_fold_idxs  # 0 for train, 1 for test
        self._load_data_(hparams)
        self.unnormalized_x_list_of_numpy = deepcopy(self.x_list)
        self.y_numpy = deepcopy(self.y_list)
        self.reference_nested_list = self._get_reference_nested_list_()
        self.split(random_state=random_state)

        self._normalize_()
        self._get_init_nested_list_(hparams)
        self._construct_()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

        on_cuda_list_x_all = []
        if torch.cuda.is_available():
            for i in range(len(self.x)):
                on_cuda_list_x_all.append(self.x[i].cuda())
            self.x = on_cuda_list_x_all
            self.y = self.y.cuda()

    def _load_data_(self, hparams):
        X_DATA_PATH = hparams['data']['features_path']
        Y_DATA_PATH = hparams['data']['labels_path']
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'

        # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
        # y: a list of labels; 1 is CLL, 0 is healthy
        with open(X_DATA_PATH, 'rb') as f:
            self.x_list = pickle.load(f)
        with open(Y_DATA_PATH, 'rb') as f:
            self.y_list = pickle.load(f)

    def _get_reference_nested_list_(self):
        reference_nested_list = \
            [
                [[u'SSC-H', 102., 921.], [u'CD45', 2048., 3891]],
                [
                    [
                        [[u'FSC-A', 921., 2150.], [u'SSC-A', 102., 921.]],
                        [
                            [
                                [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
                                [
                                    [
                                        [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                                        []
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        return reference_nested_list

    def get_unnormalized_reference_tree(self):
        unnormalized_reference_list = self._get_reference_nested_list_()
        reference_tree = ReferenceTree(unnormalized_reference_list, self.feature2id)
        return reference_tree

    def _get_init_nested_list_(self, hparams):
        if hparams['init_type'] == 'random_corner':
            self.init_nested_list = self._get_random_corner_init(size_default=hparams['corner_init_deterministic_size'])
        elif hparams['init_type'] == 'same_corners_as_DAFI':
            self.init_nested_list = self._get_same_corners_as_DAFI_init()
        elif hparams['init_type'] == 'heuristic_init':
            self.init_nested_list = self._get_heuristic_init()
        elif hparams['init_type'] == 'middle_init':
            self.init_nested_list = self._get_middle_plots_init_nested_list_()
        else:
            raise ValueError('init type not recognized')

    def _get_heuristic_init(self):
        heuristic_initializer = HeuristicInitializer(
            self.hparams['node_type'],
            self.get_gate_data_ids(),
            np.concatenate(self.get_pos_tr_data(return_numpy=True)),
            np.concatenate(self.get_neg_tr_data(return_numpy=True)),
            num_gridcells_per_axis=self.hparams['heuristic_init']['num_gridcells_per_axis'],
            greedy_filtering=self.hparams['heuristic_init']['use_greedy_filtering'],
            consider_all_gates=self.hparams['heuristic_init']['consider_all_gates']
        )
        flat_gates = heuristic_initializer.get_heuristic_gates()
        self.flat_heuristic_gates = flat_gates
        return self._convert_flattened_list_to_nested_(flat_gates)

    def get_gate_data_ids(self):
        gate_data_ids = \
            [
                [self.feature2id['SSC-H'], self.feature2id['CD45']],
                [self.feature2id['FSC-A'], self.feature2id['SSC-A']],
                [self.feature2id['CD5'], self.feature2id['CD19']],
                [self.feature2id['CD10'], self.feature2id['CD79b']],
            ]

        return gate_data_ids

    def get_pos_tr_data(self, return_numpy=False):
        pos_tr_data = \
            [
                x for idx, x in enumerate(self.x_train)
                if self.y_train[idx] == 1
            ]
        if return_numpy:
            pos_tr_data = \
                [
                    x.cpu().detach().numpy() for x in pos_tr_data
                ]
        return pos_tr_data

    def get_pos_eval_data(self):
        pos_ev_data = \
            [
                x for idx, x in enumerate(self.x_eval)
                if self.y_eval[idx] == 1
            ]
        return pos_ev_data

    def get_neg_eval_data(self):
        neg_ev_data = \
            [
                x for idx, x in enumerate(self.x_eval)
                if self.y_eval[idx] == 0
            ]
        return neg_ev_data

    def get_neg_tr_data(self, return_numpy=False):
        neg_tr_data = \
            [
                x for idx, x in enumerate(self.x_train)
                if self.y_train[idx] == 0
            ]
        if return_numpy:
            neg_tr_data = \
                [
                    x.cpu().detach().numpy() for x in neg_tr_data
                ]
        return neg_tr_data

    def _get_corner_gate(self, corner, size):
        gate = [
            corner[0] - size if corner[0] == 1 else 0.,
            corner[0] + size if corner[0] == 0 else 1.,
            corner[1] - size if corner[1] == 1 else 0.,
            corner[1] + size if corner[1] == 0 else 1.
        ]
        return gate

    def _get_same_corners_as_DAFI_init(self):

        same_corners_as_DAFI_init = \
            [
                [[u'SSC-H', 0., 0.5], [u'CD45', .5, 1.]],
                [
                    [
                        [[u'FSC-A', 0., .5], [u'SSC-A', 0., .5]],
                        [
                            [
                                [[u'CD5', .5, 1.], [u'CD19', .5, 1.]],
                                [
                                    [
                                        [[u'CD10', 0, .5], [u'CD79b', 0, .5]],
                                        []
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        return same_corners_as_DAFI_init

    def _get_random_corner_init(self, randomly_sample_size=False, size_default=0.75):
        # just using this to iterate properly over the gates
        middle_gates = self._get_middle_plots_flattened_list_()
        random_flat_gates = []
        for g in range(len(middle_gates)):
            corner = [np.random.randint(2), np.random.randint(2)]

            if randomly_sample_size:
                size = np.random.uniform(.1, .5)
            else:
                size = size_default

            random_flat_gates.append(self._get_corner_gate(corner, size))
        print('random flat gates is: ', random_flat_gates)
        return (self._convert_flattened_list_to_nested_(random_flat_gates))

    def _convert_flattened_list_to_nested_(self, random_gates):
        nested_list = \
            [
                [[u'SSC-H', random_gates[0][0], random_gates[0][1]], [u'CD45', random_gates[0][2], random_gates[0][3]]],
                [
                    [
                        [[u'FSC-A', random_gates[1][0], random_gates[1][1]],
                         [u'SSC-A', random_gates[1][2], random_gates[1][3]]],
                        [
                            [
                                [[u'CD5', random_gates[2][0], random_gates[2][1]],
                                 [u'CD19', random_gates[2][2], random_gates[2][3]]],
                                [
                                    [
                                        [[u'CD10', random_gates[3][0], random_gates[3][1]],
                                         [u'CD79b', random_gates[3][2], random_gates[3][3]]],
                                        []
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        return nested_list

    # get initializations in the middle of the plots
    def _get_middle_plots_init_nested_list_(self):
        self.init_nested_list = \
            [
                [[u'SSC-H', 1003., 3011.], [u'CD45', 1024., 3071.]],
                [
                    [
                        [[u'FSC-A', 1083., 3091.], [u'SSC-A', 1024., 3071.]],
                        [
                            [
                                [[u'CD5', 1023., 3069.], [u'CD19', 1024., 3072.]],
                                [
                                    [
                                        [[u'CD10', 1024., 3071.], [u'CD79b', 1026., 3078.]],
                                        []
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        return self.init_nested_list

    def _get_middle_plots_flattened_list_(self):
        return [[1003., 3011., 1024., 3071.], [1083., 3091., 1024., 3071.], [1023., 3069., 1024., 3072.],
                [1024., 3071., 1026, 3078.]]


class SynthInput(Cll8d1pInput):
    def __init__(self, hparams, random_state=0, augment_data_paths=None, split_fold_idxs=None, device=1):
        self.hparams = hparams
        # used to include dev data in training but not testing
        self.augment_data_paths = augment_data_paths
        self.random_state = random_state
        self.features = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
        self.features_full = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
        self.feature2id = dict((self.features[i], i) for i in range(len(self.features)))
        self.split_fold_idxs = split_fold_idxs  # 0 for train, 1 for test
        self._load_data_(hparams)
        self.unnormalized_x_list_of_numpy = deepcopy(self.x_list)
        self.y_numpy = deepcopy(self.y_list)
        self.reference_nested_list = self._get_reference_nested_list_()
        self.split(random_state=random_state)
        self._normalize_()
        print(self.reference_nested_list)
        self._get_init_nested_list_(hparams)
        self._construct_()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

        on_cuda_list_x_all = []
        if torch.cuda.is_available():
            for i in range(len(self.x)):
                on_cuda_list_x_all.append(self.x[i].cuda())
            self.x = on_cuda_list_x_all
            self.y = self.y.cuda()

    def get_gate_data_ids(self):
        gate_data_ids = \
            [
                [self.feature2id['M1'], self.feature2id['M2']],
                [self.feature2id['M3'], self.feature2id['M4']],
                [self.feature2id['M5'], self.feature2id['M6']],
                [self.feature2id['M7'], self.feature2id['M8']],
            ]
        return gate_data_ids

    def _get_heuristic_init(self):
        heuristic_initializer = HeuristicInitializer(
            self.hparams['node_type'],
            self.get_gate_data_ids(),
            np.concatenate(self.get_pos_tr_data(return_numpy=True)),
            np.concatenate(self.get_neg_tr_data(return_numpy=True)),
            num_gridcells_per_axis=self.hparams['heuristic_init']['num_gridcells_per_axis'],
            greedy_filtering=False,
            consider_all_gates=self.hparams['heuristic_init']['consider_all_gates']
        )
        flat_gates = heuristic_initializer.get_heuristic_gates()
        self.flat_heuristic_gates = flat_gates
        return self._convert_flattened_list_to_nested_(flat_gates)

    def _get_reference_nested_list_(self):
        reference_nested_list = \
            [
                [[u'M1', 0, 1.75], [u'M2', 0, 1.75]],
                [[[[u'M3', 0, 1.75], [u'M4', 0, 1.75]], []],  # left branch

                 [

                     [[u'M5', 1.75, 3.25], [u'M6', 0, 1.75]],

                     [
                         [[[u'M7', 1.75, 3.25], [u'M8', 1.75, 3.25]], []]
                     ]
                 ]

                 ]

            ]
        return reference_nested_list

    def _convert_flattened_list_to_nested_(self, flat_gates):
        gate1, gate2, gate3, gate4 = flat_gates
        converted_nested_list = \
            [
                [[u'M1', gate1[0], gate1[1]], [u'M2', gate1[2], gate1[3]]],
                [[[[u'M3', gate2[0], gate2[1]], [u'M4', gate2[2], gate2[3]]], []],  # left branch

                 [

                     [[u'M5', gate3[0], gate3[1]], [u'M6', gate3[2], gate3[3]]],

                     [
                         [[[u'M7', gate4[0], gate4[1]], [u'M8', gate4[2], gate4[3]]], []]
                     ]
                 ]

                 ]

            ]

        return converted_nested_list

    def _load_data_(self, hparams):
        X_DATA_PATH = hparams['data']['features_path']
        Y_DATA_PATH = hparams['data']['labels_path']

        with open(X_DATA_PATH, 'rb') as f:
            self.x_list = pickle.load(f)
        with open(Y_DATA_PATH, 'rb') as f:
            self.y_list = pickle.load(f)

        # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
        # y: a list of labels; 1 is CLL, 0 is healthy
        if self.hparams['load_from_pickle']:
            with open(X_DATA_PATH, 'rb') as f:
                self.x_list = pickle.load(f)
            with open(Y_DATA_PATH, 'rb') as f:
                self.y_list = pickle.load(f)


class Cll4d2pInput(CLLInputBase):
    """
    The basic idea is to replace self.feaures, self.features_full, self.feature2id, self.x_list, self.y_list etc. with
        a list of objects, where the length of the list is number of panels.
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.n_panels = 2

        self.features = [['CD5', 'CD19', 'CD10', 'CD79b'], ['CD38', 'CD20', 'Anti-Lambda', 'Anti-Kappa']]
        self.features_full = [['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3'], [
            'FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda', 'Anti-Kappa']]
        self.feature2id = [dict((self.features[0][i], i) for i in range(len(self.features[0]))),
                           dict((self.features[1][i], i) for i in range(len(self.features[1])))]

        self._load_data_()
        self._fill_empty_samples_()
        self._get_reference_nested_list_()
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [[torch.tensor(x[0], dtype=torch.float32), torch.tensor(x[1], dtype=torch.float32)] for x in
                  self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

    def _load_data_(self):
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR_PB1 = DATA_DIR + "PB1_whole_mqian/"
        CYTOMETRY_DIR_PB2 = DATA_DIR + "PB2_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
        if self.hparams['load_from_pickle']:
            with open(DATA_DIR + "filtered_4d_2p_x_list.pkl", 'rb') as f:
                self.x_list = pickle.load(f)
            with open(DATA_DIR + 'y_2p_list.pkl', 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data_2p(DIAGONOSIS_FILENAME, CYTOMETRY_DIR_PB1, CYTOMETRY_DIR_PB2,
                                       self.features_full[0], self.features_full[1])
            x_4d_pb1 = dh.filter_cll_4d_pb1([_[0] for _ in x])
            x_4d_pb2 = dh.filter_cll_4d_pb2([_[1] for _ in x])
            x_4d_2p = [[x_4d_pb1[sample_idx], x_4d_pb2[sample_idx]] for sample_idx in range(len(x_4d_pb1))]
            with open(DATA_DIR + 'filtered_4d_2p_x_list.pkl', 'wb') as f:
                pickle.dump(x_4d_2p, f)
            with open(DATA_DIR + 'y_2p_list.pkl', 'wb') as f:
                pickle.dump(y, f)
            self.x_list, self.y_list = x_4d_2p, y

    def _fill_empty_samples_(self):
        for panel_id in range(2):
            input_x_pos = np.concatenate([self.x_list[idx][panel_id] for idx in range(len(self.y_list))
                                          if self.y_list[idx] == 1])
            input_x_neg = np.concatenate([self.x_list[idx][panel_id] for idx in range(len(self.y_list))
                                          if self.y_list[idx] == 0])
            for sample_id in range(len(self.x_list)):
                if self.x_list[sample_id][panel_id].shape[0] == 0:
                    if self.y_list[sample_id] == 1:
                        self.x_list[sample_id][panel_id] = np.random.permutation(input_x_pos)[:100]
                    else:
                        self.x_list[sample_id][panel_id] = np.random.permutation(input_x_neg)[:100]

    def _get_reference_nested_list_(self):
        self.reference_nested_list = \
            [[
                [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
                [
                    [
                        [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                        []
                    ]
                ]
            ],
                [
                    [[u'CD38', 0., 1740.], [u'CD20', 614., 2252.]],
                    [
                        [
                            [[u'Anti-Kappa', 1536., 3481.], [u'Anti-Lambda', 0., 1536.]],
                            []
                        ],
                        [
                            [[u'Anti-Kappa', 0., 1536.], [u'Anti-Lambda', 1536., 3481.]],
                            []
                        ]
                    ]
                ]]

    def _get_init_nested_list_(self):
        self.init_nested_list = \
            [[
                [[u'CD5', 1019., 3056.], [u'CD19', 979., 2937.]],
                [
                    [
                        [[u'CD10', 1024., 3071.], [u'CD79b', 992., 2975.]],
                        []
                    ]
                ]
            ],
                [
                    [[u'CD38', 1024., 3071.], [u'CD20', 1024., 3071.]],
                    [
                        [
                            [[u'Anti-Kappa', 1024., 3072.], [u'Anti-Lambda', 1023., 3070.]],
                            []
                        ],
                        [
                            [[u'Anti-Kappa', 1024., 3072.], [u'Anti-Lambda', 1023., 3070.]],
                            []
                        ]
                    ]
                ]]

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list_multiple_panels(self.x_list)
        self.offset = offset
        self.scale = scale
        self.reference_nested_list = [dh.normalize_nested_tree(self.reference_nested_list[i], offset[i], scale[i],
                                                               self.feature2id[i]) for i in range(self.n_panels)]
        self.init_nested_list = [dh.normalize_nested_tree(self.init_nested_list[i], offset[i], scale[i],
                                                          self.feature2id[i]) for i in range(self.n_panels)]

    def _construct_(self):
        self.reference_tree = [ReferenceTree(self.reference_nested_list[i], self.feature2id[i])
                               for i in range(self.n_panels)]
        self.init_tree = [ReferenceTree(self.init_nested_list[i], self.feature2id[i]) for i in range(self.n_panels)]
        if self.hparams['dafi_init']:
            self.init_tree = [None] * self.n_panels

    def split(self, random_state=123):
        self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(np.array(self.x_list), self.y_list,
                                                                                test_size=self.hparams['test_size'],
                                                                                random_state=random_state)
        self.x_train = [[torch.tensor(_[0], dtype=torch.float32), torch.tensor(_[1], dtype=torch.float32)] for _ in
                        self.x_train.tolist()]
        self.x_eval = [[torch.tensor(_[0], dtype=torch.float32), torch.tensor(_[1], dtype=torch.float32)] for _ in
                       self.x_eval.tolist()]
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_eval = torch.tensor(self.y_eval, dtype=torch.float32)


class Cll2pFullInput(Cll4d2pInput):
    def __init__(self, hparams):
        self.hparams = hparams
        self.n_panels = 2

        self.features = [['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b'],
                         ['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda',
                          'Anti-Kappa']]
        self.features_full = [['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3'], [
            'FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda', 'Anti-Kappa']]
        self.feature2id = [dict((self.features[0][i], i) for i in range(len(self.features[0]))),
                           dict((self.features[1][i], i) for i in range(len(self.features[1])))]

        self._load_data_()
        self._fill_empty_samples_()
        self._get_reference_nested_list_()
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [[torch.tensor(_[0], dtype=torch.float32), torch.tensor(_[1], dtype=torch.float32)] for _ in
                  self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

    def _load_data_(self):
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR_PB1 = DATA_DIR + "PB1_whole_mqian/"
        CYTOMETRY_DIR_PB2 = DATA_DIR + "PB2_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
        if self.hparams['load_from_pickle']:
            with open(DATA_DIR + "filtered_2p_full_x_list.pkl", 'rb') as f:
                self.x_list = pickle.load(f)
            with open(DATA_DIR + 'y_2p_list.pkl', 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data_2p(DIAGONOSIS_FILENAME, CYTOMETRY_DIR_PB1, CYTOMETRY_DIR_PB2,
                                       self.features_full[0], self.features_full[1])
            x_8d_pb1 = dh.filter_cll_8d_pb1([_[0] for _ in x])
            x_10d_pb2 = dh.filter_cll_10d_pb2([_[1] for _ in x])
            x_4d_2p = [[x_8d_pb1[sample_idx], x_10d_pb2[sample_idx]] for sample_idx in range(len(x_8d_pb1))]
            with open(DATA_DIR + 'filtered_2p_full_x_list.pkl', 'wb') as f:
                pickle.dump(x_4d_2p, f)
            with open(DATA_DIR + 'y_2p_list.pkl', 'wb') as f:
                pickle.dump(y, f)
            self.x_list, self.y_list = x_4d_2p, y

    def _get_reference_nested_list_(self):
        self.reference_nested_list = \
            [
                [
                    [[u'SSC-H', 102., 921.], [u'CD45', 2048., 2560.]],
                    [
                        [
                            [[u'FSC-A', 921., 2150.], [u'SSC-A', 102., 921.]],
                            [
                                [
                                    [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
                                    [
                                        [
                                            [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                                            []
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ],
                [
                    [[u'SSC-H', 102., 921.], [u'CD45', 2048., 2560.]],
                    [
                        [
                            [[u'FSC-A', 921., 2150.], [u'SSC-A', 102., 921.]],
                            [
                                [
                                    [[u'CD38', 0., 1740.], [u'CD20', 614., 2252.]],
                                    [
                                        [
                                            [[u'Anti-Kappa', 1536., 3481.], [u'Anti-Lambda', 0., 1536.]],
                                            []
                                        ],
                                        [
                                            [[u'Anti-Kappa', 0., 1536.], [u'Anti-Lambda', 1536., 3481.]],
                                            []
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]

    def _get_init_nested_list_(self):
        self.init_nested_list = \
            [
                [
                    [[u'SSC-H', 1003., 3011.], [u'CD45', 1024., 3072.]],
                    [
                        [
                            [[u'FSC-A', 1082., 3091.], [u'SSC-A', 1023., 3071.]],
                            [
                                [
                                    [[u'CD5', 1023., 3069.], [u'CD19', 1023., 3071.]],
                                    [
                                        [
                                            [[u'CD10', 1024., 3071.], [u'CD79b', 1025., 3077.]],
                                            []
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ],
                [
                    [[u'SSC-H', 1003., 3011.], [u'CD45', 1023., 3070.]],
                    [
                        [
                            [[u'FSC-A', 1082., 3090.], [u'SSC-A', 1023., 3071.]],
                            [
                                [
                                    [[u'CD38', 1024., 3071.], [u'CD20', 1024., 3071.]],
                                    [
                                        [
                                            [[u'Anti-Kappa', 1024., 3072.], [u'Anti-Lambda', 1023., 3070.]],
                                            []
                                        ],
                                        [
                                            [[u'Anti-Kappa', 1024., 3072.], [u'Anti-Lambda', 1023., 3070.]],
                                            []
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]


class CllBothPanelsInput(Cll8d1pInput):
    def __init__(self, hparams, random_state=0, split_fold_idxs=None):
        self.augment_data_paths = hparams['augment_data_params'] if hparams['augment_training_with_dev_data'] else None

        self.split_fold_idxs = split_fold_idxs
        self.hparams = hparams
        self.n_panels = 2

        self.features = [['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b'],
                         ['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda',
                          'Anti-Kappa']]
        self.features_full = [['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3'], [
            'FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda', 'Anti-Kappa']]
        self.feature2id = [dict((self.features[0][i], i) for i in range(len(self.features[0]))),
                           dict((self.features[1][i], i) for i in range(len(self.features[1])))]
        self.random_state = random_state
        self._load_data_()
        self._fill_empty_samples_()
        self.split(random_state=random_state)
        self._get_reference_nested_list_()
        self._normalize_()
        self._get_init_nested_list_()
        self._construct_()

        self.x = [[torch.tensor(_[0], dtype=torch.float32), torch.tensor(_[1], dtype=torch.float32)] for _ in
                  self.x_list]

        self.y = torch.tensor(self.y_list, dtype=torch.float32)

    def _fill_empty_samples_(self):
        for panel_id in range(2):
            input_x_pos = np.concatenate([self.x_list[idx][panel_id] for idx in range(len(self.y_list))
                                          if self.y_list[idx] == 1])
            input_x_neg = np.concatenate([self.x_list[idx][panel_id] for idx in range(len(self.y_list))
                                          if self.y_list[idx] == 0])
            for sample_id in range(len(self.x_list)):
                if self.x_list[sample_id][panel_id].shape[0] == 0:
                    if self.y_list[sample_id] == 1:
                        self.x_list[sample_id][panel_id] = np.random.permutation(input_x_pos)[:100]
                    else:
                        self.x_list[sample_id][panel_id] = np.random.permutation(input_x_neg)[:100]

    def _load_data_(self):
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR_PB1 = DATA_DIR + "PB1_whole_mqian/"
        CYTOMETRY_DIR_PB2 = DATA_DIR + "PB2_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
        X_DATA_PATH = self.hparams['data']['features_path']
        Y_DATA_PATH = self.hparams['data']['labels_path']

        if self.hparams['load_from_pickle']:
            with open(X_DATA_PATH, 'rb') as f:
                self.x_list = pickle.load(f)
                print('Number of samples: %d' % (len(self.x_list)))
            with open(Y_DATA_PATH, 'rb') as f:
                self.y_list = pickle.load(f)

    def _construct_(self):
        self.reference_tree = ReferenceTreeBoth(self.reference_nested_list, self.feature2id)
        self.init_tree = ReferenceTreeBoth(self.init_nested_list, self.feature2id)
        if self.hparams['dafi_init']:
            self.init_tree = [None] * self.n_panels

    def _normalize_x_list_all(self):
        self.x_list, offset, scale = dh.normalize_x_list_multiple_panels(self.x_list, offset=self.offset,
                                                                         scale=self.scale)
        self.x_list = [[torch.tensor(x, dtype=torch.float32) for x in _] for _ in self.x_list]
        self.y_list = torch.tensor(self.y_list, dtype=torch.float32)
        if torch.cuda.is_available():
            on_cuda_list_x = []
            for i in range(len(self.x_list)):
                on_cuda_list_x.append([x.cuda() for x in self.x_list[i]])
            self.x_list = on_cuda_list_x
            self.y_list = self.y_list.cuda()

    def _normalize_data_tr_and_nested_list(self):
        self.x_train, offset, scale = dh.normalize_x_list_multiple_panels(self.x_train)
        self.reference_nested_list = dh.normalize_nested_tree_both_panels(self.reference_nested_list, offset, scale,
                                                                          self.feature2id)
        if not (self.hparams['init_type'] == 'random_corner' or self.hparams['init_type'] == 'same_corners_as_DAFI' or
                self.hparams['init_type'] == 'padhraics_init' or self.hparams['init_type'] == 'heuristic_init'):
            self.init_nested_list = dh.normalize_nested_tree(self.init_nested_list, offset, scale, self.feature2id)
        self.offset = offset
        self.scale = scale

    def _normalize_data_eval(self):
        self.x_eval, offset, scale = dh.normalize_x_list_multiple_panels(self.x_eval, offset=self.offset,
                                                                         scale=self.scale)

    def _normalize_(self):
        self._normalize_data_tr_and_nested_list()

        if self.hparams['test_size'] == 0 and (
                (self.split_fold_idxs is None) and (not self.hparams['use_out_of_sample_eval_data'])):
            self.x_train = [[torch.tensor(x, dtype=torch.float32) for x in _] for _ in self.x_train]
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        else:
            self._normalize_data_eval()
            self.x_train = [[torch.tensor(x, dtype=torch.float32) for x in _] for _ in self.x_train]
            self.x_eval = [[torch.tensor(x, dtype=torch.float32) for x in _] for _ in self.x_eval]
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
            self.y_eval = torch.tensor(self.y_eval, dtype=torch.float32)

        if torch.cuda.is_available():
            on_cuda_list_x_train = []
            on_cuda_list_x_eval = []
            for i in range(len(self.x_train)):
                on_cuda_list_x_train.append([x_both.cuda(self.hparams['device']) for x_both in self.x_train[i]])
            if not (self.x_eval is None):
                for i in range(len(self.x_eval)):
                    on_cuda_list_x_eval.append([x_both.cuda(self.hparams['device']) for x_both in self.x_eval[i]])
            self.x_train = on_cuda_list_x_train
            if not (self.x_eval is None):
                self.x_eval = on_cuda_list_x_eval
                self.y_eval = self.y_eval.cuda(self.hparams['device'])
            self.y_train = self.y_train.cuda(self.hparams['device'])
        self._normalize_x_list_all()

    def _get_init_nested_list_(self):
        if self.hparams['init_type'] == 'heuristic_init':
            self.init_nested_list = self._get_heuristic_init()
        else:
            raise NotImplementedError('Non-heuristic initializion not implemented for both panel input object')

    def _get_heuristic_init(self):
        # Idea is to initialize the earlier nodes that use both panels
        # using data from both panels, and then to initialize nodes
        # that just use either p1 or p2 separately
        gate_ids_both = self.get_gate_data_ids_both()
        data_for_init_pos, data_for_init_neg = self._get_concatenated_data_for_init()

        heuristic_initializer_both = HeuristicInitializer(
            self.hparams['node_type'],
            gate_ids_both,
            np.concatenate(data_for_init_pos),
            np.concatenate(data_for_init_neg),
            num_gridcells_per_axis=self.hparams['heuristic_init']['num_gridcells_per_axis'],
            greedy_filtering=self.hparams['heuristic_init']['use_greedy_filtering'],
            consider_all_gates=self.hparams['heuristic_init']['consider_all_gates']
        )

        data_p1_leaf_pos, data_p1_leaf_neg = self._get_data_p1_leaf_for_init()
        heuristic_initializer_p1_leaf = HeuristicInitializer(
            self.hparams['node_type'],
            [[self.feature2id[0]['CD10'], self.feature2id[0]['CD79b']]],
            np.concatenate([self.from_gpu_to_numpy(data) for data in data_p1_leaf_pos]),
            np.concatenate([self.from_gpu_to_numpy(data) for data in data_p1_leaf_neg]),
            num_gridcells_per_axis=self.hparams['heuristic_init']['num_gridcells_per_axis'],
            greedy_filtering=self.hparams['heuristic_init']['use_greedy_filtering'],
            consider_all_gates=self.hparams['heuristic_init']['consider_all_gates']
        )

        both_flat_gates = heuristic_initializer_both.get_heuristic_gates()
        just_p1_leaf_gates_flat = heuristic_initializer_p1_leaf.get_heuristic_gates()
        just_p2_gates_flat = self._get_flat_p2_gates_init()
        flat_gates = both_flat_gates + just_p1_leaf_gates_flat + just_p2_gates_flat
        print('Final initialized flat gates:')
        print(flat_gates)
        self.flat_heuristic_gates = flat_gates
        init_nested_list = self._convert_flattened_list_to_nested_(flat_gates)

        return init_nested_list

    def _convert_flattened_list_to_nested_(self, gates):
        nested_list = \
            [
                [[u'SSC-H', gates[0][0], gates[0][1]], [u'CD45', gates[0][2], gates[0][3]], 'both'],
                [
                    [
                        [[u'FSC-A', gates[1][0], gates[1][1]], [u'SSC-A', gates[1][2], gates[1][3]], 'both'],
                        [
                            [
                                [[u'CD5', gates[2][0], gates[2][1]], [u'CD19', gates[2][2], gates[2][3]], 'both'],
                                [
                                    [
                                        [[u'CD10', gates[3][0], gates[3][1]], [u'CD79b', gates[3][2], gates[3][3]],
                                         'p1'],
                                        []
                                    ],
                                    [
                                        [[u'CD38', gates[4][0], gates[4][1]], [u'CD20', gates[4][2], gates[4][3]],
                                         'p2'], [[[[u'Anti-Kappa', gates[5][0], gates[5][1]],
                                                   [u'Anti-Lambda', gates[5][2], gates[5][3]], 'p2'], []], [
                                                     [[u'Anti-Kappa', gates[6][0], gates[6][1]],
                                                      [u'Anti-Lambda', gates[6][2], gates[6][3]], 'p2'], []]]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        return nested_list

    def _get_data_p1_leaf_for_init(self):
        data_p1_pos = [x_both[0] for i, x_both in enumerate(self.x_train) if self.y_train[i] == 1]
        data_p1_neg = [x_both[0] for i, x_both in enumerate(self.x_train) if self.y_train[i] == 0]
        return data_p1_pos, data_p1_neg

    def _get_flat_p2_gates_init(self):
        # starting in the middle for gates which are only in p2 since the
        # feature diff hueristic makes no sense for these gates
        flat_p2_init_gates = \
            [[.25, .75, .25, .75], [.25, .75, .25, .75], [.25, .75, .25, .75]]
        return flat_p2_init_gates

    def get_gate_data_ids_both(self):
        # doesnt matter which feature2id used since both p1 and p2 have the same
        # ordering for the first five features
        gate_data_ids = \
            [
                [self.feature2id[0]['SSC-H'], self.feature2id[0]['CD45']],
                [self.feature2id[0]['FSC-A'], self.feature2id[0]['SSC-A']],
                [self.feature2id[0]['CD5'], self.feature2id[0]['CD19']],
            ]

        return gate_data_ids

    def from_gpu_to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def _get_data_for_init_both(self):
        FEATURE_IDXS_IN_BOTH = np.array([0, 1, 2, 3, 4, 5])
        x_p1_both = [self.from_gpu_to_numpy(x[0][:, FEATURE_IDXS_IN_BOTH]) for x in self.x_train]
        x_p1_both_pos = [x for i, x in enumerate(x_p1_both) if self.y_train[i] == 1]
        x_p1_both_neg = [x for i, x in enumerate(x_p1_both) if self.y_train[i] == 0]
        return x_p1_both_pos, x_p1_both_neg

    def _get_concatenated_data_for_init(self):
        FEATURE_IDXS_IN_BOTH = np.array([0, 1, 2, 3, 4, 5])
        x_cat_list = []
        for x_both in self.x_train:
            x_both = [self.from_gpu_to_numpy(x[:, FEATURE_IDXS_IN_BOTH]) for x in x_both]
            x_cat = np.concatenate(x_both, axis=0)
            x_cat_list.append(x_cat)
        x_cat_list_pos = [x_cat for i, x_cat in enumerate(x_cat_list) if self.y_train[i] == 1]
        x_cat_list_neg = [x_cat for i, x_cat in enumerate(x_cat_list) if self.y_train[i] == 0]
        return x_cat_list_pos, x_cat_list_neg

    def _get_reference_nested_list_(self):
        self.reference_nested_list = \
            [
                [[u'SSC-H', 102., 921.], [u'CD45', 2048., 3891], 'both'],
                [
                    [
                        [[u'FSC-A', 921., 2150.], [u'SSC-A', 102., 921.], 'both'],
                        [
                            [
                                [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.], 'both'],
                                [
                                    [
                                        [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.], 'p1'],
                                        []
                                    ],
                                    [
                                        [[u'CD38', 0., 1740.], [u'CD20', 614., 2252.], 'p2'],
                                        [[[[u'Anti-Kappa', 1536., 3481.], [u'Anti-Lambda', 0., 1536], 'p2'], []],
                                         [[[u'Anti-Kappa', 0., 1536.], [u'Anti-Lambda', 1536., 3481.], 'p2'], []]]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
