import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import utils.utils_load_data as dh
from utils.bayes_gate import ReferenceTree


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
        #DATA_DIR = '../data/cll/'
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
                print('Number of samples: %d' %(len(self.x_list)))
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
            size_var = 600 * 600 #about a third of the mean
            cut_var = 400 #about one tenth of the range
            min_size = 500 * 500
            self.init_nested_list = self._get_random_init_nested_list_(size_mean, size_var, cut_var, min_size=3)
        else:
            self.init_nested_list = self._get_middle_plots_init_nested_list_()
    
    # Afaik we can delete this
    #def get_random_boundary(self):
    #    b1 = np.random.randint(5)
    #    b2 = np.random.randint(b1, 5)
    #    return np.array([b1, b2])


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
#                        print(cur_cuts[i], gate[i])
                cuts_in_order = (cur_cuts[1] > cur_cuts[0]) if (last_cut_to_sample == 3 or last_cut_to_sample == 2) else (cur_cuts[3] > cur_cuts[2])
                num_iters_in_while += 1
                if num_iters_in_while > 10:
                    raise ValueError('The cut variance is way too large, try lowering it.')

            #now do the four cases from scratch work
            if last_cut_to_sample == 0:#lower boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[1] - size/(cur_cuts[3] - cur_cuts[2])
            elif last_cut_to_sample == 1:#upper boundary
                cur_cuts[last_cut_to_sample] = size/(cur_cuts[3] - cur_cuts[2]) + cur_cuts[0]
            elif last_cut_to_sample == 2:#lower boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[3] - size/(cur_cuts[1] - cur_cuts[0]) 
            elif last_cut_to_sample == 3:#upper boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[2] + size/(cur_cuts[1]- cur_cuts[0])
            assert(np.abs((cur_cuts[1] - cur_cuts[0]) * (cur_cuts[3] - cur_cuts[2]) - size) < 1e-3) #make sure it has the size
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

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list(self.x_list)
        print(self.feature2id, offset, scale, self.reference_nested_list)
        self.reference_nested_list = dh.normalize_nested_tree(self.reference_nested_list, offset, scale,
                                                              self.feature2id)
        if not self.hparams['init_type'] == 'random_corner':
            self.init_nested_list = dh.normalize_nested_tree(self.init_nested_list, offset, scale, self.feature2id)

    def _construct_(self):
        self.reference_tree = ReferenceTree(self.reference_nested_list, self.feature2id)
        self.init_tree = ReferenceTree(self.init_nested_list, self.feature2id)
        if self.hparams['dafi_init']:
            self.init_tree = None

    def split(self, random_state=123):
        if self.hparams['test_size'] == 0.:
            self.x_train = self.x_list
            self.y_train = self.y_list
            self.x_eval = None
            self.y_eval = None
            
            self.x_train = [torch.tensor(_, dtype=torch.float32) for _ in self.x_train]
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        else:
            self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(self.x_list, self.y_list,
                                                                                    test_size=self.hparams['test_size'],
                                                                                    random_state=random_state)
            self.x_train = [torch.tensor(_, dtype=torch.float32) for _ in self.x_train]
            self.x_eval = [torch.tensor(_, dtype=torch.float32) for _ in self.x_eval]
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
            self.y_eval = torch.tensor(self.y_eval, dtype=torch.float32)

        if torch.cuda.is_available():
            on_cuda_list_x_train = []
            on_cuda_list_x_eval = []
            for i in range(len(self.x_train)):
                on_cuda_list_x_train.append(self.x_train[i].cuda())
                if not (self.x_eval is None):
                    on_cuda_list_x_eval.append(self.x_eval[i].cuda())
            self.x_train = on_cuda_list_x_train
            if not (self.x_eval is None):
                self.x_eval = on_cuda_list_x_eval
                self.y_eval.cuda()
            self.y_train = self.y_train.cuda()

class Cll8d1pInput(Cll4d1pInput):
    """
    apply FSC-A and FSC-H prefiltering gate and learn other gate locations
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.features = ['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b']
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
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'

        # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
        # y: a list of labels; 1 is CLL, 0 is healthy
        if self.hparams['load_from_pickle']:
            with open(X_DATA_PATH, 'rb') as f:
                self.x_list = pickle.load(f)
            with open(Y_DATA_PATH, 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data_1p(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, self.features_full)
            x_8d = dh.filter_cll_8d_pb1(x)
            with open(DATA_DIR + 'filtered_8d_1p_x_list.pkl', 'wb') as f:
                pickle.dump(x_8d, f)
            with open(DATA_DIR + 'y_1p_list.pkl', 'wb') as f:
                pickle.dump(y, f)
            self.x_list, self.y_list = x_8d, y

    def _get_reference_nested_list_(self):
        self.reference_nested_list = \
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


    def _get_init_nested_list_(self, hparams):
        if hparams['init_type'] == 'random':
            size_mean = 2000 * 2000
            size_var = 600 * 600 #about a third of the mean
            cut_var = 400 #about one tenth of the range
            min_size = 500 * 500
            self.init_nested_list = self._get_random_init_nested_list_(size_mean, size_var, cut_var, min_size=3)
        elif hparams['init_type'] == 'random_corner':
            self.init_nested_list = self._get_random_corner_init(size_default=hparams['corner_init_deterministic_size']) 
        else:
            self.init_nested_list = self._get_middle_plots_init_nested_list_()

    def _get_corner_gate(self, corner, size):
        gate = [
            corner[0] - size if corner[0] == 1 else 0., 
            corner[0] + size if corner[0] == 0 else 1., 
            corner[1] - size if corner[1] == 1 else 0., 
            corner[1] + size if corner[1] == 0 else 1.
        ]
        return gate

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
#                        print(cur_cuts[i], gate[i])
                cuts_in_order = (cur_cuts[1] > cur_cuts[0]) if (last_cut_to_sample == 3 or last_cut_to_sample == 2) else (cur_cuts[3] > cur_cuts[2])
                num_iters_in_while += 1
                if num_iters_in_while > 10:
                    raise ValueError('The cut variance is way too large, try lowering it.')

            #now do the four cases from scratch work
            if last_cut_to_sample == 0:#lower boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[1] - size/(cur_cuts[3] - cur_cuts[2])
            elif last_cut_to_sample == 1:#upper boundary
                cur_cuts[last_cut_to_sample] = size/(cur_cuts[3] - cur_cuts[2]) + cur_cuts[0]
            elif last_cut_to_sample == 2:#lower boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[3] - size/(cur_cuts[1] - cur_cuts[0]) 
            elif last_cut_to_sample == 3:#upper boundary
                cur_cuts[last_cut_to_sample] = cur_cuts[2] + size/(cur_cuts[1]- cur_cuts[0])
            assert(np.abs((cur_cuts[1] - cur_cuts[0]) * (cur_cuts[3] - cur_cuts[2]) - size) < 1e-3) #make sure it has the size
            random_gate_flat_init.append(cur_cuts)
            if cur_cuts[last_cut_to_sample] < 0 or cur_cuts[last_cut_to_sample] > max_cut:
                print('meow')
                return self._get_random_init_nested_list_(size_mean, size_var, cut_var, min_size=min_size)
        print(random_gate_flat_init)
        return (self._convert_flattened_list_to_nested_(random_gate_flat_init))

    def _convert_flattened_list_to_nested_(self, random_gates):
        nested_list = \
            [
                [[u'SSC-H', random_gates[0][0], random_gates[0][1]], [u'CD45', random_gates[0][2], random_gates[0][3]]],
                [
                    [
                        [[u'FSC-A', random_gates[1][0], random_gates[1][1]], [u'SSC-A', random_gates[1][2], random_gates[1][3]]],
                        [
                            [
                                [[u'CD5', random_gates[2][0], random_gates[2][1]], [u'CD19', random_gates[2][2],  random_gates[2][3]]],
                                [
                                    [
                                        [[u'CD10', random_gates[3][0], random_gates[3][1]], [u'CD79b', random_gates[3][2], random_gates[3][3]]],
                                        []
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        return nested_list


    #get initializations in the middle of the plots
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
        return [[1003., 3011., 1024., 3071.],[1083., 3091., 1024., 3071.],[1023., 3069., 1024., 3072.],[1024., 3071., 1026, 3078.]]

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
        # self.x_list = [[x_pb1_idx, x_pb2_idx] for idx in range(n_samples)] # x_pb1_idx, x_pb2_idx are numpy arrays
        # self.y_list = [y_idx for idx in range(n_samples)]
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
        print("offset:", offset)
        print("scale:", scale)
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
        print(np.array(self.x_list).shape)
        print(np.array(self.y_list).shape)
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
        # self.x_list = [[x_pb1_idx, x_pb2_idx] for idx in range(n_samples)] # x_pb1_idx, x_pb2_idx are numpy arrays
        # self.y_list = [y_idx for idx in range(n_samples)]
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
