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
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

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

    def _get_init_nested_list_(self):
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

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list(self.x_list)
        print(self.feature2id, offset, scale, self.reference_nested_list)
        self.reference_nested_list = dh.normalize_nested_tree(self.reference_nested_list, offset, scale,
                                                              self.feature2id)
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


class Cll8d1pInput(Cll4d1pInput):
    """
    apply FSC-A and FSC-H prefiltering gate and learn other gate locations
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.features = ['FSC-A', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b']
        self.features_full = ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3']
        self.feature2id = dict((self.features[i], i) for i in range(len(self.features)))

        self._load_data_()
        self._get_reference_nested_list_()
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

    def _load_data_(self):
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'

        # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
        # y: a list of labels; 1 is CLL, 0 is healthy
        if self.hparams['load_from_pickle']:
            with open(DATA_DIR + "filtered_8d_1p_x_list.pkl", 'rb') as f:
                self.x_list = pickle.load(f)
            with open(DATA_DIR + 'y_1p_list.pkl', 'rb') as f:
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
            ]

    def _get_init_nested_list_(self):
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
