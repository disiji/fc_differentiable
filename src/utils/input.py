import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

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


class Cll4dInput(CLLInputBase):
    def __init__(self, hparams):
        self.hparams = hparams
        self.features = ['CD5', 'CD19', 'CD10', 'CD79b']
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
            with open(DATA_DIR + "filtered_4d_1p_x_list.pkl", 'rb') as f:
                self.x_list = pickle.load(f)
            with open(DATA_DIR + 'y_1p_list.pkl', 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data_1p(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, self.features_full)
            x_4d = dh.filter_cll_4d_pb1(x)
            with open(DATA_DIR + 'filtered_4d_1p_x_list.pkl', 'wb') as f:
                pickle.dump(x_4d, f)
            with open(DATA_DIR + 'y_1p_list.pkl', 'wb') as f:
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
        self.init_nested_list = [ \
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
        print(self.features_full, self.features, self.feature2id, offset, scale)
        self.reference_nested_list = dh.normalize_nested_tree(self.reference_nested_list, offset, scale,
                                                              self.feature2id)
        self.init_nested_list = dh.normalize_nested_tree(self.init_nested_list, offset, scale, self.feature2id)

    def _construct_(self):
        self.reference_tree = ReferenceTree(self.reference_nested_list, self.feature2id)
        self.init_tree = ReferenceTree(self.init_nested_list, self.feature2id)
        if self.hparams['dafi_init']:
            self.init_tree = None

    def split(self, random_state=123):
        self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(self.x_list, self.y_list,
                                                                                test_size=self.hparams['test_size'],
                                                                                random_state=random_state)
        self.x_train = [torch.tensor(_, dtype=torch.float32) for _ in self.x_train]
        self.x_eval = [torch.tensor(_, dtype=torch.float32) for _ in self.x_eval]
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_eval = torch.tensor(self.y_eval, dtype=torch.float32)


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
        self._get_reference_nested_list_()
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [[torch.tensor(_[0], dtype=torch.float32), torch.tensor(_[1], dtype=torch.float32)] for _ in
                  self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)


    def _get_data_(self, convert_name_func, src_dir, col_names, sep='\t'):
        files = os.listdir(src_dir)
        accepted_files = np.array(files)
        all_data = []
        ids = []
        for f in accepted_files:
            if col_names == 'all':
                all_data.append(pd.read_csv(os.path.join(src_dir, f), sep=sep).values)
            else:
                all_data.append(pd.read_csv(os.path.join(src_dir, f), sep=sep)[col_names].values)
            try:
                ids.append(int(convert_name_func(f))) #throws error if a file name needs to be edited
            except:
                print('File %s needs to be changed so the sample id is the fourth string when splitting on the underscore character' %f)
        return all_data, ids
    
    

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
        print([x[0].shape[0] for x in self.x_list])
        print([x[1].shape[0] for x in self.x_list])

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
                    [[u'CD38', 0., 500.], [u'CD20', 100., 1400.]],
                    [
                        [
                            [[u'Anti-Kappa', 800., 2300.], [u'Anti-Lambda', 0., 400.]],
                            []
                        ],
                        [
                            [[u'Anti-Kappa', 0., 800.], [u'Anti-Lambda', 800., 2300.]],
                            []
                        ]
                    ]
                ]]

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list_multiple_panels(self.x_list)
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

if __name__ == '__main__':
    #Not sure how to init your objects rn so just copied pasted to test
    def _get_data_( convert_name_func, src_dir, col_names, sep='\t'):
        files = os.listdir(src_dir)
        accepted_files = np.array(files)
        all_data = []
        ids = []
        for f in accepted_files:
            if col_names == 'all':
                all_data.append(pd.read_csv(os.path.join(src_dir, f), sep=sep).values)
            else:
                all_data.append(pd.read_csv(os.path.join(src_dir, f), sep=sep)[col_names].values)
            try:
                ids.append(int(convert_name_func(f))) #throws error if a file name needs to be edited
            except:
                print('File %s needs to be changed so the sample id is the fourth string when splitting on the underscore character' %f)
        return all_data, ids
    
    

    def _load_data_():
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR = [DATA_DIR + "PB1_whole_mqian/", DATA_DIR + "PB2_whole_mqian/"]
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
        features_full_PB1 = ('FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3')
        features_full_PB2 = ('FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda', 'Anti-Kappa')

        #Load samples from both directories-all names must be in the same format.
        PB1_samples, PB1_ids = _get_data_(lambda x: x.split('_')[3], CYTOMETRY_DIR[0], list(features_full_PB1))
        PB2_samples, PB2_ids = _get_data_(lambda x: x.split('_')[3], CYTOMETRY_DIR[1], list(features_full_PB2))

        labels = pd.read_csv(DIAGONOSIS_FILENAME, sep='\t')[['SampleID', 'Diagnosis']]
        
        #Match ids and combine into one nested list-sort by order of PB1_ids
        matched_labels = []
        PB2_in_order = []
        for i1, idx1 in enumerate(PB1_ids):
            #several samples in PB2/PB1 mqian files arent in PB.txt, so ignore these
            if labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values.shape[0] == 0:
                continue
            i2 = PB2_ids.index(idx1)
            #matched_samples.append([PB1_samples[i1], PB2_samples[i2]])
            PB2_in_order.append(PB2_samples[i2])
            print(labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values[0], idx1)
            matched_labels.append(labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values[0])
        
        #now filter the data
        PB1_filtered = dh.filter_cll_4d_PB1(PB1_samples)
        PB2_filtered = dh.filter_cll_2d_PB2(PB2_in_order)

        matched_samples = [[PB1_sample, PB2_sample] for PB1_sample, PB2_sample in zip(PB1_filtered, PB2_filtered)]

        #for m,matched_pair in enumerate(matched_samples):
        #    print(matched_pair[0].shape[0])
        #    print(matched_pair[0][0:15])
        #    PB1 = dh.filter_cll_4d_PB1(matched_pair[0])
        #    PB2 = dh.filter_cll_4d_PB2(matched_pair[0])
        #    matched_samples[m] = [PB1, PB2]
        #    

        #Might want to convert here to tensors, and to make sure the datatypes are correct for the mains

        with open('../data/cll/Two_Panel.pkl', 'wb') as f:
            pickle.dump((matched_samples, matched_labels), f)

        return matched_samples, matched_labels

    samples, labels = _load_data_()
    
