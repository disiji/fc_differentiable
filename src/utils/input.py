import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

import utils_load_data as dh
from bayes_gate_pytorch_sigmoid_trans import ReferenceTree

import os


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
        features = 'CD5', 'CD19', 'CD10', 'CD79b'
        features_full = ('FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3')
        self.hparams = hparams
        self.features = dict((i, features[i]) for i in range(len(features)))
        self.features_full = dict((i, features_full[i]) for i in range(len(features_full)))
        self.feature2id = dict((self.features[i], i) for i in self.features)

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
            with open(DATA_DIR + "filtered_4d_x_list.pkl", 'rb') as f:
                self.x_list = pickle.load(f)
            with open(DATA_DIR + 'y_list.pkl', 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, self.features_full)
            x_4d = dh.filter_cll_4d(x)
            with open(DATA_DIR + 'filtered_4d_x_list.pkl', 'wb') as f:
                pickle.dump(x_4d, f)
            with open(DATA_DIR + 'y_list.pkl', 'wb') as f:
                pickle.dump(y, f)
            self.x_list, self.y_list = x_4d, y

    def _get_reference_nested_list_(self):
        self.reference_nested_list = [
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
                [[u'CD5', 2000., 3000.], [u'CD19', 2000., 3000.]],
                [
                    [
                        [[u'CD10', 1000., 2000.], [u'CD79b', 1000., 2000.]],
                        []
                    ]
                ]
            ]

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list(self.x_list)
        print(self.features_full, self.features, self.feature2id)
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

        features_pb1 = 'CD5', 'CD19', 'CD10', 'CD79b'
        features_pb2 = 'CD5', 'CD19', 'CD10', 'CD79b'
        features_full_pb1 = ('FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3')
        features_full_pb2 = ('FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3')
        self.features_pb1 = dict((i, features_pb1[i]) for i in range(len(features_pb1)))
        self.features_pb2 = dict((i, features_pb2[i]) for i in range(len(features_pb2)))
        self.features_full_pb1 = dict((i, features_full_pb1[i]) for i in range(len(features_full_pb1)))
        self.features_full_pb2 = dict((i, features_full_pb2[i]) for i in range(len(features_full_pb2)))
        self.feature2id_pb1 = dict((self.features_pb1[i], i) for i in self.features_pb1)
        self.feature2id_pb2 = dict((self.features_pb2[i], i) for i in self.features_pb2)

        self.features = [self.features_pb1, self.features_pb2]
        self.features_full = [self.features_full_pb1, self.features_full_pb2]
        self.feature2id = [self.feature2id_pb1, self.feature2id_pb2]

        self._load_data_()
        self._get_reference_nested_list_()
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

    def _get_data_(self, convert_name_func, src_dir, sep='\t'):
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
        CYTOMETRY_DIR = [DATA_DIR + "PB1_whole_mqian/", DATA_DIR + "PB2_whole_mqian/"]
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
        
        #Load samples from both directories-all names must be in the same format.
        PB1_samples, PB1_ids = _get_data_(lambda x: x.split('_')[3], CYTOMETRY_DIR[0])
        PB2_samples, PB2_ids = _get_data_(lambda x: x.split('_')[3], CYTOMETRY_DIR[1])

        labels = pd.read_csv(DIAGONOSIS_FILENAME, sep='\t')[['SampleID', 'Diagnosis']]
        
        #Match ids and combine into one nested list-sort by order of PB1_ids
        matched_labels = []
        matched_samples = []
        for i1, idx1 in enumerate(PB1_ids):
            #several samples in PB2/PB1 mqian files arent in PB.txt
            if labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values.shape[0] == 0:
                continue
            i2 = PB2_ids.index(idx1)
            matched_samples.append([PB1_samples[i1], PB2_samples[i2]])
            print(labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values[0], idx1)
            matched_labels.append(labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values[0])
        
        with open('../data/Two_Panel', 'wb') as f:
            pickle.dump((matched_samples, matched_labels), f)

        return matched_samples, matched_labels



        # todo: load pb1 an pb2 data to x_list and write them to pickle files to avoid loading and filtering them everytime
        # self.x_list = [[x_pb1_idx, x_pb2_idx] for idx in range(n_samples)] # x_pb1_idx, x_pb2_idx are numpy arrays
        # self.y_list = [y_idx for idx in range(n_samples)]

    def _get_reference_nested_list_(self):
        # todo: extract reference tree for pb2 from the html file
        self.reference_nested_list = [
            [
                [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
                [
                    [
                        [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                        []
                    ]
                ]
            ],
            [
                [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
                [
                    [
                        [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                        []
                    ]
                ]
            ]]

    def _get_init_nested_list_(self):
        # todo: construct the init tree for pb2
        self.init_nested_list = \
            [[
                [[u'CD5', 2000., 3000.], [u'CD19', 2000., 3000.]],
                [
                    [
                        [[u'CD10', 1000., 2000.], [u'CD79b', 1000., 2000.]],
                        []
                    ]
                ]
            ],
                [
                    [[u'CD5', 2000., 3000.], [u'CD19', 2000., 3000.]],
                    [
                        [
                            [[u'CD10', 1000., 2000.], [u'CD79b', 1000., 2000.]],
                            []
                        ]
                    ]
                ]]

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list(self.x_list)
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
        self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(np.numpy(self.x_list), self.y_list,
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
    def _get_data_(convert_name_func, src_dir, sep='\t'):
        files = os.listdir(src_dir)
        accepted_files = np.array(files)
        all_data = []
        ids = []
        for f in accepted_files:
            all_data.append(pd.read_csv(os.path.join(src_dir, f), sep=sep).values)
            try:
                ids.append(int(convert_name_func(f))) #throws error if a file name needs to be edited
            except:
                print('File %s needs to be changed so the sample id is the fourth string when splitting on the underscore character' %f)
        return all_data, ids

    def _load_data_():
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR = [DATA_DIR + "PB1_whole_mqian/", DATA_DIR + "PB2_whole_mqian/"]
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
        
        #Load samples from both directories-all names must be in the same format.
        PB1_samples, PB1_ids = _get_data_(lambda x: x.split('_')[3], CYTOMETRY_DIR[0])
        PB2_samples, PB2_ids = _get_data_(lambda x: x.split('_')[3], CYTOMETRY_DIR[1])

        labels = pd.read_csv(DIAGONOSIS_FILENAME, sep='\t')[['SampleID', 'Diagnosis']]
        
        #Match ids and combine into one nested list-sort by order of PB1_ids
        matched_labels = []
        matched_samples = []
        for i1, idx1 in enumerate(PB1_ids):
            #several samples in PB2/PB1 mqian files arent in PB.txt
            if labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values.shape[0] == 0:
                continue
            i2 = PB2_ids.index(idx1)
            matched_samples.append([PB1_samples[i1], PB2_samples[i2]])
            print(labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values[0], idx1)
            matched_labels.append(labels.loc[labels['SampleID'] == idx1]['Diagnosis'].values[0])
        
        with open('../data/Two_Panel', 'wb') as f:
            pickle.dump((matched_samples, matched_labels), f)

        return matched_samples, matched_labels

    samples, labels = _load_data_()
    
