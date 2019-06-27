from __future__ import division

import os
import pickle

import numpy as np
import pandas as pd


def load_cll_data_1p(diagnosis_filename, cytometry_dir, features):
    X, y = [], []
    diagnosis_df = pd.read_csv(diagnosis_filename, sep='\t')
    for filename in sorted(os.listdir(cytometry_dir)):
        sample_id = int(filename.split('_')[3])
        # filter out PB1 samples that we do not have diagnosis information about
        if sample_id in diagnosis_df['SampleID'].values:
            X.append(pd.read_csv(os.path.join(cytometry_dir, filename), sep='\t')[features].values)
            y.append(diagnosis_df.loc[diagnosis_df['SampleID'] == sample_id]['Diagnosis'].values[0])
    d = {'no': 0, 'yes': 1}
    y = [d[_] for _ in y]
    return X, y


def load_cll_data_2p(diagnosis_filename, cytometry_dir_pb1, cytometry_dir_pb2, features_pb1, features_pb2):
    X, y = [], []
    diagnosis_df = pd.read_csv(diagnosis_filename, sep='\t')
    sample_id_list_pb2 = [int(filename.split('_')[3]) for filename in sorted(os.listdir(cytometry_dir_pb2))]
    id2filename_pb2 = dict(zip(sample_id_list_pb2, sorted(os.listdir(cytometry_dir_pb2))))
    for filename_pb1 in sorted(os.listdir(cytometry_dir_pb1)):
        sample_id = int(filename_pb1.split('_')[3])
        if sample_id in diagnosis_df['SampleID'].values and sample_id in sample_id_list_pb2:
            filename_pb2 = id2filename_pb2[sample_id]
            x_pb1 = pd.read_csv(os.path.join(cytometry_dir_pb1, filename_pb1), sep='\t')[features_pb1].values
            x_pb2 = pd.read_csv(os.path.join(cytometry_dir_pb2, filename_pb2), sep='\t')[features_pb2].values
            X.append([x_pb1, x_pb2])
            y.append(diagnosis_df.loc[diagnosis_df['SampleID'] == sample_id]['Diagnosis'].values[0])
    d = {'no': 0, 'yes': 1}
    y = [d[_] for _ in y]
    return X, y


def get_reference_tree(file):
    """
    load gating hierarchy from a pickle file to a nested list
    :param file: filename of the pickle file
    :return:
    """
    file = open(file, 'rb')
    return pickle.load(file)


def filter_slope(data, dim1, dim2, x1, x2, y1, y2):
    """
    return subset of datapoints in data that fall into the V slope formed by [(0,0),(x1, y1)] and [(0,0),(x2, y2)]
    :param data: np.array (n_datapoints, n_features)
    :param dim1: int
    :param dim2: int
    :param x1: float
    :param x2: float
    :param y1: float
    :param y2: float
    :return: (n_filtered_datapoints, n_features)
    """
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError("x1 or x2 or y1 or y2 is negative.")
    if dim1 > data.shape[1] or dim1 < 0 or dim2 > data.shape[1] or dim2 < 0:
        raise ValueError("dim1 and dim2 should be an int between 0 and data.shape[0].")
    if y1 / x1 < y2 / x2:
        raise ValueError("Slope of [(0,0), (x1, y1)] should be greater than the slope of [(0,0), (x2, y2)].")
    gradient = data[:, dim2] / data[:, dim1]
    idx = (gradient < y1 / x1) & (gradient > y2 / x2)
    return data[idx]


def filter_rectangle(data, dim1, dim2, x1, x2, y1, y2):
    """
    return subset of datapoints in data that fall into the rectangle formed by (x1, y1),(x2, y2),(x1, y2) and (x2, y1)
    :param data: np.array (n_datapoints, n_features)
    :param dim1: int
    :param dim2: int
    :param x1: float
    :param x2: float
    :param y1: float
    :param y2: float
    :return: (n_filtered_datapoints, n_features)
    """
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError("x1 or x2 or y1 or y2 is negative.")
    if dim1 > data.shape[1] or dim1 < 0 or dim2 > data.shape[1] or dim2 < 0:
        raise ValueError("dim1 and dim2 should be an int between 0 and data.shape[0].")
    if x1 > x2 or y1 > y2:
        raise ValueError("x2 should be greater than x1, y2 should be greater than y1.")
    idx = (data[:, dim1] > x1) & (data[:, dim1] < x2) & (data[:, dim2] > y1) & (data[:, dim2] < y2)
    return data[idx]


def filter_cll_4d_pb1(x_list):
    """

    :param x_list:
    :return:
    """
    idx = 3
    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 2, 3, 102, 921, 2048, 3891) for x in filtered_x_list]
    print('After second gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 0, 4, 921, 2150, 102, 921) for x in filtered_x_list]
    print('After third gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    # filtered_x_list = [filter_rectangle(x, 5, 6, 1638, 3891, 2150, 3891) for x in filtered_x_list]
    # print('After fourth gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    # filtered_x_list = [filter_rectangle(x, 7, 8, 0, 1228, 0, 1843) for x in filtered_x_list]
    # print('After fifth gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    filtered_x_list_4d = [x[:, 5:9] for x in filtered_x_list]

    return filtered_x_list_4d

def filter_cll_8d_pb1(x_list):
    """

    :param x_list:
    :return:
    """
    idx = 3
    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list_8d = [x[:, [0,2,3,4,5,6,7,8]] for x in filtered_x_list]

    return filtered_x_list_8d


def filter_cll_4d_pb2(x_list):
    """
     :param x_list: list of numpy arrays per sample
    :return: list of filtered numpy arrays per sample
    """
    idx = 3

    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 2, 3, 102, 921, 2048, 3891) for x in filtered_x_list]
    print('After second gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 0, 4, 921, 2150, 102, 921) for x in filtered_x_list]
    print('After third gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 5, 6, 1638, 3891, 2150, 3891) for x in filtered_x_list]
    print('After fourth gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    # filtered_x_list = [filter_rectangle(x, 7, 8, 0, 1740, 614, 2252) for x in filtered_x_list]
    # print('After fifth gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list_4d = [x_list[:, 7:11] for x_list in filtered_x_list]

    return filtered_x_list_4d


def filter_cll_10d_pb2(x_list):
    """
     :param x_list: list of numpy arrays per sample
    :return: list of filtered numpy arrays per sample
    """
    idx = 3

    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list_10d = [x_list[:, [0,2,3,4,5,6,7,8,9,10]] for x_list in filtered_x_list]

    return filtered_x_list_10d


def filter_cll_leaf_pb1(x_list):
    """

    :param x_list:
    :return:
    """
    idx = 3
    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 2, 3, 102, 921, 2048, 3891) for x in filtered_x_list]
    print('After second gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 0, 4, 921, 2150, 102, 921) for x in filtered_x_list]
    print('After third gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 5, 6, 1638, 3891, 2150, 3891) for x in filtered_x_list]
    print('After fourth gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 7, 8, 0, 1228, 0, 1843) for x in filtered_x_list]
    print('After fifth gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list_leaf = [x for x in filtered_x_list]

    return filtered_x_list_leaf


def filter_cll_leaf_pb2(x_list):
    """
     :param x_list: list of numpy arrays per sample
    :return: list of filtered numpy arrays per sample
    """
    idx = 3

    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 2, 3, 102, 921, 2048, 3891) for x in filtered_x_list]
    print('After second gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 0, 4, 921, 2150, 102, 921) for x in filtered_x_list]
    print('After third gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 5, 6, 1638, 3891, 2150, 3891) for x in filtered_x_list]
    print('After fourth gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 7, 8, 0, 1740, 614, 2252) for x in filtered_x_list]
    print('After fifth gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))
    filtered_x_list_leaf = [x for x in filtered_x_list]

    return filtered_x_list_leaf


def filter_cll(x_list):
    """

    :param x_list:
    :return:
    """
    idx = 3
    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first slope gate %d remain in sample %s' % (filtered_x_list[idx].shape[0], idx))

    return filtered_x_list


def normalize_x_list(x_list, offset=None, scale=None):
    """
    x_list = normalized_x_list * scale + offset;
    normalized_x_list = (x_list - offset) / scale
    :param x_list: a list of numpy array, each of shape (, n_cell_features)
    :param offset: a numpy array of shape (n_cell_features, )
    :param scale: a numpy array of shape (n_cell_featuers, )
    :return:
    """
    n_features = x_list[0].shape[1]
    if offset == None or scale == None:
        x_min = np.min(np.array([x.min(axis=0) if x.shape[0] > 0
                                 else [np.nan] * n_features for x in x_list]), axis=0)
        x_max = np.max(np.array([x.max(axis=0) if x.shape[0] > 0
                                 else [np.nan] * n_features for x in x_list]), axis=0)
        offset = x_min
        scale = x_max - x_min
    normalized_x_list = [(x - offset) / scale for x in x_list]
    return normalized_x_list, offset, scale


def normalize_x_list_multiple_panels(x_list):
    """

    :param x_list: a list of a list of numpy arrays, each numpy array is the fc measurements of one panel for one sample
    :param offset: a list of numpy arrays of shape (n_cell_features, ). The list is of length n_panels
    :param scale: a list of numpy arrays of shape (n_cell_features, ). The list is of length n_panels
    :return:
    """
    n_panels = len(x_list[0])
    x_list = list(map(list, zip(*x_list)))
    offset = [None] * n_panels
    scale = [None] * n_panels
    normalized_x_list = [None] * n_panels
    for panel_idx in range(n_panels):
        normalized_x_list[panel_idx], offset[panel_idx], scale[panel_idx] = normalize_x_list(x_list[panel_idx])
    normalized_x_list = list(map(list, zip(*normalized_x_list)))
    return normalized_x_list, offset, scale


def normalize_nested_tree(nested_tree, offset, scale, feature2id):
    """
    normalized_x = (x - offset) / scale
    :param nested_tree:
    :param offset: a numpy array of shape (n_cell_features, )
    :param scale: a numpy array of shape (n_cell_featuers, )
    :param feature2id: a dictionary that maps feature names to column idx
    :return:
    """
    if nested_tree == []:
        return []
    # normalize the root node
    gate = nested_tree[0]
    dim1, dim2 = feature2id[gate[0][0]], feature2id[gate[1][0]]
    gate[0][1] = (gate[0][1] - offset[dim1]) / scale[dim1]
    gate[0][2] = (gate[0][2] - offset[dim1]) / scale[dim1]
    gate[1][1] = (gate[1][1] - offset[dim2]) / scale[dim2]
    gate[1][2] = (gate[1][2] - offset[dim2]) / scale[dim2]

    return [gate, [normalize_nested_tree(child, offset, scale, feature2id)
                   for child in nested_tree[1]]]


if __name__ == '__main__':
    x_list = [[np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
              [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]]
    print(x_list)
    print(normalize_x_list_multiple_panels(x_list))
