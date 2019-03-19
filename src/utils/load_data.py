from __future__ import division
import pandas as pd
import os
import pickle

def load_cll_data(diagnosis_filename, cytometry_dir, features):
    X, y = [],[]
    feat_df = pd.read_csv(diagnosis_filename, sep='\t')
    for filename in os.listdir(cytometry_dir):
        sample_id = int(filename.split('_')[3])
        # filter out PB1 samples that we do not have diagnosis information about
        if sample_id in feat_df['SampleID'].values:
            X.append(pd.read_csv(os.path.join(cytometry_dir, filename), sep='\t')[features].values)
            y.append(feat_df.loc[feat_df['SampleID'] == sample_id]['Diagnosis'].values[0])
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
    if x1 <0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError("x1 or x2 or y1 or y2 is negative.")
    if dim1 > data.shape[1] or dim1 < 0 or dim2 > data.shape[1] or dim2 < 0:
        raise ValueError("dim1 and dim2 should be an int between 0 and data.shape[0].")
    if y1/x1 < y2/x2:
        raise ValueError("Slope of [(0,0), (x1, y1)] should be greater than the slope of [(0,0), (x2, y2)].")
    gradient = data[:,dim2] / data[:,dim1]
    idx = (gradient < y1/x1) & (gradient > y2/x2)
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
    if x1 <0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError("x1 or x2 or y1 or y2 is negative.")
    if dim1 > data.shape[1] or dim1 < 0 or dim2 > data.shape[1] or dim2 < 0:
        raise ValueError("dim1 and dim2 should be an int between 0 and data.shape[0].")
    if x1 > x2 or y1 > y2:
        raise ValueError("x2 should be greater than x1, y2 should be greater than y1.")
    idx = (data[:,dim1] > x1) & (data[:,dim1] < x2) & (data[:,dim2] > y1) & (data[:,dim2] < y2)
    return data[idx]


def filter_cll_4d(x_list):
    """

    :param x_list:
    :return:
    """
    idx = 3
    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 2, 3, 102, 921, 2048, 3891) for x in filtered_x_list]
    print('After second gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    filtered_x_list = [filter_rectangle(x, 0, 4, 921, 2150, 102, 921) for x in filtered_x_list]
    print('After third gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    # filtered_x_list = [filter_rectangle(x, 5, 6, 1638, 3891, 2150, 3891) for x in filtered_x_list]
    # print('After fourth gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    # filtered_x_list = [filter_rectangle(x, 7, 8, 0, 1228, 0, 1843) for x in filtered_x_list]
    # print('After fifth gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))
    filtered_x_list_4d = [x[:, 5:9] for x in filtered_x_list]

    return filtered_x_list_4d


def filter_cll(x_list):
    """

    :param x_list:
    :return:
    """
    idx = 3
    filtered_x_list = [filter_slope(x, 0, 1, 2048, 4096, 2048, 2560) for x in x_list]
    print('After first gate %d remain in sample %s' %(filtered_x_list[idx].shape[0], idx))

    return filtered_x_list


def normalize
