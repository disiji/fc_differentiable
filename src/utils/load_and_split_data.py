from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_load_data import *

# First process the data into a list of numpy arrays.

# CHANGE THE BELOW DIRECTORIES TO MATCH WHERE YOU DOWNLOADED THE DATA
# Directories containing the csv files

#raw_data_P1_dir = '../data/cll/PB1_whole_mqian'
#raw_data_P2_dir = '../data/cll/PB2_whole_mqian'
## File containing the diagnoses
#DIAGNOSIS_FILENAME = '../data/cll/PB.txt'
## Save paths for processed data
#SAVE_PATH = '../data/cll/8d_FINAL'
#SAVE_PATH_BOTH_PANELS = '../data/cll/Both_Panels_FINAL'

def save_and_preprocess_data_fcs(raw_fcs_data_P1_dir, raw_fcs_data_P2_dir, SAVE_PATH, SAVE_PATH_BOTH_PANELS):
    #raw_fcs_data_P1_dir_txt, raw_fcs_data_P2_dir_txt = generate_txt_files(raw_fcs_data_P1_dir, raw_fcs_data_P2_dir)
    DIAGNOSIS_FILENAME = os.path.join(raw_fcs_data_P1_dir, 'attachments/CLL_Diagnosis_forRevisedManuscript.txt')
    #save_and_preprocess_data_fcs(raw_fcs_data_P1_dir_txt,raw_fcs_data_P2_dir_txt, diagnosis_filename, SAVE_PATH, SAVE_PATH_BOTH_PANELS) 
    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.isdir(SAVE_PATH_BOTH_PANELS):
        os.makedirs(SAVE_PATH_BOTH_PANELS)
    random_state = 123
    dev_size = 0.32
    diagnosis_df = pd.read_csv(DIAGNOSIS_FILENAME, sep='\t')
    sorted_sample_ids = get_sample_ids_fcs(raw_fcs_data_P1_dir, diagnosis_df)

    # Load the single panel data
    #x_list, y_list= load_cll_data_1p_fcs(DIAGNOSIS_FILENAME, raw_fcs_data_P1_dir, ['FSC-A', 'FSC-H',  'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b'])
    x_list, y_list= load_cll_data_1p_fcs(DIAGNOSIS_FILENAME, raw_fcs_data_P1_dir, ['FSC-A', 'FSC-H',  'SSC-H', 'FITC-A', 'SSC-A', 'PerCP-Cy5-5-A', 'PE-Cy7-A', 'BV421-A', 'APC-A'])
    x_list =  filter_cll_8d_pb1(x_list)
    # Add sample ids so I have the sample ids after splitting
    x_list = [np.hstack([x, sorted_sample_ids[idx] * np.ones([x.shape[0], 1])]) for idx, x in enumerate(x_list)]
    # Load the two panel data
    x_list_2p , y_list_2p = load_cll_data_2p_fcs(DIAGNOSIS_FILENAME, raw_fcs_data_P1_dir, raw_fcs_data_P2_dir, ['FSC-A', 'FSC-H',  'SSC-H', 'FITC-A', 'SSC-A', 'PerCP-Cy5-5-A', 'PE-Cy7-A', 'BV421-A', 'APC-A'], ['FSC-A', 'FSC-H', 'FITC-A', 'APC-H7-A', 'SSC-A', 'PerCP-Cy5-5-A', 'PE-Cy7-A', 'BV605-A', 'BV510-A', 'PE-A', 'APC-A'])
    filtered_pb1 = filter_cll_8d_pb1([x[0] for x in x_list_2p])
    filtered_pb2 = filter_cll_10d_pb2([x[1] for x in x_list_2p])
    x_list_2p = [[filtered_pb1[i], filtered_pb2[i]] for i, x in enumerate(x_list_2p)]

    idxs = np.arange(len(x_list))

    x_val, x_dev, y_val, y_dev, x_val_2p, x_dev_2p = train_test_split(x_list, y_list, x_list_2p, test_size=dev_size, random_state=random_state, stratify= y_list)

    # Get the sample ids for dev/val
    dev_labels_after_splitting = [x[0, -1] for x in x_dev]
    val_labels_after_splitting = [x[0, -1] for x in x_val]
    print('Sample ids after splitting:', dev_labels_after_splitting, val_labels_after_splitting)

    # Remove sample ids so we only use cell marker data
    x_val = [x[:, 0:x.shape[1] - 1] for x in x_val]
    x_dev = [x[:, 0:x.shape[1] - 1] for x in x_dev]




    x_list_all = x_dev + x_val
    x_list_all_2p = x_dev_2p + x_val_2p
    y_list_all = y_dev + y_val
    sample_names_all = dev_labels_after_splitting +  val_labels_after_splitting

    # Now save the processed (still unormalized at this point) data
    with open(os.path.join(SAVE_PATH, 'x_all_8d_1p.pkl'), 'wb') as f:
        pickle.dump(x_list_all, f)
    with open(os.path.join(SAVE_PATH, 'y_all_8d_1p.pkl'), 'wb') as f:
        pickle.dump(y_list_all, f)
    with open(os.path.join(SAVE_PATH, 'x_val_8d_1p.pkl'),  'wb') as f:
        pickle.dump(x_val, f)
    with open(os.path.join(SAVE_PATH, 'x_dev_8d_1p.pkl'),  'wb') as f:
        pickle.dump(x_dev, f)
    with open(os.path.join(SAVE_PATH, 'y_val_8d_1p.pkl'),  'wb') as f:
        pickle.dump(y_val, f)
    with open(os.path.join(SAVE_PATH, 'y_dev_8d_1p.pkl'),  'wb') as f:
        pickle.dump(y_dev, f)

    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'x_all_2p.pkl'), 'wb') as f:
        pickle.dump(x_list_all_2p, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'y_all_2p.pkl'), 'wb') as f:
        pickle.dump(y_list_all, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'x_2p_dev.pkl'), 'wb') as f:
        pickle.dump(x_dev_2p, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'y_2p_dev.pkl'), 'wb') as f:
        pickle.dump(y_dev, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'x_2p_val.pkl'), 'wb') as f:
        pickle.dump(x_val_2p, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'y_2p_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    print('Sample ids at very end:', dev_labels_after_splitting, val_labels_after_splitting)

#def generate_txt_files(raw_fcs_data_P1_dir, raw_fcs_data_P2_dir):
#    generate_txt_files_dir(raw_fcs_data_P1_dir)
#    
#
#def generate_txt_files_dir(raw_fcs_data_dir):
#    raw_txt_data_dir = os.path.join(raw_fcs_data_P1_dir, 'converted_txt_files')
#    os.makedir(raw_txt_data_dir)
#    for filename in sorted(os.listdir(raw_fcs_data_dir)):
#        file_path = os.path.join(raw_fcs_data_dir, filename)
#        if os.isdir(file_path):
#            continue
#        file_df = convert_single_fcs_to_df(file_path)
#        file_df.to_csv(os.path.join(raw_txt_data_dir, file

def save_and_preprocess_data_txt(raw_data_P1_dir, raw_data_P2_dir, DIAGNOSIS_FILENAME, SAVE_PATH, SAVE_PATH_BOTH_PANELS):

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    if not os.path.isdir(SAVE_PATH_BOTH_PANELS):
        os.makedirs(SAVE_PATH_BOTH_PANELS)
    random_state = 123
    dev_size = 0.32
    diagnosis_df = pd.read_csv(DIAGNOSIS_FILENAME, sep='\t')
    sorted_sample_ids = get_sample_ids(raw_data_P1_dir, diagnosis_df)
    print(sorted_sample_ids)

    # Load the single panel data
    x_list, y_list= load_cll_data_1p(DIAGNOSIS_FILENAME, raw_data_P1_dir, ['FSC-A', 'FSC-H',  'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b'])
    x_list =  filter_cll_8d_pb1(x_list)
    # Add sample ids so I have the sample ids after splitting
    x_list = [np.hstack([x, sorted_sample_ids[idx] * np.ones([x.shape[0], 1])]) for idx, x in enumerate(x_list)]

    # Load the two panel data
    x_list_2p , y_list_2p = load_cll_data_2p(DIAGNOSIS_FILENAME, raw_data_P1_dir, raw_data_P2_dir, ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b'], ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD38', 'CD20', 'Anti-Lambda', 'Anti-Kappa'])
    filtered_pb1 = filter_cll_8d_pb1([x[0] for x in x_list_2p])
    filtered_pb2 = filter_cll_10d_pb2([x[1] for x in x_list_2p])
    x_list_2p = [[filtered_pb1[i], filtered_pb2[i]] for i, x in enumerate(x_list_2p)]

    idxs = np.arange(len(x_list))

    x_val, x_dev, y_val, y_dev, x_val_2p, x_dev_2p = train_test_split(x_list, y_list, x_list_2p, test_size=dev_size, random_state=random_state, stratify= y_list)

    # Get the sample ids for dev/val
    dev_labels_after_splitting = [x[0, -1] for x in x_dev]
    val_labels_after_splitting = [x[0, -1] for x in x_val]
    print('Sample ids after splitting:', dev_labels_after_splitting, val_labels_after_splitting)

    # Remove sample ids so we only use cell marker data
    x_val = [x[:, 0:x.shape[1] - 1] for x in x_val]
    x_dev = [x[:, 0:x.shape[1] - 1] for x in x_dev]

    # fix 24242 label to positive - has id 1
    assert(y_dev[1] == 0.)
    y_dev[1] = 1.

    # Remove id 5-discrepant result (29842)
    print(len(x_dev))
    del x_dev[5]
    del x_dev_2p[5]
    del y_dev[5]
    del dev_labels_after_splitting[5]

    # Change labels of 18726, and 29881 to match followup diagnoses
    idx1 = val_labels_after_splitting.index(18726)
    idx2 = val_labels_after_splitting.index(29881)
    y_val[idx1] = 1.
    y_val[idx2] = 1.


    #remove samples that have discrepant results at followup diagnosis
    bad_samples = [20268.0, 23172.0, 23184.0, 26129.0]
    cur_labels = val_labels_after_splitting
    for sample in bad_samples:
        sample_val_idx = cur_labels.index(sample)
        del x_val[sample_val_idx]
        del x_val_2p[sample_val_idx]
        del y_val[sample_val_idx]
        del cur_labels[sample_val_idx]

    cur_labels_val = cur_labels
    idx_18726 = cur_labels_val.index(18726)


    x_list_all = x_dev + x_val
    x_list_all_2p = x_dev_2p + x_val_2p
    y_list_all = y_dev + y_val
    sample_names_all = dev_labels_after_splitting +  cur_labels_val

    # Now save the processed (still unormalized at this point) data
    with open(os.path.join(SAVE_PATH, 'x_all_8d_1p.pkl'), 'wb') as f:
        pickle.dump(x_list_all, f)
    with open(os.path.join(SAVE_PATH, 'y_all_8d_1p.pkl'), 'wb') as f:
        pickle.dump(y_list_all, f)
    with open(os.path.join(SAVE_PATH, 'x_val_8d_1p.pkl'),  'wb') as f:
        pickle.dump(x_val, f)
    with open(os.path.join(SAVE_PATH, 'x_dev_8d_1p.pkl'),  'wb') as f:
        pickle.dump(x_dev, f)
    with open(os.path.join(SAVE_PATH, 'y_val_8d_1p.pkl'),  'wb') as f:
        pickle.dump(y_val, f)
    with open(os.path.join(SAVE_PATH, 'y_dev_8d_1p.pkl'),  'wb') as f:
        pickle.dump(y_dev, f)

    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'x_all_2p.pkl'), 'wb') as f:
        pickle.dump(x_list_all_2p, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'y_all_2p.pkl'), 'wb') as f:
        pickle.dump(y_list_all, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'x_2p_dev.pkl'), 'wb') as f:
        pickle.dump(x_dev_2p, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'y_2p_dev.pkl'), 'wb') as f:
        pickle.dump(y_dev, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'x_2p_val.pkl'), 'wb') as f:
        pickle.dump(x_val_2p, f)
    with open(os.path.join(SAVE_PATH_BOTH_PANELS, 'y_2p_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    print('Sample ids at very end:', dev_labels_after_splitting, cur_labels_val)
    print('shapes at very end for val:', [x.shape for x in x_val])


def get_sample_ids(data_dir, diagnosis_df):
    sorted_sample_ids = []
    for filename in sorted(os.listdir(data_dir)):
        sample_id = int(filename.split('_')[3])
        if sample_id in diagnosis_df['SampleID'].values:
            sorted_sample_ids.append(sample_id)
    sorted_sample_ids = np.array(sorted_sample_ids)
    return sorted_sample_ids

def get_sample_ids_fcs(data_dir, diagnosis_df):
    sorted_sample_ids = []
    for filename in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, filename)):
            continue
        #sample_id = int(filename.split('.')[0])
        sample_id = filename
        if sample_id in diagnosis_df['FileName'].values:
            sorted_sample_ids.append(sample_id)
    sorted_sample_ids = np.array([int(str(sample_id).split('.')[0]) for sample_id in sorted_sample_ids])
    return sorted_sample_ids
