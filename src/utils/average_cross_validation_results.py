import glob 
import os
import numpy as np
import torch

NUMBER_DIAGNOSTICS = 14
NUMBER_DIAGNOSTICS_FILES = 4
DIAGNOSTICS_HEADER = \
        [
            'acc_i, mean_feature_i_pos',
            'mean_feature_i_neg','log_loss_i',
            'neg_prop_reg_i','feature_diff_reg_i',
            'log_loss_f','neg_prop_reg_f','feature_diff_reg_f',
            'mean_feature_f_pos','mean_feature_f_neg',
            'expert_thresh_acc_f','wavg_thresh_acc_f','acc_f'
        ]
INIT_REG_GRID = [0.001, 0.01, 0.1, 1., 10.]
NEG_REG_GRID = [0.001, 0.01, 0.1, 1., 10.]
OUTPUT_PATH = '../../output'


LR_RESULTS_TO_AVG = \
    [
        'lr_CV_lr=0.0010',
        'lr_CV_lr=0.0050',
        'lr_CV_lr=0.0100',
        'lr_CV_lr=0.0500',
        'lr_CV_lr=0.1000',


    ]
REG_RESULTS_TO_AVG = \
    [
        'Reg_CV_neg_box_=0.0010_feat_diff=0.0010',
        'Reg_CV_neg_box_=0.0100_feat_diff=0.0010',
        'Reg_CV_neg_box_=0.1000_feat_diff=0.0010',
        'Reg_CV_neg_box_=1.0000_feat_diff=0.0010',
        'Reg_CV_neg_box_=0.0000_feat_diff=0.0010',
        'Reg_CV_neg_box_=0.0010_feat_diff=0.0100',
        'Reg_CV_neg_box_=0.0100_feat_diff=0.0100',
        'Reg_CV_neg_box_=0.1000_feat_diff=0.0100',
        'Reg_CV_neg_box_=1.0000_feat_diff=0.0100',
        'Reg_CV_neg_box_=0.0000_feat_diff=0.0100',
        'Reg_CV_neg_box_=0.0010_feat_diff=0.1000',
        'Reg_CV_neg_box_=0.0100_feat_diff=0.1000',
        'Reg_CV_neg_box_=0.1000_feat_diff=0.1000',
        'Reg_CV_neg_box_=1.0000_feat_diff=0.1000',
        'Reg_CV_neg_box_=0.0000_feat_diff=0.1000',
        'Reg_CV_neg_box_=0.0010_feat_diff=1.0000',
        'Reg_CV_neg_box_=0.0100_feat_diff=1.0000',
        'Reg_CV_neg_box_=0.1000_feat_diff=1.0000',
        'Reg_CV_neg_box_=1.0000_feat_diff=1.0000',
        'Reg_CV_neg_box_=0.0000_feat_diff=1.0000',
        'Reg_CV_neg_box_=0.0010_feat_diff=0.0000',
        'Reg_CV_neg_box_=0.0100_feat_diff=0.0000',
        'Reg_CV_neg_box_=0.1000_feat_diff=0.0000',
        'Reg_CV_neg_box_=1.0000_feat_diff=0.0000',
        'Reg_CV_neg_box_=0.0000_feat_diff=0.0000'
    ]   
#MODEL_RESULTS_TO_AVG

# Iterates through the saved CV results and averages the diagnostics
def average_diagnostics(cv_runs_exp_prefix, suffix_fmt_str, num_reruns, start_run_idxs_at_one=False):
    experiment_prefix_path = os.path.join(OUTPUT_PATH, cv_runs_exp_prefix)
    print(experiment_prefix_path + '*')
    if not start_run_idxs_at_one:
        experiment_result_dirs = \
            [
                experiment_prefix_path + suffix_fmt_str %(i)
                for i in range(num_reruns)
            ]
    else:
        experiment_result_dirs =\
            [

                experiment_prefix_path + suffix_fmt_str %(i + 1)
                for i in range(num_reruns)
            ]
    #experiment_result_dirs =  glob.glob(experiment_prefix_path + '\*')
    print(experiment_result_dirs)
    if cv_runs_exp_prefix == 'CV_just_dev/CV' or cv_runs_exp_prefix == 'CV_runs_normalization/CV':
        averaged_diagnostics = np.zeros([4, 12])
    else:
        averaged_diagnostics = np.zeros([4, NUMBER_DIAGNOSTICS])
    for experiment_dir in experiment_result_dirs:
        diagnostics_path = os.path.join(experiment_dir, 'Diagnostics')
        diagnostics_files = os.listdir(diagnostics_path)
        file_labels = []
        diagnostics_single_run = []
        for diagnostics_file in sorted(diagnostics_files):
            file_path = os.path.join(diagnostics_path, diagnostics_file)
            diagnostics = np.genfromtxt(file_path, delimiter=',')
            diagnostics_single_run.append(diagnostics)
            file_labels.append(diagnostics_file)
        # earlier runs had nan bug
        if np.sum(np.isnan(np.array(diagnostics_single_run))) == 0:
            averaged_diagnostics += np.array(diagnostics_single_run)
    averaged_diagnostics = averaged_diagnostics/len(experiment_result_dirs)
    print(averaged_diagnostics)
    return averaged_diagnostics, file_labels

def save_averaged_diagnostics(cv_runs_exp_prefix, averaged_diagnostics, labels, augmented_results=False):
    experiment_prefix_path = os.path.join(OUTPUT_PATH, cv_runs_exp_prefix)
    if augmented_results:
        average_save_dir = experiment_prefix_path + '_averaged_aug'
    else:
        average_save_dir = experiment_prefix_path + '_averaged'
    if not os.path.exists(average_save_dir):
        os.makedirs(average_save_dir)
    for l, label in enumerate(labels):
        savepath = os.path.join(average_save_dir, label)
        np.savetxt(savepath, averaged_diagnostics[l], delimiter=',', header=concat_str_list(DIAGNOSTICS_HEADER))

def concat_str_list(str_list):
    concat_str = ''
    for string in str_list:
        concat_str += string + ','
    return concat_str

def find_best_parameters_lr_CV(exp_runs_prefixes):
    best_params = {}
    best_params['model'] = {}
    best_params['model']['best_avg_acc'] = 0.
    best_params['expert_thresh_with_model'] = {}
    best_params['expert_thresh_with_model']['best_avg_acc'] = 0.
    for prefix in exp_runs_prefixes:
        split = prefix.split('=')
        lr = float(split[1][0:6])

        experiment_prefix_path = os.path.join(OUTPUT_PATH, prefix)
        average_save_dir = experiment_prefix_path + '_averaged'
        average_model_te_path = os.path.join(average_save_dir, 'te_diagnostics_model.csv')
        averages = np.genfromtxt(average_model_te_path)
        print('lr %.4f' %lr)
        
        print(averages[-1], 'model avg te across fold')
        print(averages[-3], 'model feats with expert thresh across folds')
        #parse out the reg terms

        if averages[-1] > best_params['model']['best_avg_acc']:
            best_params['model']['lr'] = lr
            best_params['model']['best_avg_acc'] = averages[-1]
        elif averages[-3] > best_params['expert_thresh_with_model']['best_avg_acc']:
            best_params['expert_thresh_with_model']['lr'] = lr
            best_params['expert_thresh_with_model']['best_avg_acc'] = averages[-3]
    print(best_params) 


def find_best_parameters_reg_CV(exp_runs_prefixes):
    best_params = {}
    best_params['model'] = {}
    best_params['model']['best_avg_acc'] = 0.
    best_params['expert_thresh_with_model'] = {}
    best_params['expert_thresh_with_model']['best_avg_acc'] = 0.
    for prefix in exp_runs_prefixes:
        split = prefix.split('=')
        neg_reg_chunk = split[1]
        if neg_reg_chunk[0:2] == '10':
            neg_reg = float(neg_reg_chunk[0:7])
        else:
            neg_reg = float(neg_reg_chunk[0:6])

        feat_diff_chunk = split[2]
        if feat_diff_chunk[0:2] == '10':
            feat_diff = float(feat_diff_chunk[0:7])
        else:
            feat_diff = float(feat_diff_chunk[0:6])
        # stopped early because results with this setting were
        # clearly bad and it was late
        if feat_diff == 10.:
            continue


        experiment_prefix_path = os.path.join(OUTPUT_PATH, prefix)
        average_save_dir = experiment_prefix_path + '_averaged'
        average_model_te_path = os.path.join(average_save_dir, 'te_diagnostics_model.csv')
        averages = np.genfromtxt(average_model_te_path)
        print('neg reg', neg_reg, 'init reg', feat_diff)
        print(averages[-1], 'model avg te across fold')
        print(averages[-3], 'model feats with expert thresh across folds')
        #parse out the reg terms

        if averages[-1] > best_params['model']['best_avg_acc']:
            best_params['model']['feat_diff'] = feat_diff
            best_params['model']['neg_reg'] = neg_reg
            best_params['model']['best_avg_acc'] = averages[-1]
        elif averages[-3] > best_params['expert_thresh_with_model']['best_avg_acc']:
            best_params['expert_thresh_with_model']['feat_diff'] = feat_diff
            best_params['expert_thresh_with_model']['neg_reg'] = neg_reg
            best_params['expert_thresh_with_model']['best_avg_acc'] = averages[-3]
    print(best_params) 



def average_and_save_diagnostics_multiple_experiments(exp_runs_prefixes, suffix_fmt_str, num_reruns):
    for exp in exp_runs_prefixes:
        averaged_diagnostics, file_labels = average_diagnostics(exp, suffix_fmt_str, num_reruns)
        save_averaged_diagnostics(exp, averaged_diagnostics, file_labels)

    


if __name__ == '__main__':
#    cv_runs_prefix = 'Reg_CV_neg_box_=0.0010_init_reg=0.0010'
    #cv_runs_prefix = 'Reg_CV_neg_box_=10.0000_init_reg=0.0010'
    #cv_runs_prefix = 'CV_just_dev/CV'
    #cv_runs_prefix =  'CV_runs_normalization/CV'
    #cv_runs_prefix = 'CV'
    cv_runs_prefix = 'CV_neg=0.001_diff=0.001'

    #averaged_diagnostics, file_labels = average_diagnostics(cv_runs_prefix, '_fold%d', 5)  #reg runs
    averaged_diagnostics, file_labels = average_diagnostics(cv_runs_prefix, '_seed%d', 9, start_run_idxs_at_one=True) #cv runs w/o augmenting tr data
    #averaged_diagnostics, file_labels = average_diagnostics(cv_runs_prefix, '_seed%d_augmented_with_dev', 14, start_run_idxs_at_one=True) #cv runs w/ augmenting tr data

    
    # save diagnositics
    #save_averaged_diagnostics(cv_runs_prefix, averaged_diagnostics, file_labels, augmented_results=False) #non augmented tr data
    #save_averaged_diagnostics(cv_runs_prefix, averaged_diagnostics, file_labels, augmented_results=True) #augmented tr data
    save_averaged_diagnostics(cv_runs_prefix, averaged_diagnostics, file_labels, augmented_results=False) #non augmented tr data

    # run to get best reg parameters printed to screen from Reg cv
    #average_and_save_diagnostics_multiple_experiments(REG_RESULTS_TO_AVG, '_fold%d', 5)
    #find_best_parameters_reg_CV(REG_RESULTS_TO_AVG)
    
    # best lr from CV
    #average_and_save_diagnostics_multiple_experiments(LR_RESULTS_TO_AVG, '_fold%d', 5)
    #find_best_parameters_lr_CV(LR_RESULTS_TO_AVG)

