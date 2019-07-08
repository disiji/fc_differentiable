import matplotlib
import pickle
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import numpy as np
column_names = ['random_state',
                'train_accuracy',
                'eval_accuracy',
                'overall_accuracy',
                'train_accuracy_dafi',
                'eval_accuracy_dafi',
                'overall_accuracy_dafi',
                'train_tracker.acc_opt'
                'eval_tracker.acc_opt',
                'train_logloss',
                'eval_logloss',
                'overall_logloss',
                'train_logloss_dafi',
                'eval_logloss_dafi',
                'overall_logloss_dafi',
                'train_auc',
                'eval_auc',
                'overall_auc',
                'train_auc_dafi',
                'eval_auc_dafi',
                'overall_auc_dafi',
                'train_brier_score',
                'eval_brier_score',
                'overall_brier_score',
                'train_brier_score_dafi',
                'eval_brier_score_dafi',
                'overall_brier_score_dafi',
                'run_time']

#  generate scatter plots of a method and dafi gates
metric_dict = ['accuracy', 'logloss', 'auc', 'brier_score']
# model_dict = ['default', 'dafi_init', 'dafi_regularization', 'default_non_alternate', 'emp_regularization_off',
#               'gate_size_regularization_off']
model_dict = ['default']


# load from csv
def scatter_vs_dafi_feature(dataname, method_name, metric_name, ax):
    filename = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name)
    if metric_name not in metric_dict:
        raise ValueError('%s is not in metric_dict.' % metric_name)
    df = pd.read_csv(filename, header=None, names=column_names)
    # stats test
    print(stats.ttest_rel(df['eval_%s' % metric_name],df['eval_%s_dafi' % metric_name]))
    print(stats.ks_2samp(df['eval_%s' % metric_name],df['eval_%s_dafi' % metric_name]))

    ax.scatter(df['eval_%s' % metric_name], df['eval_%s_dafi' % metric_name], s=5)
    ax.set_xlim(min(min(df['eval_%s' % metric_name]), min(df['eval_%s_dafi' % metric_name])),\
                max(max(df['eval_%s' % metric_name]), max(df['eval_%s_dafi' % metric_name])))
    ax.set_ylim(min(min(df['eval_%s' % metric_name]), min(df['eval_%s_dafi' % metric_name])),\
                max(max(df['eval_%s' % metric_name]), max(df['eval_%s_dafi' % metric_name])))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    return ax


def scatter_methods(dataname, method_name_1, method_name_2, metric_name):
    # todo: need to make df of different methods to plot of same length
    """

    :param dataname:
    :param method_name_1:
    :param method_name_2:
    :param metric_name:
    :return:
    """
    filename_1 = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name_1)
    filename_2 = '../output/%s/results_cll_4D.csv' % (dataname + '_' + method_name_2)
    if metric_name not in metric_dict:
        raise ValueError('%s is not in metric_dict.' % metric_name)
    df_1 = pd.read_csv(filename_1, header=None, names=column_names)
    df_2 = pd.read_csv(filename_2, header=None, names=column_names)

    figname_train = '../fig/%s/%s_vs_%s_%s_train.png' % (dataname, method_name_1, method_name_2, metric_name)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_1['train_%s' % metric_name], df_2['train_%s' % metric_name], s=5)
    ax.set_xlabel(method_name_1)
    ax.set_ylabel(method_name_2)
    if metric_name in ['accuracy', 'auc', 'brier_score']:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    elif metric_name == 'logloss':
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 5.0)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    fig.tight_layout()
    plt.savefig(figname_train)

    figname_test = '../fig/%s/%s_vs_%s_%s_test.png' % (dataname, method_name_1, method_name_2, metric_name)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_1['eval_%s' % metric_name], df_2['eval_%s' % metric_name], s=5)
    ax.set_xlabel(method_name_1)
    ax.set_ylabel(method_name_2)
    if metric_name in ['accuracy', 'auc', 'brier_score']:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.plot()
    elif metric_name == 'logloss':
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 5.0)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_title(metric_name)
    fig.tight_layout()
    plt.savefig(figname_test)

def combine_results_into_one_csv(directory_prefix, output_dir, y_data_path, corner_reg_grid=[0., .1, .2, .3, .4, .5], gate_size_reg_grid=[0., .1, .2, .3, .4, .5], decimal_points_in_dir_name=2):
    # Looks like the data is not shuffled assuming the test_size hparam is set to 0
    # Other wise this wont be matched up properly
    # I have two sets of results: one with corner reg 0->.5 and the other with corner reg
    # taking values in [.001, .050]
    str_identifiers = []
    features_list = []
    avg_results_dicts = []
    weights_list = []
    for c, corner_reg in enumerate(corner_reg_grid):
        for gate_size_reg in gate_size_reg_grid:
            if directory_prefix == '../output/logreg_to_conv_grid_search': 
                if decimal_points_in_dir_name == 2:
                    str_identifier = '_corner=%.2f_gate_size=%.2f' % (corner_reg, gate_size_reg)
                else:
                    str_identifier = '_corner=%.3f_gate_size=%.3f' % (corner_reg, gate_size_reg)
            elif directory_prefix == '../output/two_phase_logreg_to_conv_grid_search_gate_size=':
                str_identifier = '%.2f' %gate_size_reg
            directory = directory_prefix + str_identifier
            #include Dafi results as the first column
            if c == 0:
#                dafi_features = get_features(os.path.join(directory, 'features_dafi.csv'))
#                str_identifiers.append('DAFI')
#                features_list.append(dafi_features)
                #get DAFI outputs here
#                avg_results_dicts.append(None)
#                weights_list.append(None)
                pass

            avg_results_dict = avg_results(os.path.join(directory, 'results_cll_4D.csv'))
            features = get_features(os.path.join(directory, 'features_model.csv'))
            weights = get_weights(os.path.join(directory, 'model_classifier_weights.csv'))
             
            avg_results_dicts.append(avg_results_dict)
            features_list.append(features)
            weights_list.append(weights)
            
            str_identifiers.append(str_identifier)
    with open(y_data_path, 'rb') as f:
        labels = pickle.load(f)
    write_concatenated_results(output_dir, avg_results_dicts, features_list, weights_list, str_identifiers, labels)
    fig, axes = plt.subplots(1, 2, sharey=True)
    feats_noreg = np.array(features_list[0])
    labels = np.array(labels)
    features_pos = feats_noreg[labels == 0]
    features_neg = feats_noreg[labels ==1]
    axes[0].boxplot(features_pos)
    axes[1].boxplot(features_neg)
    plt.savefig(os.path.join(output_dir, 'feats_box_plot_noreg.png')) 

def write_concatenated_results(output_dir, avg_results_dicts, features_list, weights_list, str_identifiers, labels):
    with open(os.path.join(output_dir, 'concatenated_results.csv'), 'w') as f:
        #get the column labels which correspond to different settings
        col_labels = ''
        for s,string in enumerate(str_identifiers):
            if s == len(str_identifiers) - 1:
                col_labels += string + ', label' + '\n'
                continue
            col_labels += string + ','
        f.write(col_labels)
        num_samples = len(features_list[0]) #each feature is one sample in this case
        for row in range(num_samples):
            for column in range(len(features_list)):
                feature = features_list[column][row]
                if column == len(features_list) - 1:
                    f.write(str(feature) +  ',%s' %(labels[row]) + '\n') #assumes labels are matched up
                    continue
                f.write(str(feature) + ',')

    
                
        
                
            

def get_weights(weights_path):
    with open(weights_path, 'r') as f:
        last_run_weights = f.readlines()[-1][0:-2] #get rid of newline char
        last_run_weights = [float(last_run_weights.split(',')[0]), float(last_run_weights.split(',')[1])]
    return last_run_weights

def get_features(features_path):
    with open(features_path, 'r') as f:
        lines = f.readlines()
        # only giving Padhraic the last run for now
        # have 35 features/samples, and the last line is blank
        lines_last_run = lines[-36:-1]
        feats = [float(feat_str[0:-1]) for feat_str in lines_last_run]
        
    return feats
 
def avg_results(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
        #only want odd lines which have actual integers
        lines_with_results = [line.split(',') for l, line in enumerate(lines) if l % 2 == 1]
        accs = [float(split_line[1]) for split_line in lines_with_results]
        log_losses = [float(split_line[2]) for split_line in lines_with_results]
        
    return {'avg_acc' : sum(accs)/len(accs), 
            'avg_log_loss' :sum(log_losses)/len(log_losses)}
          
    


if __name__ == '__main__':
    
    #combine_results_into_one_csv('../output/logreg_to_conv_grid_search', '../output/agg_results_logreg_to_conv_gs1', '../data/cll/y_dev_4d_1p.pkl')
    #combine_results_into_one_csv('../output/logreg_to_conv_grid_search', '../output/agg_results_logreg_to_conv_gs2', '../data/cll/y_dev_4d_1p.pkl', corner_reg_grid=[0.001, 0.050], gate_size_reg_grid=[0.25, 0.5], decimal_points_in_dir_name=3)
    combine_results_into_one_csv('../output/two_phase_logreg_to_conv_grid_search_gate_size=', '../output/agg_results_two_phase', '../data/cll/y_dev_4d_1p.pkl', corner_reg_grid=[0.00], gate_size_reg_grid= [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.])
    
    

    #dataname = 'cll_4d_1p'
    #for method_name in model_dict:
    #    figname = '../output/%s/comparison.pdf' % (dataname + '_' + method_name)
    #    f, axarr = plt.subplots(1, len(metric_dict), figsize=(10, 2))
    #    for i, metric_name in enumerate(metric_dict):
    #        axarr[i] = scatter_vs_dafi_feature(dataname, method_name, metric_name, axarr[i])
    #        axarr[i].set_xlabel('Model gates')
    #    axarr[0].set_ylabel('Expert gates')
    #    f.savefig(figname, bbox_inches='tight')


    # for i in range(len(model_dict)):
    #     for j in range(i + 1, len(model_dict)):
    #         for metric_name in metric_dict:
    #             scatter_methods(dataname, model_dict[i], model_dict[j], metric_name)
