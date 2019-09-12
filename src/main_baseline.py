import pandas as pd
import pickle
import torch
from sklearn.model_selection import KFold
import os
import csv
from utils.BaselineParamsParser import BaselineParamsParser
from utils.input import Cll8d1pInput
from sklearn.metrics import roc_auc_score
from utils.Flowsom import Flowsom
from utils.KMeans import KMeans
import time
import numpy as np
import yaml
from result_analysis import make_and_write_catted_data
SEEDS = np.concatenate([np.arange(51, 72), np.arange(29) + 1], axis=0)

NUM_SEEDS = 10

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 10000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.0,
    'negative_proportion_default': 0.0001,
    'positive_box_penalty': 0.0,
    'corner_penalty': .0,
    'feature_diff_penalty': 0.,
    'gate_size_penalty': .0,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 100,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.05,
    'batch_size': 10,
    'n_epoch': 1000, 
    'seven_epochs_for_gate_motion_plot': [0, 50, 100, 200, 300, 400, 500],
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 2,
    'init_type': 'random_corner',
    'corner_init_deterministic_size': .75,
    'train_alternate': True,
    'run_logistic_to_convergence': False,
    'output': {
        'type': 'full'
    },
    'annealing': {
        'anneal_logistic_k': False,
        'final_k': 1000,
        'init_k': 1
    },
    'two_phase_training': {
        'turn_on': False,
        'num_only_log_loss_epochs': 50
    },
    'plot_params':{
        'figsize': [10, 10],
        'marker_size': .01,
    },
    'use_out_of_sample_eval_data': False,
    'dictionary_is_broken': True
}

DEV_DATA_PATHS = {
    'X': '../data/cll/8d_FINAL/x_dev_8d_1p.pkl',  
    'Y': '../data/cll/8d_FINAL/y_dev_8d_1p.pkl'
    }

def baseline_plot_main(hparams_for_input, path_to_model_params):
    start = time.time()
    parser = BaselineParamsParser(path_to_model_params)
    model_params = parser.parse_params()

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])
    default_exp_name = hparams['experiment_name']
    random_state = 1 # to match the seed used for the model cv runs
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    #folds = KFold(n_splits=hparams['n_folds_for_cluster_CV'])
    #avgs_fs = []
    #avgs_km = []

    tr_accs_fs = []
    tr_accs_km = []
    eval_accs_fs = []
    eval_accs_km = []
    for num_clusters in hparams['num_clusters_grid']:
        print('Num clusters: %d' %num_clusters)
        model_params['flowsom_params']['meta_cluster_params']['min_k'] = num_clusters
        model_params['flowsom_params']['meta_cluster_params']['max_k'] = num_clusters
        
        model_params['kmeans_params']['num_clusters'] = num_clusters
    
        default_exp_name_for_outer_loop = default_exp_name + '_num_clusters=%d' %(num_clusters)

        cll_1p_full_input = Cll8d1pInput(hparams, random_state=random_state)
        #avg_te_acc_fs_cur = 0
        #avg_te_acc_km_cur = 0
        #best_acc_fs = 0
        #best_acc_km = 0
        #best_k_km = -1
        #best_k_fs = -1

        #for fold_idx, (tr_idxs, te_idxs) in enumerate(folds.split(input_no_split.x_list)):
        start_time = time.time()
        hparams['experiment_name'] = default_exp_name_for_outer_loop
#        hparams['experiment_name'] = default_exp_name_for_outer_loop + '_fold%d' %fold_idx
        if not os.path.exists('../output/%s' % hparams['experiment_name']):
            os.makedirs('../output/%s' % hparams['experiment_name'])
        savedir = '../output/%s/' %hparams['experiment_name']
        if not(os.path.exists(savedir)):
            os.mkdir(savedir)
        

        save_files_for_flowsom(cll_1p_full_input, './temp_catted_data', model_params['flowsom_params']['num_to_subsample'], num_clusters)

        model_flowsom = Flowsom(
                    './temp_catted_data_for_flowsom.csv',
                    model_params['flowsom_params'],
                    model_params['columns_to_cluster']
            )
        #if IGNORE_KMEANS:
        #    model_params['kmeans_params']['num_clusters'] = 2
        model_kmeans = \
                KMeans(
                    './temp_catted_data_for_flowsom.csv',
                    model_params['kmeans_params'],
                    random_state=model_params['random_seed']
                )
        tr_acc_flowsom = fit_model_and_get_tr_acc(model_flowsom, 'flowsom')
        tr_acc_kmeans = fit_model_and_get_tr_acc(model_kmeans, 'kmeans')

        x_eval = [x.cpu().detach().numpy() for x in cll_1p_full_input.x_eval]
        x_eval_for_flowsom = [x[np.random.permutation(x.shape[0])][0:model_params['flowsom_params']['num_to_subsample']] for x in x_eval]
        y_eval = cll_1p_full_input.y_eval.cpu().detach().numpy()
       
        print('eval for flowsom:')
        preds_flowsom, eval_acc_flowsom = get_preds_and_eval_acc(model_flowsom, 'flowsom', x_eval_for_flowsom, y_eval)
        print('eval for kmeans:')
        preds_kmeans, eval_acc_kmeans = get_preds_and_eval_acc(model_kmeans, 'kmeans', x_eval, y_eval)
        #avg_te_acc_fs_cur += eval_acc_flowsom
        #avg_te_acc_km_cur += eval_acc_kmeans
        #print('time for fold %d: %d' %(fold_idx, time.time() - start_time))
        tr_accs_km.append(tr_acc_kmeans)
        tr_accs_fs.append(tr_acc_flowsom)

        eval_accs_fs.append(eval_acc_flowsom)
        eval_accs_km.append(eval_acc_kmeans)
        print('time for num_clusters %d: %.4f' %(num_clusters, time.time() - start_time))
        print('Kmeans tr accs so far:', tr_accs_km)
        print('Flowsom tr accs so far:', tr_accs_fs)
        print('Kmeans te accs so far:', eval_accs_km)
        print('Flowsom te accs so far:', eval_accs_fs)
        clean_up_temp_files('./temp_catted_data', num_clusters)
    np.savetxt('../output/kmeans_tr_accs_for_plot.csv', tr_accs_km, delimiter=',')
    np.savetxt('../output/flowsom_tr_accs_for_plot.csv', tr_accs_fs, delimiter=',')
    np.savetxt('../output/kmeans_eval_accs_for_plot.csv', eval_accs_km, delimiter=',')
    np.savetxt('../output/flowsom_eval_accs_for_plot.csv', eval_accs_fs, delimiter=',')
   # avg_te_acc_fs_cur = avg_te_acc_fs_cur/hparams['n_folds_for_cluster_CV']
   # avg_te_acc_km_cur = avg_te_acc_km_cur/hparams['n_folds_for_cluster_CV']
   # avgs_fs.append(avg_te_acc_fs_cur)
   # avgs_km.append(avg_te_acc_km_cur)
   # if avg_te_acc_fs_cur > best_acc_fs:
   #     best_acc_fs = avg_te_acc_fs_cur
   #     best_k_fs = num_clusters
   # if avg_te_acc_km_cur > best_acc_km:
   #     best_acc_km = avg_te_acc_km_cur
   #     best_k_km = num_clusters
   # print('Current best k for flowsom: %d with best avg acc: %.4f' %(best_k_fs, best_acc_fs))
   # print('Current best k for kmeans: %d with best avg acc: %.4f' %(best_k_km, best_acc_km))
   # print('logreg penalty is %.4f for flowsom, num clusters is %d' %(l1_penalty, num_clusters))
   # print('avgs flowsom', avgs_fs)
   # print('avgs kmeans', avgs_km)

def CV_dev_main(hparams_for_input, path_to_model_params, ignore_kmeans=False):
    start = time.time()
    parser = BaselineParamsParser(path_to_model_params)
    model_params = parser.parse_params()

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])
    default_exp_name = hparams['experiment_name']
    random_state = 1 # to match the seed used for the model cv runs
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    folds = KFold(n_splits=hparams['n_folds_for_cluster_CV'])
    avgs_fs = []
    avgs_km = []
    for l1_penalty in hparams['l1_logreg_penalty_grid']:
        model_params['flowsom_params']['L1_logreg_penalty'] = l1_penalty
        print('L1 penalty for flowsom: %.4f' %l1_penalty)
        for num_clusters in hparams['num_clusters_grid']:
            print('Num clusters: %d' %num_clusters)
            model_params['flowsom_params']['meta_cluster_params']['min_k'] = num_clusters
            model_params['flowsom_params']['meta_cluster_params']['max_k'] = num_clusters
            
            model_params['kmeans_params']['num_clusters'] = num_clusters
        
            default_exp_name_for_outer_loop = default_exp_name + '_num_clusters=%d' %(num_clusters)

            input_no_split = Cll8d1pInput(hparams)
            avg_te_acc_fs_cur = 0
            avg_te_acc_km_cur = 0
            best_acc_fs = 0
            best_acc_km = 0
            best_k_km = -1
            best_k_fs = -1

            for fold_idx, (tr_idxs, te_idxs) in enumerate(folds.split(input_no_split.x_list)):
                start_time = time.time()
                cll_1p_full_input = Cll8d1pInput(hparams, split_fold_idxs=[tr_idxs, te_idxs])
                hparams['experiment_name'] = default_exp_name_for_outer_loop + '_fold%d' %fold_idx
                if not os.path.exists('../output/%s' % hparams['experiment_name']):
                    os.makedirs('../output/%s' % hparams['experiment_name'])
                savedir = '../output/%s/' %hparams['experiment_name']
                if not(os.path.exists(savedir)):
                    os.mkdir(savedir)
                

                save_files_for_flowsom(cll_1p_full_input, './temp_catted_data', model_params['flowsom_params']['num_to_subsample'], fold_idx)

                model_flowsom = Flowsom(
                            './temp_catted_data_for_flowsom.csv',
                            model_params['flowsom_params'],
                            model_params['columns_to_cluster']
                    )
                if ignore_kmeans:
                    model_params['kmeans_params']['num_clusters'] = 2
                model_kmeans = \
                        KMeans(
                            './temp_catted_data.csv',
                            model_params['kmeans_params'],
                            random_state=model_params['random_seed']
                        )
                tr_acc_flowsom = fit_model_and_get_tr_acc(model_flowsom, 'flowsom')
                tr_acc_kmeans = fit_model_and_get_tr_acc(model_kmeans, 'kmeans')

                x_eval = [x.cpu().detach().numpy() for x in cll_1p_full_input.x_eval]
                x_eval_for_flowsom = [x[np.random.permutation(x.shape[0])][0:model_params['flowsom_params']['num_to_subsample']] for x in x_eval]
                y_eval = cll_1p_full_input.y_eval.cpu().detach().numpy()
               
                print('eval for flowsom:')
                preds_flowsom, eval_acc_flowsom = get_preds_and_eval_acc(model_flowsom, 'flowsom', x_eval_for_flowsom, y_eval)
                print('eval for kmeans:')
                preds_kmeans, eval_acc_kmeans = get_preds_and_eval_acc(model_kmeans, 'kmeans', x_eval, y_eval)
                avg_te_acc_fs_cur += eval_acc_flowsom
                avg_te_acc_km_cur += eval_acc_kmeans
                print('time for fold %d: %d' %(fold_idx, time.time() - start_time))
            avg_te_acc_fs_cur = avg_te_acc_fs_cur/hparams['n_folds_for_cluster_CV']
            avg_te_acc_km_cur = avg_te_acc_km_cur/hparams['n_folds_for_cluster_CV']
            avgs_fs.append(avg_te_acc_fs_cur)
            avgs_km.append(avg_te_acc_km_cur)
            if avg_te_acc_fs_cur > best_acc_fs:
                best_acc_fs = avg_te_acc_fs_cur
                best_k_fs = num_clusters
            if avg_te_acc_km_cur > best_acc_km:
                best_acc_km = avg_te_acc_km_cur
                best_k_km = num_clusters
            print('Current best k for flowsom: %d with best avg acc: %.4f' %(best_k_fs, best_acc_fs))
            print('Current best k for kmeans: %d with best avg acc: %.4f' %(best_k_km, best_acc_km))
            print('logreg penalty is %.4f for flowsom, num clusters is %d' %(l1_penalty, num_clusters))
            print('avgs flowsom', avgs_fs)
            print('avgs kmeans', avgs_km)


def runs_50_main(hparams_for_input, path_to_model_params):

    start = time.time()
    parser = BaselineParamsParser(path_to_model_params)
    model_params = parser.parse_params()
    if hparams_for_input['use_model_CV_seeds']:
        seeds = SEEDS
    else:
        seeds = np.arange(0, NUM_SEEDS)
    km1_tr_accs = []
    km1_tr_accs_aug = []
    km2_tr_accs = []
    km2_tr_accs_aug = []
    for s, random_seed in  enumerate(seeds):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        # using input object to normalize, and split data
        input = Cll8d1pInput(hparams_for_input, random_state=random_seed)
        input_aug = Cll8d1pInput(hparams_for_input, random_state=random_seed, augment_data_paths=DEV_DATA_PATHS) 
        # now we have to save the files into pickles, and process them into pooled csv for the flowsom code
        save_files_for_flowsom(input, './temp_catted_data', model_params['flowsom_params']['num_to_subsample'], random_seed)
        save_files_for_flowsom(input_aug, './temp_catted_data_aug', model_params['flowsom_params']['num_to_subsample'], random_seed)

        # abstract the following into a factory to make
        # cleaner
#        if params['clustering_type'] == 'flowsom':
     #   model_flowsom = Flowsom(
     #               './temp_catted_data_for_flowsom.csv',
     #               model_params['flowsom_params'],
     #               model_params['columns_to_cluster']
     #       )
     #   model_flowsom_aug = Flowsom(
     #               './temp_catted_data_aug_for_flowsom.csv',
     #               model_params['flowsom_params'],
     #               model_params['columns_to_cluster']
     #       )
        model_kmeans1 = \
                KMeans(
                    './temp_catted_data.csv',
                    model_params['kmeans_params']['model1'],
                    random_state=random_seed
                )
        model_kmeans1_aug = \
                KMeans(
                    './temp_catted_data_aug.csv',
                    model_params['kmeans_params']['model1'],
                    random_state=random_seed
                )

        model_kmeans2 = \
                KMeans(
                    './temp_catted_data.csv',
                    model_params['kmeans_params']['model2'],
                    random_state=random_seed
                )
        model_kmeans2_aug = \
                KMeans(
                    './temp_catted_data_aug.csv',
                    model_params['kmeans_params']['model2'],
                    random_state=random_seed
                )
       # else:
        #    raise ValueError('Model type not recognized')
        #tr_acc_flowsom = fit_model_and_get_tr_acc(model_flowsom, 'flowsom')
        #tr_acc_flowsom_aug = fit_model_and_get_tr_acc(model_flowsom_aug, 'flowsom aug')
        tr_acc_kmeans1 = fit_model_and_get_tr_acc(model_kmeans1, 'kmeans')
        tr_acc_kmeans1_aug = fit_model_and_get_tr_acc(model_kmeans1_aug, 'kmeans aug')
        tr_acc_kmeans2 = fit_model_and_get_tr_acc(model_kmeans2, 'kmeans')
        tr_acc_kmeans2_aug = fit_model_and_get_tr_acc(model_kmeans2_aug, 'kmeans aug')
       

        km1_tr_accs.append(tr_acc_kmeans1)
        km1_tr_accs_aug.append(tr_acc_kmeans1_aug)
        km2_tr_accs.append(tr_acc_kmeans2)
        km2_tr_accs_aug.append(tr_acc_kmeans2_aug)


        x_eval = [x.cpu().detach().numpy() for x in input.x_eval]
        y_eval = input.y_eval.cpu().detach().numpy()
        


        #preds_flowsom, eval_acc_flowsom = get_preds_and_eval_acc(model_flowsom, 'flowsom', x_eval, y_eval)
        #preds_flowsom_aug, eval_acc_flowsom_aug = get_preds_and_eval_acc(model_flowsom_aug, 'flowsom aug', x_eval, y_eval)
        preds_kmeans1, eval_acc_kmeans1 = get_preds_and_eval_acc(model_kmeans1, 'kmeans1', x_eval, y_eval)
        preds_kmeans1_aug, eval_acc_kmeans1_aug = get_preds_and_eval_acc(model_kmeans1_aug, 'kmeans1 aug', x_eval, y_eval)
        preds_kmeans2, eval_acc_kmeans2 = get_preds_and_eval_acc(model_kmeans2, 'kmeans2', x_eval, y_eval)
        preds_kmeans2_aug, eval_acc_kmeans2_aug = get_preds_and_eval_acc(model_kmeans2_aug, 'kmeans2 aug', x_eval, y_eval)

        
        clean_up_temp_files('./temp_catted_data', random_seed)
      #  model_dict = {
      #      'fs': model_flowsom,
      #      'fs_aug': model_flowsom_aug,
      #      'km': model_kmeans,
      #      'km_aug': model_kmeans_aug
      #  }


        # not running flowsom for cv runs, so put same output from
        # kmeans here in order to reuse code written for both
        # produced flowsom columns will be meaningless!
        preds_and_acc_dict = {
            'km1': [preds_kmeans1, eval_acc_kmeans1],
            'km1_aug': [preds_kmeans1_aug, eval_acc_kmeans1_aug],
            'km2': [preds_kmeans2, eval_acc_kmeans2],
            'km2_aug': [preds_kmeans2_aug, eval_acc_kmeans2_aug]
        }

        if s == 0:
            current_output = update_and_write_current_output(preds_and_acc_dict, y_eval, random_seed, '../output/%s/' %hparams_for_input['experiment_name'])
        else:
            current_output = update_and_write_current_output(preds_and_acc_dict, y_eval, random_seed, '../output/%s/' %hparams_for_input['experiment_name'], old_output=current_output)


        print('Total time taken for seed %d: %d seconds' %(random_seed, time.time() - start))
    header = 'km1, km1_aug, km2, km2_aug'
    tr_accs_results = np.concatenate( 
        [
            SEEDS.reshape(-1, 1),
            np.array(km1_tr_accs).reshape(-1, 1), np.array(km1_tr_accs_aug).reshape(-1, 1),
            np.array(km2_tr_accs).reshape(-1, 1), np.array(km2_tr_accs_aug).reshape(-1, 1)
        ], axis=1
    )
    savepath = '../output/%s/tr_accs_results_across_seeds.csv' %hparams_for_input['experiment_name']
    np.savetxt(savepath, tr_accs_results, fmt='%.5f', header=header)



def update_and_write_current_output(preds_and_acc_dict, y_eval, seed, saveprefix, old_output=None):
    auc_fs, auc_fs_aug, auc_km, auc_km_aug = get_aucs(preds_and_acc_dict, y_eval)
    keys = [key for key in preds_and_acc_dict.keys()]
    # assumes four different models in the dict
    cur_run_row_accs = np.array(
        [seed,
        preds_and_acc_dict[keys[0]][1],      
        preds_and_acc_dict[keys[1]][1],            
        preds_and_acc_dict[keys[2]][1],            
        preds_and_acc_dict[keys[3]][1]]
    ).reshape(-1, 1)

    cur_run_row_aucs = np.array(
        [seed,
        auc_fs, auc_fs_aug,
        auc_km, auc_km_aug]
    ).reshape(-1, 1)

    header = 'seed,'
    for key in preds_and_acc_dict.keys():
        header += key + ','
    header = header[0:-1] #remove extra comma
    fmt_str = '%.6f'
    if old_output is None:
        np.savetxt(saveprefix + 'accs.csv', cur_run_row_accs, header=header, fmt=fmt_str)
        np.savetxt(saveprefix + 'aucs.csv', cur_run_row_aucs, header=header, fmt=fmt_str)
        cur_run_row_accs = cur_run_row_accs.reshape(-1, 1)
        cur_run_row_aucs = cur_run_row_aucs.reshape(-1, 1)
    else:
        cur_run_row_accs = np.concatenate([old_output[0], cur_run_row_accs], axis=1)
        cur_run_row_aucs = np.concatenate([old_output[1], cur_run_row_aucs], axis=1)
        np.savetxt(saveprefix + 'accs.csv', cur_run_row_accs, header=header, fmt=fmt_str)
        np.savetxt(saveprefix + 'aucs.csv', cur_run_row_aucs, header=header, fmt=fmt_str)
    return [cur_run_row_accs, cur_run_row_aucs]


def get_aucs(preds_and_acc_dict, y_eval):
    keys = [key for key in preds_and_acc_dict.keys()]
    auc_fs = roc_auc_score(preds_and_acc_dict[keys[0]][0], y_eval, average='macro')
    auc_fs_aug = roc_auc_score(preds_and_acc_dict[keys[1]][0], y_eval, average='macro')
    auc_km = roc_auc_score(preds_and_acc_dict[keys[2]][0], y_eval, average='macro')
    auc_km_aug = roc_auc_score(preds_and_acc_dict[keys[3]][0], y_eval, average='macro')
    return auc_fs, auc_fs_aug, auc_km, auc_km_aug

def get_preds_and_eval_acc(model, model_name, x_eval, y_eval):
    preds = model.predict_testing_samples(x_eval)
    eval_acc = np.sum(preds.reshape(-1, 1) == np.array(y_eval).reshape(-1, 1))/len(y_eval)
    print('Evaluation accuracy is %.4f' %eval_acc)
    return preds, eval_acc

def fit_model_and_get_tr_acc(model, model_name):
    print('Fitting %s model...' %model_name)
    model.fit()
    print('Model fitting complete')
    model.predict_all_samples()
    print('Model prediction on train data complete')
    tr_acc = model.get_training_accuracy()
    print('Training Accuracy is %.3f' %tr_acc)
    return tr_acc


def clean_up_temp_files(saveprefix, random_seed):
    os.remove(saveprefix + 'x_train_temp_seed%d.pkl' % random_seed)
    os.remove(saveprefix + 'y_train_temp_seed%d.pkl' % random_seed)
    os.remove(saveprefix + '.csv')
    os.remove(saveprefix + '_for_flowsom.csv')

def save_files_for_flowsom(input, saveprefix, num_to_subsample,random_seed):
        x_list_path = saveprefix + 'x_train_temp_seed%d.pkl' %random_seed
        y_list_path = saveprefix + 'y_train_temp_seed%d.pkl' %random_seed
        x_train = [x.cpu().detach().numpy() for x in input.x_train]
        y_train = input.y_train.cpu().detach().numpy()
        with open(x_list_path,  'wb') as f:
            pickle.dump(x_train, f)
        with open(y_list_path, 'wb') as f:
            pickle.dump(y_train, f)
        
        catted_data = make_and_write_catted_data(x_list_path, y_list_path, saveprefix + '.csv')
        
        df = pd.read_csv(saveprefix + '.csv')
        df_subsampled = df.sample(n=int(num_to_subsample))
        df_subsampled.rename({',FSC-A':'FSC-A'})
        df_subsampled.to_csv(saveprefix + '_for_flowsom.csv', index=False)

#def main(path_to_params):
#    parser = baselineParamsParser(path_to_params)
#    params = parser.get_params()
#    model = BaselineModelFactory.create_model(params['model_params'])
#    data_loader = BaselineDataLoaderFactory.create_data_loader(
#                        params['data_loading_params'], type(model)
#    )
#    model.fit(data_loader.training_data)
#    diagnostics = BaselineDiagnostics(
#                    model, 
#                    data_loader.training_data,
#                    data_loader.test_data,
#                    hparams['diagnostic_params']
#    )
#    diagnostics.write_diagnostics(model)
#    diagnostics.write_visualizations(model)
    




if __name__ == '__main__':
    #path_to_model_params = '../configs/testing_kmeans.yaml'
    # use for CV runs with the two kmeans models selected from grid search
    #path_to_model_params = '../configs/CV_kmeans_two_models.yaml'
    #path_to_hparams_for_input = '../configs/CV_runs.yaml'

    # use these two for grid search on dev data
#    path_to_model_params = '../configs/grid_search_baseline.yaml'
#    path_to_hparams_for_input = '../configs/Reg_CV_device1.yaml'

    # use these two for baseline plots
    path_to_model_params = '../configs/default_baseline.yaml'
    path_to_hparams_for_input = '../configs/default_baseline_model.yaml'

    hparams = default_hparams
    with open(path_to_hparams_for_input, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)

    #CV_dev_main(hparams, path_to_model_params)

    baseline_plot_main(hparams, path_to_model_params)

    #runs_50_main(hparams, path_to_model_params)








