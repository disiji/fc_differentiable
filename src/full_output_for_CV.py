import numpy as np
import os
import pickle
from utils.utils_plot import make_single_iter_pos_and_neg_gates_plot
import matplotlib.pyplot as plt
from utils.CellOverlaps import CellOverlaps
from scipy.stats import kendalltau
from scipy.stats import wilcoxon

FMT_STR = '%.4f'
'''
File containing just the output functions used in
the cross validation runs
'''
def write_class_probs(model, data, labels, data_idxs, savepath):
    output = model(data, labels)
    probs = output['y_pred'].detach().cpu().numpy()
    probs_with_ids = np.hstack([probs[:, np.newaxis], data_idxs[:, np.newaxis]])
    header = 'Probabilities, Sample ids'
    np.savetxt(savepath, probs_with_ids, delimiter=',', header=header, fmt=FMT_STR)

def write_probs_for_tr_te_for_model_dafi(model, dafi_model, input, experiment_name):
    save_prefix = '../output/%s/Probabilities/' %experiment_name
    if not(os.path.exists(save_prefix)):
        os.mkdir(save_prefix)

    write_class_probs(
        model,
        input.x_train,
        input.y_train,
        input.idxs_train,
        save_prefix + 'probs_train_model.csv'
    )

    write_class_probs(
        model,
        input.x_eval,
        input.y_eval,
        input.idxs_eval,
        save_prefix + 'probs_eval_model.csv'
    )

    write_class_probs(
        dafi_model,
        input.x_train,
        input.y_train,
        input.idxs_train,
        save_prefix + 'probs_train_dafi.csv'
    )

    write_class_probs(
        dafi_model,
        input.x_eval,
        input.y_eval,
        input.idxs_eval,
        save_prefix + 'probs_eval_dafi.csv'
    )

def from_gpu_to_numpy(gpu_tensor):
    return gpu_tensor.cpu().detach().numpy()


def get_diagnostics_init_final(model, data, labels, tracker):
    diagnostics = {}

    tracker.model_init = tracker.model_init.cuda()
    output_i = tracker.model_init(data, labels)
    pos_leaf_probs_i = [from_gpu_to_numpy(out) for o,out in enumerate(output_i['leaf_probs']) if labels[o] == 1.]
    neg_leaf_probs_i = [from_gpu_to_numpy(out) for o,out in enumerate(output_i['leaf_probs']) if labels[o] == 0.]
    diagnostics['acc_i'] = tracker.acc[0]
    diagnostics['mean_feature_i_pos'] = np.mean(pos_leaf_probs_i)
    diagnostics['mean_feature_i_neg'] = np.mean(neg_leaf_probs_i)

    diagnostics['log_loss_i'] = tracker.log_loss[0]
    diagnostics['neg_prop_reg_i'] = tracker.neg_prop_loss[0]
    diagnostics['feature_diff_reg_i'] = tracker.feature_diff_loss[0]

    output_f = model(data, labels)
    pos_leaf_probs_f = [from_gpu_to_numpy(out) for o,out in enumerate(output_f['leaf_probs']) if labels[o] == 1.]
    neg_leaf_probs_f = [from_gpu_to_numpy(out) for o,out in enumerate(output_f['leaf_probs']) if labels[o] == 0.]
    all_probs_f = [from_gpu_to_numpy(out) for out in output_f['leaf_probs']]

    diagnostics['log_loss_f'] = tracker.log_loss[-1]
    diagnostics['neg_prop_reg_f'] = tracker.neg_prop_loss[-1]
    diagnostics['feature_diff_reg_f'] = tracker.feature_diff_loss[-1]
    diagnostics['mean_feature_f_pos'] = np.mean(pos_leaf_probs_f)
    diagnostics['mean_feature_f_neg'] = np.mean(neg_leaf_probs_f)
    
    expert_preds = np.array([0. if prop[0] <= .00015 else 1. for prop in all_probs_f]) # change to just prop if using ModelTree instead of ModelTreeBoth
    diagnostics['expert_thresh_acc_f'] = np.sum(expert_preds == from_gpu_to_numpy(labels))/len(expert_preds)


    diagnostics['acc_f'] = tracker.acc[-1]

    return diagnostics

def write_kendalls_tau_and_wilcoxon(model, dafi_model, input, experiment_name):
    model_output = model(input.x_list, input.y_list)
    dafi_output = dafi_model(input.x_list, input.y_list)

    model_preds = from_gpu_to_numpy(model_output['y_pred'])
    model_preds_rankings = np.argsort(model_preds)
    model_feats = from_gpu_to_numpy(model_output['leaf_probs'])
    model_feats_rankings = np.argsort(model_feats)

    dafi_preds = from_gpu_to_numpy(dafi_output['y_pred'])
    dafi_preds_rankings = np.argsort(dafi_preds)
    dafi_feats = from_gpu_to_numpy(dafi_output['leaf_probs'])
    dafi_feats_rankings = np.argsort(dafi_feats)

    wilcoxon_feats = wilcoxon(dafi_feats.reshape(-1), model_feats.reshape(-1)) 
    wilcoxon_preds = wilcoxon(dafi_preds.reshape(-1), model_preds.reshape(-1))
    kendall_feats = kendalltau(dafi_feats_rankings.reshape(-1), model_feats_rankings.reshape(-1))
    kendall_preds = kendalltau(dafi_preds_rankings.reshape(-1), model_preds_rankings.reshape(-1))

    header = 'wilcoxon_feats,wilcoxon_preds,kendall_feats,kendall_preds'
    results = np.array([wilcoxon_feats, wilcoxon_preds, kendall_feats, kendall_preds])
    savepath = '../output/%s/significance_tests.csv' %experiment_name
    #if not(os.path.exists(savepath)):
    #    os.mkdir(savepath)
    np.savetxt(savepath, results, header=header, fmt=FMT_STR)

    



    

def write_model_diagnostics(model, dafi_model, input, tracker_train_m, tracker_eval_m, tracker_train_d, tracker_eval_d, experiment_name):
    tr_diagnostics_m = get_diagnostics_init_final(model, input.x_train, input.y_train, tracker_train_m)
    te_diagnostics_m = get_diagnostics_init_final(model, input.x_eval, input.y_eval, tracker_eval_m)

    tr_diagnostics_d = get_diagnostics_init_final(dafi_model, input.x_train, input.y_train, tracker_train_d)
    te_diagnostics_d = get_diagnostics_init_final(dafi_model, input.x_eval, input.y_eval, tracker_eval_d)
    

    save_prefix = '../output/%s/Diagnostics/' %experiment_name
    if not(os.path.exists(save_prefix)):
        os.mkdir(save_prefix)

    headers = [key for key in tr_diagnostics_m.keys()]
    header = headers[0]
    for k, key in enumerate(tr_diagnostics_m.keys()):
        if k == 0:
            continue
        header += ',' + key

    np.savetxt(save_prefix + 'tr_diagnostics_model.csv', np.array(list(tr_diagnostics_m.values())).reshape([1, -1]), header=header, delimiter=',', fmt=FMT_STR)
    np.savetxt(save_prefix + 'te_diagnostics_model.csv', np.array(list(te_diagnostics_m.values())).reshape([1, -1]), header=header, delimiter=',', fmt=FMT_STR)
    np.savetxt(save_prefix + 'tr_diagnostics_dafi.csv', np.array(list(tr_diagnostics_d.values())).reshape([1, -1]), header=header,delimiter=',', fmt=FMT_STR)
    np.savetxt(save_prefix + 'te_diagnostics_dafi.csv', np.array(list(te_diagnostics_d.values())).reshape([1, -1]), header=header, delimiter=',', fmt=FMT_STR)




def write_gate_overlaps(model, dafi_tree, input, experiment_name, tracker_model):
    model_init = tracker_model.model_init



    x_list = [from_gpu_to_numpy(x) for x in input.x_list]
    overlaps_init = CellOverlaps(model_init, dafi_tree, x_list)
    overlap_diagnostics_init = overlaps_init.compute_overlap_diagnostics()

    overlaps_final = CellOverlaps(model, dafi_tree, x_list)
    overlap_diagnostics_final = overlaps_final.compute_overlap_diagnostics()

    col_names = \
            'in both,' + \
            'in model but not DAFI,' + \
            'in DAFI but not model,' + \
            'cells in model leaf gate,'+ \
            'cells in DAFI leaf gate,'

    save_prefix = '../output/%s/Overlaps/' %experiment_name
    if not(os.path.exists(save_prefix)):
        os.mkdir(save_prefix)

    np.savetxt(save_prefix + 'overlaps_init.csv', overlap_diagnostics_init, delimiter=',', header=col_names)

    np.savetxt(save_prefix + 'overlaps_final.csv', overlap_diagnostics_final, delimiter=',', header=col_names)

def write_learned_logistic_weights(model, dafi_model, experiment_name):

    model_weight, model_bias = from_gpu_to_numpy(model.linear.weight.data[0]), from_gpu_to_numpy(model.linear.bias)
    dafi_weight, dafi_bias = from_gpu_to_numpy(dafi_model.linear.weight.data[0]), from_gpu_to_numpy(dafi_model.linear.bias)

    save_path = '../output/%s/logistic_params_model_and_dafi.csv' %experiment_name
    with open(save_path, 'w') as f:
        f.write('Model Weight,Model Bias,Dafi Weight,Dafi Bias\n')
        f.write('%.4f,%.4f,%.4f,%.4f' %(model_weight, model_bias, dafi_weight, dafi_bias))

def write_features(model, dafi, input, experiment_name):

    save_path = '../output/%s/' %experiment_name

    output_model = model(input.x_list, input.y_list)
    output_dafi = dafi(input.x_list, input.y_list)

    features_model = from_gpu_to_numpy(output_model['leaf_probs'])
    features_model = np.hstack([features_model, input.sample_ids[:, np.newaxis]])
    features_dafi = from_gpu_to_numpy(output_dafi['leaf_probs'])
    features_dafi = np.hstack([features_dafi, input.sample_ids[:, np.newaxis]])
    np.savetxt(save_path + 'model_feats.csv', features_model, header='Feature,Sample_id', delimiter=',')
    np.savetxt(save_path + 'dafi_feats.csv', features_dafi, header='Feature,Sample_id', delimiter=',')


def write_gates(model, input, experiment_name):
    savepath = '../output/%s/model_gates.txt' %experiment_name
    named_tuple_gates = model.get_flattened_gates()
    gates = []
    for gate in named_tuple_gates:
        gates.append([gate.low1, gate.upp1, gate.low2, gate.upp2])
    with open(savepath, 'w') as f:
        f.write('Initial Gates\n')
        for g, gate in enumerate(input.flat_heuristic_gates):
            f.write('Gate %d: [%.4f, %.4f, %.4f, %.4f]\n' %(g, gate[0], gate[1], gate[2], gate[3]))


        f.write('Final Gates\n')
        for g,gate in enumerate(gates):
            f.write('Gate %d: [%.4f, %.4f, %.4f, %.4f]\n' %(g, gate[0], gate[1], gate[2], gate[3]))

        
def save_model_and_dafi(model, dafi_model, experiment_name):
    savepath = '../output/%s/' %experiment_name
    with open(savepath + 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(savepath + 'dafi_model.pkl', 'wb') as f:
        pickle.dump(dafi_model, f)

def run_write_full_output_for_CV(model, dafi_model, input, trackers_dict, hparams, model_checkpoint_dict, device_data=1):
    experiment_name = hparams['experiment_name']
    write_probs_for_tr_te_for_model_dafi(model, dafi_model, input, experiment_name)

    write_model_diagnostics(
            model, dafi_model, input, 
            trackers_dict['tracker_train_m'],
            trackers_dict['tracker_eval_m'],
            trackers_dict['tracker_train_d'],
            trackers_dict['tracker_eval_d'],
            experiment_name
    )

    write_learned_logistic_weights(model, dafi_model, experiment_name)

    write_features(model, dafi_model, input, experiment_name)

    write_gates(model, input, experiment_name)

    write_gate_overlaps(model, dafi_model, input, experiment_name, trackers_dict['tracker_train_m'])

    write_kendalls_tau_and_wilcoxon(model, dafi_model, input, experiment_name)

    save_model_and_dafi(model, dafi_model, experiment_name)
    
    # ouput dictionary for my plotting code
    output = {}
    output['models_per_iteration'] = [
            model_checkpoint_dict[iteration] 
            for iteration in 
            hparams['seven_epochs_for_gate_motion_plot']
    ]
    
    output['hparams'] = hparams
    output['cll_1p_full_input'] = input
    output['dafi_tree'] = dafi_model
    make_single_iter_pos_and_neg_gates_plot(output, 0, device_data=device_data)
    plt.savefig('../output/%s/pos_and_neg_gates_iter_0.png' %hparams['experiment_name'])
    plt.clf()
    make_single_iter_pos_and_neg_gates_plot(output, len(output['models_per_iteration']) - 1, device_data=device_data)
    plt.savefig('../output/%s/pos_and_neg_gates_iter_%d.png' %(hparams['experiment_name'], hparams['seven_epochs_for_gate_motion_plot'][-1]))
    plt.clf()


    make_single_iter_pos_and_neg_gates_plot(output, 0, device_data=device_data)
    plt.savefig('../output/%s/pos_and_neg_gates_iter_0.png' %hparams['experiment_name'])
    plt.clf()
    make_single_iter_pos_and_neg_gates_plot(output, len(output['models_per_iteration']) - 1, device_data=device_data)
    plt.savefig('../output/%s/pos_and_neg_gates_iter_%d.png' %(hparams['experiment_name'], hparams['seven_epochs_for_gate_motion_plot'][-1]))
    plt.clf()


    with open('../output/%s/model_checkpoints.pkl' %(experiment_name), 'wb') as f:
            pickle.dump(model_checkpoint_dict, f)
    with open('../output/%s/tracker_train_m.pkl' %(experiment_name), 'wb') as f:
            pickle.dump(trackers_dict['tracker_train_m'], f)
    with open('../output/%s/tracker_eval_m.pkl' %(experiment_name), 'wb') as f:
            pickle.dump(trackers_dict['tracker_eval_m'], f)
