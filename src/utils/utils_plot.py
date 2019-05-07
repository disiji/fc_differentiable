import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch.nn.functional as F
import torch
from math import *
import utils.utils_load_data as dh


def plot_gates(x1, x2, gates, gate_names, id2feature, ax=None, filename=None, normalized=True):
    """

    :param x1:
    :param x2:
    :param gates: a list of 2d gates, features in different gates should match
    :param gate_names: a list of gate names.
    :param id2feature:
    :param ax:
    :param filename:
    :param normalized:
    :return:
    """
    if ax == None:
        fig, ax = plt.subplots(1)
    # Add scatter plot
    ax.scatter(x1, x2, s=1)
    n_gates = len(gates)
    colors = ["black", "red", "red", "blue", "purple"]
    linestyles = ['solid', 'dashed', 'solid', 'solid', 'solid']

    for i in range(n_gates):
        dim1 = gates[i].gate_dim1
        dim2 = gates[i].gate_dim2
        if type(gates[i]).__name__ == 'ModelNode':
            gate_low1, = F.sigmoid(gates[i].gate_low1_param).item(),
            gate_low2, = F.sigmoid(gates[i].gate_low2_param).item(),
            gate_upp1, = F.sigmoid(gates[i].gate_upp1_param).item(),
            gate_upp2, = F.sigmoid(gates[i].gate_upp2_param).item(),
        if type(gates[i]).__name__ == 'Gate':
            gate_low1 = gates[i].gate_low1
            gate_low2 = gates[i].gate_low2
            gate_upp1 = gates[i].gate_upp1
            gate_upp2 = gates[i].gate_upp2
        # Create a Rectangle patch
        rect = patches.Rectangle((gate_low1, gate_low2), gate_upp1 - gate_low1, gate_upp2 - gate_low2,
                                 edgecolor=colors[i], linestyle=linestyles[i],
                                 facecolor='none', label=gate_names[i], linewidth=3, )
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.set_xlabel(id2feature[dim1])
    ax.set_ylabel(id2feature[dim2])
    ax.legend(prop={'size': 6})
    if normalized == True:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    if filename:
        fig.savefig(filename, dpi=90, bbox_inches='tight')
    return ax


def plot_metrics(x_range, train_tracker, eval_tracker, filename,
                 output_dafi_train=None,
                 output_dafi_eval=None,
                 output_metric_dict=None):
    fig_metric, ax_metric = plt.subplots(nrows=5, ncols=2, figsize=(2 * 3, 5 * 2))

    ax_metric[0, 0].plot(x_range, train_tracker.loss)
    ax_metric[0, 0].plot(x_range, eval_tracker.loss)
    ax_metric[0, 0].set_xlabel("#Epoch")
    if output_dafi_train == None:
        ax_metric[0, 0].legend(["train overall loss", "eval overall loss"], prop={'size': 6})
    else:
        ax_metric[0, 0].plot(x_range, [output_dafi_train['loss'] for _ in x_range])
        ax_metric[0, 0].plot(x_range, [output_dafi_eval['loss'] for _ in x_range])
        ax_metric[0, 0].legend(
            ["train overall loss", "eval overall loss", "train overall loss-DAfi", "eval overall loss-DAfi"],
            prop={'size': 6})

    ax_metric[0, 1].plot(x_range, train_tracker.log_loss)
    ax_metric[0, 1].plot(x_range, eval_tracker.log_loss)
    ax_metric[0, 1].set_xlabel("#Epoch")
    if output_dafi_train == None:
        ax_metric[0, 1].legend(["train logL", "eval logL"], prop={'size': 6})
    else:
        ax_metric[0, 1].plot(x_range, [output_dafi_train['log_loss'] for _ in x_range])
        ax_metric[0, 1].plot(x_range, [output_dafi_eval['log_loss'] for _ in x_range])
        ax_metric[0, 1].legend(["train logL", "eval logL", "train logL-DAFi", "eval logL-DAFi"], prop={'size': 6})

    ax_metric[1, 0].plot(x_range, train_tracker.ref_reg_loss)
    ax_metric[1, 0].set_xlabel("#Epoch")
    if output_dafi_train == None:
        ax_metric[1, 0].legend(["reference reg"], prop={'size': 6})
    else:
        ax_metric[1, 0].plot(x_range, [output_dafi_train['ref_reg_loss'] for _ in x_range])
        ax_metric[1, 0].legend(["reference reg-model", "reference reg-DAFi"], prop={'size': 6})

    ax_metric[1, 1].plot(x_range, train_tracker.size_reg_loss)
    ax_metric[1, 1].set_xlabel("#Epoch")
    if output_dafi_train == None:
        ax_metric[1, 1].legend(["gate size reg"], prop={'size': 6})
    else:
        ax_metric[1, 1].plot(x_range, [output_dafi_train['size_reg_loss'] for _ in x_range])
        ax_metric[1, 1].legend(["gate size reg-model", "gate size reg-DAFi"], prop={'size': 6})

    ax_metric[2, 0].plot(x_range, train_tracker.acc)
    ax_metric[2, 0].plot(x_range, eval_tracker.acc)
    ax_metric[2, 0].set_xlabel("#Epoch")
    if output_dafi_train == None:
        ax_metric[2, 0].legend(["train acc", "eval acc"], prop={'size': 6})
    else:
        ax_metric[2, 0].plot(x_range, [output_metric_dict['train_accuracy_dafi'] for _ in x_range])
        ax_metric[2, 0].plot(x_range, [output_metric_dict['eval_accuracy_dafi'] for _ in x_range])
        ax_metric[2, 0].legend(["train acc", "eval acc", "train acc-DAFi", "eval acc-DAFi"], prop={'size': 6})

    ax_metric[2, 1].plot(x_range, train_tracker.roc_auc_score)
    ax_metric[2, 1].plot(x_range, eval_tracker.roc_auc_score)
    if output_dafi_train == None:
        ax_metric[2, 1].legend(["train auc", "eval auc"], prop={'size': 6})
    else:
        ax_metric[2, 1].plot(x_range, [output_metric_dict['train_auc_dafi'] for _ in x_range])
        ax_metric[2, 1].plot(x_range, [output_metric_dict['eval_auc_dafi'] for _ in x_range])
        ax_metric[2, 1].legend(["train auc", "eval auc", "train auc-DAFi", "eval auc-DAFi"], prop={'size': 6})

    ax_metric[3, 0].plot(x_range, train_tracker.brier_score_loss)
    ax_metric[3, 0].plot(x_range, eval_tracker.brier_score_loss)
    if output_dafi_train == None:
        ax_metric[3, 0].legend(["train brier", "eval brier"], prop={'size': 6})
    else:
        ax_metric[3, 0].plot(x_range, [output_metric_dict['train_brier_score_dafi'] for _ in x_range])
        ax_metric[3, 0].plot(x_range, [output_metric_dict['eval_brier_score_dafi'] for _ in x_range])
        ax_metric[3, 0].legend(["train brier", "eval brier", "train brier-DAFi", "eval brier-DAFi"], prop={'size': 6})

    if train_tracker.log_decision_boundary[0].shape[1] == 1:
        ax_metric[3, 1].plot(x_range, train_tracker.log_decision_boundary)
        ax_metric[3, 1].set_xlabel("#Epoch")
        ax_metric[3, 1].legend(["log decision boundary"], prop={'size': 6})

    ax_metric[4, 0].plot(x_range, train_tracker.corner_reg_loss)
    ax_metric[4, 0].set_xlabel("#Epoch")
    if output_dafi_train == None:
        ax_metric[4, 0].legend(["corner distance reg"], prop={'size': 6})
    else:
        ax_metric[4, 0].plot(x_range, [output_dafi_train['corner_reg_loss'] for _ in x_range])
        ax_metric[4, 0].legend(["corner distance reg-model", "corner distance reg-DAFi"], prop={'size': 6})

    fig_metric.tight_layout()
    fig_metric.savefig(filename)


def plot_cll_1p(normalized_x, filtered_normalized_x, y, FEATURES, model_tree, reference_tree,
                train_tracker, model_pred, model_pred_prob, dafi_pred_prob,
                figname_root_pos, figname_root_neg, figname_leaf_pos, figname_leaf_neg):
    fig_root_pos, ax_root_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 5, ceil(sum(y) / 5) * 3))
    fig_leaf_pos, ax_leaf_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 5, ceil(sum(y) / 5) * 3))
    fig_root_neg, ax_root_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
                                             figsize=(5 * 5, ceil((len(y) - sum(y)) / 5) * 3))
    fig_leaf_neg, ax_leaf_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
                                             figsize=(5 * 5, ceil((len(y) - sum(y)) / 5) * 3))

    idx_pos = 0
    idx_neg = 0

    gate_root_init = train_tracker.model_init.root
    for item in train_tracker.model_init.children_dict:
        if len(train_tracker.model_init.children_dict[item]) > 0:
            gate_leaf_init = train_tracker.model_init.children_dict[item][0]

    for sample_idx in range(len(normalized_x)):

        if y[sample_idx] == 1:
            ax_root_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
                           [gate_root_init,
                            reference_tree.gate,
                            model_tree.root],
                           ["init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "Model: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_root_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            ax_leaf_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [gate_leaf_init,
                            reference_tree.children[0].gate,
                            model_tree.children_dict[str(id(model_tree.root))][0]],
                           ["init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "Model: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_leaf_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            if model_pred[sample_idx] == 0:
                rect_root_pos = patches.Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='red', facecolor='none')
                rect_leaf_pos = patches.Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='red', facecolor='none')
                ax_root_pos[idx_pos // 5, idx_pos % 5].add_patch(rect_root_pos)
                ax_leaf_pos[idx_pos // 5, idx_pos % 5].add_patch(rect_leaf_pos)
            idx_pos += 1

        else:
            ax_root_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
                           [gate_root_init,
                            reference_tree.gate,
                            model_tree.root],
                           ["init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "Model: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_root_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            ax_leaf_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [gate_leaf_init,
                            reference_tree.children[0].gate,
                            model_tree.children_dict[str(id(model_tree.root))][0]],
                           ["init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "Model: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_leaf_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            if model_pred[sample_idx] == 1:
                rect_root_neg = patches.Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='red', facecolor='none')
                rect_leaf_neg = patches.Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='red', facecolor='none')
                ax_root_neg[idx_neg // 5, idx_neg % 5].add_patch(rect_root_neg)
                ax_leaf_neg[idx_neg // 5, idx_neg % 5].add_patch(rect_leaf_neg)
            idx_neg += 1

    fig_root_pos.savefig(figname_root_pos)
    fig_root_neg.savefig(figname_root_neg)
    fig_leaf_pos.savefig(figname_leaf_pos)
    fig_leaf_neg.savefig(figname_leaf_neg)


def plot_cll_1p_light(normalized_x, filtered_normalized_x, y, FEATURES, model_tree, reference_tree,
                      train_tracker, figname_root_pos, figname_root_neg, figname_leaf_pos, figname_leaf_neg):
    fig_root_pos, ax_root_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 5, ceil(sum(y) / 5) * 3))
    fig_leaf_pos, ax_leaf_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 5, ceil(sum(y) / 5) * 3))
    fig_root_neg, ax_root_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
                                             figsize=(5 * 5, ceil((len(y) - sum(y)) / 5) * 3))
    fig_leaf_neg, ax_leaf_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
                                             figsize=(5 * 5, ceil((len(y) - sum(y)) / 5) * 3))

    idx_pos = 0
    idx_neg = 0

    gate_root_init = train_tracker.model_init.root
    for item in train_tracker.model_init.children_dict:
        if len(train_tracker.model_init.children_dict[item]) > 0:
            gate_leaf_init = train_tracker.model_init.children_dict[item][0]

    gate_root_model = model_tree.root
    for item in model_tree.children_dict:
        if len(model_tree.children_dict[item]) > 0:
            gate_leaf_model = model_tree.children_dict[item][0]

    for sample_idx in range(len(normalized_x)):

        if y[sample_idx] == 1:
            ax_root_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
                           [gate_root_init,
                            reference_tree.gate,
                            gate_root_model],
                           ["init", "DAFi", "Model"], FEATURES,
                           ax=ax_root_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            ax_leaf_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [gate_leaf_init,
                            reference_tree.children[0].gate,
                            gate_leaf_model],
                           ["init", "DAFi", "Model"], FEATURES,
                           ax=ax_leaf_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            idx_pos += 1

        else:
            ax_root_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
                           [gate_root_init,
                            reference_tree.gate,
                            gate_root_model],
                           ["init", "DAFi", "Model"], FEATURES,
                           ax=ax_root_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            ax_leaf_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [gate_leaf_init,
                            reference_tree.children[0].gate,
                            gate_leaf_model],
                           ["init", "DAFi", "Model"], FEATURES,
                           ax=ax_leaf_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            idx_neg += 1

    fig_root_pos.savefig(figname_root_pos)
    fig_root_neg.savefig(figname_root_neg)
    fig_leaf_pos.savefig(figname_leaf_pos)
    fig_leaf_neg.savefig(figname_leaf_neg)


def plot_motion_p1(input, epoch_list, model_checkpoint_dict, filename):
    # data to plot on
    input_x_pos = torch.cat([input.x[idx] for idx in range(len(input.y)) if input.y[idx] == 1], dim=0)
    input_x_pos_subsample = input_x_pos[torch.randperm(input_x_pos.size()[0])][:10_000]
    input_x_neg = torch.cat([input.x[idx] for idx in range(len(input.y)) if input.y[idx] == 0], dim=0)
    input_x_neg_subsample = input_x_neg[torch.randperm(input_x_neg.size()[0])][:10_000]

    gate_root_ref = input.reference_tree.gate
    gate_leaf_ref = input.reference_tree.children[0].gate

    fig, axarr = plt.subplots(nrows=4, ncols=len(model_checkpoint_dict),
                              figsize=(10 / 4 * len(model_checkpoint_dict), 10), sharex=True, sharey=True)

    col_titles = ["Iterations: %d" % epoch for epoch in epoch_list]
    row_titles = ["Positive Root", "Positive Leaf", "Negative Root", "Negative Leaf"]

    # plot the first row: positive root
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        gate_root_model = model_tree.root
        axarr[0, idx] = plot_gates(input_x_pos_subsample[:, 0], input_x_pos_subsample[:, 1],
                                   [gate_root_ref, gate_root_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[0, idx], filename=None, normalized=True)
        axarr[0, idx].get_legend().remove()

    # plot the second row: positive leaf
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        filtered_input_x_pos = dh.filter_rectangle(input_x_pos_subsample, 0, 1,
                                                   F.sigmoid(model_tree.root.gate_low1_param),
                                                   F.sigmoid(model_tree.root.gate_upp1_param),
                                                   F.sigmoid(model_tree.root.gate_low2_param),
                                                   F.sigmoid(model_tree.root.gate_upp2_param))
        for item in model_tree.children_dict:
            if len(model_tree.children_dict[item]) > 0:
                gate_leaf_model = model_tree.children_dict[item][0]
        axarr[1, idx] = plot_gates(filtered_input_x_pos[:, 2], filtered_input_x_pos[:, 3],
                                   [gate_leaf_ref, gate_leaf_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[1, idx], filename=None, normalized=True)
        axarr[1, idx].get_legend().remove()

    # plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)

    # plot the third row: negative root
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        gate_root_model = model_tree.root
        axarr[2, idx] = plot_gates(input_x_neg_subsample[:, 0], input_x_neg_subsample[:, 1],
                                   [gate_root_ref, gate_root_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[2, idx], filename=None, normalized=True)
        axarr[2, idx].get_legend().remove()

    # plot the fourth row: negative leaf
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        filtered_input_x_neg = dh.filter_rectangle(input_x_neg_subsample, 0, 1,
                                                   F.sigmoid(model_tree.root.gate_low1_param),
                                                   F.sigmoid(model_tree.root.gate_upp1_param),
                                                   F.sigmoid(model_tree.root.gate_low2_param),
                                                   F.sigmoid(model_tree.root.gate_upp2_param))
        for item in model_tree.children_dict:
            if len(model_tree.children_dict[item]) > 0:
                gate_leaf_model = model_tree.children_dict[item][0]
        axarr[3, idx] = plot_gates(filtered_input_x_neg[:, 2], filtered_input_x_neg[:, 3],
                                   [gate_leaf_ref, gate_leaf_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[3, idx], filename=None, normalized=True)
        axarr[3, idx].legend(loc=1, prop={'size': 10})
        if idx < len(epoch_list) - 1:
            axarr[3, idx].get_legend().remove()

    pad = 5
    for ax, col in zip(axarr[0], col_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axarr[:,0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    fig.savefig(filename, format='png', bbox_inches='tight')



def plot_motion_p2(input, epoch_list, model_checkpoint_dict, filename):
    # data to plot on
    input_x_pos = torch.cat([input.x[idx] for idx in range(len(input.y)) if input.y[idx] == 1], dim=0)
    input_x_pos_subsample = input_x_pos[torch.randperm(input_x_pos.size()[0])][:10_000]
    input_x_neg = torch.cat([input.x[idx] for idx in range(len(input.y)) if input.y[idx] == 0], dim=0)
    input_x_neg_subsample = input_x_neg[torch.randperm(input_x_neg.size()[0])][:10_000]

    gate_root_ref = input.reference_tree.gate
    gate_left_ref = input.reference_tree.children[0].gate
    gate_right_ref = input.reference_tree.children[1].gate

    fig, axarr = plt.subplots(nrows=6, ncols=len(model_checkpoint_dict),
                              figsize=(10 / 4 * len(model_checkpoint_dict), 10. / 4 * 6), sharex=True, sharey=True)

    col_titles = ["Iterations: %d" % epoch for epoch in epoch_list]
    row_titles = ["Positive Root", "Positive Left", "Positive Right", "Negative Root", "Negative Left", "Negative Right"]

    # plot the first row: positive root
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        gate_root_model = model_tree.root
        axarr[0, idx] = plot_gates(input_x_pos_subsample[:, 0], input_x_pos_subsample[:, 1],
                                   [gate_root_ref, gate_root_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[0, idx], filename=None, normalized=True)
        axarr[0, idx].get_legend().remove()

    # plot the second row: positive leaf
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        filtered_input_x_pos = dh.filter_rectangle(input_x_pos_subsample, 0, 1,
                                                   F.sigmoid(model_tree.root.gate_low1_param),
                                                   F.sigmoid(model_tree.root.gate_upp1_param),
                                                   F.sigmoid(model_tree.root.gate_low2_param),
                                                   F.sigmoid(model_tree.root.gate_upp2_param))

        for item in model_tree.children_dict:
            if len(model_tree.children_dict[item]) > 0:
                gate_left_model, gate_right_model = model_tree.children_dict[item]
        axarr[1, idx] = plot_gates(filtered_input_x_pos[:, 2], filtered_input_x_pos[:, 3],
                                   [gate_left_ref, gate_left_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[1, idx], filename=None, normalized=True)
        axarr[1, idx].get_legend().remove()
        axarr[2, idx] = plot_gates(filtered_input_x_pos[:, 2], filtered_input_x_pos[:, 3],
                                   [gate_right_ref, gate_right_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[2, idx], filename=None, normalized=True)
        axarr[2, idx].get_legend().remove()

    # plot the third row: negative root
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        gate_root_model = model_tree.root
        axarr[3, idx] = plot_gates(input_x_neg_subsample[:, 0], input_x_neg_subsample[:, 1],
                                   [gate_root_ref, gate_root_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[3, idx], filename=None, normalized=True)
        axarr[3, idx].get_legend().remove()

    # plot the fourth row: negative leaf
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        filtered_input_x_neg = dh.filter_rectangle(input_x_neg_subsample, 0, 1,
                                                   F.sigmoid(model_tree.root.gate_low1_param),
                                                   F.sigmoid(model_tree.root.gate_upp1_param),
                                                   F.sigmoid(model_tree.root.gate_low2_param),
                                                   F.sigmoid(model_tree.root.gate_upp2_param))
        for item in model_tree.children_dict:
            if len(model_tree.children_dict[item]) > 0:
                gate_left_model, gate_right_model = model_tree.children_dict[item]
        axarr[4, idx] = plot_gates(filtered_input_x_neg[:, 2], filtered_input_x_neg[:, 3],
                                   [gate_left_ref, gate_left_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[4, idx], filename=None, normalized=True)
        axarr[4, idx].get_legend().remove()
        axarr[5, idx] = plot_gates(filtered_input_x_neg[:, 2], filtered_input_x_neg[:, 3],
                                   [gate_right_ref, gate_right_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[5, idx], filename=None, normalized=True)
        axarr[5, idx].legend(loc=1, prop={'size': 10})
        if idx < len(epoch_list) - 1:
            axarr[5, idx].get_legend().remove()

    pad = 5
    for ax, col in zip(axarr[0], col_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axarr[:,0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    fig.savefig(filename, format='png', bbox_inches='tight')


def plot_motion_p2_swap(input, epoch_list, model_checkpoint_dict, filename):
    # data to plot on
    input_x_pos = torch.cat([input.x[idx] for idx in range(len(input.y)) if input.y[idx] == 1], dim=0)
    input_x_pos_subsample = input_x_pos[torch.randperm(input_x_pos.size()[0])][:10_000]
    input_x_neg = torch.cat([input.x[idx] for idx in range(len(input.y)) if input.y[idx] == 0], dim=0)
    input_x_neg_subsample = input_x_neg[torch.randperm(input_x_neg.size()[0])][:10_000]

    gate_root_ref = input.reference_tree.gate
    gate_left_ref = input.reference_tree.children[0].gate
    gate_right_ref = input.reference_tree.children[1].gate

    fig, axarr = plt.subplots(nrows=6, ncols=len(model_checkpoint_dict),
                              figsize=(10 / 4 * len(model_checkpoint_dict), 10. / 4 * 6), sharex=True, sharey=True)

    col_titles = ["Iterations: %d" % epoch for epoch in epoch_list]
    row_titles = ["Positive Root", "Positive Left", "Positive Right", "Negative Root", "Negative Left", "Negative Right"]

    # plot the first row: positive root
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        gate_root_model = model_tree.root
        axarr[0, idx] = plot_gates(input_x_pos_subsample[:, 0], input_x_pos_subsample[:, 1],
                                   [gate_root_ref, gate_root_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[0, idx], filename=None, normalized=True)
        axarr[0, idx].get_legend().remove()

    # plot the second row: positive leaf
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        filtered_input_x_pos = dh.filter_rectangle(input_x_pos_subsample, 0, 1,
                                                   F.sigmoid(model_tree.root.gate_low1_param),
                                                   F.sigmoid(model_tree.root.gate_upp1_param),
                                                   F.sigmoid(model_tree.root.gate_low2_param),
                                                   F.sigmoid(model_tree.root.gate_upp2_param))

        for item in model_tree.children_dict:
            if len(model_tree.children_dict[item]) > 0:
                gate_right_model, gate_left_model = model_tree.children_dict[item]
        axarr[1, idx] = plot_gates(filtered_input_x_pos[:, 2], filtered_input_x_pos[:, 3],
                                   [gate_left_ref, gate_left_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[1, idx], filename=None, normalized=True)
        axarr[1, idx].get_legend().remove()
        axarr[2, idx] = plot_gates(filtered_input_x_pos[:, 2], filtered_input_x_pos[:, 3],
                                   [gate_right_ref, gate_right_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[2, idx], filename=None, normalized=True)
        axarr[2, idx].get_legend().remove()

    # plot the third row: negative root
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        gate_root_model = model_tree.root
        axarr[3, idx] = plot_gates(input_x_neg_subsample[:, 0], input_x_neg_subsample[:, 1],
                                   [gate_root_ref, gate_root_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[3, idx], filename=None, normalized=True)
        axarr[3, idx].get_legend().remove()

    # plot the fourth row: negative leaf
    for idx, epoch in enumerate(epoch_list):
        # find model gates to plot
        model_tree = model_checkpoint_dict[epoch]
        filtered_input_x_neg = dh.filter_rectangle(input_x_neg_subsample, 0, 1,
                                                   F.sigmoid(model_tree.root.gate_low1_param),
                                                   F.sigmoid(model_tree.root.gate_upp1_param),
                                                   F.sigmoid(model_tree.root.gate_low2_param),
                                                   F.sigmoid(model_tree.root.gate_upp2_param))
        for item in model_tree.children_dict:
            if len(model_tree.children_dict[item]) > 0:
                gate_right_model, gate_left_model = model_tree.children_dict[item]
        axarr[4, idx] = plot_gates(filtered_input_x_neg[:, 2], filtered_input_x_neg[:, 3],
                                   [gate_left_ref, gate_left_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[4, idx], filename=None, normalized=True)
        axarr[4, idx].get_legend().remove()
        axarr[5, idx] = plot_gates(filtered_input_x_neg[:, 2], filtered_input_x_neg[:, 3],
                                   [gate_right_ref, gate_right_model],
                                   ["Expert", "Model"], input.features,
                                   ax=axarr[5, idx], filename=None, normalized=True)
        axarr[5, idx].legend(loc=1, prop={'size': 10})
        if idx < len(epoch_list) - 1:
            axarr[5, idx].get_legend().remove()

    pad = 5
    for ax, col in zip(axarr[0], col_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axarr[:,0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    fig.savefig(filename, format='png', bbox_inches='tight')

