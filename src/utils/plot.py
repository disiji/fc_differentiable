import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from math import *


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
    colors = ["red", "orange", "green", "blue", "purple", "yellow", "black"]

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
        rect = patches.Rectangle((gate_low1, gate_low2), gate_upp1 - gate_low1, gate_upp2 - gate_low2, linewidth=2,
                                 edgecolor=colors[i], facecolor='none', label=gate_names[i], alpha=0.5)
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


def plot_metrics(x_range, train_loss, eval_loss, train_log_loss, eval_log_loss, train_reg_loss, eval_reg_loss,
                 train_acc, eval_acc, log_decision_boundary, filename):
    """

    :param x_range:
    :param train_loss:
    :param eval_loss:
    :param train_log_loss:
    :param eval_log_loss:
    :param train_reg_loss:
    :param eval_reg_loss:
    :param train_acc:
    :param eval_acc:
    :param log_decision_boundary:
    :param filename:
    :return:
    """
    fig_metric, ax_metric = plt.subplots(nrows=3, ncols=2, figsize=(2 * 3, 3 * 2))

    print("x range:", x_range)
    print("train_loss:", train_loss)
    print("eval_loss:", eval_loss)
    print("train_log_loss:", train_log_loss)
    print("eval_log_loss:", eval_log_loss)
    print("train_reg_loss:", train_reg_loss)
    print("eval_reg_loss:", eval_reg_loss)
    print("train_acc:", train_acc)
    print("eval_acc:", eval_acc)
    print("log_decision_boundary:", log_decision_boundary)

    ax_metric[0, 0].plot(x_range, train_loss)
    ax_metric[0, 0].plot(x_range, eval_loss)
    ax_metric[0, 0].set_xlabel("#Epoch")
    ax_metric[0, 0].legend(["train overall loss", "eval overall loss"], prop={'size': 6})

    ax_metric[0, 1].plot(x_range, train_log_loss)
    ax_metric[0, 1].plot(x_range, eval_log_loss)
    ax_metric[0, 1].set_xlabel("#Epoch")
    ax_metric[0, 1].legend(["train logL", "eval logL"], prop={'size': 6})

    ax_metric[1, 0].plot(x_range, train_reg_loss)
    ax_metric[1, 0].plot(x_range, eval_reg_loss)
    ax_metric[1, 0].set_xlabel("#Epoch")
    ax_metric[1, 0].legend(["train reg loss", "eval reg loss"], prop={'size': 6})

    ax_metric[1, 1].plot(x_range, train_acc)
    ax_metric[1, 1].plot(x_range, eval_acc)
    ax_metric[1, 1].set_xlabel("#Epoch")
    ax_metric[1, 1].legend(["train acc", "eval acc"], prop={'size': 6})

    ax_metric[2, 0].plot(x_range, log_decision_boundary)
    ax_metric[2, 0].set_xlabel("#Epoch")
    ax_metric[2, 0].legend(["log decision boundary"], prop={'size': 6})

    fig_metric.tight_layout()
    fig_metric.savefig(filename)


def plot_cll(normalized_x, filtered_normalized_x, y, FEATURES, model_tree, reference_tree,
             train_root_gate_opt, eval_root_gate_opt, root_gate_init,
             train_leaf_gate_opt, eval_leaf_gate_opt, leaf_gate_init,
             model_pred, model_pred_prob, dafi_pred, dafi_pred_prob,
             figname_root_pos, figname_root_neg, figname_leaf_pos, figname_leaf_neg):
    """

    :param normalized_x:
    :param filtered_normalized_x:
    :param y:
    :param FEATURES:
    :param model_tree:
    :param reference_tree:
    :param train_root_gate_opt:
    :param eval_root_gate_opt:
    :param root_gate_init:
    :param train_leaf_gate_opt:
    :param eval_leaf_gate_opt:
    :param leaf_gate_init:
    :param model_pred:
    :param model_pred_prob:
    :param dafi_pred:
    :param dafi_pred_prob:
    :param figname_root_pos:
    :param figname_root_neg:
    :param figname_leaf_pos:
    :param figname_leaf_neg:
    :return:
    """

    fig_root_pos, ax_root_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 5, ceil(sum(y) / 5) * 3))
    fig_leaf_pos, ax_leaf_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 5, ceil(sum(y) / 5) * 3))
    fig_root_neg, ax_root_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
                                             figsize=(5 * 5, ceil((len(y) - sum(y)) / 5) * 3))
    fig_leaf_neg, ax_leaf_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
                                             figsize=(5 * 5, ceil((len(y) - sum(y)) / 5) * 3))

    idx_pos = 0
    idx_neg = 0

    for sample_idx in range(len(normalized_x)):

        if y[sample_idx] == 1:
            ax_root_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
                           [model_tree.root, train_root_gate_opt, eval_root_gate_opt, root_gate_init,
                            reference_tree.gate],
                           ["converge: p=%.3f" % model_pred_prob[sample_idx], "opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx]], FEATURES,
                           ax=ax_root_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            ax_leaf_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [model_tree.children_dict[str(id(model_tree.root))][0], train_leaf_gate_opt,
                            eval_leaf_gate_opt,
                            leaf_gate_init, reference_tree.children[0].gate],
                           ["converge: p=%.3f" % model_pred_prob[sample_idx], "opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx]], FEATURES,
                           ax=ax_leaf_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            if model_pred[sample_idx] == 0:
                rect_root_pos = patches.Rectangle((0, 0), 1, 1, linewidth=4, edgecolor='red', facecolor='none')
                rect_leaf_pos = patches.Rectangle((0, 0), 1, 1, linewidth=4, edgecolor='red', facecolor='none')
                ax_root_pos[idx_pos // 5, idx_pos % 5].add_patch(rect_root_pos)
                ax_leaf_pos[idx_pos // 5, idx_pos % 5].add_patch(rect_leaf_pos)
            idx_pos += 1

        else:
            ax_root_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
                           [model_tree.root, train_root_gate_opt, eval_root_gate_opt, root_gate_init,
                            reference_tree.gate],
                           ["converge: p=%.3f" % model_pred_prob[sample_idx], "opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx]], FEATURES,
                           ax=ax_root_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            ax_leaf_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [model_tree.children_dict[str(id(model_tree.root))][0], train_leaf_gate_opt,
                            eval_leaf_gate_opt,
                            leaf_gate_init, reference_tree.children[0].gate],
                           ["converge: p=%.3f" % model_pred_prob[sample_idx], "opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx]], FEATURES,
                           ax=ax_leaf_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            if model_pred[sample_idx] == 1:
                rect_root_neg = patches.Rectangle((0, 0), 1, 1, linewidth=4, edgecolor='red', facecolor='none')
                rect_leaf_neg = patches.Rectangle((0, 0), 1, 1, linewidth=4, edgecolor='red', facecolor='none')
                ax_root_neg[idx_neg // 5, idx_neg % 5].add_patch(rect_root_neg)
                ax_leaf_neg[idx_neg // 5, idx_neg % 5].add_patch(rect_leaf_neg)
            idx_neg += 1

    # fig_root_pos.tight_layout()
    # fig_leaf_pos.tight_layout()
    # fig_root_neg.tight_layout()
    # fig_leaf_neg.tight_layout()

    fig_root_pos.savefig(figname_root_pos)
    fig_root_neg.savefig(figname_root_neg)
    fig_leaf_pos.savefig(figname_leaf_pos)
    fig_leaf_neg.savefig(figname_leaf_neg)
