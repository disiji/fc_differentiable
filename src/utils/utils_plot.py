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


def plot_metrics(x_range, train_tracker, eval_tracker, filename,
                 output_dafi_train=None,
                 output_dafi_eval=None,
                 output_metric_dict = None):

    fig_metric, ax_metric = plt.subplots(nrows=4, ncols=2, figsize=(2 * 3, 4 * 2))

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
        ax_metric[1, 1].legend(["gate size reg-model", "gate size reg-DAFi"],prop={'size': 6})

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

    ax_metric[3, 1].plot(x_range, train_tracker.log_decision_boundary)
    ax_metric[3, 1].set_xlabel("#Epoch")
    ax_metric[3, 1].legend(["log decision boundary"], prop={'size': 6})

    fig_metric.tight_layout()
    fig_metric.savefig(filename)


def plot_cll(normalized_x, filtered_normalized_x, y, FEATURES, model_tree, reference_tree,
             train_tracker, eval_tracker,
             model_pred, model_pred_prob, dafi_pred, dafi_pred_prob,
             figname_root_pos, figname_root_neg, figname_leaf_pos, figname_leaf_neg):

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
                           [train_tracker.root_gate_opt, eval_tracker.root_gate_opt, train_tracker.root_gate_init,
                            reference_tree.gate, model_tree.root],
                           ["opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "converge: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_root_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
            ax_leaf_pos[idx_pos // 5, idx_pos % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [train_tracker.leaf_gate_opt,
                            eval_tracker.leaf_gate_opt,
                            train_tracker.leaf_gate_init, reference_tree.children[0].gate,
                            model_tree.children_dict[str(id(model_tree.root))][0]],
                           ["opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "converge: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
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
                           [train_tracker.root_gate_opt, eval_tracker.root_gate_opt, train_tracker.root_gate_init,
                            reference_tree.gate, model_tree.root],
                           ["opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "converge: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_root_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            ax_leaf_neg[idx_neg // 5, idx_neg % 5] = \
                plot_gates(filtered_normalized_x[sample_idx][:, 2],
                           filtered_normalized_x[sample_idx][:, 3],
                           [train_tracker.leaf_gate_opt,
                            eval_tracker.leaf_gate_opt,
                            train_tracker.leaf_gate_init, reference_tree.children[0].gate,
                            model_tree.children_dict[str(id(model_tree.root))][0]],
                           ["opt on train", "opt on eval", "init",
                            "DAFi: p=%.3f" % dafi_pred_prob[sample_idx],
                            "converge: p=%.3f" % model_pred_prob[sample_idx]], FEATURES,
                           ax=ax_leaf_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
            if model_pred[sample_idx] == 1:
                rect_root_neg = patches.Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='red', facecolor='none')
                rect_leaf_neg = patches.Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='red', facecolor='none')
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
