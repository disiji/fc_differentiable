from random import shuffle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from utils.bayes_gate_pytorch_sigmoid_trans import *
import utils.load_data as dh
from sklearn.model_selection import train_test_split
import time
import torch
import pickle
from copy import deepcopy
from utils import plot as util_plot
from math import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    start = time.time()

    DATA_DIR = '../data/cll/'
    CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
    DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
    REFERENCE_TREE_FILENAME = DATA_DIR + 'ref_gate.pkl'
    FEATURES = ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3']
    FEATURE2ID = dict((FEATURES[i], i) for i in range(len(FEATURES)))
    LOGISTIC_K = 100
    REGULARIZATION_PENALTY = 1
    EMPTYNESS_PENALTY = 50
    LOAD_DATA_FROM_PICKLE = True
    n_epoch = 1000
    batch_size = 20
    n_epoch_eval = 10
    # update classifier parameter and boundary parameter alternatively;
    # update boundary parameters after every n_mini_batch_update_gates iterations of updating the classifer parameters
    n_mini_batch_update_gates = 100
    learning_rate_classifier = 0.05
    learning_rate_gates = 0.5

    # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
    # y: a list of labels; 1 is CLL, 0 is healthy
    if LOAD_DATA_FROM_PICKLE:
        with open(DATA_DIR + "filtered_cll_x_list.pkl", 'rb') as f:
            x = pickle.load(f)
        with open(DATA_DIR + 'y_list.pkl', 'rb') as f:
            y = pickle.load(f)
    else:
        x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, FEATURES)
        x = dh.filter_cll(x)
        with open(DATA_DIR + 'filtered_cll_x_list.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open(DATA_DIR + 'y_list.pkl', 'wb') as f:
            pickle.dump(y, f)
    # scale the data
    normalized_x, offset, scale = dh.normalize_x_list(x)
    print("Number of cells in each sample after filtering:", [_.shape[0] for _ in normalized_x])
    x_train, x_eval, y_train, y_eval = train_test_split(normalized_x, y, test_size=0.30, random_state=123)
    x_train = [torch.tensor(_, dtype=torch.float32) for _ in x_train]
    x_eval = [torch.tensor(_, dtype=torch.float32) for _ in x_eval]
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)
    normalized_x = [torch.tensor(_, dtype=torch.float32) for _ in normalized_x]
    y = torch.tensor(y, dtype=torch.float32)
    print("Running time for loading the data: %.3f seconds." % (time.time() - start))

    nested_list = dh.get_reference_tree(REFERENCE_TREE_FILENAME)
    nested_list = nested_list[1][0] # fix the slope gate
    nested_list = dh.normalize_nested_tree(nested_list, offset, scale, FEATURE2ID)
    reference_tree = ReferenceTree(nested_list, FEATURE2ID)
    init_tree = None

    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY,
                           emptyness_penalty=EMPTYNESS_PENALTY,init_tree=init_tree)
    print("Initialize the model tree as:", model_tree)

    # Keep track of losses for plotting
    train_loss = []
    train_log_loss = []
    train_reg_loss = []
    eval_loss = []
    eval_log_loss = []
    eval_reg_loss = []
    train_acc = []
    eval_acc = []
    train_precision = []
    eval_precision = []
    train_recall = []
    eval_recall = []

    # optimal gates
    root_gate_init = deepcopy(model_tree.root)
    leaf_gate_init = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
    train_root_gate_opt = None
    train_leaf_gate_opt = None
    eval_root_gate_opt = None
    eval_leaf_gate_opt = None
    train_acc_opt = 0
    eval_acc_opt = 0
    train_n_iter_opt = 0
    eval_n_iter_opt = 0

    # optimizer
    classifier_params = [model_tree.linear.weight, model_tree.linear.bias]
    gates_params = list(set(model_tree.parameters()) - set(classifier_params))
    optimizer_classifier = torch.optim.SGD(classifier_params, lr=learning_rate_classifier)
    optimizer_gates = torch.optim.SGD(gates_params, lr=learning_rate_gates)

    start = time.time()

    for epoch in range(n_epoch):

        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(x_train))])
        shuffle(idx_shuffle)
        x_train = [x_train[_] for _ in idx_shuffle]
        y_train = y_train[idx_shuffle]

        n_mini_batch = len(x_train) // batch_size

        for i in range(n_mini_batch):
            # generate mini batch data
            idx_batch = [j for j in range(batch_size * i, batch_size * (i + 1))]
            x_batch = [x_train[j] for j in idx_batch]
            y_batch = y_train[idx_batch]

            # zero the parameter gradients
            optimizer_gates.zero_grad()
            optimizer_classifier.zero_grad()

            # forward + backward + optimize
            output = model_tree(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
            if (n_mini_batch * epoch + i) % n_mini_batch_update_gates == 0:
                optimizer_gates.step()
            else:
                optimizer_classifier.step()

            # statistics
            y_pred = (output['y_pred'].data.numpy() > 0.5) * 1.0
            y_batch = y_batch.data.numpy()
            # leaf_probs = output['leaf_probs']
            train_loss.append(output['loss'])
            train_log_loss.append(output['log_loss'])
            train_reg_loss.append(output['reg_loss'])
            train_acc.append(sum(y_pred == y_batch) * 1.0 / batch_size)
            train_precision.append(precision_score(y_batch, y_pred, average='macro'))
            train_recall.append(recall_score(y_batch, y_pred, average='macro'))

        # print every n_batch_print mini-batches
        if epoch % n_epoch_eval == 0:
            print(model_tree)
            train_loss_avg = sum(train_loss[-n_mini_batch:]) * 1.0 / n_mini_batch
            train_reg_loss_avg = sum(train_reg_loss[-n_mini_batch:]) * 1.0 / n_mini_batch
            train_acc_avg = sum(train_acc[-n_mini_batch:]) * 1.0 / n_mini_batch
            # eval
            output_eval = model_tree(x_eval, y_eval)
            # leaf_probs = output_eval['leaf_probs']
            y_eval_pred = (output_eval['y_pred'].detach().numpy() > 0.5) * 1.0
            eval_loss.append(output_eval['loss'])
            eval_log_loss.append(output_eval['log_loss'])
            eval_reg_loss.append(output_eval['reg_loss'])
            eval_acc.append(sum(y_eval_pred == y_eval.numpy()) * 1.0 / len(x_eval))
            eval_precision.append(precision_score(y_eval.numpy(), y_eval_pred, average='macro'))
            eval_recall.append(recall_score(y_eval.numpy(), y_eval_pred, average='macro'))

            # keep track of optimal gates for train and eval set
            if train_acc_avg > train_acc_opt:
                train_root_gate_opt = deepcopy(model_tree.root)
                train_leaf_gate_opt = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
                train_acc_opt = train_acc_avg
                train_n_iter_opt = (epoch, i)
            if eval_acc[-1] > eval_acc_opt:
                eval_root_gate_opt = deepcopy(model_tree.root)
                eval_leaf_gate_opt = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
                eval_acc_opt = eval_acc[-1]
                eval_n_iter_opt = (epoch, i)

            # compute
            print(model_tree)
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (epoch, i, train_loss_avg, eval_loss[-1]))
            print('[Epoch %d, batch %d] training, eval reg loss: %.3f, %.3f' % (
                epoch, i, train_reg_loss_avg, eval_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (epoch, i, train_acc_avg, eval_acc[-1]))

    ##################### write results
    print("Running time for training %d epoch: %.3f seconds" % (n_epoch, time.time() - start))
    print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_acc_opt, train_n_iter_opt[0], train_n_iter_opt[1], eval_acc_opt, eval_n_iter_opt[0],
              eval_n_iter_opt[1],))
    with open('../output/results_cll.csv', "a+") as file:
        y_train_pred = (model_tree(x_train, y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
        y_eval_pred = (model_tree(x_eval, y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
        y_pred = (model_tree(normalized_x, y)['y_pred'].detach().numpy() > 0.5) * 1.0
        train_accuracy = sum(y_train_pred == y_train.numpy()) * 1.0 / len(x_train)
        eval_accuracy = sum(y_eval_pred == y_eval.numpy()) * 1.0 / len(x_eval)
        overall_accuracy = sum(y_pred == y.numpy()) * 1.0 / len(x)
        file.write(
            "%d, %d, %d, %d, %d, %d, %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f([%d, %d]), %.3f([%d, %d]), %.3f\n" % (
                LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY,
                n_epoch, batch_size, n_epoch_eval, n_mini_batch_update_gates,
                learning_rate_classifier, learning_rate_gates,
                train_accuracy, eval_accuracy, overall_accuracy,
                train_acc_opt, train_n_iter_opt[0], train_n_iter_opt[1],
                eval_acc_opt, eval_n_iter_opt[0], eval_n_iter_opt[1],
                time.time() - start))

        ##################### visualization
        plt.figure(1)

        plt.subplot(221)
        plt.plot(train_loss)
        plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_loss))], eval_loss)
        plt.legend(["train overall loss", "eval overall loss"])

        plt.subplot(222)
        plt.plot(train_reg_loss)
        plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_reg_loss))], eval_reg_loss)
        plt.xlabel("#minibatch")
        plt.legend(["train reg loss", "eval reg loss"])

        plt.subplot(223)
        plt.plot(train_acc)
        plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_acc))], eval_acc)
        plt.xlabel("#minibatch")
        plt.legend(["train acc", "eval acc"])

        plt.subplot(224)
        plt.plot(train_log_loss)
        plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_log_loss))], eval_log_loss)
        plt.xlabel("#minibatch")
        plt.legend(["train logL", "eval logL"])

        plt.savefig("../fig/CLL_metrics_k%d_reg%d_emp%d_nepoch%d.png" % (
        LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, n_epoch))
        #
        # fig_root_pos, ax_root_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 4, ceil(sum(y) / 5) * 3))
        # fig_leaf_pos, ax_leaf_pos = plt.subplots(nrows=ceil(sum(y) / 5), ncols=5, figsize=(5 * 4, ceil(sum(y) / 5) * 3))
        # fig_root_neg, ax_root_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
        #                                          figsize=(5 * 4, ceil((len(y) - sum(y)) / 5) * 3))
        # fig_leaf_neg, ax_leaf_neg = plt.subplots(nrows=ceil((len(y) - sum(y)) / 5), ncols=5,
        #                                          figsize=(5 * 4, ceil((len(y) - sum(y)) / 5) * 3))
        #
        # idx_pos = 0
        # idx_neg = 0
        #
        # # filter out samples according DAFI gate at root for visualization at leaf
        # filtered_normalized_x = [dh.filter_rectangle(x, 0, 1, 0.402, 0.955, 0.549, 0.99) for x in normalized_x]
        #
        # for sample_idx in range(len(normalized_x)):
        #     if y[sample_idx] == 1:
        #         ax_root_pos[idx_pos // 5, idx_pos % 5] = \
        #             util_plot.plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
        #                                  [model_tree.root, train_root_gate_opt, eval_root_gate_opt, root_gate_init,
        #                                   reference_tree.gate],
        #                                  ["converge", "opt on train", "opt on eval", "init", "DAFi"], FEATURES,
        #                                  ax=ax_root_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
        #
        #         ax_leaf_pos[idx_pos // 5, idx_pos % 5] = \
        #             util_plot.plot_gates(filtered_normalized_x[sample_idx][:, 2],
        #                                  filtered_normalized_x[sample_idx][:, 3],
        #                                  [model_tree.children_dict[str(id(model_tree.root))][0], train_leaf_gate_opt,
        #                                   eval_leaf_gate_opt,
        #                                   leaf_gate_init, reference_tree.children[0].gate],
        #                                  ["converge", "opt on train", "opt on eval", "init", "DAFi"], FEATURES,
        #                                  ax=ax_leaf_pos[idx_pos // 5, idx_pos % 5], filename=None, normalized=True)
        #         idx_pos += 1
        #     else:
        #         ax_root_neg[idx_neg // 5, idx_neg % 5] = \
        #             util_plot.plot_gates(normalized_x[sample_idx][:, 0], normalized_x[sample_idx][:, 1],
        #                                  [model_tree.root, train_root_gate_opt, eval_root_gate_opt, root_gate_init,
        #                                   reference_tree.gate],
        #                                  ["converge", "opt on train", "opt on eval", "init", "DAFi"], FEATURES,
        #                                  ax=ax_root_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
        #         ax_leaf_neg[idx_neg // 5, idx_neg % 5] = \
        #             util_plot.plot_gates(filtered_normalized_x[sample_idx][:, 2],
        #                                  filtered_normalized_x[sample_idx][:, 3],
        #                                  [model_tree.children_dict[str(id(model_tree.root))][0], train_leaf_gate_opt,
        #                                   eval_leaf_gate_opt,
        #                                   leaf_gate_init, reference_tree.children[0].gate],
        #                                  ["converge", "opt on train", "opt on eval", "init", "DAFi"], FEATURES,
        #                                  ax=ax_leaf_neg[idx_neg // 5, idx_neg % 5], filename=None, normalized=True)
        #         idx_neg += 1
        #
        # fig_root_pos.tight_layout()
        # fig_leaf_pos.tight_layout()
        # fig_root_neg.tight_layout()
        # fig_leaf_neg.tight_layout()
        #
        # fig_root_pos.savefig(
        #     "../fig/4D_root_pos_k%d_reg%d_emp%d_nepoch%d.png" % (
        #     LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, n_epoch))
        # fig_root_neg.savefig(
        #     "../fig/4D_root_neg_k%d_reg%d_emp%d_nepoch%d.png" % (
        #     LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, n_epoch))
        # fig_leaf_pos.savefig(
        #     "../fig/4D_leaf_pos_k%d_reg%d_emp%d_nepoch%d.png" % (
        #     LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, n_epoch))
        # fig_leaf_neg.savefig(
        #     "../fig/4D_leaf_neg_k%d_reg%d_emp%d_nepoch%d.png" % (
        #     LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, n_epoch))
