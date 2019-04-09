import csv
import os
import pickle
import time
from copy import deepcopy
from random import shuffle

import numpy as np
import torch
import yaml
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import utils.load_data as dh
from utils import plot as util_plot
from utils.bayes_gate_pytorch_sigmoid_trans import ModelTree, ReferenceTree

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'emptyness_penalty': 10,
    'gate_size_penalty': 1,
    'gate_size_default': 1. / 4,
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "SGD",  # or Adam
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 20,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.01,
    'learning_rate_gates': 0.5,
    'batch_size': 85,
    'n_epoch': 1000,
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 100,
}


class Cll4dInput:
    def __init__(self, hparams):
        features = 'CD5', 'CD19', 'CD10', 'CD79b'
        features_full = ('FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3')
        self.hparams = hparams
        self.features = dict((i, features[i]) for i in range(len(features)))
        self.features_full = dict((i, features_full[i]) for i in range(len(features_full)))
        self.feature2id = dict((self.features[i], i) for i in self.features)
        self.x_list = None,
        self.y_list = None,
        self.x = None,
        self.y = None,
        self.x_train = None
        self.x_eval = None
        self.y_train = None
        self.y_eval = None
        self.reference_nested_list = None,
        self.reference_tree = None,
        self.init_nested_list = None
        self.init_tree = None,

        self._load_data_()
        self._get_reference_nested_list_()
        self._get_init_nested_list_()
        self._normalize_()
        self._construct_()
        self.split()

        self.x = [torch.tensor(_, dtype=torch.float32) for _ in self.x_list]
        self.y = torch.tensor(self.y_list, dtype=torch.float32)

    def _load_data_(self):
        DATA_DIR = '../data/cll/'
        CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
        DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'

        # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
        # y: a list of labels; 1 is CLL, 0 is healthy
        if self.hparams['load_from_pickle']:
            with open(DATA_DIR + "filtered_4d_x_list.pkl", 'rb') as f:
                self.x_list = pickle.load(f)
            with open(DATA_DIR + 'y_list.pkl', 'rb') as f:
                self.y_list = pickle.load(f)
        else:
            x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, self.features_full)
            x_4d = dh.filter_cll_4d(x)
            with open(DATA_DIR + 'filtered_4d_x_list.pkl', 'wb') as f:
                pickle.dump(x_4d, f)
            with open(DATA_DIR + 'y_list.pkl', 'wb') as f:
                pickle.dump(y, f)
            self.x_list, self.y_list = x_4d, y

    def _get_reference_nested_list_(self):
        self.reference_nested_list = [
            [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
            [
                [
                    [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                    []
                ]
            ]
        ]

    def _get_init_nested_list_(self):
        self.init_nested_list = \
            [
                [[u'CD5', 2000., 3000.], [u'CD19', 2000., 3000.]],
                [
                    [
                        [[u'CD10', 1000., 2000.], [u'CD79b', 1000., 2000.]],
                        []
                    ]
                ]
            ]

    def _normalize_(self):
        self.x_list, offset, scale = dh.normalize_x_list(self.x_list)
        print(self.features_full, self.features, self.feature2id)
        self.reference_nested_list = dh.normalize_nested_tree(self.reference_nested_list, offset, scale,
                                                              self.feature2id)
        self.init_nested_list = dh.normalize_nested_tree(self.init_nested_list, offset, scale, self.feature2id)

    def _construct_(self):
        self.reference_tree = ReferenceTree(self.reference_nested_list, self.feature2id)
        self.init_tree = ReferenceTree(self.init_nested_list, self.feature2id)
        if self.hparams['dafi_init']:
            self.init_tree = None

    def split(self, random_state=123):
        self.x_train, self.x_eval, self.y_train, self.y_eval = train_test_split(self.x_list, self.y_list,
                                                                                test_size=self.hparams['test_size'],
                                                                                random_state=random_state)
        self.x_train = [torch.tensor(_, dtype=torch.float32) for _ in self.x_train]
        self.x_eval = [torch.tensor(_, dtype=torch.float32) for _ in self.x_eval]
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_eval = torch.tensor(self.y_eval, dtype=torch.float32)


class Tracker():
    def __init__(self):
        # Keep track of losses for plotting
        self.loss = []
        self.log_loss = []
        self.ref_reg_loss = []
        self.size_reg_loss = []
        self.acc = []
        self.precision = []
        self.recall = []
        self.roc_auc_score = []
        self.brier_score_loss = []
        self.log_decision_boundary = []
        self.root_gate_opt = None
        self.leaf_gate_opt = None
        self.root_gate_init = None
        self.leaf_gate_init = None
        self.acc_opt = 0
        self.n_iter_opt = (0, 0)

    def update(self, model_tree, output, y_true, epoch, i):
        y_pred = (output['y_pred'].detach().numpy() > 0.5) * 1.0
        self.loss.append(output['loss'])
        self.log_loss.append(output['log_loss'])
        self.ref_reg_loss.append(output['ref_reg_loss'])
        self.size_reg_loss.append(output['size_reg_loss'])
        self.acc.append(sum(y_pred == y_true.numpy()) * 1.0 / y_true.shape[0])
        self.precision.append(precision_score(y_true.numpy(), y_pred, average='macro'))
        self.recall.append(recall_score(y_true.numpy(), y_pred, average='macro'))
        self.roc_auc_score.append(roc_auc_score(y_true.numpy(), y_pred, average='macro'))
        self.brier_score_loss.append(brier_score_loss(y_true.numpy(), y_pred))
        self.log_decision_boundary.append(
            (-model_tree.linear.bias.detach() / model_tree.linear.weight.detach()))
        # keep track of optimal gates for train and eval set
        if self.acc[-1] > self.acc_opt:
            self.root_gate_opt = deepcopy(model_tree.root)
            self.leaf_gate_opt = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
            self.acc_opt = self.acc[-1]
            self.n_iter_opt = (epoch, i)


def run_train_dafi(dafi_tree, hparams, input):
    """
    train a classifier on the top of DAFi features
    :param dafi_tree:
    :param hparams:
    :param input:
    :return:
    """
    start = time.time()
    if hparams['optimizer'] == "SGD":
        dafi_optimizer_classifier = torch.optim.SGD([dafi_tree.linear.weight, dafi_tree.linear.bias],
                                                    lr=hparams['learning_rate_classifier'])
    else:
        dafi_optimizer_classifier = torch.optim.Adam([dafi_tree.linear.weight, dafi_tree.linear.bias],
                                                     lr=hparams['learning_rate_classifier'])

    for epoch in range(hparams['n_epoch_dafi']):
        idx_shuffle = np.array([i for i in range(len(input.x_train))])
        shuffle(idx_shuffle)
        x_train = [input.x_train[_] for _ in idx_shuffle]
        y_train = input.y_train[idx_shuffle]
        for i in range(len(x_train) // hparams['batch_size']):
            idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
            x_batch = [x_train[j] for j in idx_batch]
            y_batch = y_train[idx_batch]
            dafi_optimizer_classifier.zero_grad()
            output = dafi_tree(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
            dafi_optimizer_classifier.step()
    print("Running time for training classifier with DAFi gates: %.3f seconds." % (time.time() - start))
    return dafi_tree


def run_train_model(model_tree, hparams, input):
    """

    :param model_tree:
    :param hparams:
    :param input:
    :return:
    """
    start = time.time()
    classifier_params = [model_tree.linear.weight, model_tree.linear.bias]
    gates_params = [p for p in model_tree.parameters() if p not in classifier_params]
    if hparams['optimizer'] == "SGD":
        optimizer_classifier = torch.optim.SGD(classifier_params, lr=hparams['learning_rate_classifier'])
        optimizer_gates = torch.optim.SGD(gates_params, lr=hparams['learning_rate_gates'])
    else:
        optimizer_classifier = torch.optim.Adam(classifier_params, lr=hparams['learning_rate_classifier'])
        optimizer_gates = torch.optim.Adam(gates_params, lr=hparams['learning_rate_gates'])

    # optimal gates
    train_tracker = Tracker()
    eval_tracker = Tracker()
    train_tracker.root_gate_init = deepcopy(model_tree.root)
    train_tracker.leaf_gate_init = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])

    for epoch in range(hparams['n_epoch']):
        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(input.x_train))])
        shuffle(idx_shuffle)
        x_train = [input.x_train[_] for _ in idx_shuffle]
        y_train = input.y_train[idx_shuffle]

        for i in range(len(x_train) // hparams['batch_size']):
            idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
            optimizer_gates.zero_grad()
            optimizer_classifier.zero_grad()
            output = model_tree([x_train[j] for j in idx_batch], y_train[idx_batch])
            loss = output['loss']
            loss.backward()
            if (len(x_train) // hparams['batch_size'] * epoch + i) % hparams['n_mini_batch_update_gates'] == 0:
                print("optimizing gates...")
                optimizer_gates.step()
            else:
                optimizer_classifier.step()

        # print every n_batch_print mini-batches
        if epoch % hparams['n_epoch_eval'] == 0:
            # stats on train
            train_tracker.update(model_tree, model_tree(input.x_train, input.y_train), input.y_train, epoch, i)
            eval_tracker.update(model_tree, model_tree(input.x_eval, input.y_eval), input.y_eval, epoch, i)

            # compute
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (
                epoch, i, train_tracker.loss[-1], eval_tracker.loss[-1]))
            print('[Epoch %d, batch %d] training, eval ref_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.ref_reg_loss[-1], eval_tracker.ref_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval size_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.size_reg_loss[-1], eval_tracker.size_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (
                epoch, i, train_tracker.acc[-1], eval_tracker.acc[-1]))

    print("Running time for training %d epoch: %.3f seconds" % (hparams['n_epoch'], time.time() - start))
    print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1], eval_tracker.acc_opt,
              eval_tracker.n_iter_opt[0],
              eval_tracker.n_iter_opt[1],))

    return model_tree, train_tracker, eval_tracker, time.time() - start


def run_output(model_tree, dafi_tree, hparams, input, train_tracker, eval_tracker, run_time):
    y_pred_train = (model_tree(input.x_train, input.y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_eval = (model_tree(input.x_eval, input.y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred = (model_tree(input.x, input.y)['y_pred'].detach().numpy() > 0.5) * 1.0
    train_accuracy = sum(y_pred_train == input.y_train.numpy()) * 1.0 / len(input.x_train)
    eval_accuracy = sum(y_pred_eval == input.y_eval.numpy()) * 1.0 / len(input.x_eval)
    overall_accuracy = sum(y_pred == input.y.numpy()) * 1.0 / len(input.x)
    train_auc = roc_auc_score(input.y_train.numpy(), y_pred_train, average='macro')
    eval_auc = roc_auc_score(input.y_eval.numpy(), y_pred_eval, average='macro')
    overall_auc = roc_auc_score(input.y.numpy(), y_pred, average='macro')
    train_brier_score = brier_score_loss(input.y_train.numpy(), y_pred_train)
    eval_brier_score = brier_score_loss(input.y_eval.numpy(), y_pred_eval)
    overall_brier_score = brier_score_loss(input.y.numpy(), y_pred)

    y_pred_train_dafi = (dafi_tree(input.x_train, input.y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_eval_dafi = (dafi_tree(input.x_eval, input.y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_dafi = (dafi_tree(input.x, input.y)['y_pred'].detach().numpy() > 0.5) * 1.0
    train_accuracy_dafi = sum(y_pred_train_dafi == input.y_train.numpy()) * 1.0 / len(input.x_train)
    eval_accuracy_dafi = sum(y_pred_eval_dafi == input.y_eval.numpy()) * 1.0 / len(input.x_eval)
    overall_accuracy_dafi = sum(y_pred_dafi == input.y.numpy()) * 1.0 / len(input.x)
    train_auc_dafi = roc_auc_score(input.y_train.numpy(), y_pred_train_dafi, average='macro')
    eval_auc_dafi = roc_auc_score(input.y_eval.numpy(), y_pred_eval_dafi, average='macro')
    overall_auc_dafi = roc_auc_score(input.y.numpy(), y_pred_dafi, average='macro')
    train_brier_score_dafi = brier_score_loss(input.y_train.numpy(), y_pred_train_dafi)
    eval_brier_score_dafi = brier_score_loss(input.y_eval.numpy(), y_pred_eval_dafi)
    overall_brier_score_dafi = brier_score_loss(input.y.numpy(), y_pred_dafi)

    with open('../output/%s/results_cll_4D.csv' % hparams['experiment_name'], "a+") as file:
        file.write(
            "%d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f([%d; %d]), %.3f([%d; %d]), %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f\n" % (
                hparams['random_state'],
                train_accuracy, eval_accuracy, overall_accuracy,
                train_accuracy_dafi, eval_accuracy_dafi, overall_accuracy_dafi,
                train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1],
                eval_tracker.acc_opt, eval_tracker.n_iter_opt[0], eval_tracker.n_iter_opt[1],
                model_tree(input.x_train, input.y_train)['log_loss'].detach().numpy(),
                model_tree(input.x_eval, input.y_eval)['log_loss'].detach().numpy(),
                model_tree(input.x, input.y)['log_loss'].detach().numpy(),
                dafi_tree(input.x_train, input.y_train)['log_loss'].detach().numpy(),
                dafi_tree(input.x_eval, input.y_eval)['log_loss'].detach().numpy(),
                dafi_tree(input.x, input.y)['log_loss'].detach().numpy(),
                train_auc, eval_auc, overall_auc, train_auc_dafi, eval_auc_dafi, overall_auc_dafi,
                train_brier_score, eval_brier_score, overall_brier_score,
                train_brier_score_dafi, eval_brier_score_dafi, overall_brier_score_dafi,
                run_time
            ))

    return {
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "overall_accuracy": overall_accuracy,
        "train_accuracy_dafi": train_accuracy_dafi,
        "eval_accuracy_dafi": eval_accuracy_dafi,
        "overall_accuracy_dafi": overall_accuracy_dafi,
        "train_auc": train_auc,
        "eval_auc": eval_auc,
        "overall_auc": overall_auc,
        "train_auc_dafi": train_auc_dafi,
        "eval_auc_dafi": eval_auc_dafi,
        "overall_auc_dafi": overall_auc_dafi,
        "train_brier_score": train_brier_score,
        "eval_brier_score": eval_brier_score,
        "overall_brier_score": overall_brier_score,
        "train_brier_score_dafi": train_brier_score_dafi,
        "eval_brier_score_dafi": eval_brier_score_dafi,
        "overall_brier_score_dafi": overall_brier_score_dafi,

    }


def run_plot_metric(hparams, train_tracker, eval_tracker, dafi_tree, input, output_metric_dict):
    x_range = [i * hparams['n_epoch_eval'] for i in range(hparams['n_epoch'] // hparams['n_epoch_eval'])]
    filename_metric = "../output/%s/metrics.png" % (hparams['experiment_name'])
    util_plot.plot_metrics(x_range, train_tracker, eval_tracker, filename_metric,
                           dafi_tree(input.x_train, input.y_train),
                           dafi_tree(input.x_eval, input.y_eval),
                           output_metric_dict)


def run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, input):
    filename_root_pas = "../output/%s/root_pos.png" % (hparams['experiment_name'])
    filename_root_neg = "../output/%s/root_neg.png" % (hparams['experiment_name'])
    filename_leaf_pas = "../output/%s/leaf_pos.png" % (hparams['experiment_name'])
    filename_leaf_neg = "../output/%s/leaf_neg.png" % (hparams['experiment_name'])

    ####### compute model_pred_prob
    model_pred_prob = model_tree(input.x, input.y)['y_pred'].detach().numpy()
    model_pred = (model_pred_prob > 0.5) * 1.0
    dafi_pred_prob = dafi_tree(input.x, input.y)['y_pred'].detach().numpy()
    dafi_pred = (dafi_pred_prob > 0.5) * 1.0

    # filter out samples according DAFI gate at root for visualization at leaf
    filtered_normalized_x = [dh.filter_rectangle(x, 0, 1, 0.402, 0.955, 0.549, 0.99) for x in input.x]
    util_plot.plot_cll(input.x, filtered_normalized_x, input.y, input.features, model_tree, input.reference_tree,
                       train_tracker, eval_tracker, model_pred, model_pred_prob, dafi_pred, dafi_pred_prob,
                       filename_root_pas, filename_root_neg, filename_leaf_pas, filename_leaf_neg)


def run(yaml_filename):
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
            hparams['n_mini_batch_update_gates'] - 1)

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])

    cll_4d_input = Cll4dInput(hparams)

    for random_state in range(hparams['n_run']):
        hparams['random_state'] = random_state
        cll_4d_input.split(random_state)

        model_tree = ModelTree(cll_4d_input.reference_tree,
                               logistic_k=hparams['logistic_k'],
                               regularisation_penalty=hparams['regularization_penalty'],
                               emptyness_penalty=hparams['emptyness_penalty'],
                               gate_size_penalty=hparams['gate_size_penalty'],
                               init_tree=cll_4d_input.init_tree,
                               loss_type=hparams['loss_type'],
                               gate_size_default=hparams['gate_size_default'])

        dafi_tree = ModelTree(cll_4d_input.reference_tree,
                              logistic_k=hparams['logistic_k_dafi'],
                              regularisation_penalty=hparams['regularization_penalty'],
                              emptyness_penalty=hparams['emptyness_penalty'],
                              gate_size_penalty=hparams['gate_size_penalty'],
                              init_tree=None,
                              loss_type=hparams['loss_type'],
                              gate_size_default=hparams['gate_size_default'])

        dafi_tree = run_train_dafi(dafi_tree, hparams, cll_4d_input)
        model_tree, train_tracker, eval_tracker, run_time = run_train_model(model_tree, hparams, cll_4d_input)
        output_metric_dict = run_output(
            model_tree, dafi_tree, hparams, cll_4d_input, train_tracker, eval_tracker, run_time)

    # only plot once
    run_plot_metric(hparams, train_tracker, eval_tracker, dafi_tree, cll_4d_input, output_metric_dict)
    run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, cll_4d_input)
    print("end")


if __name__ == '__main__':
    # run(sys.argv[1])
    run("../configs/gate_size_regularization_off.yaml")
