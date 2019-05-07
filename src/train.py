import time
from copy import deepcopy
from random import shuffle

from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from utils import utils_plot as util_plot
from utils.input import *
from utils.utils_train import Tracker
from utils.input import CLLInputBase


def run_train_dafi(dafi_model, hparams, input):
    """
    train a classifier on the top of DAFi features
    :param dafi_model:
    :param hparams:
    :param input:
    :return:
    """
    start = time.time()
    if hparams['optimizer'] == "SGD":
        dafi_optimizer_classifier = torch.optim.SGD([dafi_model.linear.weight, dafi_model.linear.bias],
                                                    lr=hparams['learning_rate_classifier'])
    else:
        dafi_optimizer_classifier = torch.optim.Adam([dafi_model.linear.weight, dafi_model.linear.bias],
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
            output = dafi_model(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
            dafi_optimizer_classifier.step()
        if epoch % hparams['n_epoch_eval'] == 0:
            y_pred = (dafi_model(input.x, input.y)['y_pred'].detach().numpy() > 0.5) * 1.0
            print("Accuracy as Epoch %d: %.3f" % (epoch, sum((y_pred == input.y.numpy()) ) / input.y.shape[0]))
    print("Running time for training classifier with DAFi gates: %.3f seconds." % (time.time() - start))
    return dafi_model


def run_train_model(model, hparams, input, model_checkpoint=False):
    """

    :param model:
    :param hparams:
    :param input:
    :return:
    """
    start = time.time()
    classifier_params = [model.linear.weight, model.linear.bias]
    gates_params = [p for p in model.parameters() if not any(p is d_ for d_ in classifier_params)]

    if hparams['optimizer'] == "SGD":
        optimizer_classifier = torch.optim.SGD(classifier_params, lr=hparams['learning_rate_classifier'])
        optimizer_gates = torch.optim.SGD(gates_params, lr=hparams['learning_rate_gates'])
    else:
        optimizer_classifier = torch.optim.Adam(classifier_params, lr=hparams['learning_rate_classifier'])
        optimizer_gates = torch.optim.Adam(gates_params, lr=hparams['learning_rate_gates'])

    # optimal gates
    train_tracker = Tracker()
    eval_tracker = Tracker()
    train_tracker.model_init = deepcopy(model)
    eval_tracker.model_init = deepcopy(model)
    model_checkpoint_dict = {}

    if model_checkpoint:
        model_checkpoint_dict[0] = deepcopy(model)

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
            output = model([x_train[j] for j in idx_batch], y_train[idx_batch])
            loss = output['loss']
            loss.backward()
            if hparams['train_alternate'] == True:
                if (len(x_train) // hparams['batch_size'] * epoch + i) % hparams['n_mini_batch_update_gates'] == 0:
                    print("optimizing gates...")
                    optimizer_gates.step()
                else:
                    optimizer_classifier.step()
            else:
                optimizer_gates.step()
                optimizer_classifier.step()

        # print every n_batch_print mini-batches
        if epoch % hparams['n_epoch_eval'] == 0:
            # stats on train
            train_tracker.update(model, model(input.x_train, input.y_train), input.y_train, epoch, i)
            eval_tracker.update(model, model(input.x_eval, input.y_eval), input.y_eval, epoch, i)

            # compute
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (
                epoch, i, train_tracker.loss[-1], eval_tracker.loss[-1]))
            print('[Epoch %d, batch %d] training, eval ref_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.ref_reg_loss[-1], eval_tracker.ref_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval size_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.size_reg_loss[-1], eval_tracker.size_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval corner_reg_loss: %.3f, %.3f' % (
                epoch, i, train_tracker.corner_reg_loss[-1], eval_tracker.corner_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (
                epoch, i, train_tracker.acc[-1], eval_tracker.acc[-1]))

        # epoch_list = [0, 100, 300, 500, 1000, 1500, 2000]
        # epoch_list = [0, 100, 200, 300, 500, 700, 1000]
        epoch_list = [0, 50, 100, 200, 300, 400, 500]

        if model_checkpoint:
            if epoch+1 in epoch_list:#[100, 200, 300, 400, 500, 600]:
                model_checkpoint_dict[epoch+1] = deepcopy(model)

    print("Running time for training %d epoch: %.3f seconds" % (hparams['n_epoch'], time.time() - start))
    print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1], eval_tracker.acc_opt,
              eval_tracker.n_iter_opt[0],
              eval_tracker.n_iter_opt[1],))

    return model, train_tracker, eval_tracker, time.time() - start, model_checkpoint_dict


def run_output(model, dafi_tree, hparams, input, train_tracker, eval_tracker, run_time):
    """

    :param model:
    :param dafi_tree:
    :param hparams:
    :param input:
    :param train_tracker:
    :param eval_tracker:
    :param run_time:
    :return:
    """
    y_score_train = model(input.x_train, input.y_train)['y_pred'].detach().numpy()
    y_score_eval = model(input.x_eval, input.y_eval)['y_pred'].detach().numpy()
    y_score = model(input.x, input.y)['y_pred'].detach().numpy()
    y_pred_train = (y_score_train > 0.5) * 1.0
    y_pred_eval = (y_score_eval > 0.5) * 1.0
    y_pred = (y_score > 0.5) * 1.0
    train_accuracy = sum(y_pred_train == input.y_train.numpy()) * 1.0 / len(input.x_train)
    eval_accuracy = sum(y_pred_eval == input.y_eval.numpy()) * 1.0 / len(input.x_eval)
    overall_accuracy = sum(y_pred == input.y.numpy()) * 1.0 / len(input.x)
    train_auc = roc_auc_score(input.y_train.numpy(), y_pred_train, average='macro')
    eval_auc = roc_auc_score(input.y_eval.numpy(), y_pred_eval, average='macro')
    overall_auc = roc_auc_score(input.y.numpy(), y_pred, average='macro')
    train_brier_score = brier_score_loss(input.y_train.numpy(), y_score_train)
    eval_brier_score = brier_score_loss(input.y_eval.numpy(), y_score_eval)
    overall_brier_score = brier_score_loss(input.y.numpy(), y_score)

    y_score_train_dafi = dafi_tree(input.x_train, input.y_train)['y_pred'].detach().numpy()
    y_score_eval_dafi = dafi_tree(input.x_eval, input.y_eval)['y_pred'].detach().numpy()
    y_score_dafi = dafi_tree(input.x, input.y)['y_pred'].detach().numpy()
    y_pred_train_dafi = (y_score_train_dafi > 0.5) * 1.0
    y_pred_eval_dafi = (y_score_eval_dafi > 0.5) * 1.0
    y_pred_dafi = (y_score_dafi > 0.5) * 1.0
    train_accuracy_dafi = sum(y_pred_train_dafi == input.y_train.numpy()) * 1.0 / len(input.x_train)
    eval_accuracy_dafi = sum(y_pred_eval_dafi == input.y_eval.numpy()) * 1.0 / len(input.x_eval)
    overall_accuracy_dafi = sum(y_pred_dafi == input.y.numpy()) * 1.0 / len(input.x)
    train_auc_dafi = roc_auc_score(input.y_train.numpy(), y_pred_train_dafi, average='macro')
    eval_auc_dafi = roc_auc_score(input.y_eval.numpy(), y_pred_eval_dafi, average='macro')
    overall_auc_dafi = roc_auc_score(input.y.numpy(), y_pred_dafi, average='macro')
    train_brier_score_dafi = brier_score_loss(input.y_train.numpy(), y_score_train_dafi)
    eval_brier_score_dafi = brier_score_loss(input.y_eval.numpy(), y_score_eval_dafi)
    overall_brier_score_dafi = brier_score_loss(input.y.numpy(), y_score_dafi)

    with open('../output/%s/model_classifier_weights.csv' % hparams['experiment_name'], "a+") as file:
        bias = str(model.linear.bias.detach().item())
        weights = ', '.join(map(str, model.linear.weight.data[0].numpy()))
        file.write('%d, %s, %s\n' % (hparams['random_state'], bias, weights))

    with open('../output/%s/dafi_classifier_weights.csv' % hparams['experiment_name'], "a+") as file:
        bias = str(dafi_tree.linear.bias.detach().item())
        weights = ', '.join(map(str, dafi_tree.linear.weight.data[0].numpy()))
        file.write('%d, %s, %s\n' % (hparams['random_state'], bias, weights))

    with open('../output/%s/results_cll_4D.csv' % hparams['experiment_name'], "a+") as file:
        file.write(
            "%d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f([%d; %d]), %.3f([%d; %d]), %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f\n" % (
                hparams['random_state'],
                train_accuracy, eval_accuracy, overall_accuracy,
                train_accuracy_dafi, eval_accuracy_dafi, overall_accuracy_dafi,
                train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1],
                eval_tracker.acc_opt, eval_tracker.n_iter_opt[0], eval_tracker.n_iter_opt[1],
                model(input.x_train, input.y_train)['log_loss'].detach().numpy(),
                model(input.x_eval, input.y_eval)['log_loss'].detach().numpy(),
                model(input.x, input.y)['log_loss'].detach().numpy(),
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
    """

    :param hparams:
    :param train_tracker:
    :param eval_tracker:
    :param dafi_tree:
    :param input:
    :param output_metric_dict:
    :return:
    """
    x_range = [i * hparams['n_epoch_eval'] for i in range(hparams['n_epoch'] // hparams['n_epoch_eval'])]
    filename_metric = "../output/%s/metrics.png" % (hparams['experiment_name'])
    util_plot.plot_metrics(x_range, train_tracker, eval_tracker, filename_metric,
                           dafi_tree(input.x_train, input.y_train),
                           dafi_tree(input.x_eval, input.y_eval),
                           output_metric_dict)


def run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, input):
    """

    :param hparams:
    :param train_tracker:
    :param eval_tracker:
    :param model_tree:
    :param dafi_tree:
    :param input:
    :return:
    """
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
    util_plot.plot_cll_1p(input.x, filtered_normalized_x, input.y, input.features, model_tree, input.reference_tree,
                          train_tracker, model_pred, model_pred_prob, dafi_pred_prob,
                          filename_root_pas, filename_root_neg, filename_leaf_pas, filename_leaf_neg)


def run_gate_motion_step(hparams, input, model_checkpoint_dict, train_tracker, n_samples_plot=20):
    # select 10 samples for plotting
    idx_pos = [i for i in range(len(input.y)) if input.y[i] == 1][:(n_samples_plot // 2)]
    idx_neg = [i for i in range(len(input.y)) if input.y[i] == 0][:(n_samples_plot // 2)]
    idx_mask = sorted(idx_pos + idx_neg)
    input.x = [input.x[idx] for idx in idx_mask]
    input.y = torch.index_select(input.y, 0, torch.LongTensor(idx_mask))

    for epoch in model_checkpoint_dict:
        model_tree = model_checkpoint_dict[epoch]
        filename_root_pas = "../output/%s/gate_motion_root_pos_epoch%d.png" % (hparams['experiment_name'], epoch)
        filename_root_neg = "../output/%s/gate_motion_root_neg_epoch%d.png" % (hparams['experiment_name'], epoch)
        filename_leaf_pas = "../output/%s/gate_motion_leaf_pos_epoch%d.png" % (hparams['experiment_name'], epoch)
        filename_leaf_neg = "../output/%s/gate_motion_leaf_neg_epoch%d.png" % (hparams['experiment_name'], epoch)

        # filter out samples according DAFI gate at root for visualization at leaf
        #filtered_normalized_x = [dh.filter_rectangle(x, 0, 1, 0.402, 0.955, 0.549, 0.99) for x in input.x]
        filtered_normalized_x = [dh.filter_rectangle(x, 0, 1,
                                           F.sigmoid(model_tree.root.gate_low1_param),
                                           F.sigmoid(model_tree.root.gate_upp1_param),
                                           F.sigmoid(model_tree.root.gate_low2_param),
                                           F.sigmoid(model_tree.root.gate_upp2_param)) for x in input.x]
        util_plot.plot_cll_1p_light(input.x, filtered_normalized_x, input.y, input.features, model_tree,
                                    input.reference_tree,
                                    train_tracker, filename_root_pas, filename_root_neg, filename_leaf_pas,
                                    filename_leaf_neg)

def run_gate_motion_1p(hparams, input, model_checkpoint_dict):

    filename = "../output/%s/gate_motion.png" % hparams['experiment_name']
    # select checkpoints to plot, limit the length to 4
    epoch_list = [0, 100, 300, 500, 1000, 1500, 2000]#[100, 200, 300, 400, 500, 600]
    util_plot.plot_motion_p1(input, epoch_list, model_checkpoint_dict, filename)

def run_gate_motion_2p(hparams, input, model_checkpoint_dict):

    filename_p1 = "../output/%s/gate_motion_p1.png" % hparams['experiment_name']
    filename_p2 = "../output/%s/gate_motion_p2.png" % hparams['experiment_name']
    filename_p2_swap = "../output/%s/gate_motion_p2_swap.png" % hparams['experiment_name']
    # select checkpoints to plot, limit the length to 4
    # epoch_list = [0, 100, 300, 500, 1000, 1500, 2000]#[100, 200, 300, 400, 500, 600]
    epoch_list = [0, 50, 100, 200, 300, 400, 500]
    model_checkpoint_dict_p1 = {epoch: model_checkpoint_dict[epoch].model_trees[0] for epoch in epoch_list}
    model_checkpoint_dict_p2 = {epoch: model_checkpoint_dict[epoch].model_trees[1] for epoch in epoch_list}
    input_p1 = CLLInputBase()
    input_p1.x = [_[0] for _ in input.x]
    input_p1.y = input.y
    input_p1.features = input.features[0]
    input_p1.reference_tree = input.reference_tree[0]
    input_p2 = CLLInputBase()
    input_p2.x = [_[1] for _ in input.x]
    input_p2.y = input.y
    input_p2.features = input.features[1]
    input_p2.reference_tree = input.reference_tree[1]

    # input.x, input.y, input.features, input.reference_
    util_plot.plot_motion_p1(input_p1, epoch_list, model_checkpoint_dict_p1, filename_p1)
    util_plot.plot_motion_p2(input_p2, epoch_list, model_checkpoint_dict_p2, filename_p2)
    util_plot.plot_motion_p2_swap(input_p2, epoch_list, model_checkpoint_dict_p2, filename_p2_swap)



def run_write_prediction(model_tree, dafi_tree, input, hparams):

    with open("../output/%s/features_model.csv" % hparams['experiment_name'], "a+") as file:
        file.write("%d\n" % hparams['random_state'])
        np.savetxt(file, model_tree(input.x, input.y)['leaf_logp'].detach().numpy(), delimiter=',')
        file.write('\n')
    with open("../output/%s/features_dafi.csv" % hparams['experiment_name'], "a+") as file:
        file.write("%d\n" % hparams['random_state'])
        np.savetxt(file, dafi_tree(input.x, input.y)['leaf_logp'].detach().numpy(), delimiter=',')
        file.write('\n')
