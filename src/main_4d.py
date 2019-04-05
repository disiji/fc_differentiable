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

if __name__ == '__main__':

    start = time.time()

    DATA_DIR = '../data/cll/'
    CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
    DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
    FEATURES = ['CD5', 'CD19', 'CD10', 'CD79b']
    FEATURES_FULL = ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3']
    FEATURE2ID = dict((FEATURES[i], i) for i in range(len(FEATURES)))
    LOGISTIC_K = 100
    LOGISTIC_K_DAFI = 1000
    REGULARIZATION_PENALTY = 0
    EMPTYNESS_PENALTY = 20
    GATE_SIZE_PENALTY = 0
    GATE_SIZE_DAFAULT = 1./4
    LOAD_DATA_FROM_PICKLE = True
    DAFI_INIT = False
    OPTIMIZER = "SGD"
    # OPTIMIZER = "Adam"
    if DAFI_INIT:
        INIT_METHOD = "dafi_init"
    else:
        INIT_METHOD = "random_init"
    LOSS_TYPE = 'logistic'  # or MSE
    # LOSS_TYPE = 'MSE"
    n_epoch_eval = 20
    # update classifier parameter and boundary parameter alternatively;
    # update boundary parameters after every 4 iterations of updating the classifer parameters
    n_mini_batch_update_gates = 50
    learning_rate_classifier = 0.01
    learning_rate_gates = 0.5
    # batch_size = 74
    batch_size = 10
    n_epoch = 1000
    n_epoch_dafi = n_epoch // n_mini_batch_update_gates * (n_mini_batch_update_gates - 1)

    # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
    # y: a list of labels; 1 is CLL, 0 is healthy
    if LOAD_DATA_FROM_PICKLE:
        with open(DATA_DIR + "filtered_4d_x_list.pkl", 'rb') as f:
            x = pickle.load(f)
        with open(DATA_DIR + 'y_list.pkl', 'rb') as f:
            y = pickle.load(f)
    else:
        x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, FEATURES_FULL)
        x_4d = dh.filter_cll_4d(x)
        with open(DATA_DIR + 'filtered_4d_x_list.pkl', 'wb') as f:
            pickle.dump(x_4d, f)
        with open(DATA_DIR + 'y_list.pkl', 'wb') as f:
            pickle.dump(y, f)
        x = x_4d  # rename for consistency

    # scale the data
    normalized_x, offset, scale = dh.normalize_x_list(x)
    print("Number of cells in each sample after filtering:", [_.shape[0] for _ in normalized_x])
    x_train, x_eval, y_train, y_eval = train_test_split(normalized_x, y, test_size=0.10, random_state=123)
    x_train = [torch.tensor(_, dtype=torch.float32) for _ in x_train]
    x_eval = [torch.tensor(_, dtype=torch.float32) for _ in x_eval]
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)
    normalized_x = [torch.tensor(_, dtype=torch.float32) for _ in normalized_x]
    y = torch.tensor(y, dtype=torch.float32)

    n_mini_batch = len(x_train) // batch_size

    print("Running time for loading the data: %.3f seconds." % (time.time() - start))

    nested_list = \
        [
            [[u'CD5', 1638., 3891], [u'CD19', 2150., 3891.]],
            [
                [
                    [[u'CD10', 0, 1228.], [u'CD79b', 0, 1843.]],
                    []
                ]
            ]
        ]
    nested_list_init = \
        [
            [[u'CD5', 2000., 3000.], [u'CD19', 2000., 3000.]],
            [
                [
                    [[u'CD10', 1000., 2000.], [u'CD79b', 1000., 2000.]],
                    []
                ]
            ]
        ]
    nested_list = dh.normalize_nested_tree(nested_list, offset, scale, FEATURE2ID)
    nested_list_init = dh.normalize_nested_tree(nested_list_init, offset, scale, FEATURE2ID)
    # AFTER NORMALIZATION...
    # nested_list = \
    #     [
    #         [[u'CD5', 0.402, 0.955], [u'CD19', 0.549, 0.99]],
    #         [
    #             [
    #                 [[u'CD10', 0, 0.300], [u'CD79b', 0, 0.465]],
    #                 []
    #             ]
    #         ]
    #     ]
    # nested_list_init = \
    #     [
    #         [[u'CD5', 0.490, 0.736], [u'CD19', 0.510, 0.766]],
    #         [
    #             [
    #                 [[u'CD10', 0.244, 0.488], [u'CD79b', 0.252, 0.504]],
    #                 []
    #             ]
    #         ]
    #     ]
    reference_tree = ReferenceTree(nested_list, FEATURE2ID)
    init_tree = ReferenceTree(nested_list_init, FEATURE2ID)
    if DAFI_INIT:
        init_tree = None

    # # Just for sanity check...
    # for logistic_k in [1, 10, 100, 1000, 10000, 100000]:
    #     model_tree = ModelTree(reference_tree, logistic_k=logistic_k, regularisation_penalty=REGULARIZATION_PENALTY,
    #                            emptyness_penalty=EMPTYNESS_PENALTY)
    #     # extract features with bounding boxes in the reference tree;
    #     # threshold = 0.0252
    #     threshold = 0.01
    #     features_train = model_tree(x_train, y_train)['leaf_probs'].detach().numpy()[:, 0]
    #     features_eval = model_tree(x_eval, y_eval)['leaf_probs'].detach().numpy()[:, 0]
    #     y_pred_train = (features_train > threshold) * 1.0
    #     y_pred_eval = (features_eval > threshold) * 1.0
    #     print("With SOFT features(steepness = %d) extracted with bounding boxes in reference tree..." % logistic_k)
    #     print("Acc on training and eval data: %.3f, %.3f" % (
    #         sum((y_pred_train == y_train.numpy())) * 1.0 / len(x_train),
    #         sum((y_pred_eval == y_eval.numpy())) * 1.0 / len(x_eval)))


    # train differentiable gates model
    start = time.time()
    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY,
                           emptyness_penalty=EMPTYNESS_PENALTY, gate_size_penalty=GATE_SIZE_PENALTY,
                           init_tree=init_tree, loss_type=LOSS_TYPE, gate_size_default=GATE_SIZE_DAFAULT)

    # Keep track of losses for plotting
    train_loss = []
    train_log_loss = []
    train_ref_reg_loss = []
    train_size_reg_loss = []
    eval_loss = []
    eval_log_loss = []
    eval_ref_reg_loss = []
    eval_size_reg_loss = []
    train_acc = []
    eval_acc = []
    train_precision = []
    eval_precision = []
    train_recall = []
    eval_recall = []
    log_decision_boundary = []

    # optimal gates
    root_gate_init = deepcopy(model_tree.root)
    leaf_gate_init = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
    train_root_gate_opt = None
    train_leaf_gate_opt = None
    eval_root_gate_opt = None
    eval_leaf_gate_opt = None
    train_acc_opt = 0
    eval_acc_opt = 0
    train_n_iter_opt = (0, 0)
    eval_n_iter_opt = (0, 0)

    # optimizer
    classifier_params = [model_tree.linear.weight, model_tree.linear.bias]
    gates_params = [p for p in model_tree.parameters() if p not in classifier_params]
    if OPTIMIZER == "SGD":
        optimizer_classifier = torch.optim.SGD(classifier_params, lr=learning_rate_classifier)
        optimizer_gates = torch.optim.SGD(gates_params, lr=learning_rate_gates)
    else:
        optimizer_classifier = torch.optim.Adam(classifier_params, lr=learning_rate_classifier)
        optimizer_gates = torch.optim.Adam(gates_params, lr=learning_rate_gates)

    for epoch in range(n_epoch):
        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(x_train))])
        shuffle(idx_shuffle)
        x_train = [x_train[_] for _ in idx_shuffle]
        y_train = y_train[idx_shuffle]

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
                print("optimizing gates...")
                optimizer_gates.step()
            else:
                optimizer_classifier.step()

        # print every n_batch_print mini-batches
        if epoch % n_epoch_eval == 0:
            print(model_tree)
            log_decision_boundary.append((-model_tree.linear.bias.detach() / model_tree.linear.weight.detach()))
            # stats on train
            output_train = model_tree(x_train, y_train)
            y_train_pred = (output_train['y_pred'].detach().numpy() > 0.5) * 1.0
            train_loss.append(output_train['loss'])
            train_log_loss.append(output_train['log_loss'])
            train_ref_reg_loss.append(output_train['ref_reg_loss'])
            train_size_reg_loss.append(output_train['size_reg_loss'])
            train_acc.append(sum(y_train_pred == y_train.numpy()) * 1.0 / len(x_train))
            train_precision.append(precision_score(y_train.numpy(), y_train_pred, average='macro'))
            train_recall.append(recall_score(y_train.numpy(), y_train_pred, average='macro'))

            # stats on eval
            output_eval = model_tree(x_eval, y_eval)
            # leaf_probs = output_eval['leaf_probs']
            y_eval_pred = (output_eval['y_pred'].detach().numpy() > 0.5) * 1.0
            eval_loss.append(output_eval['loss'])
            eval_log_loss.append(output_eval['log_loss'])
            eval_ref_reg_loss.append(output_eval['ref_reg_loss'])
            eval_size_reg_loss.append(output_eval['size_reg_loss'])
            eval_acc.append(sum(y_eval_pred == y_eval.numpy()) * 1.0 / len(x_eval))
            eval_precision.append(precision_score(y_eval.numpy(), y_eval_pred, average='macro'))
            eval_recall.append(recall_score(y_eval.numpy(), y_eval_pred, average='macro'))

            # keep track of optimal gates for train and eval set
            if train_acc[-1] > train_acc_opt:
                train_root_gate_opt = deepcopy(model_tree.root)
                train_leaf_gate_opt = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
                train_acc_opt = train_acc[-1]
                train_n_iter_opt = (epoch, i)
            if eval_acc[-1] > eval_acc_opt:
                eval_root_gate_opt = deepcopy(model_tree.root)
                eval_leaf_gate_opt = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
                eval_acc_opt = eval_acc[-1]
                eval_n_iter_opt = (epoch, i)

            # compute
            print(output_eval['ref_reg_loss'], output_eval['size_reg_loss'], output_eval['loss'])
            print("w, b, (-b/w):", model_tree.linear.weight.detach().numpy(),
                  model_tree.linear.bias.detach().numpy(), log_decision_boundary[-1])
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (
                epoch, i, train_loss[-1], eval_loss[-1]))
            print('[Epoch %d, batch %d] training, eval ref_reg_loss: %.3f, %.3f' % (
                epoch, i, train_ref_reg_loss[-1], eval_ref_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval size_reg_loss: %.3f, %.3f' % (
                epoch, i, train_size_reg_loss[-1], eval_size_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (
                epoch, i, train_acc[-1], eval_acc[-1]))


    ########### train a classifier on the top of DAFi features
    start = time.time()
    dafi_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K_DAFI, regularisation_penalty=REGULARIZATION_PENALTY,
                          emptyness_penalty=EMPTYNESS_PENALTY, gate_size_penalty=GATE_SIZE_PENALTY, init_tree=None,
                          loss_type=LOSS_TYPE, gate_size_default=GATE_SIZE_DAFAULT)
    if OPTIMIZER == "SGD":
        dafi_optimizer_classifier = torch.optim.SGD([dafi_tree.linear.weight, dafi_tree.linear.bias],
                                                lr=learning_rate_classifier)
    else:
        dafi_optimizer_classifier = torch.optim.Adam([dafi_tree.linear.weight, dafi_tree.linear.bias],
                                                lr=learning_rate_classifier)

    for epoch in range(n_epoch_dafi):
        idx_shuffle = np.array([i for i in range(len(x_train))])
        shuffle(idx_shuffle)
        x_train = [x_train[_] for _ in idx_shuffle]
        y_train = y_train[idx_shuffle]
        for i in range(n_mini_batch):
            idx_batch = [j for j in range(batch_size * i, batch_size * (i + 1))]
            x_batch = [x_train[j] for j in idx_batch]
            y_batch = y_train[idx_batch]
            dafi_optimizer_classifier.zero_grad()
            output = dafi_tree(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
            dafi_optimizer_classifier.step()
    print("Running time for training classifier with DAFi gates: %.3f seconds." % (time.time() - start))

    ####### compute model_pred_prob
    model_pred_prob = model_tree(normalized_x, y)['y_pred'].detach().numpy()
    model_pred = (model_pred_prob > 0.5) * 1.0
    dafi_pred_prob = dafi_tree(normalized_x, y)['y_pred'].detach().numpy()
    dafi_pred = (dafi_pred_prob > 0.5) * 1.0

    y_pred_train_dafi = (dafi_tree(x_train, y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
    y_pred_eval_dafi = (dafi_tree(x_eval, y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
    train_acc_dafi = sum(y_pred_train_dafi == y_train.numpy()) * 1.0 / len(x_train)
    eval_acc_dafi = sum(y_pred_eval_dafi == y_eval.numpy()) * 1.0 / len(x_eval)
    overall_acc_dafi = sum(
        (dafi_tree(normalized_x, y)['y_pred'].detach().numpy() > 0.5) * 1.0 == y.numpy()) * 1.0 / len(x)

    ##################### write results
    print("Running time for training %d epoch: %.3f seconds" % (n_epoch, time.time() - start))
    print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_acc_opt, train_n_iter_opt[0], train_n_iter_opt[1], eval_acc_opt, eval_n_iter_opt[0],
              eval_n_iter_opt[1],))
    with open('../output/results_cll_4D.csv', "a+") as file:
        y_train_pred = (model_tree(x_train, y_train)['y_pred'].detach().numpy() > 0.5) * 1.0
        y_eval_pred = (model_tree(x_eval, y_eval)['y_pred'].detach().numpy() > 0.5) * 1.0
        y_pred = (model_tree(normalized_x, y)['y_pred'].detach().numpy() > 0.5) * 1.0
        train_accuracy = sum(y_train_pred == y_train.numpy()) * 1.0 / len(x_train)
        eval_accuracy = sum(y_eval_pred == y_eval.numpy()) * 1.0 / len(x_eval)
        overall_accuracy = sum(y_pred == y.numpy()) * 1.0 / len(x)
        file.write(
            "%d, %d, %.3f, %d, %d, %s, %s, %d, %d, %d, %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f([%d; %d]), %.3f([%d; %d]), %.3f, %.3f, %.3f, %.3f,  %.3f, %.3f, %.3f\n" % (
                LOGISTIC_K, LOGISTIC_K_DAFI, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, GATE_SIZE_PENALTY, INIT_METHOD, LOSS_TYPE,
                n_epoch, batch_size, n_epoch_eval, n_mini_batch_update_gates,
                learning_rate_classifier, learning_rate_gates,
                train_accuracy, eval_accuracy, overall_accuracy,
                train_acc_dafi, eval_acc_dafi, overall_acc_dafi,
                train_acc_opt, train_n_iter_opt[0], train_n_iter_opt[1],
                eval_acc_opt, eval_n_iter_opt[0], eval_n_iter_opt[1],
                model_tree(x_train, y_train)['log_loss'].detach().numpy(),
                model_tree(x_eval, y_eval)['log_loss'].detach().numpy(),
                model_tree(normalized_x, y)['log_loss'].detach().numpy(),
                dafi_tree(x_train, y_train)['log_loss'].detach().numpy(),
                dafi_tree(x_eval, y_eval)['log_loss'].detach().numpy(),
                dafi_tree(normalized_x, y)['log_loss'].detach().numpy(),
                time.time() - start))

    ##################### visualization

    ##### plot metrics
    x_range = [i * n_epoch_eval for i in range(n_epoch // n_epoch_eval)]
    figname_metric = "../fig/4D_k%d_reg%.1f_emp%d_gatesize%d_nepoch%d_batchsize%d_%s_%s_metrics.png" % (
        LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, GATE_SIZE_PENALTY, n_epoch, batch_size, INIT_METHOD,
        LOSS_TYPE)
    util_plot.plot_metrics(x_range, train_loss, eval_loss, train_log_loss, eval_log_loss, train_ref_reg_loss,
                           eval_ref_reg_loss, train_size_reg_loss, eval_size_reg_loss,
                           train_acc, eval_acc, log_decision_boundary, figname_metric, dafi_tree(x_train, y_train),
                           dafi_tree(x_eval, y_eval), train_acc_dafi, eval_acc_dafi)

    ##### plot gates
    figname_root_pos = "../fig/4D_k%d_reg%.1f_emp%d_gatesize%d_nepoch%d_batchsize%d_%s_%s_root_pos.png" % (
        LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, GATE_SIZE_PENALTY, n_epoch, batch_size, INIT_METHOD,
        LOSS_TYPE)
    figname_root_neg = "../fig/4D_k%d_reg%.1f_emp%d_gatesize%d_nepoch%d_batchsize%d_%s_%s_root_neg.png" % (
        LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, GATE_SIZE_PENALTY, n_epoch, batch_size, INIT_METHOD,
        LOSS_TYPE)
    figname_leaf_pos = "../fig/4D_k%d_reg%.1f_emp%d_gatesize%d_nepoch%d_batchsize%d_%s_%s_leaf_pos.png" % (
        LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, GATE_SIZE_PENALTY, n_epoch, batch_size, INIT_METHOD,
        LOSS_TYPE)
    figname_leaf_neg = "../fig/4D_k%d_reg%.1f_emp%d_gatesize%d_nepoch%d_batchsize%d_%s_%s_leaf_neg.png" % (
        LOGISTIC_K, REGULARIZATION_PENALTY, EMPTYNESS_PENALTY, GATE_SIZE_PENALTY, n_epoch, batch_size, INIT_METHOD,
        LOSS_TYPE)

    # filter out samples according DAFI gate at root for visualization at leaf
    filtered_normalized_x = [dh.filter_rectangle(x, 0, 1, 0.402, 0.955, 0.549, 0.99) for x in normalized_x]

    util_plot.plot_cll(normalized_x, filtered_normalized_x, y, FEATURES, model_tree, reference_tree,
                       train_root_gate_opt, eval_root_gate_opt, root_gate_init,
                       train_leaf_gate_opt, eval_leaf_gate_opt, leaf_gate_init,
                       model_pred, model_pred_prob, dafi_pred, dafi_pred_prob,
                       figname_root_pos, figname_root_neg, figname_leaf_pos, figname_leaf_neg)

    print("end")
