import torch
from random import shuffle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from utils.bayes_gate_pytorch import *
from utils import plot as util_plot
import time
import torch.nn as nn

if __name__ == '__main__':

    # done: generate fake data
    x_train = [torch.rand((100, 4)) for i in range(1000)]
    y_train = torch.tensor([0 for i in range(400)] + [1 for i in range(600)], dtype=torch.float32)
    sample_id_train = [i for i in range(1000)]

    x_eval = [torch.rand((100, 4)), torch.rand((100, 4)), torch.rand((100, 4))]
    y_eval = torch.tensor([0, 0, 1], dtype=torch.float32)
    sample_id_eval = [1001, 1002, 1003]

    # done: create some kind of reference tree
    nested_list = \
        [
            [[u'CD5', 0.2, 0.8], [u'CD19', 0.2, 0.8]],
            [
                [
                    [[u'CD10', 0, 1], [u'CD79b', 0, 1]],
                    []
                ]
            ]
        ]

    nested_list_init = \
        [
            [[u'CD5', 0.5, 0.6], [u'CD19', 0.5, 0.6]],
            [
                [
                    [[u'CD10', 0.4, 0.8], [u'CD79b', 0.4, 0.8]],
                    []
                ]
            ]
        ]
    FEATURES = ['CD5', 'CD19', 'CD10', 'CD79b']
    FEATURE2ID = dict((FEATURES[i], i) for i in range(len(FEATURES)))
    LOGISTIC_K = 10
    REGULARIZATION_PENALTY = 1.
    reference_tree = ReferenceTree(nested_list, FEATURE2ID)
    init_tree = ReferenceTree(nested_list_init, FEATURE2ID)

    # done: Test ModelNode works: only the first gate [[u'CD5', 1638, 3891], [u'CD19', 2150, 3891]] is used.
    model_node = ModelNode(LOGISTIC_K, reference_tree, init_tree)
    for name, param in model_node.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # done: check results
    logp, reg_penalty = model_node(x_train[0])
    print("reg_penalty of root node with a different init_tree:", reg_penalty)
    model_node = ModelNode(LOGISTIC_K, reference_tree, init_tree=None)
    for name, param in model_node.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # done: check results
    logp, reg_penalty = model_node(x_train[0])
    print("reg_penalty of root node with the reference tree:", reg_penalty)

    # done: test ModelTree
    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY,
                           init_tree=init_tree)
    output = model_tree(x_train, y_train)
    print("reg_penalty and loss with a different init_tree:", output['reg_loss'], output['loss'])
    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY,
                           init_tree=None)
    output = model_tree(x_train, y_train)
    print("reg_penalty and loss with the reference tree:", output['reg_loss'], output['loss'])
    print(model_tree)
    print(model_tree.linear.weight, model_tree.linear.bias)
    print(model_tree.children_dict.items())
    output = model_tree(x_train, y_train)
    # todo: check results
    leaf_probs = output['leaf_probs']
    y_pred = output['y_pred']
    loss = output['loss']
    # print("leaf_probs:", leaf_probs)

    # todo: train the model
    # Keep track of losses for plotting
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []
    train_precision = []
    eval_precision = []
    train_recall = []
    eval_recall = []

    n_epoch = 10
    batch_size = int(len(x_train) / 10)
    n_epoch_print = 50

    # optimizer
    learning_rate = 3.14
    # optimizer = torch.optim.Adam(model_tree.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model_tree.parameters(), lr=learning_rate)

    start = time.time()

    for epoch in range(n_epoch):

        # shuffle training data
        idx_shuffle = np.array([_ for _ in range(len(x_train))])
        shuffle(idx_shuffle)
        x_train = [x_train[_] for _ in idx_shuffle]
        y_train = y_train[idx_shuffle]

        n_mini_batch = len(x_train) // batch_size

        for i in range(n_mini_batch):
            # generate mini batch data
            idx_batch = [_ for _ in range(batch_size * i, batch_size * (i + 1))]
            x_batch = [x_train[_] for _ in idx_batch]
            y_batch = y_train[idx_batch]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model_tree(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
            optimizer.step()

            # statistics
            y_pred = (output['y_pred'].data.numpy() > 0.5) * 1.0
            y_batch = y_batch.data.numpy()
            # leaf_probs = output['leaf_probs']
            train_loss.append(output['loss'])
            train_acc.append(sum(y_pred == y_batch) * 1.0 / batch_size)
            train_precision.append(precision_score(y_batch, y_pred, average='macro'))
            train_recall.append(recall_score(y_batch, y_pred, average='macro'))

        # print every n_batch_print mini-batches
        if epoch % n_epoch_print == n_epoch_print - 1:
            train_loss_avg = sum(train_loss[-n_mini_batch:]) * 1.0 / n_mini_batch
            train_acc_avg = sum(train_acc[-n_mini_batch:]) * 1.0 / n_mini_batch
            # eval
            output_eval = model_tree(x_eval, y_eval)
            # leaf_probs = output_eval['leaf_probs']
            y_pred = (output_eval['y_pred'].data.numpy() > 0.5) * 1.0
            eval_loss.append(output_eval['loss'])
            eval_acc.append(sum(y_pred == y_eval.data.numpy()) * 1.0 / len(x_eval))
            eval_precision.append(precision_score(y_eval.data.numpy(), y_pred, average='macro'))
            eval_recall.append(recall_score(y_eval.data.numpy(), y_pred, average='macro'))
            print(model_tree)
            print("=================")
            # print(output_eval['reg_loss'], output_eval['loss'],
            #       nn.BCEWithLogitsLoss(output_eval['leaf_probs'].squeeze(1), y_eval))
            # print(model_tree.linear.weight, model_tree.linear.bias, -model_tree.linear.bias/model_tree.linear.weight)
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (epoch, i, train_loss_avg, eval_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (epoch, i, train_acc_avg, eval_acc[-1]))

    # visualization
    ax = util_plot.plot_gate(x_train[0][:, 0], x_train[0][:,1 ], 'CD5', 'CD19', 0.2, 0.8, 0.2, 0.8, filename="../fig/test.png",
         normalized=True)

    print("Running time for training %d epoch: %.3f seconds" % (n_epoch, time.time() - start))
    print('Finished Training')
