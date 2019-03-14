from random import shuffle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from utils.bayes_gate_pytorch import *
import utils.load_data as dh
from sklearn.model_selection import train_test_split
import time
import torch

if __name__ == '__main__':

    start = time.time()

    DATA_DIR = '../data/cll/'
    CYTOMETRY_DIR = DATA_DIR + "PB1_whole_mqian/"
    DIAGONOSIS_FILENAME = DATA_DIR + 'PB.txt'
    FEATURES = ['CD5', 'CD19', 'CD10', 'CD79b']
    FEATURES_FULL = ['FSC-A', 'FSC-H', 'SSC-H', 'CD45', 'SSC-A', 'CD5', 'CD19', 'CD10', 'CD79b', 'CD3']
    FEATURE2ID = dict((FEATURES[i], i) for i in range(len(FEATURES)))
    LOGISTIC_K = 1000
    REGULARIZATION_PENALTY = 1.

    # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
    # y: a list of labels; 1 is CLL, 0 is healthy
    x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, FEATURES_FULL)
    x = dh.filter_cll_4d(x)
    # scale the data
    x = [_/1000 for _ in x]
    print("number of cells in each sample after filtering:", [_.shape[0] for _ in x])
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.20, random_state=42)
    x_train = [torch.tensor(_, dtype=torch.float32) for _ in x_train]
    x_eval = [torch.tensor(_, dtype=torch.float32) for _ in x_eval]
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)

    print("Running time for loading the data: %.3f seconds." % (time.time() - start))

    nested_list = \
        [
            [[u'CD5', 1638./1000, 3891./1000], [u'CD19', 2150./1000, 3891./1000]],
            [
                [
                    [[u'CD10', 0, 1228./1000], [u'CD79b', 0, 1843./1000]],
                    []
                ]
            ]
        ]
    reference_tree = ReferenceTree(nested_list, FEATURE2ID)
    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY)

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

    n_epoch = 1000
    batch_size = 10
    n_batch_print = 100

    # optimizer
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model_tree.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model_tree.parameters(), lr=0.001, momentum=0.9)

    start = time.time()

    for epoch in range(n_epoch):

        running_loss_train = 0.0

        # shuffle training data
        idx_shuffle = np.array([_ for _ in range(len(x_train))])
        shuffle(idx_shuffle)

        x_train = [x_train[_] for _ in idx_shuffle]
        y_train = y_train[idx_shuffle]

        for i in range(len(x_train) // batch_size):

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
            y_pred = (output['y_pred'].data.numpy() > 0) * 1.0
            y_batch = y_batch.data.numpy()
            # leaf_probs = output['leaf_probs']
            train_loss.append(output['loss'])
            train_acc.append(sum(y_pred == y_batch) * 1.0 / batch_size)
            train_precision.append(precision_score(y_batch, y_pred, average='macro'))
            train_recall.append(recall_score(y_batch, y_pred, average='macro'))

            # print every n_batch_print mini-batches
            if (i + epoch * len(x_train) // batch_size) % n_batch_print == n_batch_print - 1:
                train_loss_avg = sum(train_loss[-n_batch_print:]) * 1.0 / n_batch_print
                train_acc_avg = sum(train_acc[-n_batch_print:]) * 1.0 / n_batch_print
                # eval
                output_eval = model_tree(x_eval, y_eval)
                # leaf_probs = output_eval['leaf_probs']
                y_pred = (output_eval['y_pred'].data.numpy() > 0) * 1.0
                eval_loss.append(output_eval['loss'])
                eval_acc.append(sum(y_pred == y_eval.data.numpy()) * 1.0 / len(x_eval))
                eval_precision.append(precision_score(y_eval.data.numpy(), y_pred, average='macro'))
                eval_recall.append(recall_score(y_eval.data.numpy(), y_pred, average='macro'))

                print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (epoch, i, train_loss_avg, eval_loss[-1]))
                print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (epoch, i, train_acc_avg, eval_acc[-1]))

    print("Running time for training %d epoch: %.3f seconds" % (n_epoch, time.time() - start))
    print('Finished Training')


