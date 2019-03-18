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
    LOGISTIC_K = 10.0
    REGULARIZATION_PENALTY = 1.

    # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
    # y: a list of labels; 1 is CLL, 0 is healthy
    x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, FEATURES_FULL)
    x = dh.filter_cll_4d(x)
    # scale the data
    x = [_/1000.0 for _ in x]
    print("Number of cells in each sample after filtering:", [_.shape[0] for _ in x])
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.10, random_state=123)
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
    # nested_list = \
    #     [
    #         [[u'CD5', 1000./1000, 3891./1000], [u'CD19', 1000./1000, 3891./1000]],
    #         [
    #             [
    #                 [[u'CD10', 1000./1000, 1228./1000], [u'CD79b', 1000./1000, 1843./1000]],
    #                 []
    #             ]
    #         ]
    #     ]
    reference_tree = ReferenceTree(nested_list, FEATURE2ID)

    # Just for sanity check...
    # for logistic_k in [1, 10, 100, 1000, 10000]:
    #     model_tree = ModelTree(reference_tree, logistic_k=logistic_k, regularisation_penalty=REGULARIZATION_PENALTY)
    #     # extract features with bounding boxes in the reference tree;
    #     threshold = 0.0252
    #     features_train = model_tree(x_train, y_train)['leaf_probs'].numpy()[:, 0]
    #     features_eval = model_tree(x_eval, y_eval)['leaf_probs'].numpy()[:, 0]
    #     y_pred_train = (features_train > threshold) * 1.0
    #     y_pred_eval = (features_eval > threshold) * 1.0
    #     print("With SOFT features(steepness = %d) extracted with bounding boxes in reference tree..." % logistic_k)
    #     print("Acc on trainning and eval data: %.3f, %.3f"% (
    #         sum((y_pred_train == y_train.numpy())) * 1.0 / len(x_train),
    #         sum((y_pred_eval == y_eval.numpy())) * 1.0 / len(x_eval)))

    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY)


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
    batch_size = len(x_train)
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
            print(output_eval['y_pred'])
            y_pred = (output_eval['y_pred'].data.numpy() > 0.5) * 1.0
            eval_loss.append(output_eval['loss'])
            eval_acc.append(sum(y_pred == y_eval.data.numpy()) * 1.0 / len(x_eval))
            eval_precision.append(precision_score(y_eval.data.numpy(), y_pred, average='macro'))
            eval_recall.append(recall_score(y_eval.data.numpy(), y_pred, average='macro'))
            print(model_tree)
            # print(model_tree.linear.weight, model_tree.linear.bias, -model_tree.linear.bias/model_tree.linear.weight)
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (epoch, i, train_loss_avg, eval_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (epoch, i, train_acc_avg, eval_acc[-1]))

    print("Running time for training %d epoch: %.3f seconds" % (n_epoch, time.time() - start))
    print('Finished Training')


