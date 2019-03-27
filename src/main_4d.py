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
from math import *

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
    LOAD_DATA_FROM_PICKLE = True

    # x: a list of samples, each entry is a numpy array of shape n_cells * n_features
    # y: a list of labels; 1 is CLL, 0 is healthy
    if LOAD_DATA_FROM_PICKLE:
        with open(DATA_DIR + "filtered_4d_x_list.pkl", 'rb') as f:
            x = pickle.load(f)
        with open(DATA_DIR + 'y_list.pkl', 'rb') as f:
            y = pickle.load(f)
    else:
        x, y = dh.load_cll_data(DIAGONOSIS_FILENAME, CYTOMETRY_DIR, FEATURES_FULL)
        x = dh.filter_cll_4d(x)
        with open(DATA_DIR + 'filtered_4d_x_list.pkl', 'wb') as f:
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
    # x_train = [torch.tensor(_, dtype=torch.float32) for _ in x_train]
    # x_eval = [torch.tensor(_, dtype=torch.float32) for _ in normalized_x]
    # y_train = torch.tensor(y_train, dtype=torch.float32)
    # y_eval = torch.tensor(y, dtype=torch.float32)

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
    nested_list = \
        [
            [[u'CD5', 0.402, 0.955], [u'CD19', 0.549, 0.99]],
            [
                [
                    [[u'CD10', 0, 0.300], [u'CD79b', 0, 0.465]],
                    []
                ]
            ]
        ]
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
    init_tree = None

    # Just for sanity check...
    for logistic_k in [1, 10, 100, 1000, 10000, 100000]:
        model_tree = ModelTree(reference_tree, logistic_k=logistic_k, regularisation_penalty=REGULARIZATION_PENALTY)
        # extract features with bounding boxes in the reference tree;
        # threshold = 0.0252
        threshold = 0.01
        features_train = model_tree(x_train, y_train)['leaf_probs'].detach().numpy()[:, 0]
        features_eval = model_tree(x_eval, y_eval)['leaf_probs'].detach().numpy()[:, 0]
        y_pred_train = (features_train > threshold) * 1.0
        y_pred_eval = (features_eval > threshold) * 1.0
        print("With SOFT features(steepness = %d) extracted with bounding boxes in reference tree..." % logistic_k)
        print("Acc on training and eval data: %.3f, %.3f" % (
            sum((y_pred_train == y_train.numpy())) * 1.0 / len(x_train),
            sum((y_pred_eval == y_eval.numpy())) * 1.0 / len(x_eval)))

    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY,
                           init_tree=init_tree)
    print("Initialize the model tree as:", model_tree)

    # Keep track of losses for plotting
    train_loss = []
    train_reg_loss = []
    eval_loss = []
    eval_reg_loss = []
    train_acc = []
    eval_acc = []
    train_precision = []
    eval_precision = []
    train_recall = []
    eval_recall = []
    log_decision_boundary = []

    n_epoch = 1000
    batch_size = 5
    n_epoch_eval = 10
    n_mini_batch_update_gates = 100
    # update classifier parameter and boundary parameter alternatively;
    # update boundary parameters after every 4 iterations of updating the classifer parameters

    # optimizer
    learning_rate_classifier = 0.05
    learning_rate_gates = 0.5
    # optimizer = torch.optim.Adam(model_tree.parameters(), lr=learning_rate)
    classifier_params = [model_tree.linear.weight, model_tree.linear.bias]
    gates_params = [p for p in model_tree.parameters() if p not in classifier_params]
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
                print("optimizing gates...")
                optimizer_gates.step()
            else:
                optimizer_classifier.step()

            # statistics
            y_pred = (output['y_pred'].data.numpy() > 0.5) * 1.0
            y_batch = y_batch.data.numpy()
            # leaf_probs = output['leaf_probs']
            train_loss.append(output['loss'])
            train_reg_loss.append(output['reg_loss'])
            train_acc.append(sum(y_pred == y_batch) * 1.0 / batch_size)
            train_precision.append(precision_score(y_batch, y_pred, average='macro'))
            train_recall.append(recall_score(y_batch, y_pred, average='macro'))
            log_decision_boundary.append((-model_tree.linear.bias.detach() / model_tree.linear.weight.detach()))

        # print every n_batch_print mini-batches
        if epoch % n_epoch_eval == 0:
            print(model_tree)
            train_loss_avg = sum(train_loss[-n_mini_batch:]) * 1.0 / n_mini_batch
            train_reg_loss_avg = sum(train_reg_loss[-n_mini_batch:]) * 1.0 / n_mini_batch
            train_acc_avg = sum(train_acc[-n_mini_batch:]) * 1.0 / n_mini_batch
            # eval
            output_eval = model_tree(x_eval, y_eval)
            # leaf_probs = output_eval['leaf_probs']
            print(output_eval['y_pred'])
            y_eval_pred = (output_eval['y_pred'].detach().numpy() > 0.5) * 1.0
            eval_loss.append(output_eval['loss'])
            eval_reg_loss.append(output_eval['reg_loss'])
            eval_acc.append(sum(y_eval_pred == y_eval.numpy()) * 1.0 / len(x_eval))
            eval_precision.append(precision_score(y_eval.numpy(), y_eval_pred, average='macro'))
            eval_recall.append(recall_score(y_eval.numpy(), y_eval_pred, average='macro'))


            print(model_tree)
            print(output_eval['reg_loss'], output_eval['loss'])
            print("w, b, (-b/w):", model_tree.linear.weight.detach().numpy(),
                  model_tree.linear.bias.detach().numpy(),
                  log_decision_boundary[-1])
            print('[Epoch %d, batch %d] training, eval loss: %.3f, %.3f' % (epoch, i, train_loss_avg, eval_loss[-1]))
            print('[Epoch %d, batch %d] training, eval reg loss: %.3f, %.3f' % (
                epoch, i, train_reg_loss_avg, eval_reg_loss[-1]))
            print('[Epoch %d, batch %d] training, eval acc: %.3f, %.3f' % (epoch, i, train_acc_avg, eval_acc[-1]))
            # fix the threshold
            features_train = model_tree(x_train, y_train)['leaf_probs'].detach().numpy()[:, 0]
            features_eval = model_tree(x_eval, y_eval)['leaf_probs'].detach().numpy()[:, 0]
            threshold = 0.01
            y_pred_train = (features_train > threshold) * 1.0
            y_pred_eval = (features_eval > threshold) * 1.0
            print("With features learned and set decision boundary at 0.01...")
            print("Acc on training and eval data: %.3f, %.3f" % (
                sum((y_pred_train == y_train.numpy())) * 1.0 / len(x_train),
                sum((y_pred_eval == y_eval.numpy())) * 1.0 / len(x_eval)))
            threshold = 0.028
            y_pred_train = (features_train > threshold) * 1.0
            y_pred_eval = (features_eval > threshold) * 1.0
            print("With features learned and set decision boundary at 0.028...")
            print("Acc on training and eval data: %.3f, %.3f" % (
                sum((y_pred_train == y_train.numpy())) * 1.0 / len(x_train),
                sum((y_pred_eval == y_eval.numpy())) * 1.0 / len(x_eval)))

    print("Running time for training %d epoch: %.3f seconds" % (n_epoch, time.time() - start))
    import matplotlib.pyplot as plt

    plt.plot(train_loss)
    plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_loss))], eval_loss)
    plt.legend(["train loss", "eval loss"])
    plt.show()
    plt.plot(train_reg_loss)
    plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_reg_loss))], eval_reg_loss)
    plt.legend(["train reg loss", "eval reg loss"])
    plt.show()
    plt.plot(train_acc)
    plt.plot([i * n_epoch_eval * n_mini_batch for i in range(len(eval_acc))], eval_acc)
    plt.legend(["train acc", "eval acc"])
    plt.show()
    plt.plot(log_decision_boundary)
    plt.legend(["log decision boundary"])
    plt.show()
    print('Finished Training')
