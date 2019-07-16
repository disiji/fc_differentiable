import sys
import time
from copy import deepcopy
from random import shuffle

from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from utils import utils_plot as util_plot
from utils.input import *
from utils.utils_train import Tracker
from utils.input import CLLInputBase
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.bayes_gate import ModelTree
from utils.input import Cll4d1pInput
from utils.input import Cll8d1pInput

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

#Does training with full_batch for just the classifier weights until convergence with adam. Also no tr/te split-for us with just dev data
def run_train_only_logistic_regression(model, x_tensor_list, y, adam_lr, conv_thresh=1e-10, verbose=True, log_features=None):
    start = time.time() 
    classifier_params = [model.linear.weight, model.linear.bias]
    optimizer_classifier = torch.optim.Adam(classifier_params, lr=adam_lr)
    if log_features is None:
        output = model(x_tensor_list, y, detach_logistic_params=True)
        log_features = output['leaf_logp']
    BCEWithLogits = nn.BCEWithLogitsLoss()
    #these are called log_probs in the forwards function, but
    #calling them log_features is more consistent with our
    #previous usages in the paper and code
    
    prev_loss = -10 #making sure the loop starts
    delta = 50
    iters = 0
    while delta > conv_thresh:
        #features are fixed here, the only thing we need is the change in log loss from logistic params
        #forward pass through entire model is uneccessary!
        log_loss = BCEWithLogits(model.linear(log_features).squeeze(1), y)
        optimizer_classifier.zero_grad()
        log_loss.backward()
        optimizer_classifier.step()
        delta = torch.abs(log_loss - prev_loss)
        prev_loss = log_loss
        iters += 1
        if verbose:
            print(log_loss.item())
            if iters%100 == 0:
                print('%.6f ' %(delta), end='')
                if iters%500 == 0:
                    print('\n')
            print('\n')
            print('time taken %d, with loss %.2f' %(time.time() - start, log_loss.detach().item()))
    return model

def init_model_trackers_and_optimizers(hparams, input, model_checkpoint):
    model = ModelTree(input.reference_tree,
                           logistic_k=hparams['logistic_k'],
                           regularisation_penalty=hparams['regularization_penalty'],
                           negative_box_penalty=hparams['negative_box_penalty'],
                           positive_box_penalty=hparams['positive_box_penalty'],
                           corner_penalty=hparams['corner_penalty'],
                           gate_size_penalty=hparams['gate_size_penalty'],
                           init_tree=input.init_tree,
                           loss_type=hparams['loss_type'],
                           gate_size_default=hparams['gate_size_default'])
    if hparams['two_phase_training'] == False:
        raise ValueError('Only call run_train_model_two_phase with two phase setup in yaml!')
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
    eval_tracker = None
    if not(hparams['test_size'] == 0.):
        eval_tracker = Tracker()
        eval_tracker.model_init = deepcopy(model)
    train_tracker.model_init = deepcopy(model)
    model_checkpoint_dict = {}

    if model_checkpoint:
        model_checkpoint_dict[0] = deepcopy(model)

    if hparams['annealing']['anneal_logistic_k']:
        model.logistic_k = hparams['annealing']['init_k']

    if torch.cuda.is_available():
        model.cuda()
        #train_tracker.cuda()
        #eval_tracker.cuda()
        #optimizer_classifier.cuda()
        #optimizer_gates.cuda()
        #model_checkpoint_dict.cuda()
    return model, train_tracker, eval_tracker, optimizer_classifier, optimizer_gates, model_checkpoint_dict


def free_memory(variables_to_free):
    for var in variables_to_free:
        del var
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_train_model_two_phase(hparams, input, model_checkpoint=False):
    best_model_so_far = None
    best_log_loss_so_far = 1e10 #just a large number
    for random_init in range(hparams['two_phase_training']['num_random_inits_for_log_loss_only']):
        start = time.time()
        if random_init > 0:
            # have to free memory to avoid overflowing gpu with uneeded copies of the data and trackers
            #free_memory([model, train_tracker, eval_tracker, optimizer_classifier, 
            #    optimizer_gates, model_checkpoint_dict, input, output, output_detached])
            if type(input) == Cll8d1pInput:
                input = Cll8d1pInput(hparams)
            elif type(input) == Cll4d1pInput:
                input = Cll4d1pInput(hparams)
            else:
                raise ValueError('Class of input object not yet supported, update train.py')
        model, train_tracker, eval_tracker, optimizer_classifier, optimizer_gates, model_checkpoint_dict = init_model_trackers_and_optimizers(hparams, input, model_checkpoint)
        # First train just the logistic regressor for each init
        for epoch in range(hparams['two_phase_training']['num_only_log_loss_epochs']):
            
            # shuffle training data
            idx_shuffle = np.array([i for i in range(len(input.x_train))])
            shuffle(idx_shuffle)
            x_train = [input.x_train[_] for _ in idx_shuffle]
            y_train = input.y_train[idx_shuffle]
            
            if hparams['annealing']['anneal_logistic_k']:
                #reach the final k by the last epoch starting at init_k
                final_k = hparams['annealing']['final_k']
                init_k = hparams['annealing']['init_k']
                rate = np.log(final_k/init_k) * 1./(hparams['n_epoch'])
                model.logistic_k = hparams['annealing']['init_k'] * np.exp(rate * epoch)
                print('Current sharpness %.2f' %(model.logistic_k))

            for i in range(len(x_train) // hparams['batch_size']):
                idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
                optimizer_gates.zero_grad()
                optimizer_classifier.zero_grad()
                x_batch = [x_train[j] for j in idx_batch]
                y_batch = y_train[idx_batch]
    #            output = model([x_train[j] for j in idx_batch], y_train[idx_batch])
                if hparams['run_logistic_to_convergence']:
                    output_detached = model(x_batch, y_batch, detach_logistic_params=True)
                    output = model(x_batch, y_batch)
                else:
                    output = model(x_batch, y_batch)
                loss = output['log_loss']
                loss.backward()
                if hparams['train_alternate'] == True:
                    if hparams['run_logistic_to_convergence'] == True:
                        #kinda odd that this function uses its own optimizer in this case, may want to scrutinize this later
                        run_train_only_logistic_regression(model, x_batch, y_batch, hparams['learning_rate_classifier'], verbose=False, log_features=output_detached['leaf_logp'])
                        optimizer_gates.step()
                    else:
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
                if not(hparams['test_size'] == 0.):
                    eval_tracker.update(model, model(input.x_eval, input.y_eval), input.y_eval, epoch, i)

                # compute
                if hparams['test_size'] == 0.:
                    loss_tuple = (epoch, i, 'log loss:', train_tracker.log_loss[-1], 'acc:', train_tracker.acc[-1])
                    print('[Epoch %d, batch %d] %s %.3f, %s, %.3f' %loss_tuple)
                else:
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
        #epoch_list = [0, 50, 100, 200, 300, 400, 500]
        
            epoch_list = hparams['seven_epochs_for_gate_motion_plot']
            if model_checkpoint:
                if epoch+1 in epoch_list:#[100, 200, 300, 400, 500, 600]:
                    model_checkpoint_dict[epoch+1] = deepcopy(model)
        print("Running time for training %d epochs: %.3f seconds" % (hparams['two_phase_training']['num_only_log_loss_epochs'], time.time() - start))
        if not(hparams['test_size']) == 0.:
            print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
                  train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1], eval_tracker.acc_opt,
                    eval_tracker.n_iter_opt[0],
                    eval_tracker.n_iter_opt[1],))
        output_detached = model(input.x_train, input.y_train, detach_logistic_params=True)
        run_train_only_logistic_regression(model, input.x_train, input.y_train, hparams['learning_rate_classifier'], verbose=False, log_features=output_detached['leaf_logp'])
        output = model(input.x_train, input.y_train)
        if best_log_loss_so_far > output['log_loss']:
            best_log_loss_so_far = output['log_loss']
            best_model_so_far = model
        if best_log_loss_so_far < 1e-4:
            break

    print('Best loss obtained within %d random initializationss: %.3f' %(hparams['two_phase_training']['num_random_inits_for_log_loss_only'], best_log_loss_so_far))

    # Now train using the regularization terms as well as the log loss
    
    start = time.time()
    
    #only train the second part with the best model from before
    model = best_model_so_far
    for epoch in range(hparams['n_epoch'] - hparams['two_phase_training']['num_only_log_loss_epochs']):
        #free_memory([output, output_detached, x_train, y_train, x_batch, y_batch])
        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(input.x_train))])
        shuffle(idx_shuffle)
        x_train = [input.x_train[_] for _ in idx_shuffle]
        y_train = input.y_train[idx_shuffle]
    
        
        if hparams['annealing']['anneal_logistic_k']:
            #reach the final k by the last epoch starting at init_k
            final_k = hparams['annealing']['final_k']
            init_k = hparams['annealing']['init_k']
            rate = np.log(final_k/init_k) * 1./(hparams['n_epoch'])
            model.logistic_k = hparams['annealing']['init_k'] * np.exp(rate * epoch)
            print('Current sharpness %.2f' %(model.logistic_k))

        for i in range(len(x_train) // hparams['batch_size']):
            idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
            optimizer_gates.zero_grad()
            optimizer_classifier.zero_grad()
            x_batch = [x_train[j] for j in idx_batch]
            y_batch = y_train[idx_batch]
#            output = model([x_train[j] for j in idx_batch], y_train[idx_batch])
            if hparams['run_logistic_to_convergence']:
                output_detached = model(x_batch, y_batch, detach_logistic_params=True)
                output = model(x_batch, y_batch)
            else:
                output = model(x_batch, y_batch)

            loss = output['loss']
            loss.backward()
            if hparams['train_alternate'] == True:
                if hparams['run_logistic_to_convergence'] == True:
                    #kinda odd that this function uses its own optimizer in this case, may want to scrutinize this later
                    run_train_only_logistic_regression(model, x_batch, y_batch, hparams['learning_rate_classifier'], verbose=False, log_features=output_detached['leaf_logp'])
                    optimizer_gates.step()
                else:
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
            if not(hparams['test_size'] == 0.):
                eval_tracker.update(model, model(input.x_eval, input.y_eval), input.y_eval, epoch, i)

            # compute
            if hparams['test_size'] == 0.:
                loss_tuple = (epoch, i, 'full loss:', train_tracker.loss[-1], 'ref_reg:', train_tracker.ref_reg_loss[-1], 'size_reg:', train_tracker.size_reg_loss[-1], 'corner_reg:', train_tracker.corner_reg_loss[-1], 'acc:', train_tracker.acc[-1])
                print('[Epoch %d, batch %d] %s %.3f, %s, %.3f, %s, %.3f, %s, %.3f, %s, %.3f' %loss_tuple)
            else:
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
        #epoch_list = [0, 50, 100, 200, 300, 400, 500]
        
        # train the logistic regressor one more time
        output_detached = model(input.x_train, input.y_train, detach_logistic_params=True)
        run_train_only_logistic_regression(model, input.x_train, input.y_train, hparams['learning_rate_classifier'], verbose=False, log_features=output_detached['leaf_logp'])


        epoch_list = hparams['seven_epochs_for_gate_motion_plot']
        if model_checkpoint:
            if epoch+1 + hparams['two_phase_training']['num_only_log_loss_epochs'] in epoch_list:#[100, 200, 300, 400, 500, 600]:
                model_checkpoint_dict[epoch+1 + hparams['two_phase_training']['num_only_log_loss_epochs']] = deepcopy(model)
    print("Running time for training %d epoch: %.3f seconds" % (hparams['n_epoch'], time.time() - start))
    if not(hparams['test_size']) == 0.:
        print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1], eval_tracker.acc_opt,
              eval_tracker.n_iter_opt[0],
              eval_tracker.n_iter_opt[1],))

    return model, train_tracker, eval_tracker, time.time() - start, model_checkpoint_dict
        





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
    eval_tracker = None
    if not(hparams['test_size'] == 0.):
        eval_tracker = Tracker()
        eval_tracker.model_init = deepcopy(model)
    train_tracker.model_init = deepcopy(model)
    model_checkpoint_dict = {}

    if model_checkpoint:
        model_checkpoint_dict[0] = deepcopy(model)

    if hparams['annealing']['anneal_logistic_k']:
        model.logistic_k = hparams['annealing']['init_k']
    for epoch in range(hparams['n_epoch']):
        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(input.x_train))])
        shuffle(idx_shuffle)
        x_train = [input.x_train[_] for _ in idx_shuffle]
        y_train = input.y_train[idx_shuffle]
    
        
        if hparams['annealing']['anneal_logistic_k']:
            #reach the final k by the last epoch starting at init_k
            final_k = hparams['annealing']['final_k']
            init_k = hparams['annealing']['init_k']
            rate = np.log(final_k/init_k) * 1./(hparams['n_epoch'])
            model.logistic_k = hparams['annealing']['init_k'] * np.exp(rate * epoch)
            print('Current sharpness %.2f' %(model.logistic_k))

        for i in range(len(x_train) // hparams['batch_size']):
            idx_batch = [j for j in range(hparams['batch_size'] * i, hparams['batch_size'] * (i + 1))]
            optimizer_gates.zero_grad()
            optimizer_classifier.zero_grad()
            x_batch = [x_train[j] for j in idx_batch]
            y_batch = y_train[idx_batch]
#            output = model([x_train[j] for j in idx_batch], y_train[idx_batch])
            if hparams['run_logistic_to_convergence']:
                output_detached = model(x_batch, y_batch, detach_logistic_params=True)
                output = model(x_batch, y_batch)
            else:
                output = model(x_batch, y_batch)
            if hparams['two_phase_training']['turn_on'] == True:
                if epoch < hparams['two_phase_training']['num_only_log_loss_epochs']:
                    loss = output['log_loss']
                else:
                    loss = output['loss']
            else:
                loss = output['loss']
            loss.backward()
            if hparams['train_alternate'] == True:
                if hparams['run_logistic_to_convergence'] == True:
                    #kinda odd that this function uses its own optimizer in this case, may want to scrutinize this later
                    run_train_only_logistic_regression(model, x_batch, y_batch, hparams['learning_rate_classifier'], verbose=False, log_features=output_detached['leaf_logp'])
                    optimizer_gates.step()
                else:
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
            if not(hparams['test_size'] == 0.):
                eval_tracker.update(model, model(input.x_eval, input.y_eval), input.y_eval, epoch, i)

            # compute
            if hparams['test_size'] == 0.:
                loss_tuple = (epoch, i, 'full loss:', train_tracker.loss[-1], 'ref_reg:', train_tracker.ref_reg_loss[-1], 'size_reg:', train_tracker.size_reg_loss[-1], 'corner_reg:', train_tracker.corner_reg_loss[-1], 'acc:', train_tracker.acc[-1])
                print('[Epoch %d, batch %d] %s %.3f, %s, %.3f, %s, %.3f, %s, %.3f, %s, %.3f' %loss_tuple)
            else:
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
        #epoch_list = [0, 50, 100, 200, 300, 400, 500]
        epoch_list = hparams['seven_epochs_for_gate_motion_plot']
        if model_checkpoint:
            if epoch+1 in epoch_list:#[100, 200, 300, 400, 500, 600]:
                model_checkpoint_dict[epoch+1] = deepcopy(model)
    print("Running time for training %d epoch: %.3f seconds" % (hparams['n_epoch'], time.time() - start))
    if not(hparams['test_size']) == 0.:
        print("Optimal acc on train and eval during training process: %.3f at [Epoch %d, batch %d] "
          "and %.3f at [Epoch %d, batch %d]" % (
              train_tracker.acc_opt, train_tracker.n_iter_opt[0], train_tracker.n_iter_opt[1], eval_tracker.acc_opt,
              eval_tracker.n_iter_opt[0],
              eval_tracker.n_iter_opt[1],))

    return model, train_tracker, eval_tracker, time.time() - start, model_checkpoint_dict

def convert_gate(gate):
    if type(gate).__name__ == 'ModelNode':
        gate_low1 = torch.sigmoid(gate.gate_low1_param).item()

        gate_low2 = torch.sigmoid(gate.gate_low2_param).item()

        gate_upp1 = torch.sigmoid(gate.gate_upp1_param).item()

        gate_upp2 = torch.sigmoid(gate.gate_upp2_param).item()
    
    else:
        gate_low1 = gate.gate_low1

        gate_low2 = gate.gate_low2

        gate_upp1 = gate.gate_upp1

        gate_upp2 = gate.gate_upp2

    return([gate_low1, gate_upp1, gate_low2, gate_upp2])

def get_dafi_intersection_over_union_p1_avg(model, dafi_tree):
    root_model = model.root
    root_dafi = dafi_tree.root
    ratio = 0.
    print(model.root, dafi_tree.root)
    ratio += get_intersection_over_union(root_model, root_dafi)
    

    keys_model = [key for key in model.children_dict.keys()]
    keys_DAFI = [key for key in dafi_tree.children_dict.keys()]
 

    child_model = model.children_dict[keys_model[1]][0]
    child_dafi = dafi_tree.children_dict[keys_DAFI[1]][0]
    ratio += get_intersection_over_union(child_model, child_dafi)
    return ratio/2.
    


def get_intersection_over_union(gate_model, gate_dafi):
    if not((gate_model.gate_dim1 == gate_dafi.gate_dim1) and (gate_model.gate_dim2 == gate_dafi.gate_dim2)):
        raise ValueError('Gates are not from the same pair of axes/makers, so doesnt make sense to compute overlap-they\'re on different scatter plots!')

    flat_gate_model = convert_gate(gate_model)
    flat_gate_dafi = convert_gate(gate_dafi)

    dafi_overlap = get_overlap_p1_single_node(flat_gate_model, flat_gate_dafi)
    gate_model_area = (flat_gate_model[1] - flat_gate_model[0]) * (flat_gate_model[3] - flat_gate_model[2])
    gate_dafi_area = (flat_gate_dafi[1] - flat_gate_dafi[0]) * (flat_gate_dafi[3] - flat_gate_dafi[2])

    ratio_inter_union = dafi_overlap/(gate_dafi_area + gate_model_area - dafi_overlap)

    return ratio_inter_union
   
def test_overlap_p1_single_node():
    test_gate1 = [0, .5, 0, .25]
    test_gate2 = [.25, .5, 0, .25]
    test_gate3 = [0, .25, 0, .1]
    test_gate4 = [0, 1, 0, 1]
    assert(get_overlap_p1_single_node(test_gate1, test_gate1) == 1/8)
    assert(get_overlap_p1_single_node(test_gate1, test_gate2) == 1/16)
    assert(get_overlap_p1_single_node(test_gate2, test_gate1) == 1/16)
    assert(get_overlap_p1_single_node(test_gate1, test_gate3) == 1/40)
    assert(get_overlap_p1_single_node(test_gate3, test_gate1) == 1/40)
    assert(get_overlap_p1_single_node(test_gate1, test_gate4) == 1/8) 

def get_overlap_p1_single_node(flat_gate_model, flat_gate_dafi):
    if no_overlap(flat_gate_model, flat_gate_dafi):
        return 0.
    d1_cuts = [flat_gate_model[0], flat_gate_dafi[0], flat_gate_model[1], flat_gate_dafi[1]]
    d2_cuts = [flat_gate_model[2], flat_gate_dafi[2], flat_gate_model[3], flat_gate_dafi[3]]

    d1_length = sorted(d1_cuts)[2] - sorted(d1_cuts)[1]
    d2_length = sorted(d2_cuts)[2] - sorted(d2_cuts)[1]

    return d1_length * d2_length
    
    
def no_overlap(flat_gate_model, flat_gate_dafi):
    no_d1_overlap = (flat_gate_model[1] < flat_gate_dafi[0]) or (flat_gate_dafi[1] < flat_gate_model[0])

    no_d2_overlap = (flat_gate_model[3] < flat_gate_dafi[2]) or (flat_gate_dafi[3] < flat_gate_model[2])

    return (no_d1_overlap or no_d2_overlap)
    
        
        
#def get_dafi_overlap_all_nodes(model, dafi_tree):
#    this_level_model = [model.root]
#    this_level_dafi = dafi_tree
#    while this_level_model:
#        next_level_model = list()
#        next_level_dafi = list()
#        for model_node, dafi_tree_node in zip(this_level_model, :
#            dafi_tree_node = dafi_tree[0]
#            get_dafi_overlap(model_node, dafi_tree_node)
def run_lightweight_output_no_split_no_dafi(model, dafi_tree, hparams, input, train_tracker, eval_tracker, run_time, model_checkpoint_dict):
    y_score = model(input.x_train, input.y_train)['y_pred'].cpu().detach().numpy()
    y_pred = (y_score > 0.5) * 1.0
    overall_accuracy = sum(y_pred == input.y.cpu().numpy()) * 1.0 / len(input.x)
    if not type(input) == Cll8d1pInput:
        dafi_ratio_inter_union = get_dafi_intersection_over_union_p1_avg(model, dafi_tree)
    log_loss = model(input.x_train, input.y_train)['log_loss'].cpu().detach().numpy() 

    with open('../output/%s/model_classifier_weights.csv' % hparams['experiment_name'], "a+") as file:
        bias = str(model.linear.bias.detach().item())
        weights = ', '.join(map(str, model.linear.weight.data[0].cpu().numpy()))
        file.write('%d, %s, %s\n' % (hparams['random_state'], bias, weights))

    if not type(input) == Cll8d1pInput:
        with open('../output/%s/results_cll_4D.csv' % hparams['experiment_name'], "a+") as file:
            results_names = 'seed, overall_acc, log_loss, dafi_ratio_inter_union, run_time'
            file.write(results_names + '\n')
            file.write(
                "%d, %.3f, %.3f, %.3f, %.3f\n" % (
                    hparams['random_state'], overall_accuracy, log_loss, dafi_ratio_inter_union, run_time))
        output = {
            "overall_accuracy": overall_accuracy,
            'log_loss': log_loss,
            'dafi_overlap_ratio': dafi_ratio_inter_union,
            'run_time': run_time
            }
                
    else:
        with open('../output/%s/results_cll_4D.csv' % hparams['experiment_name'], "a+") as file:
            results_names = 'seed, overall_acc, log_loss, run_time'
            file.write(results_names + '\n')
            file.write(
                "%d, %.3f, %.3f, %.3f\n" % (
                    hparams['random_state'], overall_accuracy,
                    log_loss, run_time
                ))
    
        output = {
            "overall_accuracy": overall_accuracy,
            'log_loss': log_loss,
            'run_time': run_time
            }
    with open('../output/%s/model.pkl' %(hparams['experiment_name']), 'wb') as f:
        pickle.dump(model, f)

    with open('../output/%s/model_checkpoints.pkl' %(hparams['experiment_name']), 'wb') as f:
            pickle.dump(model_checkpoint_dict, f)


    return output


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
    print(filename)
    # select checkpoints to plot, limit the length to 4
    #epoch_list = [0, 100, 300, 500, 1000, 1500, 2000]#[100, 200, 300, 400, 500, 600]
    epoch_list = hparams['seven_epochs_for_gate_motion_plot']
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
        np.savetxt(file, model_tree(input.x, input.y)['leaf_logp'].cpu().detach().numpy(), delimiter=',')
        file.write('\n')
    with open("../output/%s/features_dafi.csv" % hparams['experiment_name'], "a+") as file:
        file.write("%d\n" % hparams['random_state'])
        np.savetxt(file, dafi_tree(input.x, input.y)['leaf_logp'].cpu().detach().numpy(), delimiter=',')
        file.write('\n')

    feats = model_tree(input.x, input.y)['leaf_logp'].cpu().detach().numpy()
    plt.clf()
    plt.scatter(np.arange(feats.shape[0]), feats)
    plt.title('features for each sample')
    plt.savefig('../output/%s/features_scatter_for_most_recent_run.png' %hparams['experiment_name'])

if __name__ == '__main__':
    test_overlap_p1_single_node()
