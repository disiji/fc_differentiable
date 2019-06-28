from __future__ import division

from math import *

import torch
import torch.nn as nn
import torch.nn.functional as F

#used only for referenceTree object, cuts here don't have to be passed through sigmoid activations
class Gate(object):
    def __init__(self, gate_tuple, features2id):
        """
        :param gate_tuple: [[dim1, low1, upp1], [dim2, low2, upp2]]
        self.gate_dim1: a string of the feature name
        self.gate_dim2: a string of the feature name
        self.gate_low1: float
        self.gate_low2: float
        self.gate_upp1: float
        self.gate_upp2: float
        """
        if len(gate_tuple) == 0:
            raise ValueError("Input 'gate_tuple' can not be an empty list.")
        else:
            self.gate_dim1, self.gate_low1, self.gate_upp1 = gate_tuple[0]
            self.gate_dim2, self.gate_low2, self.gate_upp2 = gate_tuple[1]
            self.gate_dim1 = features2id[self.gate_dim1]
            self.gate_dim2 = features2id[self.gate_dim2]


class ReferenceTree(object):
    def __init__(self, nested_list, features2id):
        if len(nested_list) != 2:
            raise ValueError("Input 'nested_list' is not properly defined.")
        self.gate = Gate(nested_list[0], features2id)
        self.n_children = len(nested_list[1])
        self.children = [None] * self.n_children
        self.isLeaf = self.n_children == 0
        self.n_leafs = 1 if self.n_children == 0 else 0
        for idx in range(self.n_children):
            self.children[idx] = ReferenceTree(nested_list[1][idx], features2id)
            self.n_leafs += self.children[idx].n_leafs
        print("n_children and n_leafs in reference tree: (%d, %d), and isLeaf = %r" % (
            self.n_children, self.n_leafs, self.isLeaf))

#cuts here have to passed through sigmoid activation to get boundaries in (0, 1)
class ModelNode(nn.Module):
    def __init__(self, logistic_k, reference_tree, init_tree=None, gate_size_default=1. / 4):
        """
        :param logistic_k:
        :param reference_tree:
        """
        # variables for gates]
        super(ModelNode, self).__init__()
        self.logistic_k = logistic_k
        self.reference_tree = reference_tree
        self.gate_dim1 = self.reference_tree.gate.gate_dim1
        self.gate_dim2 = self.reference_tree.gate.gate_dim2
        self.gate_size_default = gate_size_default
        if init_tree == None:
            self.gate_low1_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(self.reference_tree.gate.gate_low1), dtype=torch.float32))
            self.gate_low2_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(self.reference_tree.gate.gate_low2), dtype=torch.float32))
            self.gate_upp1_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(self.reference_tree.gate.gate_upp1), dtype=torch.float32))
            self.gate_upp2_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(self.reference_tree.gate.gate_upp2), dtype=torch.float32))
        else:
            self.gate_low1_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(init_tree.gate.gate_low1), dtype=torch.float32))
            self.gate_low2_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(init_tree.gate.gate_low2), dtype=torch.float32))
            self.gate_upp1_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(init_tree.gate.gate_upp1), dtype=torch.float32))
            self.gate_upp2_param = nn.Parameter(
                torch.tensor(self.__log_odds_ratio__(init_tree.gate.gate_upp2), dtype=torch.float32))

    def __log_odds_ratio__(self, p):
        """
        return log(p/1-p)
        :param p: a float
        :return: a float
        """
        if p < 1e-10:
            return torch.tensor([-10.0])
        if p > 1.0 - 1e-10:
            return torch.tensor([10.0])
        return log(p / (1 - p)),

    def __repr__(self):
        repr_string = ('ModelNode(\n'
                       '  dims=({dim1}, {dim2}),\n'
                       '  gate_dim1=({low1:0.4f}, {high1:0.4f}),\n'
                       '  gate_dim2=({low2:0.4f}, {high2:0.4f}),\n'
                       ')\n')
        return repr_string.format(
            dim1=self.gate_dim1,
            dim2=self.gate_dim2,
            low1=F.sigmoid(self.gate_low1_param).item(),
            high1=F.sigmoid(self.gate_upp1_param).item(),
            low2=F.sigmoid(self.gate_low2_param).item(),
            high2=F.sigmoid(self.gate_upp2_param).item()
        )

    def forward(self, x):
        """
        compute the log probability that each cell passes the gate
        :param x: (n_cell, n_cell_features)
        :return: (logp, reg_penalty)
        """
        gate_low1 = F.sigmoid(self.gate_low1_param)
        gate_low2 = F.sigmoid(self.gate_low2_param)
        gate_upp1 = F.sigmoid(self.gate_upp1_param)
        gate_upp2 = F.sigmoid(self.gate_upp2_param)

        logp = F.logsigmoid(self.logistic_k * ((x[:, self.gate_dim1] - gate_low1))) \
               + F.logsigmoid(- self.logistic_k * ((x[:, self.gate_dim1] - gate_upp1))) \
               + F.logsigmoid(self.logistic_k * ((x[:, self.gate_dim2] - gate_low2))) \
               + F.logsigmoid(- self.logistic_k * ((x[:, self.gate_dim2] - gate_upp2)))
        ref_reg_penalty = (gate_low1 - self.reference_tree.gate.gate_low1) ** 2 \
                          + (gate_low2 - self.reference_tree.gate.gate_low2) ** 2 \
                          + (gate_upp1 - self.reference_tree.gate.gate_upp1) ** 2 \
                          + (gate_upp2 - self.reference_tree.gate.gate_upp2) ** 2
        size_reg_penalty = (abs(gate_upp1 - gate_low1) - self.gate_size_default[0]) ** 2 + \
                           (abs(gate_upp2 - gate_low2) - self.gate_size_default[1]) ** 2
        corner_reg_penalty = torch.sqrt(torch.min(torch.tensor([gate_low1 ** 2 + gate_low2 ** 2,
                                                                 gate_low1 ** 2 + (1 - gate_upp2) ** 2,
                                                                 (1 - gate_upp1) ** 2 + gate_low2 ** 2,
                                                                 (1 - gate_upp1) ** 2 + (1 - gate_upp2) ** 2])))
        return logp, ref_reg_penalty, size_reg_penalty, corner_reg_penalty


class ModelTree(nn.Module):

    def __init__(self, reference_tree,
                 logistic_k=10,
                 regularisation_penalty=10.,
                 positive_box_penalty=10.,
                 negative_box_penalty=10.,
                 corner_penalty=1.,
                 gate_size_penalty=10,
                 init_tree=None,
                 loss_type='logistic',
                 gate_size_default=(1. / 2, 1. / 2),
                 classifier=True):
        """
        :param args: pass values for variable n_cell_features, n_sample_features,
        :param kwargs: pass keyworded values for variable logistic_k=?, regularisation_penality=?.
        """
        super(ModelTree, self).__init__()
        self.logistic_k = logistic_k
        self.regularisation_penalty = regularisation_penalty
        self.positive_box_penalty = positive_box_penalty
        self.negative_box_penalty = negative_box_penalty
        self.corner_penalty = corner_penalty
        self.gate_size_penalty = gate_size_penalty
        self.gate_size_default = gate_size_default
        self.loss_type = loss_type
        self.classifier = classifier
        self.children_dict = nn.ModuleDict()
        self.root = self.add(reference_tree, init_tree)
        self.n_sample_features = reference_tree.n_leafs
        # define parameters in the logistic regression model
        if self.classifier:
            self.linear = nn.Linear(self.n_sample_features, 1) #default behavior is probably Guassian- check this
            #self.linear.weight.data.exponential_(1.0) #TODO add option for initialization with normal as well and uncomment
            #self.linear.bias.data.fill_(-1.0)
            if self.loss_type == "logistic":
                self.criterion = nn.BCEWithLogitsLoss()
            elif self.loss_type == "MSE":
                self.criterion = nn.MSELoss()

    def add(self, reference_tree, init_tree=None):
        """
        construct self.children_dict dictionary for all nodes in the tree structure.
        :param reference_tree:
        :return:
        """
        node = ModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default)
        child_list = nn.ModuleList()
        if init_tree == None:
            for child in reference_tree.children:
                child_node = self.add(child)
                child_list.append(child_node)
        else:
            for _ in range(len(reference_tree.children)):
                child_ref = reference_tree.children[_]
                child_init = init_tree.children[_]
                child_node = self.add(child_ref, child_init)
                child_list.append(child_node)
        self.children_dict.update({str(id(node)): child_list})
        return node

    def forward(self, x, y=None, detach_logistic_params=False):
        """

        :param x: a list of tensors
        :param y:
        :return:
            output:
                leaf_probs: torch.tensor (n_samples, n_leaf_nodes)
                y_pred: torch.tensor(n_samples)
                loss: float
        """
        output = {'leaf_probs': None,
                  'leaf_logp': None,
                  'y_pred': None,
                  'ref_reg_loss': 0,
                  'size_reg_loss': 0,
                  'emp_reg_loss': 0,
                  'corner_reg_loss': 0,
                  'log_loss': None,
                  'loss': None
                  }

        tensor = torch.tensor((), dtype=torch.float32)
        leaf_probs = tensor.new_zeros((len(x), self.n_sample_features))

        for sample_idx in range(len(x)):

            this_level = [(self.root, torch.zeros((x[sample_idx].shape[0],)))]
            leaf_idx = 0
            while this_level:
                next_level = list()
                for (node, pathlogp) in this_level:
                    logp, ref_reg_penalty, size_reg_penalty, corner_reg_penalty = node(x[sample_idx])
                    output['ref_reg_loss'] += ref_reg_penalty * self.regularisation_penalty / len(x)
                    output['size_reg_loss'] += size_reg_penalty * self.gate_size_penalty / len(x)
                    output['corner_reg_loss'] += corner_reg_penalty * self.corner_penalty / len(x)
                    pathlogp = pathlogp + logp 
                    if len(self.children_dict[str(id(node))]) > 0:
                        for child_node in self.children_dict[str(id(node))]:
                            next_level.append((child_node, pathlogp))
                    else:
                        leaf_probs[sample_idx, leaf_idx] = pathlogp.exp().sum(dim=0) / x[sample_idx].shape[0]
                        leaf_idx += 1
                this_level = next_level

        loss = output['ref_reg_loss'] + output['size_reg_loss'] + output['corner_reg_loss']

        output['leaf_probs'] = leaf_probs
        output['leaf_logp'] = torch.log(leaf_probs)  # Rob: This is weird...

        if self.classifier:
            if detach_logistic_params:
                output['y_pred'] = torch.sigmoid(self.linear(output['leaf_logp'].detach())).squeeze(1)
            else:
                output['y_pred'] = torch.sigmoid(self.linear(output['leaf_logp'])).squeeze(1)

        if y is not None:
            if self.classifier:
                if self.loss_type == "logistic":
                    output['log_loss'] = self.criterion(self.linear(output['leaf_logp']).squeeze(1), y)
                elif self.loss_type == "MSE":
                    output['log_loss'] = self.criterion(output['y_pred'], y)
                loss = loss + output['log_loss']
            # add regularization on the number of cells fall into the leaf gate of negative samples;
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             output['leaf_probs'][sample_idx][0] / (len(y) - sum(y))
                else:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.positive_box_penalty * \
                                             output['leaf_probs'][sample_idx][0] / sum(y)

        output['loss'] = loss + output['emp_reg_loss']

        return output


class ModelForest(nn.Module):

    def __init__(self, reference_tree_list,
                 logistic_k=10,
                 regularisation_penalty=10.,
                 positive_box_penalty=10.,
                 negative_box_penalty=10.,
                 corner_penalty=1.,
                 gate_size_penalty=10,
                 init_tree_list=None,
                 loss_type='logistic',
                 gate_size_default=1. / 4):
        """

        :param reference_tree_list:
        :param logistic_k:
        :param regularisation_penalty:
        :param emptyness_penalty:
        :param gate_size_penalty:
        :param init_tree_list:
        :param loss_type:
        :param gate_size_default:
        """
        super(ModelForest, self).__init__()
        self.logistic_k = logistic_k
        self.regularisation_penalty = regularisation_penalty
        self.positive_box_penalty = positive_box_penalty
        self.negative_box_penalty = negative_box_penalty
        self.corner_penalty = corner_penalty
        self.gate_size_penalty = gate_size_penalty
        self.gate_size_default = gate_size_default
        self.loss_type = loss_type
        self.model_trees = nn.ModuleList()
        self.n_panels = len(reference_tree_list)
        self.build(reference_tree_list, init_tree_list)

        # define parameters in the logistic regression model
        self.n_sample_features = sum([ref_tree.n_leafs for ref_tree in reference_tree_list])
        self.linear = nn.Linear(self.n_sample_features, 1)
        self.linear.weight.data.exponential_(1.0)
        self.linear.bias.data.fill_(-1.0)
        if self.loss_type == "logistic":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == "MSE":
            self.criterion = nn.MSELoss()

    def build(self, reference_tree_list, init_tree_list):
        for idx in range(self.n_panels):
            self.model_trees.append(ModelTree(reference_tree_list[idx],
                                              logistic_k=self.logistic_k,
                                              regularisation_penalty=self.regularisation_penalty,
                                              positive_box_penalty=self.positive_box_penalty,
                                              negative_box_penalty=self.negative_box_penalty,
                                              corner_penalty=self.corner_penalty,
                                              gate_size_penalty=self.gate_size_penalty,
                                              init_tree=init_tree_list[idx],
                                              loss_type=self.loss_type,
                                              gate_size_default=self.gate_size_default,
                                              classifier=False))

    def forward(self, x, y=None):
        """

        :param x: a list of a list of tensors, each list is of same length. (n_samples, n_panels, (n_cell * n_marker))
        :param y: a tensor of (n_samples, )
        :return:
        """
        output = {'leaf_probs': None,
                  'leaf_logp': None,
                  'y_pred': None,
                  'ref_reg_loss': 0,
                  'size_reg_loss': 0,
                  'emp_reg_loss': 0,
                  'log_loss': None,
                  'corner_reg_loss': 0,
                  'loss': None
                  }

        tensor = torch.tensor((), dtype=torch.float32)
        output['leaf_probs'] = tensor.new_zeros((len(x), self.n_sample_features))

        feature_counter = 0
        for panel_idx in range(self.n_panels):
            x_panel = [x_sample[panel_idx] for x_sample in x]
            output_panel = self.model_trees[panel_idx](x_panel)
            n_panel_features = output_panel['leaf_probs'].shape[1]
            output['leaf_probs'][:, feature_counter: (feature_counter + n_panel_features)] = output_panel['leaf_probs']
            feature_counter += n_panel_features
            output['ref_reg_loss'] += output_panel['ref_reg_loss']
            output['size_reg_loss'] += output_panel['size_reg_loss']
            output['corner_reg_loss'] += output_panel['corner_reg_loss']


        # output['leaf_logp'] = torch.log(output['leaf_probs'])
        output['leaf_logp'] = torch.log(torch.clamp(output['leaf_probs'], min=1e-10, max=1-(1e-10)))
        output['y_pred'] = torch.sigmoid(self.linear(output['leaf_logp'])).squeeze(1)

        if len(y) > 0:
            if self.loss_type == "logistic":
                output['log_loss'] = self.criterion(self.linear(output['leaf_logp']).squeeze(1), y)
            elif self.loss_type == "MSE":
                output['log_loss'] = self.criterion(output['y_pred'], y)
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             output['leaf_probs'][sample_idx][0] / (len(y) - sum(y))
                else:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.positive_box_penalty * \
                                             output['leaf_probs'][sample_idx][0] / sum(y)
            output['loss'] = output['ref_reg_loss'] + output['size_reg_loss'] + \
                             output['emp_reg_loss'] + output['log_loss'] + output['corner_reg_loss']
        return output
