from __future__ import division

from collections import namedtuple
from math import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils_load_data as dh


# used only for referenceTree object, cuts here don't have to be passed through sigmoid activations
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

    def __repr__(self):
        repr_string = ('ModelNode(\n'
                       '  dims=({dim1}, {dim2}),\n'
                       '  gate_dim1=({low1:0.4f}, {high1:0.4f}),\n'
                       '  gate_dim2=({low2:0.4f}, {high2:0.4f}),\n'
                       ')\n')
        return repr_string.format(
            dim1=self.gate_dim1,
            dim2=self.gate_dim2,
            low1=self.gate_low1,
            high1=self.gate_upp1,
            low2=self.gate_low2,
            high2=self.gate_upp2
        )


class GateBothPanels(Gate):
    def __init__(self, gate_tuple, features2id):
        self.panel = gate_tuple[2]
        self.panel_id = 1 if self.panel == 'p2' else 0
        super().__init__(gate_tuple, features2id[self.panel_id])


class ReferenceTreeBoth(object):
    def __init__(self, nested_list, features2idBoth):
        if len(nested_list) != 2:
            raise ValueError("Input 'nested_list' is not properly defined.")
        self.gate = GateBothPanels(nested_list[0], features2idBoth)
        panel_id = 1 if self.gate.panel == 'p2' else 0
        self.features2idBoth = features2idBoth[panel_id]
        self.ids2features = dict((idx, feature) for feature, idx in self.features2idBoth.items())
        self.n_children = len(nested_list[1])
        self.children = [None] * self.n_children
        self.isLeaf = self.n_children == 0
        self.n_leafs = 1 if self.n_children == 0 else 0
        for idx in range(self.n_children):
            self.children[idx] = ReferenceTreeBoth(nested_list[1][idx], features2idBoth)
            self.n_leafs += self.children[idx].n_leafs

    def __repr__(self):
        print(self.gate)
        for child in self.children:
            print(child.gate)
        return ''


class ReferenceTree(object):
    def __init__(self, nested_list, features2id):
        if len(nested_list) != 2:
            raise ValueError("Input 'nested_list' is not properly defined.")
        self.features2id = features2id
        self.ids2features = dict((idx, feature) for feature, idx in self.features2id.items())
        self.gate = Gate(nested_list[0], features2id)
        self.n_children = len(nested_list[1])
        self.children = [None] * self.n_children
        self.isLeaf = self.n_children == 0
        self.n_leafs = 1 if self.n_children == 0 else 0
        for idx in range(self.n_children):
            self.children[idx] = ReferenceTree(nested_list[1][idx], features2id)
            self.n_leafs += self.children[idx].n_leafs


# cuts here have to passed through sigmoid activation to get boundaries in (0, 1)
class ModelNode(nn.Module):
    def __init__(self, logistic_k, reference_tree, init_tree=None, gate_size_default=1. / 4, is_root=False,
                 panel='both'):
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
        self.is_root = is_root
        self.init_tree = init_tree
        self.panel = panel
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
        if self.init_tree is None:
            init_reg_penalty = 0.0
        else:
            init_reg_penalty = (gate_low1 - self.init_tree.gate.gate_low1) ** 2 \
                               + (gate_low2 - self.init_tree.gate.gate_low2) ** 2 \
                               + (gate_upp1 - self.init_tree.gate.gate_upp1) ** 2 \
                               + (gate_upp2 - self.init_tree.gate.gate_upp2) ** 2
        size_reg_penalty = (abs(gate_upp1 - gate_low1) - self.gate_size_default[0]) ** 2 + \
                           (abs(gate_upp2 - gate_low2) - self.gate_size_default[1]) ** 2

        corner_reg_penalty = torch.sqrt(torch.min(torch.stack([gate_low1 ** 2 + gate_low2 ** 2,
                                                               gate_low1 ** 2 + (1 - gate_upp2) ** 2,
                                                               (1 - gate_upp1) ** 2 + gate_low2 ** 2,
                                                               (1 - gate_upp1) ** 2 + (1 - gate_upp2) ** 2], dim=0)))
        return logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty


class SquareModelNode(ModelNode):
    def __init__(self, logistic_k, reference_tree, init_tree=None, gate_size_default=(1. / 4, 1. / 4), is_root=False,
                 panel='both'):
        super(ModelNode, self).__init__()
        self.logistic_k = logistic_k
        self.reference_tree = reference_tree
        self.gate_dim1 = self.reference_tree.gate.gate_dim1
        self.gate_dim2 = self.reference_tree.gate.gate_dim2
        self.gate_size_default = gate_size_default
        self.is_root = is_root
        self.panel = panel

        if init_tree == None:
            self.init_gate_params(reference_tree)
            self.init_tree = reference_tree
        else:
            self.init_gate_params(init_tree)
            self.init_tree = init_tree

    @staticmethod
    def load_tree_into_gate(tree):
        low1 = tree.gate.gate_low1
        low2 = tree.gate.gate_low2
        upp1 = tree.gate.gate_upp1
        upp2 = tree.gate.gate_upp2

        gate = namedtuple('gate', ['low1', 'upp1', 'low2', 'upp2'])
        gate.low1 = low1
        gate.low2 = low2
        gate.upp1 = upp1
        gate.upp2 = upp2
        return gate

    def get_expanded_beyond_range_square_gate(self, tree):
        rectangle_gate = SquareModelNode.load_tree_into_gate(tree)
        x_len = rectangle_gate.upp1 - rectangle_gate.low1
        y_len = rectangle_gate.upp2 - rectangle_gate.low2
        x_is_bigger = False
        if x_len > y_len:
            x_is_bigger = True

        square_side_len = x_len if x_is_bigger else y_len

        square_gate = namedtuple('gate', ['low1', 'upp1', 'low2', 'upp2'])
        if x_is_bigger:
            # check which side we are in contact with
            if rectangle_gate.low2 == 0.0:
                # contact is with bottom
                square_gate.upp2 = rectangle_gate.upp2
                square_gate.low2 = rectangle_gate.upp2 - square_side_len
            else:
                # contact is with the top
                square_gate.low2 = rectangle_gate.low2
                square_gate.upp2 = rectangle_gate.low2 + square_side_len

            square_gate.upp1 = rectangle_gate.upp1
            square_gate.low1 = rectangle_gate.low1
        else:
            if rectangle_gate.low1 == 0.0:
                # contact is with left side
                square_gate.upp1 = rectangle_gate.upp1
                square_gate.low1 = rectangle_gate.upp1 - square_side_len
            else:
                # contact is with right side
                square_gate.low1 = rectangle_gate.low1
                square_gate.upp1 = rectangle_gate.low1 + square_side_len

            square_gate.low2 = rectangle_gate.low2
            square_gate.upp2 = rectangle_gate.upp2

        return square_gate

    def init_gate_params(self, tree):
        # first make the gate into a square
        # note that the gate here will have negative values
        # reminder: the tree gates are normalized to 0,1 at this point
        square_gate = self.get_expanded_beyond_range_square_gate(tree)

        self.center1_param = nn.Parameter(
            torch.tensor(
                self.__log_odds_ratio__(
                    square_gate.low1 + \
                    (square_gate.upp1 - square_gate.low1) / 2.
                ),
                dtype=torch.float32
            )
        )
        self.center2_param = nn.Parameter(
            torch.tensor(
                self.__log_odds_ratio__(
                    square_gate.low2 + \
                    (square_gate.upp2 - square_gate.low2) / 2.
                ),
                dtype=torch.float32
            )
        )

        s1 = square_gate.upp1 - square_gate.low1
        s2 = square_gate.upp2 - square_gate.low2
        # should be a square after expanding into a square
        assert (np.round(s1 * 10 ** 4) / 10 ** 4 == np.round(s2 * 10 ** 4) / 10 ** 4)

        self.side_length_param = nn.Parameter(
            torch.tensor(
                self.__log_odds_ratio__(s1),
                dtype=torch.float32
            )
        )

    def replace_nans_with_0(self, grad):
        if torch.isnan(grad):
            if torch.cuda.is_available():
                grad = torch.tensor([0.]).cuda()
        return torch.autograd.Variable(grad)

    def forward(self, x):
        """
        compute the log probability that each cell passes the gate
        :param x: (n_cell, n_cell_features)
        :return: (logp, reg_penalty)
        """
        self.side_length_param.register_hook(self.replace_nans_with_0)
        self.center1_param.register_hook(self.replace_nans_with_0)
        self.center2_param.register_hook(self.replace_nans_with_0)

        gate_upp1 = F.sigmoid(self.center1_param) + F.sigmoid(self.side_length_param) / 2.
        gate_low1 = F.sigmoid(self.center1_param) - F.sigmoid(self.side_length_param) / 2.
        gate_upp2 = F.sigmoid(self.center2_param) + F.sigmoid(self.side_length_param) / 2.
        gate_low2 = F.sigmoid(self.center2_param) - F.sigmoid(self.side_length_param) / 2.
        # gate_low1.register_hook(self.replace_nans_with_0)

        logp = F.logsigmoid(self.logistic_k * ((x[:, self.gate_dim1] - gate_low1))) \
               + F.logsigmoid(- self.logistic_k * ((x[:, self.gate_dim1] - gate_upp1))) \
               + F.logsigmoid(self.logistic_k * ((x[:, self.gate_dim2] - gate_low2))) \
               + F.logsigmoid(- self.logistic_k * ((x[:, self.gate_dim2] - gate_upp2)))

        ref_reg_penalty = (gate_low1 - self.reference_tree.gate.gate_low1) ** 2 \
                          + (gate_low2 - self.reference_tree.gate.gate_low2) ** 2 \
                          + (gate_upp1 - self.reference_tree.gate.gate_upp1) ** 2 \
                          + (gate_upp2 - self.reference_tree.gate.gate_upp2) ** 2
        init_reg_penalty = (gate_low1 - self.init_tree.gate.gate_low1) ** 2 \
                           + (gate_low2 - self.init_tree.gate.gate_low2) ** 2 \
                           + (gate_upp1 - self.init_tree.gate.gate_upp1) ** 2 \
                           + (gate_upp2 - self.init_tree.gate.gate_upp2) ** 2
        size_reg_penalty = (abs(gate_upp1 - gate_low1) - self.gate_size_default[0]) ** 2 + \
                           (abs(gate_upp2 - gate_low2) - self.gate_size_default[1]) ** 2

        corner_reg_penalty = torch.sqrt(torch.min(torch.stack([gate_low1 ** 2 + gate_low2 ** 2,
                                                               gate_low1 ** 2 + (1 - gate_upp2) ** 2,
                                                               (1 - gate_upp1) ** 2 + gate_low2 ** 2,
                                                               (1 - gate_upp1) ** 2 + (1 - gate_upp2) ** 2], dim=0)))

        return logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty

    def __repr__(self):
        repr_string = ('ModelNode(\n'
                       '  dims=({dim1}, {dim2}),\n'
                       '  gate_dim1=({low1:0.4f}, {high1:0.4f}),\n'
                       '  gate_dim2=({low2:0.4f}, {high2:0.4f}),\n'
                       '  panel={panel},\n'
                       ')\n')
        gate_low1_param = F.sigmoid(self.center1_param) - F.sigmoid(self.side_length_param) / 2.
        gate_low2_param = F.sigmoid(self.center2_param) - F.sigmoid(self.side_length_param) / 2.
        gate_upp1_param = F.sigmoid(self.center1_param) + F.sigmoid(self.side_length_param) / 2.
        gate_upp2_param = F.sigmoid(self.center2_param) + F.sigmoid(self.side_length_param) / 2.

        return repr_string.format(
            dim1=self.gate_dim1,
            dim2=self.gate_dim2,
            low1=gate_low1_param.item(),
            high1=gate_upp1_param.item(),
            low2=gate_low2_param.item(),
            high2=gate_upp2_param.item(),
            panel=self.panel
        )


class ModelTree(nn.Module):

    def __init__(self, reference_tree,
                 logistic_k=100,
                 regularisation_penalty=0.,
                 positive_box_penalty=0.,
                 negative_box_penalty=0.,
                 corner_penalty=0.,
                 gate_size_penalty=0.,
                 feature_diff_penalty=0.,
                 init_reg_penalty=0.,
                 init_tree=None,
                 loss_type='logistic',
                 gate_size_default=(1. / 2, 1. / 2),
                 neg_proportion_default=.0001,
                 classifier=True,
                 node_type='rectangle'):
        """
        :param args: pass values for variable n_cell_features, n_sample_features,
        :param kwargs: pass keyworded values for variable logistic_k=?, regularisation_penality=?.
        """
        super(ModelTree, self).__init__()
        self.neg_proportion_default = neg_proportion_default
        self.logistic_k = logistic_k
        self.regularisation_penalty = regularisation_penalty
        self.positive_box_penalty = positive_box_penalty
        self.negative_box_penalty = negative_box_penalty
        self.init_reg_penalty = init_reg_penalty
        self.corner_penalty = corner_penalty
        self.feature_diff_penalty = feature_diff_penalty
        self.gate_size_penalty = gate_size_penalty
        self.gate_size_default = gate_size_default
        self.loss_type = loss_type
        self.classifier = classifier
        self.children_dict = nn.ModuleDict()
        self.node_type = node_type
        self.last_node_idx = 0
        self.node_idx_dict = {}
        self.root = self.add(reference_tree, init_tree)
        self.root.is_root = True
        self.n_sample_features = reference_tree.n_leafs
        # define parameters in the logistic regression model
        if self.classifier:
            self.linear = nn.Linear(self.n_sample_features, 1)
            if self.loss_type == "logistic":
                self.criterion = nn.BCEWithLogitsLoss()
            elif self.loss_type == "MSE":
                self.criterion = nn.MSELoss()

    '''
    loads the nodes gate params into a namedtuple after converting
    to the true gate locations
    '''

    @staticmethod
    def get_gate(node):
        # model nodes save the cuts as logits
        if type(node).__name__ == 'ModelNode':
            gate_low1 = F.sigmoid(node.gate_low1_param).cpu().detach().numpy()
            gate_low2 = F.sigmoid(node.gate_low2_param).cpu().detach().numpy()
            gate_upp1 = F.sigmoid(node.gate_upp1_param).cpu().detach().numpy()
            gate_upp2 = F.sigmoid(node.gate_upp2_param).cpu().detach().numpy()
        elif type(node).__name__ == 'SquareModelNode':
            gate_upp1 = torch.clamp(
                (F.sigmoid(node.center1_param) + F.sigmoid(node.side_length_param) / 2.),
                0., 1.
            ).cpu().detach().numpy()
            gate_low1 = torch.clamp(
                (F.sigmoid(node.center1_param) - F.sigmoid(node.side_length_param) / 2.),
                0., 1.
            ).cpu().detach().numpy()
            gate_upp2 = torch.clamp(
                (F.sigmoid(node.center2_param) + F.sigmoid(node.side_length_param) / 2.),
                0., 1.
            ).cpu().detach().numpy()
            gate_low2 = torch.clamp(
                (F.sigmoid(node.center2_param) - F.sigmoid(node.side_length_param) / 2.),
                0., 1.
            ).cpu().detach().numpy()
        else:
            gate_low1 = node.gate_low1_param.cpu().detach().numpy()
            gate_low2 = node.gate_low2_param.cpu().detach().numpy()
            gate_upp1 = node.gate_upp1_param.cpu().detach().numpy()
            gate_upp2 = node.gate_upp2_param.cpu().detach().numpy()

        gate = namedtuple('gate', ['low1', 'upp1', 'low2', 'upp2'])
        gate.low1 = gate_low1
        gate.low2 = gate_low2
        gate.upp1 = gate_upp1
        gate.upp2 = gate_upp2

        return gate

    @staticmethod
    def filter_data_at_single_node(data, node):
        gate = ModelTree.get_gate(node)
        filtered_data = dh.filter_rectangle(
            data, node.gate_dim1,
            node.gate_dim2, gate.low1, gate.upp1,
            gate.low2, gate.upp2
        )
        return filtered_data

    def filter_data_to_leaf(self, data):
        # the second to last entry is the leaf node's data, while the last is the data filtered by the leaf node's gate
        return self.filter_data(data)[-2]

    def get_data_inside_all_gates(self, data):
        return self.filter_data(data)[-1]

    def get_data_inside_all_gates_4chain(self, data):
        nodes = self.get_nodes_four_chain()
        gates = [ModelTree.get_gate(node) for node in nodes]
        for node in nodes:
            data = self.filter_data_at_single_node(data, node)
        return data

    def filter_data(self, data):
        # lists easily function as stacks in python
        node_stack = [self.root]
        # keep track of each's node parent data after filtering
        data_stack = [data]

        filtered_data = [data]

        while len(node_stack) > 0:
            node = node_stack.pop()
            cur_data = data_stack.pop()
            filtered_data.append(self.filter_data_at_single_node(cur_data, node))

            for child in self.children_dict[self.get_node_idx(node)]:
                node_stack.append(child)
                # push the same data onto the stack since the
                # children share the same parent
                data_stack.append(filtered_data[-1])

        return filtered_data

    def get_list_of_model_nodes(self):
        return self.apply_function_depth_first(lambda x: x)

    '''
    applies a function to each node in the
    model and appends the output to a list
    '''

    def apply_function_depth_first(self, function):
        # lists easily function as stacks in python
        node_stack = [self.root]
        outputs = []

        while len(node_stack) > 0:
            node = node_stack.pop()
            outputs.append(function(node))

            for child in self.children_dict[self.get_node_idx(node)]:
                node_stack.append(child)

        return outputs

    def get_nodes_synth(self):
        root = self.root
        children_dict_values = [value for value in self.children_dict.values()]

        for value in children_dict_values:
            if len(value) == 0:
                continue
            # this means were at the first children of root
            if value[0].gate_dim1 == 2:
                child1 = value[0]
                child2 = value[1]
            elif value[0].gate_dim1 == 6:
                child3 = value[0]

        return [root, child1, child2, child3]

    '''
    returns a list of pairs of ids for each gate in depth-first order
    '''

    def get_flat_ids(self):
        def get_ids(node):
            return [node.gate_dim1, node.gate_dim2]

        return self.apply_function_depth_first(get_ids)

    '''
    returns a depth first ordering of the
    model gates as a list of name_tuples 
    '''

    def get_flattened_gates(self):
        return self.apply_function_depth_first(ModelTree.get_gate)

    def get_flattened_gates_numbers(self):
        gates_with_objects = self.apply_function_depth_first(ModelTree.get_gate)
        flat_gates = []
        for gate in gates_with_objects:
            flat_gates.append([gate.low1, gate.upp1, gate.low2, gate.upp2])
        return flat_gates

    def add(self, reference_tree, init_tree=None):
        """
        construct self.children_dict dictionary for all nodes in the tree structure.
        :param reference_tree:
        :return:
        """
        if self.node_type == 'rectangle':
            node = ModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default)
        elif self.node_type == 'square':
            node = SquareModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default)
        else:
            raise ValueError('Node type %s not implemented' % (self.node_type))

        if torch.cuda.is_available():
            node.cuda()

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
        # str(id(node)) was old code
        # but it doesn't allow easy saving and loading of the model
        self.children_dict.update({self.get_node_idx(node): child_list})
        return node

    # Could add case here where an additional number is appended if 
    # gates share the same dimensions
    def get_node_idx(self, node):
        node_gate = str(node.gate_dim1) + str(node.gate_dim2)
        return node_gate

    def make_gates_hard(self):
        self.logistic_k = 1e4

    def get_hard_proportions(self, data):
        inside_all_gates = [self.get_data_inside_all_gates(d.cpu().detach().numpy()) for d in data]
        proportions = np.array(
            [
                x.shape[0] / data[idx].shape[0]
                for idx, x in enumerate(inside_all_gates)
            ]
        )
        return proportions

    def register_nan_hook(self, tensor, string='hiss'):
        tensor.register_hook(lambda x: print(torch.sum(torch.isnan(x)), string))

    def get_nodes_four_chain(self):
        root = self.root
        children_dict_values = [value for value in self.children_dict.values()]

        for value in children_dict_values:
            if len(value) == 0:
                continue
            if value[0].gate_dim1 == 6:
                child3 = value[0]
            elif value[0].gate_dim1 == 4:
                child2 = value[0]
            elif value[0].gate_dim1 == 0:
                child1 = value[0]

        return [root, child1, child2, child3]

    # only use if model is a chain of four nodes-this needs to be refactored into a new object probably
    def forward_4chain(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False, device=0):
        output = {'leaf_probs': None,
                  'leaf_logp': None,
                  'y_pred': None,
                  'ref_reg_loss': 0,
                  'size_reg_loss': 0,
                  'init_reg_loss': 0,
                  'emp_reg_loss': 0,
                  'corner_reg_loss': 0,
                  'log_loss': None,
                  'loss': None
                  }

        DEVICE = device
        tensor = torch.tensor((), dtype=torch.float32)
        leaf_probs = tensor.new_zeros((len(x), self.n_sample_features)).cuda(DEVICE)
        if torch.cuda.is_available():
            leaf_probs.cuda(DEVICE)

        leaf_idx = 0
        for sample_idx in range(len(x)):

            pathlogp = 0
            nodes = self.get_nodes_four_chain()
            for node in nodes:
                logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty = node(x[sample_idx])
                output['ref_reg_loss'] += ref_reg_penalty * self.regularisation_penalty / len(x)
                output['size_reg_loss'] += size_reg_penalty * self.gate_size_penalty / len(x)
                output['corner_reg_loss'] += corner_reg_penalty * self.corner_penalty / len(x)
                output['init_reg_loss'] += init_reg_penalty * self.init_reg_penalty / len(x)
                pathlogp = pathlogp + logp

            if x[sample_idx].shape[0] == 0.:
                raise ValueError('Some sample has shape 0 ie no cells!!')
            leaf_probs[sample_idx, leaf_idx] = pathlogp.exp().sum(dim=0) / x[sample_idx].shape[0]

        loss = output['ref_reg_loss'] + output['size_reg_loss'] + output['corner_reg_loss'] + output['init_reg_loss']
        if use_hard_proportions:
            output['leaf_probs'] = torch.tensor(self.get_hard_proportions_4chain(x)[:, np.newaxis],
                                                dtype=torch.float32).cuda(DEVICE)
            output['leaf_logp'] = torch.log(output['leaf_probs']).clamp(min=-1000)
        else:
            output['leaf_probs'] = leaf_probs
            output['leaf_logp'] = torch.log(leaf_probs).clamp(min=-1000)

        if y is not None:
            pos_mean = 0.
            neg_mean = 0.
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             torch.abs(output['leaf_logp'][sample_idx][0] - np.log(
                                                 self.neg_proportion_default)) / (len(y) - sum(y))
                    neg_mean = neg_mean + output['leaf_probs'][sample_idx][0]
                else:
                    pos_mean = pos_mean + output['leaf_probs'][sample_idx][0]
            # use the average mean to normalize the difference so the square isn't so tiny
            output['feature_diff_reg'] = self.feature_diff_penalty * \
                                         -torch.log(
                                             (((1. / (len(y) - sum(y))) * neg_mean - (1. / (sum(y))) * pos_mean)) ** 2)
            loss = loss + output['feature_diff_reg']
        loss = loss + output['emp_reg_loss']

        if self.classifier:
            if detach_logistic_params:
                output['leaf_logp'] = output['leaf_logp'].detach()
            output['y_pred'] = torch.sigmoid(self.linear(output['leaf_logp'])).squeeze(1)

        if y is not None:
            if self.classifier:
                if self.loss_type == "logistic":
                    output['log_loss'] = self.criterion(self.linear(output['leaf_logp']).squeeze(1), y)
                elif self.loss_type == "MSE":
                    output['log_loss'] = self.criterion(output['y_pred'], y)
                loss = loss + output['log_loss']
        output['loss'] = loss
        return output

    def forward(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False, device=0):
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
                  'init_reg_loss': 0,
                  'emp_reg_loss': 0,
                  'corner_reg_loss': 0,
                  'log_loss': None,
                  'loss': None
                  }

        tensor = torch.tensor((), dtype=torch.float32)
        leaf_probs = tensor.new_zeros((len(x), self.n_sample_features)).cuda(device)
        if torch.cuda.is_available():
            leaf_probs.cuda(device)

        for sample_idx in range(len(x)):

            this_level = [(self.root, torch.zeros((x[sample_idx].shape[0],)))]
            if torch.cuda.is_available():
                this_level = [(self.root, torch.zeros((x[sample_idx].shape[0],)).cuda(device))]
            leaf_idx = 0
            while this_level:
                next_level = list()
                for (node, pathlogp) in this_level:
                    logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty = node(x[sample_idx])
                    output['ref_reg_loss'] += ref_reg_penalty * self.regularisation_penalty / len(x)
                    output['size_reg_loss'] += size_reg_penalty * self.gate_size_penalty / len(x)
                    output['corner_reg_loss'] += corner_reg_penalty * self.corner_penalty / len(x)
                    output['init_reg_loss'] += init_reg_penalty * self.init_reg_penalty / len(x)
                    pathlogp = pathlogp + logp
                    if len(self.children_dict[self.get_node_idx(node)]) > 0:
                        for child_node in self.children_dict[self.get_node_idx(node)]:
                            next_level.append((child_node, pathlogp))
                    else:
                        if x[sample_idx].shape[0] == 0.:
                            raise ValueError('Some sample has shape 0 ie no cells!!')
                        leaf_probs[sample_idx, leaf_idx] = pathlogp.exp().sum(dim=0) / x[sample_idx].shape[0]
                        leaf_idx += 1
                this_level = next_level

        loss = output['ref_reg_loss'] + output['size_reg_loss'] + output['corner_reg_loss'] + output['init_reg_loss']

        if use_hard_proportions:
            output['leaf_probs'] = torch.tensor(self.get_hard_proportions(x)[:, np.newaxis], dtype=torch.float32).cuda()
            output['leaf_logp'] = torch.log(output['leaf_probs']).clamp(min=-1000)
        else:
            output['leaf_probs'] = leaf_probs
            output['leaf_logp'] = torch.log(leaf_probs).clamp(min=-1000)

        if y is not None:
            pos_mean = 0.
            neg_mean = 0.
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             torch.abs(output['leaf_logp'][sample_idx][0] - np.log(
                                                 self.neg_proportion_default)) / (len(y) - sum(y))
                    neg_mean = neg_mean + output['leaf_probs'][sample_idx][0]
                else:
                    pos_mean = pos_mean + output['leaf_probs'][sample_idx][0]
            # use the average mean to normalize the difference so the square isn't so tiny
            output['feature_diff_reg'] = self.feature_diff_penalty * \
                                         -torch.log(
                                             (((1. / (len(y) - sum(y))) * neg_mean - (1. / (sum(y))) * pos_mean)) ** 2)
            loss = loss + output['feature_diff_reg']
        loss = loss + output['emp_reg_loss']

        if self.classifier:
            if detach_logistic_params:
                output['leaf_logp'] = output['leaf_logp'].detach()
            output['y_pred'] = torch.sigmoid(self.linear(output['leaf_logp'])).squeeze(1)

        if y is not None:
            if self.classifier:
                if self.loss_type == "logistic":
                    output['log_loss'] = self.criterion(self.linear(output['leaf_logp']).squeeze(1), y)
                elif self.loss_type == "MSE":
                    output['log_loss'] = self.criterion(output['y_pred'], y)
                loss = loss + output['log_loss']
        output['loss'] = loss
        return output


class ModelTreeBothPanels(ModelTree):

    def __init__(self, hparams, features_by_panel, reference_tree, init_tree=None):
        self.features_by_panel = features_by_panel
        self.kappa_lambda1 = None
        super().__init__(
            reference_tree,
            logistic_k=hparams['logistic_k'],
            regularisation_penalty=hparams['regularization_penalty'],
            positive_box_penalty=0.,
            negative_box_penalty=hparams['negative_box_penalty'],
            corner_penalty=0.,
            gate_size_penalty=0.,
            feature_diff_penalty=hparams['feature_diff_penalty'],
            init_reg_penalty=0.,
            init_tree=init_tree,
            loss_type=hparams['loss_type'],
            gate_size_default=(1. / 2, 1. / 2),
            neg_proportion_default=hparams['neg_proportion_default'],
            classifier=True,
            node_type=hparams['node_type']
        )

        self.kappa_lambda1 = None
        self.hparams = hparams
        self.root = self.add(reference_tree, init_tree=init_tree)

    def add(self, reference_tree, init_tree=None, node_idx=-1, panel='both'):
        """
        construct self.children_dict dictionary for all nodes in the tree structure.
        keys are just ints with lower keys being higher up in the tree
        :param reference_tree:
        :return:
        """
        if self.node_type == 'rectangle':
            node = ModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default, panel=panel)
        elif self.node_type == 'square':
            node = SquareModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default, panel=panel)
        else:
            raise ValueError('Node type %s not implemented' % (self.node_type))

        if torch.cuda.is_available():
            node.cuda()

        child_list = nn.ModuleList()
        if init_tree == None:
            for child in reference_tree.children:
                panel = self.get_panel(child)
                child_node = self.add(child, init_tree=None, panel=panel)
                child_list.append(child_node)
        else:
            for _ in range(len(reference_tree.children)):
                child_ref = reference_tree.children[_]
                child_init = init_tree.children[_]
                panel = self.get_panel(child_ref)
                child_node = self.add(child_ref, child_init, panel=panel)
                child_list.append(child_node)
        # str(id(node)) was old code for saved models from CV results, see fix children dict function
        node_idx = self.get_node_idx(node)
        self.children_dict.update({str(node_idx): child_list})
        return node

    def get_node_idx(self, node):
        # kappa/labmda leaf
        if node.gate_dim1 == 9:
            if self.kappa_lambda1 is None:
                self.kappa_lambda1 = node
            if node == self.kappa_lambda1:
                return str(5)
            else:
                return str(6)
        # root
        if node.gate_dim1 == 0:
            return str(0)
        # cd 45 ssc-h:
        if node.gate_dim1 == 1:
            return str(1)
        # cd 19 cd 5
        if node.gate_dim1 == 4:
            return str(2)
        if node.gate_dim1 == 6 and node.panel == 'panel1':
            return str(3)
        if node.gate_dim1 == 6 and node.panel == 'panel2':
            return str(4)

    def get_panel(self, ref_tree):
        dim_ids = [ref_tree.gate.gate_dim1, ref_tree.gate.gate_dim2]
        feature_names = [ref_tree.ids2features[dim_ids[0]], ref_tree.ids2features[dim_ids[1]]]
        if feature_names[0] in self.features_by_panel[0] and feature_names[1] in self.features_by_panel[0]:
            return 'panel1'
        elif feature_names[1] in self.features_by_panel[1] and feature_names[0] in self.features_by_panel[1]:
            return 'panel2'
        else:
            return 'both'

    def get_correct_panel_data(self, node, x):
        panel = node.panel
        if panel == 'both':
            return np.concatenate(x[0], x[1])
        elif panel == 'panel1':
            return x[1]
        else:
            return x[0]

    def get_first_child_node(self, node):
        return self.children_dict[self.get_node_idx(node)][0]

    def get_second_child_node(self, node):
        return self.children_dict[self.get_node_idx(node)][1]

    def is_leaf_node(self, node):
        return (len(self.children_dict[self.get_node_idx(node)]) == 0)

    def get_flat_nodes(self):
        p1_nodes = self.get_p1_nodes()
        p2_nodes = self.get_p2_nodes()
        return p1_nodes[0:4] + p2_nodes[3:]

    def get_p1_nodes(self):
        nodes = [self.root]
        for i in range(3):
            nodes.append(self.get_first_child_node(nodes[-1]))
        # check that the leaf is actaully a leaf
        assert self.is_leaf_node(nodes[-1])
        return nodes

    def get_p2_nodes(self):
        nodes = [self.root]
        for i in range(2):
            nodes.append(self.get_first_child_node(nodes[-1]))
        # panel two CD38 should be second child node of cd19 node (node idx 2, and the last node in nodes at this step)
        nodes.append(self.get_second_child_node(nodes[-1]))
        # panel two has two leaves, so add both children of CD38 node
        nodes.append(self.get_first_child_node(nodes[-1]))
        nodes.append(self.get_second_child_node(nodes[-2]))
        return nodes

    def filter_data_line_graph_single_panel(self, nodes, panel_data):
        cur_data = panel_data
        filtered_data = [cur_data]
        for node in nodes:
            cur_data = ModelTree.filter_data_at_single_node(cur_data, node)
            filtered_data.append(cur_data)
        return filtered_data

    def get_filtered_data_all_nodes(self, data_both):
        PANEL1_IDX = 0
        PANEL2_IDX = 1
        p1_nodes = self.get_p1_nodes()
        p1_filtered_data = self.filter_data_line_graph_single_panel(p1_nodes, data_both[PANEL1_IDX])

        p2_nodes = self.get_p2_nodes()
        p2_nodes_all_but_one_leaf = p2_nodes[0:-1]

        p2_filtered_data_except_last_leaf = self.filter_data_line_graph_single_panel(p2_nodes_all_but_one_leaf,
                                                                                     data_both[PANEL2_IDX])
        p2_last_leaf_data = ModelTree.filter_data_at_single_node(p2_filtered_data_except_last_leaf[-2], p2_nodes[-1])
        return p1_filtered_data[0:-1] + [p2_filtered_data_except_last_leaf[-3]] + [
            p2_filtered_data_except_last_leaf[-2], p2_filtered_data_except_last_leaf[-2]]

    def get_data_inside_leaves_one_sample(self, data_both):
        PANEL1_IDX = 0
        PANEL2_IDX = 1
        p1_nodes = self.get_p1_nodes()
        p1_filtered_data = self.filter_data_line_graph_single_panel(p1_nodes, data_both[PANEL1_IDX])

        p2_nodes = self.get_p2_nodes()
        p2_nodes_all_but_one_leaf = p2_nodes[0:-1]

        p2_filtered_data_except_last_leaf = self.filter_data_line_graph_single_panel(p2_nodes_all_but_one_leaf,
                                                                                     data_both[PANEL2_IDX])
        p2_last_leaf_data = ModelTree.filter_data_at_single_node(p2_filtered_data_except_last_leaf[-2], p2_nodes[-1])

        return [p1_filtered_data[-1], p2_filtered_data_except_last_leaf[-1], p2_last_leaf_data]

    def get_data_inside_leaves(self, data_list_both):
        inside_leaves = []
        for data_both in data_list_both:
            inside_leaves.append(self.get_data_inside_leaves_one_sample(data_both))
        return inside_leaves

    def get_hard_proportions(self, data_list_both):
        # idxs for which panel the three leaves are from
        PANEL_IDXS = [0, 1, 1]
        data_list_both_np = [[panel_data.detach().cpu().numpy() for panel_data in data_both] for data_both in
                             data_list_both]
        inside_leaves = self.get_data_inside_leaves(data_list_both_np)
        proportions = np.array(
            [
                [x_leaf.shape[0] / data_list_both[idx][PANEL_IDXS[leaf_idx]].shape[0]
                 for leaf_idx, x_leaf in enumerate(leaves_data)]
                for idx, leaves_data in enumerate(inside_leaves)
            ]
        )
        return proportions

    def forward(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False, device=1):
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
                  'init_reg_loss': 0,
                  'emp_reg_loss': 0,
                  'corner_reg_loss': 0,
                  'log_loss': None,
                  'loss': None
                  }

        tensor = torch.tensor((), dtype=torch.float32)
        leaf_probs = tensor.new_zeros((len(x), self.n_sample_features)).cuda(device)
        if torch.cuda.is_available():
            leaf_probs.cuda(device)
        for sample_idx in range(len(x)):
            # this_level = [(self.root, torch.zeros((x[sample_idx][0].shape[0], x[sample_idx][1].shape[0])))]
            if torch.cuda.is_available():
                this_level = [(self.root, [torch.zeros((x[sample_idx][0].shape[0],)).cuda(device),
                                           torch.zeros((x[sample_idx][1].shape[0],)).cuda(device)])]
            else:
                this_level = [
                    (self.root, [torch.zeros((x[sample_idx][0].shape[0],)), torch.zeros((x[sample_idx][1].shape[0],))])]
            leaf_idx = 0
            while this_level:
                next_level = list()
                for (node, pathlogp) in this_level:
                    self.update_pathlogp(pathlogp, node, x[sample_idx])
                    # pathlogp = pathlogp + logp
                    if len(self.children_dict[str(self.get_node_idx(node))]) > 0:
                        for child_node in self.children_dict[str(self.get_node_idx(node))]:
                            next_level.append((child_node, pathlogp))
                    else:
                        if (x[sample_idx][0].shape[0] == 0.) or (x[sample_idx][1].shape[0] == 0.):
                            raise ValueError('Some sample has shape 0 ie no cells!!')
                        if node.panel == 'panel1':
                            leaf_probs[sample_idx, leaf_idx] = pathlogp[0].exp().sum(dim=0) / x[sample_idx][0].shape[0]
                        elif node.panel == 'panel2':
                            leaf_probs[sample_idx, leaf_idx] = pathlogp[1].exp().sum(dim=0) / x[sample_idx][1].shape[0]
                        leaf_idx += 1
                this_level = next_level

        loss = output['ref_reg_loss'] + output['size_reg_loss'] + output['corner_reg_loss'] + output['init_reg_loss']
        # self.register_nan_hook(output['size_reg_loss'], string='size reg')
        # self.register_nan_hook(output['ref_reg_loss'], string='reference reg')
        # self.register_nan_hook(output['init_reg_loss'], string='init_reg')
        if use_hard_proportions:
            output['leaf_probs'] = torch.tensor(self.get_hard_proportions(x)[:, np.newaxis], dtype=torch.float32).cuda()
            output['leaf_logp'] = torch.log(output['leaf_probs']).clamp(min=-1000)
        else:
            output['leaf_probs'] = leaf_probs
            output['leaf_logp'] = torch.log(leaf_probs).clamp(min=-1000)  # Rob: This is weird...

        if y is not None:
            pos_mean = 0.
            neg_mean = 0.
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             torch.abs(output['leaf_logp'][sample_idx][0] - np.log(
                                                 self.neg_proportion_default)) / (len(y) - sum(y))
                    neg_mean = neg_mean + output['leaf_probs'][sample_idx][0]
                else:
                    pos_mean = pos_mean + output['leaf_probs'][sample_idx][0]
            # use the average mean to normalize the difference so the square isn't so tiny
            output['feature_diff_reg'] = self.feature_diff_penalty * \
                                         -torch.log(
                                             (((1. / (len(y) - sum(y))) * neg_mean - (1. / (sum(y))) * pos_mean)) ** 2)
            loss = loss + output['feature_diff_reg']
        loss = loss + output['emp_reg_loss']

        if self.classifier:
            if detach_logistic_params:
                output['leaf_logp'] = output['leaf_logp'].detach()
            output['y_pred'] = torch.sigmoid(self.linear(output['leaf_logp'])).squeeze(1)

        if y is not None:
            if self.classifier:
                if self.loss_type == "logistic":
                    output['log_loss'] = self.criterion(self.linear(output['leaf_logp']).squeeze(), y)
                elif self.loss_type == "MSE":
                    output['log_loss'] = self.criterion(output['y_pred'], y)
                loss = loss + output['log_loss']
            # add regularization on the number of cells fall into the leaf gate of negative samples;
        output['loss'] = loss
        return output

    def update_pathlogp(self, logp, node, x):
        if node.panel == 'both':
            logp_p1, _, _, _, _ = node(x[0])
            logp_p2, _, _, _, _ = node(x[1])
            logp[0] = logp[0] + logp_p1
            logp[1] = logp[1] + logp_p2
        elif node.panel == 'panel1':
            logp_p1, _, _, _, _ = node(x[0])
            logp[0] = logp[0] + logp_p1
        elif node.panel == 'panel2':
            logp_p2, _, _, _, _ = node(x[1])
            logp[1] = logp[1] + logp_p2
        return logp


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

        output['leaf_logp'] = torch.log(torch.clamp(output['leaf_probs'], min=1e-10, max=1 - (1e-10)))
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
