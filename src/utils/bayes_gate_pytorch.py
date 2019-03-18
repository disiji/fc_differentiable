from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.gate_low1 = torch.tensor(self.gate_low1, dtype=torch.float32)
            self.gate_low2 = torch.tensor(self.gate_low2, dtype=torch.float32)
            self.gate_upp1 = torch.tensor(self.gate_upp1, dtype=torch.float32)
            self.gate_upp2 = torch.tensor(self.gate_upp2, dtype=torch.float32)


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


class ModelNode(nn.Module):
    def __init__(self, logistic_k, reference_tree):
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
        self.gate_low1 = nn.Parameter(self.reference_tree.gate.gate_low1)
        self.gate_low2 = nn.Parameter(self.reference_tree.gate.gate_low2)
        self.gate_upp1 = nn.Parameter(self.reference_tree.gate.gate_upp1)
        self.gate_upp2 = nn.Parameter(self.reference_tree.gate.gate_upp2)

    def __repr__(self):
        repr_string = ('ModelNode(\n'
                       '  dims=({dim1}, {dim2}),\n'
                       '  gate_dim1=({low1:0.4f}, {high1:0.4f}),\n'
                       '  gate_dim2=({low2:0.4f}, {high2:0.4f}),\n'
                       ')\n')
        return repr_string.format(
            dim1=self.gate_dim1,
            dim2=self.gate_dim2,
            low1=self.gate_low1.item(),
            high1=self.gate_upp1.item(),
            low2=self.gate_low2.item(),
            high2=self.gate_upp2.item()
        )

    def forward(self, x):
        """
        compute the log probability that each cell passes the gate
        :param x: (n_cell, n_cell_features)
        :return: (logp, reg_penalty)
        """
        logp = F.logsigmoid(self.logistic_k * ((x[:, self.gate_dim1] - self.gate_low1))) \
               + F.logsigmoid(- self.logistic_k * ((x[:, self.gate_dim1] - self.gate_upp1))) \
               + F.logsigmoid(self.logistic_k * ((x[:, self.gate_dim2] - self.gate_low2))) \
               + F.logsigmoid(- self.logistic_k * ((x[:, self.gate_dim2] - self.gate_upp2)))
        reg_penalty = F.mse_loss(self.gate_low1, self.reference_tree.gate.gate_low1) \
                      + F.mse_loss(self.gate_low2, self.reference_tree.gate.gate_low2) \
                      + F.mse_loss(self.gate_upp1, self.reference_tree.gate.gate_upp1) \
                      + F.mse_loss(self.gate_upp2, self.reference_tree.gate.gate_upp2)
        return logp, reg_penalty


class ModelTree(nn.Module):

    def __init__(self, reference_tree, logistic_k=10, regularisation_penalty=10.):
        """
        :param args: pass values for variable n_cell_features, n_sample_features,
        :param kwargs: pass keyworded values for variable logistic_k=?, regularisation_penality=?.
        """
        super(ModelTree, self).__init__()
        self.logistic_k = logistic_k
        self.regularisation_penalty = regularisation_penalty
        self.children_dict = nn.ModuleDict()
        self.root = self.add(reference_tree)
        self.n_sample_features = reference_tree.n_leafs

        # define parameters in the logistic regression model
        self.linear = torch.nn.Linear(self.n_sample_features, 1)
        self.criterion = nn.BCELoss()

    def add(self, reference_tree):
        """
        construct self.children_dict dictionary for all nodes in the tree structure.
        :param reference_tree:
        :return:
        """
        node = ModelNode(self.logistic_k, reference_tree)
        child_list = nn.ModuleList()
        for child in reference_tree.children:
            child_node = self.add(child)
            child_list.append(child_node)
        self.children_dict.update({str(id(node)): child_list})
        return node

    def forward(self, x, y=[]):
        """

        :param x: a list of tensors
        :param y:
        :return:
            output:
                leaf_probs: torch.tensor (n_samples, n_leaf_nodes)
                y_pred: torch.tensor(n_samples)
                loss: float
        """
        output = {'leaf_probs': [],
                  'y_pred': None}
        loss = 0.0

        for idx in range(len(x)):
            leaf_probs = []

            thislevel = [(self.root, torch.zeros((x[idx].shape[0],)))]
            while thislevel:
                nextlevel = list()
                for (node, pathlogp) in thislevel:
                    if len(self.children_dict[str(id(node))]) > 0:
                        logp, reg_penalty = node.forward(x[idx])
                        loss += reg_penalty * self.regularisation_penalty
                        for child_node in self.children_dict[str(id(node))]:
                            nextlevel.append((child_node, pathlogp + logp))
                    else:
                        leaf_probs.append(torch.sum(torch.exp(pathlogp)) * 1.0 / x[idx].shape[0])
                thislevel = nextlevel

            output['leaf_probs'].append(leaf_probs)

        output['leaf_probs'] = torch.tensor(output['leaf_probs'])
        output['y_pred'] = torch.sigmoid(self.linear(output['leaf_probs'])).squeeze(1)

        output['reg_loss'] = loss

        if len(y) == 0:
            loss = None
        else:
            loss += self.criterion(output['y_pred'], y)

        output['loss'] = loss
        return output
