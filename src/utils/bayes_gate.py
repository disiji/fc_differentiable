from __future__ import division
from collections import namedtuple

from math import *
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import utils.utils_load_data as dh

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
#        print("n_children and n_leafs in reference tree: (%d, %d), and isLeaf = %r" % (
#            self.n_children, self.n_leafs, self.isLeaf))


        



#cuts here have to passed through sigmoid activation to get boundaries in (0, 1)
class ModelNode(nn.Module):
    def __init__(self, logistic_k, reference_tree, init_tree=None, gate_size_default=1. / 4, is_root=False):
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
        self.init_tree=init_tree
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

#        print([gate_low1, gate_low2, gate_upp1, gate_upp2])
#        print([
#                self.gate_low1_param.detach().item(), self.gate_low2_param.detach().item(), 
#                self.gate_upp1_param.detach().item(), self.gate_upp2_param.detach().item()
#        ])

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
        
#        torch.cat([gate_low1 ** 2 + gate_low2 ** 2, gate_low1 ** 2 + (1 - gate_upp2) ** 2, \
#            (1 - gate_upp1) ** 2 + gate_low2 ** 2, \
#            (1 - gate_upp1) ** 2 + (1 - gate_upp2) ** 2])
        
        corner_reg_penalty = torch.sqrt(torch.min(torch.stack([gate_low1 ** 2 + gate_low2 ** 2,
                                                                 gate_low1 ** 2 + (1 - gate_upp2) ** 2,
                                                                 (1 - gate_upp1) ** 2 + gate_low2 ** 2,
                                                                 (1 - gate_upp1) ** 2 + (1 - gate_upp2) ** 2], dim=0)))
        return logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty

class SquareModelNode(ModelNode):
    def __init__(self, logistic_k, reference_tree, init_tree=None, gate_size_default=(1./4, 1./4), is_root=False):
        super(ModelNode, self).__init__()
        self.logistic_k = logistic_k
        self.reference_tree = reference_tree
        self.gate_dim1 = self.reference_tree.gate.gate_dim1
        self.gate_dim2 = self.reference_tree.gate.gate_dim2
        self.gate_size_default = gate_size_default
        self.is_root = is_root


        if init_tree == None:
            self.init_gate_params(reference_tree)
        else:
            self.init_gate_params(init_tree)
            self.init_tree = init_tree





        #if init_tree == None:
        #    self.center1_param = nn.Parameter(
        #            torch.tensor(
        #            self.__log_odds_ratio__(
        #                self.reference_tree.gate.gate_low1 + \
        #                (self.reference_tree.gate.gate_upp1 - self.reference_tree.gate.gate_low1)/2.
        #            ), 
        #            dtype=torch.float32
        #        )
        #    )
        #    self.center2_param = nn.Parameter(
        #            torch.tensor(
        #            self.__log_odds_ratio__(
        #                self.reference_tree.gate.gate_low2 + \
        #                (self.reference_tree.gate.gate_upp2 - self.reference_tree.gate.gate_low2)/2.
        #            ), 
        #            dtype=torch.float32
        #        )
        #    )
        #    diff1 = self.reference_tree.gate.gate_upp1 - self.reference_tree.gate.gate_low1
        #    diff2 = self.reference_tree.gate.gate_upp2 - self.reference_tree.gate.gate_low2
        #    # really should be an average and then lowering value appropiately
        #    self.side_length_param = nn.Parameter(
        #            torch.tensor(
        #            self.__log_odds_ratio__(
        #                (diff1 if diff1 < diff2 else diff2)
        #            ), 
        #            dtype=torch.float32
        #        )
        #    )

        #else:
        #    self.center1_param = nn.Parameter(
        #            torch.tensor(
        #            self.__log_odds_ratio__(
        #                init_tree.gate.gate_low1 + \
        #                (init_tree.gate.gate_upp1 - init_tree.gate.gate_low1)/2.
        #            ), 
        #            dtype=torch.float32
        #        )
        #    )
        #    self.center2_param = nn.Parameter(
        #            torch.tensor(
        #            self.__log_odds_ratio__(
        #                init_tree.gate.gate_low2 + \
        #                (init_tree.gate.gate_upp2 - init_tree.gate.gate_low2)/2.
        #            ), 
        #            dtype=torch.float32
        #        )
        #    )
        #    diff1 = init_tree.gate.gate_upp1 - init_tree.gate.gate_low1
        #    diff2 = init_tree.gate.gate_upp2 - init_tree.gate.gate_low2
        #    # really should be an average and then lowering value appropiately
        #    self.side_length_param = nn.Parameter(
        #            torch.tensor(
        #            self.__log_odds_ratio__(
        #                (diff1 if diff1 < diff2 else diff2)
        #            ), 
        #            dtype=torch.float32
        #        )
        #    )

        #
        #
        #print(
        #        'center [%.3f, %.3f], side_length %.3f' 
        #        %(self.center1_param.detach().item(), self.center2_param.detach().item(), self.side_length_param.detach().item())
        #) 

#        self.gate_upp1_param = self.__log_odds_ratio__(nn.Parameter(self.center1 + self.side_length/2.))
#        self.gate_low1_param = self.__log_odds_ratio__(nn.Parameter(self.center1 - self.side_length/2.))
#        self.gate_upp2_param = self.__log_odds_ratio__(nn.Parameter(self.center2 + self.side_length/2.))
#        self.gate_low2_param = self.__log_odds_ratio__(nn.Parameter(self.center2 - self.side_length/2.))
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
                
#        square_gate.upp1 = rectangle_gate.upp1
#        square_gate.low1 = rectangle_gate.upp1 - square_side_len
#        square_gate.upp2 = rectangle_gate.upp2
#        square_gate.low2 = rectangle_gate.upp2 - square_side_len
#        print('Node at dims: ', self.gate_dim1, self.gate_dim2)
#        print([square_gate.low1, square_gate.upp1, square_gate.low2, square_gate.upp2])
        
        #print([rectangle_gate.low1, rectangle_gate.upp1, rectangle_gate.low2, rectangle_gate.upp2])

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
                    (square_gate.upp1 - square_gate.low1)/2.
                ), 
                dtype=torch.float32
            )
        )
        self.center2_param = nn.Parameter(
                torch.tensor(
                self.__log_odds_ratio__(
                    square_gate.low2 + \
                    (square_gate.upp2 - square_gate.low2)/2.
                ), 
                dtype=torch.float32
            )
        )

        s1 = square_gate.upp1 - square_gate.low1
        s2 = square_gate.upp2 - square_gate.low2
        # should be a square after expanding into a square
        assert(np.round(s1 * 10**4)/10**4 == np.round(s2 * 10**4)/10**4)

        self.side_length_param = nn.Parameter(
                torch.tensor(
                self.__log_odds_ratio__(s1), 
                dtype=torch.float32
                )
        )

        #print(F.sigmoid(self.center1_param), F.sigmoid(self.center2_param))
        #print(F.sigmoid(self.side_length_param))
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
        #self.side_length_param.register_hook(lambda x: print(x, 'side param'))
        self.side_length_param.register_hook(self.replace_nans_with_0)
        self.center1_param.register_hook(self.replace_nans_with_0)
        self.center2_param.register_hook(self.replace_nans_with_0)
        gate_upp1 = F.sigmoid(self.center1_param) + F.sigmoid(self.side_length_param)/2.
        gate_low1 = F.sigmoid(self.center1_param) - F.sigmoid(self.side_length_param)/2.
        gate_upp2 = F.sigmoid(self.center2_param) +  F.sigmoid(self.side_length_param)/2.
        gate_low2 = F.sigmoid(self.center2_param) -  F.sigmoid(self.side_length_param)/2.
        gate_low1.register_hook(self.replace_nans_with_0)
        #print(self.gate_dim1, self.gate_dim2)
        #print(self.center1_param, self.side_length_param)
        #print(self.center1_param.grad, self.side_length_param.grad, 'huh betta not be none')
        #print('upp1', gate_upp1)
        #print('low1', gate_low1)
        #print('low2', gate_low2)
        #print('upp2', gate_upp2)
        #if not self.center1_param.grad is None:
        #    if np.isnan(self.center1_param.grad.detach().cpu().numpy()):
        #        raise RuntimeError('nannanananan')
        
        
#        gate_low1 = F.sigmoid(gate_low1_param)
#        gate_low2 = F.sigmoid(gate_low2_param)
#        gate_upp1 = F.sigmoid(gate_upp1_param)
#        gate_upp2 = F.sigmoid(gate_upp2_param)
#
#        print([gate_low1, gate_low2, gate_upp1, gate_upp2])
#        print([
#                self.gate_low1_param.detach().item(), self.gate_low2_param.detach().item(), 
#                self.gate_upp1_param.detach().item(), self.gate_upp2_param.detach().item()
#        ])
        #A = F.logsigmoid(self.logistic_k  * ((x[:, self.gate_dim1] - gate_low1)))
        #A.register_hook(lambda x: torch.clamp(x, min=1e-10, max=1e10))
        #
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
        
#        torch.cat([gate_low1 ** 2 + gate_low2 ** 2, gate_low1 ** 2 + (1 - gate_upp2) ** 2, \
#            (1 - gate_upp1) ** 2 + gate_low2 ** 2, \
#            (1 - gate_upp1) ** 2 + (1 - gate_upp2) ** 2])
        
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
                       ')\n')
        gate_low1_param = F.sigmoid(self.center1_param) - F.sigmoid(self.side_length_param)/2.
        gate_low2_param = F.sigmoid(self.center2_param) - F.sigmoid(self.side_length_param)/2.
        gate_upp1_param = F.sigmoid(self.center1_param) + F.sigmoid(self.side_length_param)/2.
        gate_upp2_param = F.sigmoid(self.center2_param) + F.sigmoid(self.side_length_param)/2.

        return repr_string.format(
            dim1=self.gate_dim1,
            dim2=self.gate_dim2,
            low1=F.sigmoid(gate_low1_param).item(),
            high1=F.sigmoid(gate_upp1_param).item(),
            low2=F.sigmoid(gate_low2_param).item(),
            high2=F.sigmoid(gate_upp2_param).item()
        )
#    def __init__(self, logistic_k, reference_tree, init_tree=None, gate_size_default=(1./4, 1./4), is_root=False):
#        super(SquareModelNode, self).__init__(
#                logistic_k, reference_tree, init_tree=init_tree, 
#                gate_size_default=gate_size_default, is_root=is_root
#        )
#
#        # overwrite the parent classes four sides to be a function of a center point
#        # and a side length (since this is a square node only need those three params)
#        # this is a bit hacky, but doing it this way avoids having to change code elsewhere
#        self.float_gate_upp1_param = F.sigmoid(self.gate_upp1_param).detach().item()
#        self.float_gate_low1_param = F.sigmoid(self.gate_low1_param).detach().item()
#        self.float_gate_upp2_param = F.sigmoid(self.gate_upp2_param).detach().item()
#        self.float_gate_low2_param = F.sigmoid(self.gate_low2_param).detach().item()
#
#        self.center1 = nn.Parameter(torch.tensor((self.float_gate_upp1_param - self.float_gate_low1_param)/2., dtype=torch.float32))
#        self.center2 = nn.Parameter(torch.tensor((self.float_gate_upp2_param - self.float_gate_low2_param)/2., dtype=torch.float32))
#        self.side_length_float = \
#                torch.tensor(
#                ((self.float_gate_upp1_param - self.float_gate_low1_param) + (self.float_gate_upp2_param - self.float_gate_low2_param))/2.,
#                dtype=torch.float32)
#        
#
#        upp1 = self.center1 + self.side_length_float/2.
#        upp2 = self.center2 + self.side_length_float/2.
#        low1 = self.center1 - self.side_length_float/2.
#        low2 = self.center2 - self.side_length_float/2.
#        # make sure the resulting square is inside range (0, 1)
#        while not ((upp1 <= 1.) and (upp2 <= 1.) and (low1 >= 0.) and (low2 >= 0.)):
#            self.side_length_float = self.side_length_float - 1e-3
#            upp1 = self.center1 + self.side_length_float/2.
#            upp2 = self.center2 + self.side_length_float/2.
#            low1 = self.center1 - self.side_length_float/2.
#            low2 = self.center2 - self.side_length_float/2.
#            
#        self.side_length = nn.Parameter(torch.tensor(self.side_length_float, dtype=torch.float32))
#
#        self.gate_upp1_param = self.center1 + self.side_length/2.
#        self.gate_low1_param = self.center1 - self.side_length/2.
#        self.gate_upp2_param = self.center2 + self.side_length/2.
#        self.gate_low2_param = self.center2 - self.side_length/2.
#        print(
#            'center: [%.3f, %.3f], side_length: %.3f'
#            %(self.center1.detach().item(), self.center2.detach().item(), self.side_length.detach().item())
#        )

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
        self.root = self.add(reference_tree, init_tree)
        self.root.is_root = True
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

    '''
    loads the nodes gate params into a namedtuple
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
                            (F.sigmoid(node.center1_param) + F.sigmoid(node.side_length_param)/2.),
                            0., 1.
                        ).cpu().detach().numpy()
            gate_low1 = torch.clamp(
                            (F.sigmoid(node.center1_param) - F.sigmoid(node.side_length_param)/2.),
                            0., 1.
                        ).cpu().detach().numpy()
            gate_upp2 = torch.clamp(
                            (F.sigmoid(node.center2_param) +  F.sigmoid(node.side_length_param)/2.),
                            0., 1.
                        ).cpu().detach().numpy()
            gate_low2 = torch.clamp(
                            (F.sigmoid(node.center2_param) -  F.sigmoid(node.side_length_param)/2.),
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
        #print(node)
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

            for child in self.children_dict[str(id(node))]:
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

            for child in self.children_dict[str(id(node))]:
                node_stack.append(child)

        return outputs

    def fix_dictionary_4chain(self):
        print('before fixing')
        print(self.children_dict)
        fixed_children_dict = nn.ModuleDict()
        #nodes = self.get_nodes_four_chain_unfixed()
        nodes = self.get_nodes_four_chain()
        print(nodes)
        #order if magic is needed : 0, 2, 3, 1 -> 3, 1, 2, ()
        fixed_children_dict[str(id(nodes[0]))] = nn.ModuleList([nodes[1]])
        fixed_children_dict[str(id(nodes[1]))] = nn.ModuleList([nodes[2]])
        fixed_children_dict[str(id(nodes[2]))] = nn.ModuleList([nodes[3]])
        fixed_children_dict[str(id(nodes[3]))] = nn.ModuleList()
        self.children_dict = fixed_children_dict
        print('After fixing')
        print(self.get_nodes_four_chain())
    
    def get_nodes_synth(self):
        root = self.root
        children_dict_values = [value for value in self.children_dict.values()]
        print(children_dict_values, 'unfixed')
        
        for value in children_dict_values:
            if len(value) == 0:
                continue
            # this means were at the first children of root
            if value[0].gate_dim1 == 2:
                child1 = value[0]
                child2 = value[1]
            elif value[0].gate_dim1 == 6:
                child3 = value[0]

        #child1 = children_dict_values[1][0]
        #child2 = children_dict_values[2][0]
        #child3 = children_dict_values[3][0]
        return [root, child1, child2, child3]


    def fix_children_dict_synth(self):
        nodes = self.get_nodes_synth()
        fixed_dict = nn.ModuleDict()
        fixed_dict[str(id(nodes[0]))] = nn.ModuleList([nodes[1], nodes[2]])
        fixed_dict[str(id(nodes[1]))] = nn.ModuleList()
        fixed_dict[str(id(nodes[2]))]= nn.ModuleList([nodes[3]])
        fixed_dict[str(id(nodes[3]))] = nn.ModuleList()
        print('Fixed')
        print(fixed_dict)
        self.children_dict = fixed_dict
        

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
    # have to custom implement deepcopy so 
    # the keys for the dictionary are updated
    # to match the id of the new copied nodes
    def __deepcopy__(self, memo):
        # These four lines just use the 
        # default implementation to copy
        # everything
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        cp = deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method

        # Update the ids in the children_dict
        # to match the new copies
        root_copy = deepcopy(self.root)
        cp.root = root_copy
        cp.children_dict  = self._deepcopy_children_dict(root_copy)
   #     print('Copy:')
   #     print(cp.children_dict)
   #     print('copy root %s, model root %s' %(str(id(cp.root)), str(id(self.root))))
   #     print('Model:')
   #     print(self.children_dict)
        return cp

    def _deepcopy_children_dict(self, root_copy):
        dict_copy = nn.ModuleDict()
        node_stack = [self.root]
        node_copy_stack = [root_copy]
        
        while len(node_stack) > 0:
            node = node_stack.pop()
            node_copy = node_copy_stack.pop()

            children_list_copy = nn.ModuleList()
            children_list = self.children_dict[str(id(node))]

            # Handle case that node is a leaf
            #if len(children_list) == 0:
            #    dict_copy[str(id(node_copy))] = children_list_copy
            #    continue

            for child in children_list:
                child_copy = deepcopy(child)
                children_list_copy.append(child_copy)

                node_stack.append(child)
                node_copy_stack.append(child_copy)
            #node_copy = deepcopy(node)

            dict_copy[str(id(node_copy))] = children_list_copy
            #if node == self.root:
            #    dict_copy[str(id(root_copy))] = children_list_copy
            #else:
            #    dict_copy[str(id(node_copy))] = children_list_copy

            #print(dict_copy)
            #print(self.children_dict)
#            print('next '+ str(id(node_copy_stack[0])) if len(node_copy_stack) > 0 else 'meow')
#            print('cur: ' + str(id(node_copy)))
#            pdb.set_trace()
#        print('done copying')
        return dict_copy 
    

    def add(self, reference_tree, init_tree=None, node_idx=-1):
        """
        construct self.children_dict dictionary for all nodes in the tree structure.
        keys are just ints with lower keys being higher up in the tree
        :param reference_tree:
        :return:
        """
        if self.node_type == 'rectangle':
            node = ModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default)
        elif self.node_type == 'square':
            node = SquareModelNode(self.logistic_k, reference_tree, init_tree, self.gate_size_default)
        else:
            raise ValueError('Node type %s not implemented' %(self.node_type))

        if torch.cuda.is_available():
            node.cuda()

        child_list = nn.ModuleList()
        if init_tree == None:
            for child in reference_tree.children:
                node_idx += 1
                child_node = self.add(child, node_idx)
                child_list.append(child_node)
        else:
            for _ in range(len(reference_tree.children)):
                child_ref = reference_tree.children[_]
                child_init = init_tree.children[_]
                node_idx += 1
                child_node = self.add(child_ref, child_init, node_idx)
                child_list.append(child_node)
        #str(id(node)) was old code for saved models from CV results, see fix children dict function
        self.children_dict.update({node_idx: child_list}) 
        return node

    def make_gates_hard(self):
        self.logistic_k = 1e4

    def get_hard_proportions(self, data):
        inside_all_gates = [self.get_data_inside_all_gates(d.cpu().detach().numpy()) for d in data]
        proportions = np.array(
            [        
                x.shape[0]/data[idx].shape[0]
                for idx, x in enumerate(inside_all_gates)
            ]
        )
        return proportions

    def register_nan_hook(self, tensor, string='hiss'):
        tensor.register_hook(lambda x: print(torch.sum(torch.isnan(x)), string))

    def get_hard_proportions_4chain(self, x_list):
        x_list = [x.cpu().detach().numpy() for x in x_list]
        nodes = self.get_nodes_four_chain()
        proportions = []
        for x in x_list:
            tot_num_in_x = len(x)
            for node in nodes:
                x = self.filter_data_at_single_node(x, node)
            proportions.append(x.shape[0]/tot_num_in_x)
        return np.array(proportions)


    #def get_nodes_four_chain(self):
    #    root = self.root
    #    children_dict_values = [value for value in self.children_dict.values()]
    #    #print(children_dict_values, 'fixed')
    #    child1 = children_dict_values[2][0]
    #    child2 = children_dict_values[1][0]
    #    child3 = children_dict_values[0][0]
    #    return [root, child1, child2, child3]

    def get_nodes_four_chain(self):
        root = self.root
        children_dict_values = [value for value in self.children_dict.values()]
        #print(children_dict_values, 'unfixed')
        
        for value in children_dict_values:
            if len(value) == 0:
                continue
            if value[0].gate_dim1 == 6:
                child3 = value[0]
            elif value[0].gate_dim1 == 4:
                child2 = value[0]
            elif value[0].gate_dim1 == 0:
                child1 = value[0]

        #child1 = children_dict_values[1][0]
        #child2 = children_dict_values[2][0]
        #child3 = children_dict_values[3][0]
        return [root, child1, child2, child3]

    #only use if model is a chain of four nodes-this needs to be refactored in a new object probably
    def forward_4chain(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False, device=1):
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
                output['init_reg_loss'] += init_reg_penalty * self.init_reg_penalty/ len(x)
                pathlogp = pathlogp + logp
               
            if x[sample_idx].shape[0] == 0.:
                raise ValueError('Some sample has shape 0 ie no cells!!')
            leaf_probs[sample_idx, leaf_idx] = pathlogp.exp().sum(dim=0) / x[sample_idx].shape[0]


        loss = output['ref_reg_loss'] + output['size_reg_loss'] + output['corner_reg_loss'] + output['init_reg_loss']
        #self.register_nan_hook(output['size_reg_loss'], string='size reg')
        #self.register_nan_hook(output['ref_reg_loss'], string='reference reg')
        #self.register_nan_hook(output['init_reg_loss'], string='init_reg')
        if use_hard_proportions:
            output['leaf_probs'] = torch.tensor(self.get_hard_proportions_4chain(x)[:, np.newaxis], dtype=torch.float32).cuda(DEVICE)
            output['leaf_logp'] = torch.log(output['leaf_probs']).clamp(min=-1000)
        else:
            output['leaf_probs'] = leaf_probs
            output['leaf_logp'] = torch.log(leaf_probs).clamp(min=-1000)  # Rob: This is weird...

#        if y is not None:
#            for sample_idx in range(len(y)):
#                if y[sample_idx] == 0:
#                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
#                                             torch.abs(output['leaf_probs'][sample_idx][0] - self.neg_proportion_default)/ (len(y) - sum(y))
#                else:
#                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.positive_box_penalty * \
#                                             output['leaf_probs'][sample_idx][0] / sum(y)
#        output['loss'] = loss + output['emp_reg_loss']

        # Note: it doesn't look like this is implemented for more than one leaf node!
        # replacing with torch sum should fix this
        if y is not None:
            pos_mean = 0.
            neg_mean = 0.
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             torch.abs(output['leaf_logp'][sample_idx][0] - np.log(self.neg_proportion_default))/ (len(y) - sum(y))
                    neg_mean = neg_mean + output['leaf_probs'][sample_idx][0]
                else:
                    pos_mean = pos_mean + output['leaf_probs'][sample_idx][0]
            # use the average mean to normalize the difference so the square isn't so tiny
            output['feature_diff_reg'] = self.feature_diff_penalty * \
                                         -torch.log((((1./(len(y) - sum(y))) * neg_mean - (1./(sum(y))) * pos_mean))**2)
            loss = loss + output['feature_diff_reg']
            #self.register_nan_hook(output['feature_diff_reg'], string='feature diff')
            #self.register_nan_hook(output['emp_reg_loss'], string='emptiness')

                #else:
                #    output['emp_reg_loss'] = output['emp_reg_loss'] + self.positive_box_penalty * \
                #                             output['leaf_probs'][sample_idx][0] / sum(y)
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
            # add regularization on the number of cells fall into the leaf gate of negative samples;
        output['loss'] = loss
        #self.register_nan_hook(loss, 'total loss')
        #print(self.get_flattened_gates_numbers())
        #print(loss.requires_grad)
        #hard_props = self.get_hard_proportions(x)
        #soft_props = output['leaf_probs']
        #to_print = [(hard_prop, soft_prop.cpu().detach().numpy()[0]) for hard_prop, soft_prop in zip(hard_props, soft_props)]
        #print(to_print)
        return output


    def forward(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False):
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
        leaf_probs = tensor.new_zeros((len(x), self.n_sample_features)).cuda()
        if torch.cuda.is_available():
            leaf_probs.cuda()

        for sample_idx in range(len(x)):

            this_level = [(self.root, torch.zeros((x[sample_idx].shape[0],)))]
            if torch.cuda.is_available():
                this_level = [(self.root, torch.zeros((x[sample_idx].shape[0],)).cuda())]
            leaf_idx = 0
            while this_level:
                next_level = list()
                for (node, pathlogp) in this_level:
                    logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty = node(x[sample_idx])
                    output['ref_reg_loss'] += ref_reg_penalty * self.regularisation_penalty / len(x) 
                    output['size_reg_loss'] += size_reg_penalty * self.gate_size_penalty / len(x)
                    output['corner_reg_loss'] += corner_reg_penalty * self.corner_penalty / len(x)
                    output['init_reg_loss'] += init_reg_penalty * self.init_reg_penalty/ len(x)
                    pathlogp = pathlogp + logp 
                    if len(self.children_dict[str(id(node))]) > 0:
                        for child_node in self.children_dict[str(id(node))]:
                            next_level.append((child_node, pathlogp))
                    else:
                        if x[sample_idx].shape[0] == 0.:
                            raise ValueError('Some sample has shape 0 ie no cells!!')
                        leaf_probs[sample_idx, leaf_idx] = pathlogp.exp().sum(dim=0) / x[sample_idx].shape[0]
                        leaf_idx += 1
                this_level = next_level

        loss = output['ref_reg_loss'] + output['size_reg_loss'] + output['corner_reg_loss'] + output['init_reg_loss']
        #self.register_nan_hook(output['size_reg_loss'], string='size reg')
        #self.register_nan_hook(output['ref_reg_loss'], string='reference reg')
        #self.register_nan_hook(output['init_reg_loss'], string='init_reg')
        if use_hard_proportions:
            output['leaf_probs'] = torch.tensor(self.get_hard_proportions(x)[:, np.newaxis], dtype=torch.float32).cuda()
            output['leaf_logp'] = torch.log(output['leaf_probs']).clamp(min=-1000)
        else:
            output['leaf_probs'] = leaf_probs
            output['leaf_logp'] = torch.log(leaf_probs).clamp(min=-1000)  # Rob: This is weird...

#        if y is not None:
#            for sample_idx in range(len(y)):
#                if y[sample_idx] == 0:
#                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
#                                             torch.abs(output['leaf_probs'][sample_idx][0] - self.neg_proportion_default)/ (len(y) - sum(y))
#                else:
#                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.positive_box_penalty * \
#                                             output['leaf_probs'][sample_idx][0] / sum(y)
#        output['loss'] = loss + output['emp_reg_loss']

        if y is not None:
            pos_mean = 0.
            neg_mean = 0.
            for sample_idx in range(len(y)):
                if y[sample_idx] == 0:
                    output['emp_reg_loss'] = output['emp_reg_loss'] + self.negative_box_penalty * \
                                             torch.abs(output['leaf_logp'][sample_idx][0] - np.log(self.neg_proportion_default))/ (len(y) - sum(y))
                    neg_mean = neg_mean + output['leaf_probs'][sample_idx][0]
                else:
                    pos_mean = pos_mean + output['leaf_probs'][sample_idx][0]
            # use the average mean to normalize the difference so the square isn't so tiny
            output['feature_diff_reg'] = self.feature_diff_penalty * \
                                         -torch.log((((1./(len(y) - sum(y))) * neg_mean - (1./(sum(y))) * pos_mean))**2)
            loss = loss + output['feature_diff_reg']
            #self.register_nan_hook(output['feature_diff_reg'], string='feature diff')
            #self.register_nan_hook(output['emp_reg_loss'], string='emptiness')

                #else:
                #    output['emp_reg_loss'] = output['emp_reg_loss'] + self.positive_box_penalty * \
                #                             output['leaf_probs'][sample_idx][0] / sum(y)
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
            # add regularization on the number of cells fall into the leaf gate of negative samples;
        output['loss'] = loss
        #self.register_nan_hook(loss, 'total loss')
        #print(self.get_flattened_gates_numbers())
        #print(loss.requires_grad)
        #hard_props = self.get_hard_proportions(x)
        #soft_props = output['leaf_probs']
        #to_print = [(hard_prop, soft_prop.cpu().detach().numpy()[0]) for hard_prop, soft_prop in zip(hard_props, soft_props)]
        #print(to_print)
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
