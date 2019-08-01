import numpy as np
import utils.utils_load_data as dh
import matplotlib.pyplot as plt

class HeuristicInitializer:
    '''
        data is a numpy array of all the cells normalized marker data
    '''
    def __init__(self, node_type, gate_data_axes_ids, pos_data, neg_data, 
                    num_gridcells_per_axis=20, consider_all_gates=False, greedy_filtering=False, verbose=False):
        self.node_type = node_type
        self.num_gates = len(gate_data_axes_ids)
        self.gate_data_axes_ids = gate_data_axes_ids
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.num_gridcells_per_axis = num_gridcells_per_axis
        self.heuristic_gates = None
        self.consider_all_gates = consider_all_gates
        self.greedy_filtering = greedy_filtering
        self.NUM_CORNERS = 4
        self.verbose = verbose

    def construct_heuristic_gates(self):
        heuristic_gates = []
        self.cur_data_from_parent_pos= self.pos_data
        self.cur_data_from_parent_neg = self.neg_data
        for gate_idx in range(self.num_gates):
            heuristic_gates.append(self.find_best_gate(gate_idx))
            if self.greedy_filtering:
                gate_data_axes = self.gate_data_axes_ids[gate_idx]
                best_gate = heuristic_gates[-1]
                self.cur_data_from_parent_pos = \
                    dh.filter_rectangle(
                        self.cur_data_from_parent_pos,
                        gate_data_axes[0], gate_data_axes[1],
                        best_gate[0], best_gate[1],
                        best_gate[2], best_gate[3]
                    )
                self.cur_data_from_parent_neg = \
                    dh.filter_rectangle(
                        self.cur_data_from_parent_neg,
                        gate_data_axes[0], gate_data_axes[1],
                        best_gate[0], best_gate[1],
                        best_gate[2], best_gate[3]
                    )

        self.heuristic_gates = heuristic_gates
        print(self.heuristic_gates)

    def get_heuristic_gates(self):
        if self.heuristic_gates:
            return self.heuristic_gates
        else:
            self.construct_heuristic_gates()
            return self.heuristic_gates


    def find_best_gate(self, gate_idx):
        cell_locs_x = np.linspace(0., 1., num=self.num_gridcells_per_axis + 1)[1:-1]
        cell_locs_y = np.linspace(0., 1., num=self.num_gridcells_per_axis + 1)[1:-1]
        best_heuristic_score = -1.
        best_gate = None
        gate_data_axes = self.gate_data_axes_ids[gate_idx]
        heuristic_at_each_cell = -1. * np.ones([self.num_gridcells_per_axis, self.num_gridcells_per_axis, self.NUM_CORNERS])
        for x in cell_locs_x:
            for y in cell_locs_y:
                for corner_idx in range(self.NUM_CORNERS):
                    gate = self.extend_gridcell_to_corner_gate(x, y, corner_idx)
                    if self.is_gate_acceptable(gate):
                        heuristic_score = self.compute_heuristic(gate, gate_data_axes)
                        heuristic_at_each_cell[int(x * self.num_gridcells_per_axis) - 1, int(y * self.num_gridcells_per_axis) - 1, corner_idx] = np.around(heuristic_score, 3)
                        if heuristic_score > best_heuristic_score:
                            best_heuristic_score = heuristic_score
                            best_gate = gate
        if self.verbose:
            print('Upper Right:')
            print(heuristic_at_each_cell[:, :, 0])
            print('Upper Left:')
            print(heuristic_at_each_cell[:, :, 1])
            print('Lower Left:')
            print(heuristic_at_each_cell[:, :, 2])
            print('Lower Right:')
            print(heuristic_at_each_cell[:, :, 3])
            print('Best Gate:', best_gate)

        return best_gate
    
    def is_gate_acceptable(self, gate):
        # not actually a gate
        if (gate[0] == gate[1]) or (gate[2] == gate[3]):
            return False
        # will return gates which may change if you use square nodes
        elif self.consider_all_gates:
            return True
        # smaller side length must be at least one half the other side length
        # for square gates
        elif self.node_type == 'square':
            s1 = gate[1] - gate[0]
            s2 = gate[3] - gate[2]
            s1_bigger = s1 > s2
            smaller_side = s2 if s1_bigger else s1
            bigger_side = s1 if s1_bigger else s2
            return smaller_side >= (bigger_side/2.)
        else:
            return True

    def extend_gridcell_to_corner_gate(self, x, y, corner_idx):
        UPPER_RIGHT = 0
        UPPER_LEFT = 1
        LOWER_LEFT = 2
        LOWER_RIGHT = 3

        if corner_idx == UPPER_RIGHT:
            return [x, 1., y, 1.]
        elif corner_idx == UPPER_LEFT:
            return [0., x, y, 1.]
        elif corner_idx == LOWER_LEFT:
            return [0., x, 0., y]
        elif corner_idx == LOWER_RIGHT:
            return [x, 1., 0., y]

    def compute_heuristic(self, gate, gate_data_axes):
        pos_data = self.cur_data_from_parent_pos if self.greedy_filtering else self.pos_data
        neg_data = self.cur_data_from_parent_neg if self.greedy_filtering else self.neg_data

#        print(pos_data.shape[0], 'pos')
#        print(neg_data.shape[0])
        pos_data_inside_gate_bool_idxs = dh.filter_rectangle(
                            pos_data,
                            gate_data_axes[0], gate_data_axes[1],
                            gate[0], gate[1],
                            gate[2], gate[3],
                            return_idx=True,
                        )

        neg_data_inside_gate_bool_idxs = dh.filter_rectangle(
                            neg_data,
                            gate_data_axes[0], gate_data_axes[1],
                            gate[0], gate[1],
                            gate[2], gate[3],
                            return_idx=True
                        )

#        print(np.sum(pos_data_inside_gate_bool_idxs))
#        print(np.sum(neg_data_inside_gate_bool_idxs))
        
        
        
        # this should be correct
        pos_prop = np.sum(pos_data_inside_gate_bool_idxs)/self.pos_data.shape[0]
        neg_prop = np.sum(neg_data_inside_gate_bool_idxs)/self.neg_data.shape[0]

        
        return (pos_prop - neg_prop)




