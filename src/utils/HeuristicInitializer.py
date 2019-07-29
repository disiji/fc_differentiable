import numpy as np
import utils.utils_load_data as dh


class HeuristicInitializer:
    '''
        data is a numpy array of all the cells normalized marker data
    '''
    def __init__(self, node_type, gate_data_ids, pos_data, neg_data, num_gridcells_per_axis=20):
        self.node_type = node_type
        self.num_gates = len(gate_data_ids)
        self.gate_data_ids = gate_data_ids
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.num_gridcells_per_axis = 20
        self.heuristic_gates = None
        self.pos_data_inside_gate_dict = {i:None for i in range(self.num_gates)}
        self.neg_data_inside_gate_dict = {i:None for i in range(self.num_gates)}
        self.NUM_CORNERS = 4

    def construct_heuristic_gates(self):
        heuristic_gates = []
        for gate_idx in range(self.num_gates):
            heuristic_gates.append(self.find_best_gate(gate_idx))
        self.heuristic_gates= heuristic_gates

    def get_heuristic_gates(self):
        if self.heuristic_gates:
            return self.heuristic_gates
        else:
            self.construct_heuristic_gates()
            return self.heuristic_gates


    def find_best_gate(self, gate_idx):
        cell_locs_x = np.linspace(0., 1., self.num_gridcells_per_axis)
        cell_locs_y = np.linspace(0., 1., self.num_gridcells_per_axis)
        best_heuristic_score = -1.
        best_gate = None
        for x in cell_locs_x:
            for y in cell_locs_y:
                for corner_idx in range(self.NUM_CORNERS):
                    gate = self.extend_gridcell_to_corner_gate(x, y, corner_idx)
                    if self.is_gate_acceptable(gate):
                        heuristic_score = self.compute_heuristic(gate, gate_idx)
                        if heuristic_score > best_heuristic_score:
                            best_heuristic_score = heuristic_score
                            best_gate = gate
        return best_gate
    
    def is_gate_acceptable(self, gate):
        # smaller side length must be at least one half the other side length
        # for square gates
        if self.node_type == 'square':
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

    def compute_heuristic(self, gate, gate_idx):
        data_dims = self.gate_data_ids[gate_idx]
        pos_data_at_gate = self.pos_data[:, [data_dims[0], data_dims[1]]]
        neg_data_at_gate = self.neg_data[:, [data_dims[0], data_dims[1]]]

        if (self.pos_data_inside_gate_dict[gate_idx]) is None:
            pos_data_inside_gate = dh.filter_rectangle(
                                pos_data_at_gate,
                                0, 1,
                                gate[0], gate[1],
                                gate[2], gate[3],
                            )
            self.pos_data_inside_gate_dict[gate_idx] = pos_data_inside_gate
        else:
            pos_data_inside_gate = self.pos_data_inside_gate_dict[gate_idx]

        if (self.neg_data_inside_gate_dict[gate_idx]) is None:
            neg_data_inside_gate = dh.filter_rectangle(
                                neg_data_at_gate,
                                0, 1,
                                gate[0], gate[1],
                                gate[2], gate[3],
                            )
            self.neg_data_inside_gate_dict[gate_idx] = neg_data_inside_gate
        else:
            neg_data_inside_gate = self.neg_data_inside_gate_dict[gate_idx]

        pos_prop = pos_data_inside_gate.shape[0]/pos_data_at_gate.shape[0]
        neg_prop = neg_data_inside_gate.shape[0]/neg_data_at_gate.shape[0]

        return (pos_prop - neg_prop)**2




