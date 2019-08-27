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
        # will return gates which wont be as expected if you use square nodes
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


def from_gpu_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


#class HeuristicInitializerPanel2(HeuristicInitializer):
#
#    def __init__(self, node_type, gate_data_axes_ids, pos_data, neg_data, 
#            num_gridcells_per_axis=20, consider_all_gates=False, greedy_filtering=False, verbose=False, filter_func=lambda x: False):
#        super().__init__(node_type, gate_data_axes_ids, pos_data, neg_data, num_gridcells_per_axis=num_gridcells_per_axis, verbose=verbose, filter_func=filter_func)
#
#    def find_best_gate(self, gate_idx):
#        cell_locs_x = np.linspace(0., 1., num=self.num_gridcells_per_axis + 1)[1:-1]
#        cell_locs_y = np.linspace(0., 1., num=self.num_gridcells_per_axis + 1)[1:-1]
#        best_heuristic_score = -1.
#        best_gate = None
#        gate_data_axes = self.gate_data_axes_ids[gate_idx]
#        heuristic_at_each_cell = -1. * np.ones([self.num_gridcells_per_axis, self.num_gridcells_per_axis, self.NUM_CORNERS])
#        for x in cell_locs_x:
#            for y in cell_locs_y:
#                for corner_idx in range(self.NUM_CORNERS):
#                    gate = self.extend_gridcell_to_corner_gate(x, y, corner_idx)
#                    if self.is_gate_acceptable(gate, gate_idx):
#                        heuristic_score = self.compute_heuristic(gate, gate_data_axes)
#                        heuristic_at_each_cell[int(x * self.num_gridcells_per_axis) - 1, int(y * self.num_gridcells_per_axis) - 1, corner_idx] = np.around(heuristic_score, 3)
#                        if heuristic_score > best_heuristic_score:
#                            best_heuristic_score = heuristic_score
#                            best_gate = gate
#        if self.verbose:
#            print('Upper Right:')
#            print(heuristic_at_each_cell[:, :, 0])
#            print('Upper Left:')
#            print(heuristic_at_each_cell[:, :, 1])
#            print('Lower Left:')
#            print(heuristic_at_each_cell[:, :, 2])
#            print('Lower Right:')
#            print(heuristic_at_each_cell[:, :, 3])
#            print('Best Gate:', best_gate)
#
#        return best_gate
#    #method here to filter bad gates for upper/lower in anti-kapp/anti/lambda bottom gates
#    def is_gate_acceptable(self, gate, gate_idx):
#        # not actually a gate
#        if (gate[0] == gate[1]) or (gate[2] == gate[3]):
#            return False
#        # filter func for more complicated filterings
#        elif self.filter_func(gate):
#            return False
#        # handle two leaf nodes
#        elif (gate_idx == -1) or (gate_idx == -2):
#            pass
#        # will return gates which wont be as expected if you use square nodes
#        elif self.consider_all_gates:
#            return True
#        # smaller side length must be at least one half the other side length
#        # for square gates
#        elif self.node_type == 'square':
#            s1 = gate[1] - gate[0]
#            s2 = gate[3] - gate[2]
#            s1_bigger = s1 > s2
#            smaller_side = s2 if s1_bigger else s1
#            bigger_side = s1 if s1_bigger else s2
#            return smaller_side >= (bigger_side/2.)
#        else:
#            return True




#class HeuristicInitializerBoth:
#    # since this is both panel code, x_list is a list of length two lists
#    # with the first entry of each len two list containing the p1
#    # data and the second containig p2 data
#    def __init__(self, node_type, gate_ids, x, y, num_gridcells_per_axis=4, greedy_filtering=False, consider_all_gates=False, filter_func_p2=lambda x: True):
#        self.x_list = x
#        self.y = from_gpu_to_numpy(y)
#        self.gate_ids = gate_ids
#        self.node_type = node_type
#        self.num_gridcells_per_axis = num_gridcells_per_axis
#        self.greedy_filtering = greedy_filtering
#        self.consider_all_gates = consider_all_gates
#        self.filter_func_p2
#    
#    def prepare_data_and_gates(self):
#         self._construct_x_pos_and_neg()
#         self._initializer_p1_and_38_20 = \
#            HeuristicInitializer(
#                self.node_type,
#                self.gate_ids[0:-2], #double check this ordering,
#                self.x_pos_p1,
#                self.x_neg,
#                num_gridcells_per_axis=self.num_gridcells_per_axis,
#                greedy_filtering=self.greedy_filtering,
#                consider_all_gates=self.consider_all_gates
#            )
#
#         self._initializer_p2_leaves = \
#            HeuristicInitializer(
#                self.node_type,
#                self.gate_ids[-2:], #double check this ordering,
#                self.x_pos,
#                self.x_neg,
#                num_gridcells_per_axis=self.num_gridcells_per_axis,
#                greedy_filtering=self.greedy_filtering,
#                consider_all_gates=self.consider_all_gates
#                filter_func=self.filter_func
#            )
#    def get_flat_heuristic_gates(self):
#        gates_p1_and_38_20 = self._initializer_p1_and_38_20.get_heuristic_gates()
#        gates_p2_leaves = self._initializer_p2_leaves.get_heuristic_gates()
#        return gates_p1_and_38_20.extend(gates_p2_leaves)
#
#
#    def _construct_x_pos_and_neg(self):
#        self.x_pos = self._make_x_pos()
#        self.x_neg =self._make_x_neg()
#        return x_pos, x_neg
#
#    def _make_x_pos(self):
#        return [[from_gpu_to_numpy(panel) for panel in x] for i, x in enumerate(self.x_list) if self.y[i] == 1]
#
#    def _make_x_neg(self):
#        return [[from_gpu_to_numpy(panel) for panel in x] for i, x in enumerate(self.x_list) if self.y[i] == 0]
    


















