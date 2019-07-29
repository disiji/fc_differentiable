import numpy as np

'''
model1/model2: ModelTree objects to compare the overlap from their gates
data_list: list of numpy arrays of data per sample
'''
class CellOverlaps:

    def __init__(self, model1, model2, data_list):
        self.model1 = model1
        self.model2 = model2
        self.data_list_with_ids = self._init_data_with_ids(data_list)

    '''
    gives each cell a unique id which is appended to the 
    end of each cells marker values. 
    (ie an extra row is added for ids)
    '''
    def _init_data_with_ids(self, data_list):
        last_cell_idx = 0
        data_list_with_ids = []
        for data in data_list:
            cur_data_idxs = last_cell_idx + np.arange(data.shape[0])
            data_with_ids = np.hstack([data, cur_data_idxs[:, np.newaxis]])
            data_list_with_ids.append(data_with_ids)
            last_cell_idx = last_cell_idx + len(cur_data_idxs)        
        return data_list_with_ids

    def compute_overlap_diagnostics(self):
        overlap_diagnostics = []
        for data in self.data_list_with_ids:
            model1_leaf_data = self.model1.filter_data_to_leaf(data)
            model2_leaf_data = self.model2.filter_data_to_leaf(data)
            num_overlap = self.compute_overlaps_single_data_array(
                model1_leaf_data,
                model2_leaf_data,
            )
            in_m1_leaf_not_m2 = self.compute_in_m1_not_m2(model1_leaf_data, model2_leaf_data)
            in_m2_leaf_not_m1 = self.compute_in_m2_not_m1(model1_leaf_data, model2_leaf_data)
            overlap_diagnostics.append(
                (
                    num_overlap, 
                    in_m1_leaf_not_m2, 
                    in_m2_leaf_not_m1,
                    model1_leaf_data.shape[0],
                    model2_leaf_data.shape[0]
                )
            )
            assert(in_m1_leaf_not_m2 + in_m2_leaf_not_m1 + 2 * num_overlap == model1_leaf_data.shape[0] + model2_leaf_data.shape[0])
        return np.array(overlap_diagnostics)
    
    def compute_overlaps_single_data_array(self, data1, data2):
        ids1 = data1[:, -1]
        ids2 = data2[:, -1]
        num_overlap = np.sum(np.in1d(ids1, ids2))
        return num_overlap

    def compute_in_m1_not_m2(self, data1, data2):
        ids1 = data1[:, -1]
        ids2 = data2[:, -1]
        num_in_d1_not_d2 = np.sum(np.in1d(ids1, ids2) == False)
        return num_in_d1_not_d2

    def compute_in_m2_not_m1(self, data1, data2):
        # just flip argument order
        return self.compute_in_m1_not_m2(data2, data1)


            

        

        


        
