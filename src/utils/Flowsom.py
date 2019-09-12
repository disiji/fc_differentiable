from flowsom import flowsom as _Flowsom
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
import numpy as np


#make this inherit from baseline model later
class Flowsom:

    def __init__(self, path_to_data_csv, model_params, cols_to_cluster, random_seed=1):
        self._fsom = _Flowsom(path_to_data_csv, if_fcs=False, drop_col=[])
        # get rid of extra space in the file
        self._fsom.df.columns = [column[1:] if not(col_idx == 0) else column for col_idx, column in enumerate(self._fsom.df.columns)]
        # TODO: add a check that ids and labels are here too
        # contains sample ids and labels as well
        self.df = deepcopy(self._fsom.df)
        self.num_samples = self.df[['sample_ids']].nunique().values[0]
        self.sample_ids = self.df.sample_ids.unique().astype(int)
        self.params = model_params
        self.random_seed = random_seed
        self.labels = self.get_labels()
        if cols_to_cluster:
            # the _Flowsom class uses all columns to cluster so have to get
            # rid of labels to avoid clustering based on them.
            self._fsom.df = self._fsom.df[cols_to_cluster]


    def fit(self):
        som_params = self.params['som_params']
        self._fsom.som_mapping(
            som_params['grid_size'],
            som_params['grid_size'],
            len(self._fsom.df.columns),
            som_params['weight_init_stdev'],
            som_params['som_lr'],
            som_params['num_iters'],
            if_fcs=False,
            seed=self.random_seed #check that this seed is used in meta clustering too!!
        )
        
        meta_cluster_params = self.params['meta_cluster_params']
        algorithm = self.get_algorithm_class()
        self._fsom.meta_clustering(
                algorithm,
                meta_cluster_params['min_k'],
                meta_cluster_params['max_k'],
                meta_cluster_params['iters_per_random_restart']
        )
    
        self.construct_sample_level_proportions()
        self.logistic_regressor = LogisticRegression(C=self.params['L1_logreg_penalty'],penalty='l1')
        self.fit_logistic_regressor()

    def get_algorithm_class(self):
        if self.params['meta_cluster_params']['algorithm'] == 'AgglomerativeClustering':
            return AgglomerativeClustering
    def construct_sample_level_proportions(self):
        # labels each row in self._fsom.df according to which meta 
        # cluster it's the closest to, labels range from 0 to
        # self._fsom.bestk
        self._fsom.labeling()
        self.proportions = -1 * np.ones([self.num_samples, self._fsom.bestk])
        for sample_id in self.sample_ids:
            sample = self._fsom.df.loc[self.df['sample_ids'] == sample_id]
            self.proportions[sample_id - 1] = self.get_proportions_single_sample(sample)

    def get_proportions_single_sample(self, sample):
        proportions = np.zeros([self._fsom.bestk])
        tot_cells = len(sample)
        counts = sample['category'].value_counts()
        cell_types_in_sample = sample['category'].unique()
        for i in cell_types_in_sample:
            proportions[i] = (counts[i]/tot_cells)
        return proportions

    def fit_logistic_regressor(self):
        self.logistic_regressor.fit(self.proportions, self.labels)

    def predict_single_testing_sample(self, sample):
        counts = np.zeros([1, self._fsom.bestk])
#        closest_nodes_in_som = self._fsom.map_som.winner(sample.reshape([-1, 1, 1, 8]))
#        meta_clusters = self._fsom.map_class[closest_nodes_in_som]
#        counts = np.bincount(meta_clusters)
        for cell in sample:
            closest_node_in_som = self._fsom.map_som.winner(cell)
            meta_cluster = self._fsom.map_class[closest_node_in_som]
            counts[:, meta_cluster] = counts[:, meta_cluster] + 1
        proportions = counts/np.sum(counts)
        return self.logistic_regressor.predict(proportions)



    def predict_all_samples(self):
        self.predictions = self.logistic_regressor.predict(self.proportions)
       # predictions = []
       # for sample_id in range(self.num_samples):
       #     pred = self.logistic_regressor.predict(self.proportions[sample_id])
       #     predictions.append(pred)
       # self.predictions = np.array(predictions)
        return self.predictions
    
    def predict_testing_samples(self, testing_samples):
        predictions = []
        for sample in testing_samples:
            predictions.append(self.predict_single_testing_sample(sample))
        print(predictions)
        return np.concatenate(predictions).reshape(-1, 1)

    
    def get_training_accuracy(self):
        tr_acc = 1./(self.num_samples) * np.sum(self.predictions ==  self.labels)
        return tr_acc

    def get_labels(self):
        labels = []
        for sample_id in self.sample_ids:
            sample = self.df[self.df['sample_ids'] == sample_id]
            labels.append(sample['labels'].values[0])
        return np.array(labels)

        

        
