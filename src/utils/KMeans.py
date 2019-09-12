from sklearn.cluster import KMeans as sk_KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

class KMeans:

    def __init__(self, path_to_data_csv, model_params, random_state=0):
        self._kmeans = \
            sk_KMeans(
                n_clusters = model_params['num_clusters'],
                n_init = model_params['n_init'],
                random_state=random_state
            )
        self.concat_data, self.labels, self.data_by_sample = self._load_data(path_to_data_csv, model_params)
        self.logistic_regressor = LogisticRegression(penalty='l1', C=1e10)


    def _load_data(self, path_to_data_csv, model_params):
        pd_data_with_labels = pd.read_csv(path_to_data_csv)
        # get rid of any extra spaces in the column names
        pd_data_with_labels.columns = [col.replace(' ', '') for col in pd_data_with_labels.columns]
        data_by_sample = []
        labels = []
        for i in range(pd_data_with_labels['sample_ids'].nunique()):

            sample_df = pd_data_with_labels[pd_data_with_labels['sample_ids'] == i]
            if not(len(sample_df) * model_params['subsample_frac'] < model_params['min_number_of_cells']):
                sample = sample_df.values[np.random.permutation(len(sample_df))][0: int(len(sample_df) * model_params['subsample_frac'])]
            else:
                sample = sample_df.values[np.random.permutation(len(sample_df))][0: model_params['min_number_of_cells']]
            data_by_sample.append(sample[:, 0:8])
            labels.append(sample_df['labels'].values[0])


        data_with_labels = pd_data_with_labels.values
        concat_data = data_with_labels[:, 0:8]
        return concat_data, labels, data_by_sample

    def fit(self):
        self._kmeans.fit(self.concat_data)
        self.proportions = self._construct_proportions()
        self.logistic_regressor.fit(self.proportions, self.labels)

    def _construct_proportions(self, testing_data=None):
        proportions = []
        if testing_data is None:
            for sample in self.data_by_sample:
                proportions.append(self.get_single_sample_proportion(sample))
        else:
            for sample in testing_data:
                proportions.append(self.get_single_sample_proportion(sample))
        return proportions

    def get_single_sample_proportion(self, sample):
        cluster_centers = self._kmeans.cluster_centers_
        closest_clusters = self._kmeans.predict(sample)

        proportions = []
        for cluster_idx in range(cluster_centers.shape[0]):
            cur_prop = np.sum(closest_clusters == cluster_idx)/sample.shape[0]
            proportions.append(cur_prop)
        return proportions

    def predict_all_samples(self):
        self.predictions = self.logistic_regressor.predict(self.proportions)
        return self.predictions

    def predict_single_testing_sample(self, sample):
        proportion = self._get_single_proportion(sample)
        prediction = self.logistic_regressor.predict(proportion)
        return prediction

    def predict_testing_samples(self, samples_list):
        testing_proportions = self._construct_proportions(testing_data=samples_list)
        self.testing_predictions = self.logistic_regressor.predict(testing_proportions)
        return self.testing_predictions

    def get_training_accuracy(self):
        if self.predictions is None:
            preds = self.predict_all_samples()
            return np.sum(self.labels == preds)/(len(self.labels))
        else:
            return np.sum(self.labels == self.predictions)/len(self.labels)



















