from sklearn.cluster import KMeans as sk_KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

class KMeans:

    def __init__(self, path_to_data_csv, model_params):
        self._kmeans = \
            sk_KMeans(
                n_clusters = model_params['num_clusters']
            )
        self.concat_data, self.labels, self.data_by_sample = self._load_data(path_to_data_csv)
        self.logistic_regressor = LogisticRegression()


    def _load_data(self, path_to_data_csv):
        pd_data_with_labels = pd.read_csv(path_to_data_csv)
        # get rid of any extra spaces in the column names
        pd_data_with_labels.columns = [col.replace(' ', '') for col in pd_data_with_labels.columns]
        data_by_sample = []
        labels = []
        for i in range(pd_data_with_labels['sample_ids'].nunique()):

            sample = pd_data_with_labels[pd_data_with_labels['sample_ids'] == i]
            data_by_sample.append(sample.values[:, 0:8])
            labels.append(sample['labels'].values[0])


        data_with_labels = pd_data_with_labels.values
        concat_data = data_with_labels[:, 0:8]
        return concat_data, labels, data_by_sample

    def fit(self):
        self._kmeans.fit(self.concat_data)
        self.proportions = self._construct_proportions()
        self.logistic_regressor.fit(self.proportions, self.labels)

    def _construct_proportions(self, data_for_getting_proportions=None):
        proportions = []
        if data_for_getting_proportions is None:
            for sample in self.data_by_sample:
                proportions.append(self.get_single_sample_proportion(sample))
        else:
            for sample in data_for_getting_proportions:
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

    def predict_testing_samples(self, concatenated_samples):
        testing_proportions = self._construct_proportions(testing_data=concatenated_samples)
        self.testing_predictions = self.logistic_regressor(testing_proportions)
        return self.testing_predictions

    def get_training_accuracy(self):
        if self.predictions is None:
            preds = self.predict_all_samples()
            return np.sum(self.labels == preds)/(len(self.labels))
        else:
            return np.sum(self.labels == self.predictions)/len(self.labels)



















