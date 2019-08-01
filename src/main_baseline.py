from utils.BaselineParamsParser import BaselineParamsParser
from utils.Flowsom import Flowsom
from utils.KMeans import KMeans

def main(path_to_params):
    parser = BaselineParamsParser(path_to_params)
    params = parser.parse_params()
    # abstract the following into a factory to make
    # cleaner
    if params['clustering_type'] == 'flowsom':
        model = Flowsom(
                params['path_to_data_csv'],
                params['flowsom_params'],
                params['columns_to_cluster']
        )
    elif params['clustering_type'] == 'kmeans':
        model = \
            KMeans(
                params['path_to_data_csv'],
                params['kmeans_params']
            )
    else:
        raise ValueError('Model type not recognized')

    model.fit()
    print('Model fitting complete')
    model.predict_all_samples()
    print('Model prediction on train data complete')
    tr_acc = model.get_training_accuracy()
    print('Training Accuracy is %.3f' %tr_acc)




#def main(path_to_params):
#    parser = baselineParamsParser(path_to_params)
#    params = parser.get_params()
#    model = BaselineModelFactory.create_model(params['model_params'])
#    data_loader = BaselineDataLoaderFactory.create_data_loader(
#                        params['data_loading_params'], type(model)
#    )
#    model.fit(data_loader.training_data)
#    diagnostics = BaselineDiagnostics(
#                    model, 
#                    data_loader.training_data,
#                    data_loader.test_data,
#                    hparams['diagnostic_params']
#    )
#    diagnostics.write_diagnostics(model)
#    diagnostics.write_visualizations(model)
    




if __name__ == '__main__':
    path_to_params = '../configs/testing_kmeans.yaml'
    main(path_to_params)









