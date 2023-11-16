import sys
import uuid
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
import pycaret.clustering
import pycaret.datasets
if sys.platform == 'win32':
    pytest.skip('Skipping test module on Windows', allow_module_level=True)

@pytest.fixture(scope='module')
def data():
    if False:
        while True:
            i = 10
    return pycaret.datasets.get_data('jewellery')

def test_clustering(data):
    if False:
        for i in range(10):
            print('nop')
    experiment_name = uuid.uuid4().hex
    pycaret.clustering.setup(data, normalize=True, log_experiment=True, experiment_name=experiment_name, experiment_custom_tags={'tag': 1}, log_plots=True, html=False, session_id=123, n_jobs=1)
    kmeans = pycaret.clustering.create_model('kmeans', experiment_custom_tags={'tag': 1})
    kmodes = pycaret.clustering.create_model('kmodes', experiment_custom_tags={'tag': 1})
    pycaret.clustering.plot_model(kmeans)
    pycaret.clustering.plot_model(kmodes)
    kmeans_results = pycaret.clustering.assign_model(kmeans)
    kmodes_results = pycaret.clustering.assign_model(kmodes)
    assert isinstance(kmeans_results, pd.DataFrame)
    assert isinstance(kmodes_results, pd.DataFrame)
    kmeans_predictions = pycaret.clustering.predict_model(model=kmeans, data=data)
    assert isinstance(kmeans_predictions, pd.DataFrame)
    all_models = pycaret.clustering.models()
    assert isinstance(all_models, pd.DataFrame)
    X = pycaret.clustering.get_config('X')
    seed = pycaret.clustering.get_config('seed')
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)
    pycaret.clustering.set_config('seed', 124)
    seed = pycaret.clustering.get_config('seed')
    assert seed == 124
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.list_run_infos(experiment.experiment_id):
        run = client.get_run(experiment_run.run_id)
        assert run.data.tags.get('tag') == '1'
    pycaret.clustering.save_model(kmeans, 'kmeans_model_23122019')
    pycaret.clustering.set_current_experiment(pycaret.clustering.ClusteringExperiment())
    kmeans = pycaret.clustering.load_model('kmeans_model_23122019')
    kmeans_predictions = pycaret.clustering.predict_model(model=kmeans, data=data)
    assert isinstance(kmeans_predictions, pd.DataFrame)
if __name__ == '__main__':
    test_clustering()