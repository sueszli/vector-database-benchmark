import uuid
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
import pycaret.anomaly
import pycaret.datasets

@pytest.fixture(scope='module')
def data():
    if False:
        while True:
            i = 10
    return pycaret.datasets.get_data('anomaly')

def test_anomaly(data):
    if False:
        i = 10
        return i + 15
    experiment_name = uuid.uuid4().hex
    pycaret.anomaly.setup(data, normalize=True, log_experiment=True, experiment_name=experiment_name, experiment_custom_tags={'tag': 1}, log_plots=True, html=False, session_id=123, n_jobs=1)
    iforest = pycaret.anomaly.create_model('iforest', experiment_custom_tags={'tag': 1})
    knn = pycaret.anomaly.create_model('knn', experiment_custom_tags={'tag': 1})
    cluster = pycaret.anomaly.create_model('cluster', experiment_custom_tags={'tag': 1})
    pycaret.anomaly.plot_model(iforest)
    pycaret.anomaly.plot_model(knn)
    iforest_results = pycaret.anomaly.assign_model(iforest)
    knn_results = pycaret.anomaly.assign_model(knn)
    cluster_results = pycaret.anomaly.assign_model(cluster)
    assert isinstance(iforest_results, pd.DataFrame)
    assert isinstance(knn_results, pd.DataFrame)
    assert isinstance(cluster_results, pd.DataFrame)
    iforest_predictions = pycaret.anomaly.predict_model(model=iforest, data=data)
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)
    cluster_predictions = pycaret.anomaly.predict_model(model=cluster, data=data)
    assert isinstance(iforest_predictions, pd.DataFrame)
    assert isinstance(knn_predictions, pd.DataFrame)
    assert isinstance(cluster_predictions, pd.DataFrame)
    X = pycaret.anomaly.get_config('X')
    seed = pycaret.anomaly.get_config('seed')
    assert isinstance(X, pd.DataFrame)
    assert isinstance(seed, int)
    pycaret.anomaly.set_config('seed', 124)
    seed = pycaret.anomaly.get_config('seed')
    assert seed == 124
    all_models = pycaret.anomaly.models()
    assert isinstance(all_models, pd.DataFrame)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for experiment_run in client.list_run_infos(experiment.experiment_id):
        run = client.get_run(experiment_run.run_id)
        assert run.data.tags.get('tag') == '1'
    pycaret.anomaly.save_model(knn, 'knn_model_23122019')
    pycaret.anomaly.set_current_experiment(pycaret.anomaly.AnomalyExperiment())
    knn = pycaret.anomaly.load_model('knn_model_23122019')
    knn_predictions = pycaret.anomaly.predict_model(model=knn, data=data)
    assert isinstance(knn_predictions, pd.DataFrame)
if __name__ == '__main__':
    test_anomaly()