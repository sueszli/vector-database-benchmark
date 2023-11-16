from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.orca.automl.search import SearchEngineFactory
from bigdl.orca.automl.search.ray_tune import RayTuneSearchEngine
from bigdl.orca.automl.model.base_pytorch_model import PytorchModelBuilder
import torch
import torch.nn as nn
from bigdl.orca.automl import hp
import numpy as np
from bigdl.orca import init_orca_context, stop_orca_context

def create_simple_search_space():
    if False:
        print('Hello World!')
    return {'lr': hp.uniform(0.001, 0.01), 'batch_size': hp.choice([32, 64])}

def create_stop(stop_metric=None):
    if False:
        i = 10
        return i + 15
    stop = {'training_iteration': 20}
    stop.update({'reward_metric': stop_metric})
    return stop

def linear_model_creator(config):
    if False:
        print('Hello World!')
    'Returns a torch.nn.Module object.'
    return nn.Linear(config.get('input_dim', 1), config.get('output_dim', 1))

def optimizer_creator(model, config):
    if False:
        return 10
    'Returns optimizer defined upon the model parameters.'
    return torch.optim.SGD(model.parameters(), lr=config.get('lr', 0.01))

def loss_creator(config):
    if False:
        print('Hello World!')
    return nn.MSELoss()

def prepare_searcher(data, search_space, stop, validation_data=None, model_creator=linear_model_creator, optimizer_creator=optimizer_creator, loss_creator=loss_creator, metric='mse', metric_mode='min', name='demo'):
    if False:
        i = 10
        return i + 15
    modelBuilder = PytorchModelBuilder(model_creator=model_creator, optimizer_creator=optimizer_creator, loss_creator=loss_creator)
    searcher = SearchEngineFactory.create_engine(backend='ray', logs_dir='~/zoo_automl_logs', resources_per_trial={'cpu': 2}, name=name)
    searcher.compile(data=data, validation_data=validation_data, model_builder=modelBuilder, search_space=search_space, n_sampling=2, epochs=stop['training_iteration'], metric_threshold=stop['reward_metric'], metric_mode=metric_mode, metric=metric)
    return searcher

def get_np_input():
    if False:
        print('Hello World!')

    def get_linear_data(a, b, size):
        if False:
            i = 10
            return i + 15
        x = np.arange(0, 10, 10 / size)
        y = a * x + b
        return (x, y)
    (train_x, train_y) = get_linear_data(2, 5, 1000)
    (val_x, val_y) = get_linear_data(2, 5, 400)
    return (train_x, train_y, val_x, val_y)

class TestRayTuneSearchEngine(ZooTestCase):

    def setup_method(self, method):
        if False:
            return 10
        init_orca_context(init_ray_on_spark=True)

    def teardown_method(self, method):
        if False:
            while True:
                i = 10
        stop_orca_context()

    def test_numpy_input(self):
        if False:
            print('Hello World!')
        (train_x, train_y, val_x, val_y) = get_np_input()
        data = (train_x, train_y)
        val_data = (val_x, val_y)
        searcher = prepare_searcher(data=data, validation_data=val_data, name='test_ray_numpy_with_val', search_space=create_simple_search_space(), stop=create_stop())
        searcher.run()
        best_trials = searcher.get_best_trials(k=1)
        assert best_trials is not None

    def test_searcher_metric(self):
        if False:
            i = 10
            return i + 15
        (train_x, train_y, val_x, val_y) = get_np_input()
        data = (train_x, train_y)
        val_data = (val_x, val_y)
        searcher = prepare_searcher(data=data, validation_data=val_data, name='test_searcher_metric_name', metric='mse', metric_mode='min', search_space=create_simple_search_space(), stop=create_stop(float('inf')))
        analysis = searcher.run()
        sorted_results = list(map(lambda x: x.last_result['mse'], RayTuneSearchEngine._get_sorted_trials(analysis.trials, metric='mse', mode='min')))
        assert 'mse' in analysis.trials[0].last_result.keys()
        assert all((sorted_results[i] <= sorted_results[i + 1] for i in range(len(sorted_results) - 1)))
        assert RayTuneSearchEngine._get_best_result(analysis.trials, metric='mse', mode='min')['mse'] == sorted_results[0]
        assert all((analysis.trials[i].last_result['mse'] >= analysis.trials[i].last_result['best_mse'] for i in range(len(sorted_results))))
        assert analysis.trials[0].last_result['iterations_since_restore'] == 1
        searcher = prepare_searcher(data=data, validation_data=val_data, name='test_searcher_metric_name', metric='r2', metric_mode='max', search_space=create_simple_search_space(), stop=create_stop(0))
        analysis = searcher.run()
        sorted_results = list(map(lambda x: x.last_result['r2'], RayTuneSearchEngine._get_sorted_trials(analysis.trials, metric='r2', mode='max')))
        assert 'r2' in analysis.trials[0].last_result.keys()
        assert all((sorted_results[i] >= sorted_results[i + 1] for i in range(len(sorted_results) - 1)))
        assert RayTuneSearchEngine._get_best_result(analysis.trials, metric='r2', mode='max')['r2'] == sorted_results[0]
        assert all((analysis.trials[i].last_result['r2'] <= analysis.trials[i].last_result['best_r2'] for i in range(len(sorted_results))))
        assert analysis.trials[0].last_result['iterations_since_restore'] == 1
        searcher = prepare_searcher(data=data, validation_data=val_data, name='test_searcher_metric_name', metric='mae', metric_mode='min', search_space=create_simple_search_space(), stop=create_stop(0))
        analysis = searcher.run()
        sorted_results = list(map(lambda x: x.last_result['mae'], RayTuneSearchEngine._get_sorted_trials(analysis.trials, metric='mae', mode='min')))
        assert 'mae' in analysis.trials[0].last_result.keys()
        assert all((sorted_results[i] <= sorted_results[i + 1] for i in range(len(sorted_results) - 1)))
        assert RayTuneSearchEngine._get_best_result(analysis.trials, metric='mae', mode='min')['mae'] == sorted_results[0]
        assert all((analysis.trials[i].last_result['mae'] >= analysis.trials[i].last_result['best_mae'] for i in range(len(sorted_results))))
        assert analysis.trials[0].last_result['iterations_since_restore'] == 20