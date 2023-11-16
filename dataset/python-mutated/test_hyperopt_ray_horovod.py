import os.path
import shutil
import uuid
from unittest.mock import patch
import pytest
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.constants import ACCURACY, AUTO, BATCH_SIZE, EXECUTOR, MAX_CONCURRENT_TRIALS, TRAINER
from ludwig.globals import HYPEROPT_STATISTICS_FILE_NAME
from ludwig.hyperopt.results import HyperoptResults
from ludwig.hyperopt.run import hyperopt
from ludwig.hyperopt.utils import update_hyperopt_params_with_defaults
from ludwig.schema.model_config import ModelConfig
from tests.integration_tests.utils import binary_feature, create_data_set_to_use, generate_data, number_feature
try:
    import ray
    from ray.tune.syncer import get_node_to_storage_syncer, SyncConfig
    from ludwig.backend.ray import RayBackend
    from ludwig.hyperopt.execution import _get_relative_checkpoints_dir_parts, RayTuneExecutor
except ImportError:
    ray = None
    RayTuneExecutor = object
pytestmark = pytest.mark.integration_tests_a
LOCAL_SYNC_TEMPLATE = 'echo {source}/ {target}/'
LOCAL_DELETE_TEMPLATE = 'echo {target}'

def mock_storage_client(path):
    if False:
        print('Hello World!')
    'Mocks storage client that treats a local dir as durable storage.'
    os.makedirs(path, exist_ok=True)
    syncer = get_node_to_storage_syncer(SyncConfig(upload_dir=path))
    return syncer
HYPEROPT_CONFIG = {'parameters': {'trainer.learning_rate': {'space': 'loguniform', 'lower': 0.001, 'upper': 0.1}, 'combiner.output_size': {'space': 'grid_search', 'values': [4, 8]}}, 'goal': 'minimize'}
SCENARIOS = [{'executor': {'type': 'ray', 'num_samples': 2, 'trial_driver_resources': {'hyperopt_resources': 1}, 'cpu_resources_per_trial': 1}, 'search_alg': {'type': 'variant_generator'}}, {'executor': {'type': 'ray', 'num_samples': 2, 'scheduler': {'type': 'hb_bohb', 'time_attr': 'training_iteration', 'reduction_factor': 4}, 'trial_driver_resources': {'hyperopt_resources': 1}, 'cpu_resources_per_trial': 1}, 'search_alg': {'type': 'bohb'}}]
RAY_BACKEND_KWARGS = {'processor': {'parallelism': 1}}

def _get_config(search_alg, executor):
    if False:
        for i in range(10):
            print('nop')
    input_features = [number_feature()]
    output_features = [binary_feature()]
    num_epochs = 1 if search_alg['type'] == 'variant_generator' else 81
    return {'input_features': input_features, 'output_features': output_features, 'combiner': {'type': 'concat'}, TRAINER: {'epochs': num_epochs, 'learning_rate': 0.001, BATCH_SIZE: 128}, 'hyperopt': {**HYPEROPT_CONFIG, 'executor': executor, 'search_alg': search_alg}}

class MockRayTuneExecutor(RayTuneExecutor):

    def _get_sync_client_and_remote_checkpoint_dir(self, trial_dir):
        if False:
            i = 10
            return i + 15
        remote_checkpoint_dir = os.path.join(self.mock_path, *_get_relative_checkpoints_dir_parts(trial_dir))
        return (mock_storage_client(remote_checkpoint_dir), remote_checkpoint_dir)

class CustomTestCallback(Callback):

    def __init__(self):
        if False:
            print('Hello World!')
        self.preprocessed = False

    def on_hyperopt_preprocessing_start(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.preprocessed = True

    def on_hyperopt_start(self, *args, **kwargs):
        if False:
            print('Hello World!')
        assert self.preprocessed

@pytest.fixture
def ray_mock_dir():
    if False:
        print('Hello World!')
    path = os.path.join(ray._private.utils.get_user_temp_dir(), f'mock-client-{uuid.uuid4().hex[:4]}') + os.sep
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)

def run_hyperopt_executor(search_alg, executor, csv_filename, ray_mock_dir, validate_output_feature=False, validation_metric=None):
    if False:
        while True:
            i = 10
    config = _get_config(search_alg, executor)
    csv_filename = os.path.join(ray_mock_dir, 'dataset.csv')
    dataset_csv = generate_data(config['input_features'], config['output_features'], csv_filename, num_examples=25)
    dataset_parquet = create_data_set_to_use('parquet', dataset_csv)
    config = ModelConfig.from_dict(config).to_dict()
    hyperopt_config = config['hyperopt']
    if validate_output_feature:
        hyperopt_config['output_feature'] = config['output_features'][0]['name']
    if validation_metric:
        hyperopt_config['validation_metric'] = validation_metric
    backend = RayBackend(**RAY_BACKEND_KWARGS)
    update_hyperopt_params_with_defaults(hyperopt_config)
    if hyperopt_config[EXECUTOR].get(MAX_CONCURRENT_TRIALS) == AUTO:
        hyperopt_config[EXECUTOR][MAX_CONCURRENT_TRIALS] = backend.max_concurrent_trials(hyperopt_config)
    parameters = hyperopt_config['parameters']
    if search_alg.get('type', '') == 'bohb':
        del parameters['combiner.output_size']
        hyperopt_config['parameters'] = parameters
    split = hyperopt_config['split']
    output_feature = hyperopt_config['output_feature']
    metric = hyperopt_config['metric']
    goal = hyperopt_config['goal']
    search_alg = hyperopt_config['search_alg']
    model = LudwigModel(config=config, backend=backend)
    (training_set, validation_set, test_set, training_set_metadata) = model.preprocess(dataset=dataset_parquet)
    hyperopt_executor = MockRayTuneExecutor(parameters, output_feature, metric, goal, split, search_alg=search_alg, **hyperopt_config[EXECUTOR])
    hyperopt_executor.mock_path = os.path.join(ray_mock_dir, 'bucket')
    hyperopt_executor.execute(config, training_set=training_set, validation_set=validation_set, test_set=test_set, training_set_metadata=training_set_metadata, backend=backend, output_directory=ray_mock_dir, skip_save_processed_input=True, skip_save_unprocessed_output=True, resume=False)

@pytest.mark.slow
@pytest.mark.distributed
def test_hyperopt_executor_variant_generator(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    if False:
        i = 10
        return i + 15
    search_alg = SCENARIOS[0]['search_alg']
    executor = SCENARIOS[0]['executor']
    run_hyperopt_executor(search_alg, executor, csv_filename, ray_mock_dir)

@pytest.mark.skip(reason='PG/resource cleanup bugs in Ray 2.x: https://github.com/ray-project/ray/issues/31738')
@pytest.mark.distributed
def test_hyperopt_executor_bohb(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    if False:
        return 10
    search_alg = SCENARIOS[1]['search_alg']
    executor = SCENARIOS[1]['executor']
    run_hyperopt_executor(search_alg, executor, csv_filename, ray_mock_dir)

@pytest.mark.distributed
@pytest.mark.skip(reason='https://github.com/ludwig-ai/ludwig/issues/1441')
@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    if False:
        for i in range(10):
            print('nop')
    run_hyperopt_executor({'type': 'variant_generator'}, {'type': 'ray', 'num_samples': 2}, csv_filename, ray_mock_dir, validate_output_feature=True, validation_metric=ACCURACY)

@pytest.mark.skip(reason='https://github.com/ludwig-ai/ludwig/issues/1441')
@pytest.mark.distributed
@patch('ludwig.hyperopt.execution.RayTuneExecutor', MockRayTuneExecutor)
def test_hyperopt_run_hyperopt(csv_filename, ray_mock_dir, ray_cluster_7cpu):
    if False:
        while True:
            i = 10
    input_features = [number_feature()]
    output_features = [binary_feature()]
    csv_filename = os.path.join(ray_mock_dir, 'dataset.csv')
    dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=100)
    dataset_parquet = create_data_set_to_use('parquet', dataset_csv)
    config = {'input_features': input_features, 'output_features': output_features, 'combiner': {'type': 'concat'}, TRAINER: {'epochs': 1, 'learning_rate': 0.001, BATCH_SIZE: 128}, 'backend': {'type': 'ray', **RAY_BACKEND_KWARGS}}
    output_feature_name = output_features[0]['name']
    hyperopt_configs = {'parameters': {'trainer.learning_rate': {'space': 'loguniform', 'lower': 0.001, 'upper': 0.1}, output_feature_name + '.output_size': {'space': 'randint', 'lower': 2, 'upper': 8}}, 'goal': 'minimize', 'output_feature': output_feature_name, 'validation_metrics': 'loss', 'executor': {'type': 'ray', 'num_samples': 2}, 'search_alg': {'type': 'variant_generator'}}
    config['hyperopt'] = hyperopt_configs
    run_hyperopt(config, dataset_parquet, ray_mock_dir)

def run_hyperopt(config, rel_path, out_dir, experiment_name='ray_hyperopt'):
    if False:
        i = 10
        return i + 15
    callback = CustomTestCallback()
    hyperopt_results = hyperopt(config, dataset=rel_path, output_directory=out_dir, experiment_name=experiment_name, callbacks=[callback])
    assert isinstance(hyperopt_results, HyperoptResults)
    assert os.path.isfile(os.path.join(out_dir, experiment_name, HYPEROPT_STATISTICS_FILE_NAME))