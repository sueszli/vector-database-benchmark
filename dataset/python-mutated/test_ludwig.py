import contextlib
import os
import tempfile
import pytest
import ray
ludwig_installed = True
tf_installed = True
try:
    import ludwig
except (ImportError, ModuleNotFoundError):
    ludwig_installed = False
try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError):
    tf_installed = False
skip = not ludwig_installed or not tf_installed
pytestmark = pytest.mark.skipif(skip, reason='Missing Ludwig dependency')
if not skip:
    from ludwig.backend.ray import RayBackend, get_horovod_kwargs
    from ray.tests.ludwig.ludwig_test_utils import create_data_set_to_use, spawn
    from ray.tests.ludwig.ludwig_test_utils import bag_feature
    from ray.tests.ludwig.ludwig_test_utils import binary_feature
    from ray.tests.ludwig.ludwig_test_utils import category_feature
    from ray.tests.ludwig.ludwig_test_utils import date_feature
    from ray.tests.ludwig.ludwig_test_utils import generate_data
    from ray.tests.ludwig.ludwig_test_utils import h3_feature
    from ray.tests.ludwig.ludwig_test_utils import numerical_feature
    from ray.tests.ludwig.ludwig_test_utils import sequence_feature
    from ray.tests.ludwig.ludwig_test_utils import set_feature
    from ray.tests.ludwig.ludwig_test_utils import train_with_backend
    from ray.tests.ludwig.ludwig_test_utils import vector_feature
else:

    def spawn(func):
        if False:
            while True:
                i = 10
        return func

@contextlib.contextmanager
def ray_start_2_cpus():
    if False:
        for i in range(10):
            print('nop')
    is_ray_initialized = ray.is_initialized()
    with tempfile.TemporaryDirectory() as tmpdir:
        if not is_ray_initialized:
            res = ray.init(num_cpus=2, include_dashboard=False, object_store_memory=150 * 1024 * 1024, _temp_dir=tmpdir)
        else:
            res = None
        try:
            yield res
        finally:
            if not is_ray_initialized:
                ray.shutdown()

def run_api_experiment(config, data_parquet):
    if False:
        print('Hello World!')
    kwargs = get_horovod_kwargs()
    assert kwargs.get('num_workers') == 2
    dask_backend = RayBackend()
    assert train_with_backend(dask_backend, config, dataset=data_parquet, evaluate=False)

@spawn
def run_test_parquet(input_features, output_features, num_examples=100, run_fn=run_api_experiment, expect_error=False):
    if False:
        while True:
            i = 10
    tf.config.experimental_run_functions_eagerly(True)
    with ray_start_2_cpus():
        config = {'input_features': input_features, 'output_features': output_features, 'combiner': {'type': 'concat', 'fc_size': 14}, 'training': {'epochs': 2, 'batch_size': 8}}
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_filename = os.path.join(tmpdir, 'dataset.csv')
            dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=num_examples)
            dataset_parquet = create_data_set_to_use('parquet', dataset_csv)
            if expect_error:
                with pytest.raises(ValueError):
                    run_fn(config, data_parquet=dataset_parquet)
            else:
                run_fn(config, data_parquet=dataset_parquet)

def test_ray_tabular():
    if False:
        print('Hello World!')
    input_features = [sequence_feature(reduce_output='sum'), numerical_feature(normalization='zscore'), set_feature(), binary_feature(), bag_feature(), vector_feature(), h3_feature(), date_feature()]
    output_features = [category_feature(vocab_size=2, reduce_input='sum'), binary_feature(), set_feature(max_len=3, vocab_size=5), numerical_feature(normalization='zscore'), vector_feature()]
    run_test_parquet(input_features, output_features)

def test_ray_tabular_client():
    if False:
        print('Hello World!')
    from ray.util.client.ray_client_helpers import ray_start_client_server
    with ray_start_2_cpus():
        assert not ray.util.client.ray.is_connected()
        with ray_start_client_server():
            assert ray.util.client.ray.is_connected()
            test_ray_tabular()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', '-x', __file__]))