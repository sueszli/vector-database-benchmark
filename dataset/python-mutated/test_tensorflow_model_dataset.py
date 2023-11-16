from pathlib import PurePosixPath
import numpy as np
import pytest
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER, Version

@pytest.fixture(scope='module')
def tf():
    if False:
        for i in range(10):
            print('nop')
    import tensorflow as tf
    return tf

@pytest.fixture(scope='module')
def tensorflow_model_dataset():
    if False:
        for i in range(10):
            print('nop')
    from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
    return TensorFlowModelDataset

@pytest.fixture
def filepath(tmp_path):
    if False:
        while True:
            i = 10
    return (tmp_path / 'test_tf').as_posix()

@pytest.fixture
def dummy_x_train():
    if False:
        print('Hello World!')
    return np.array([[[1.0], [1.0]], [[0.0], [0.0]]])

@pytest.fixture
def dummy_y_train():
    if False:
        while True:
            i = 10
    return np.array([[[1], [1]], [[1], [1]]])

@pytest.fixture
def dummy_x_test():
    if False:
        return 10
    return np.array([[[0.0], [0.0]], [[1.0], [1.0]]])

@pytest.fixture
def tf_model_dataset(filepath, load_args, save_args, fs_args, tensorflow_model_dataset):
    if False:
        return 10
    return tensorflow_model_dataset(filepath=filepath, load_args=load_args, save_args=save_args, fs_args=fs_args)

@pytest.fixture
def versioned_tf_model_dataset(filepath, load_version, save_version, tensorflow_model_dataset):
    if False:
        i = 10
        return i + 15
    return tensorflow_model_dataset(filepath=filepath, version=Version(load_version, save_version))

@pytest.fixture
def dummy_tf_base_model(dummy_x_train, dummy_y_train, tf):
    if False:
        print('Hello World!')
    inputs = tf.keras.Input(shape=(2, 1))
    x = tf.keras.layers.Dense(1)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='1_layer_dummy')
    model.compile('rmsprop', 'mse')
    model.fit(dummy_x_train, dummy_y_train, batch_size=64, epochs=1)
    model.reset_metrics()
    return model

@pytest.fixture
def dummy_tf_base_model_new(dummy_x_train, dummy_y_train, tf):
    if False:
        while True:
            i = 10
    inputs = tf.keras.Input(shape=(2, 1))
    x = tf.keras.layers.Dense(1)(inputs)
    x = tf.keras.layers.Dense(1)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='2_layer_dummy')
    model.compile('rmsprop', 'mse')
    model.fit(dummy_x_train, dummy_y_train, batch_size=64, epochs=1)
    model.reset_metrics()
    return model

@pytest.fixture
def dummy_tf_subclassed_model(dummy_x_train, dummy_y_train, tf):
    if False:
        while True:
            i = 10
    'Demonstrate that own class models cannot be saved\n    using HDF5 format but can using TF format\n    '

    class MyModel(tf.keras.Model):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

        def call(self, inputs, training=None, mask=None):
            if False:
                for i in range(10):
                    print('nop')
            x = self.dense1(inputs)
            return self.dense2(x)
    model = MyModel()
    model.compile('rmsprop', 'mse')
    model.fit(dummy_x_train, dummy_y_train, batch_size=64, epochs=1)
    return model

class TestTensorFlowModelDataset:
    """No versioning passed to creator"""

    def test_save_and_load(self, tf_model_dataset, dummy_tf_base_model, dummy_x_test):
        if False:
            return 10
        'Test saving and reloading the data set.'
        predictions = dummy_tf_base_model.predict(dummy_x_test)
        tf_model_dataset.save(dummy_tf_base_model)
        reloaded = tf_model_dataset.load()
        new_predictions = reloaded.predict(dummy_x_test)
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-06, atol=1e-06)
        assert tf_model_dataset._load_args == {}
        assert tf_model_dataset._save_args == {'save_format': 'tf'}

    def test_load_missing_model(self, tf_model_dataset):
        if False:
            while True:
                i = 10
        'Test error message when trying to load missing model.'
        pattern = 'Failed while loading data from data set TensorFlowModelDataset\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            tf_model_dataset.load()

    def test_exists(self, tf_model_dataset, dummy_tf_base_model):
        if False:
            return 10
        'Test `exists` method invocation for both existing and nonexistent data set.'
        assert not tf_model_dataset.exists()
        tf_model_dataset.save(dummy_tf_base_model)
        assert tf_model_dataset.exists()

    def test_hdf5_save_format(self, dummy_tf_base_model, dummy_x_test, filepath, tensorflow_model_dataset):
        if False:
            while True:
                i = 10
        'Test TensorflowModelDataset can save TF graph models in HDF5 format'
        hdf5_dataset = tensorflow_model_dataset(filepath=filepath, save_args={'save_format': 'h5'})
        predictions = dummy_tf_base_model.predict(dummy_x_test)
        hdf5_dataset.save(dummy_tf_base_model)
        reloaded = hdf5_dataset.load()
        new_predictions = reloaded.predict(dummy_x_test)
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-06, atol=1e-06)

    def test_unused_subclass_model_hdf5_save_format(self, dummy_tf_subclassed_model, dummy_x_train, dummy_y_train, dummy_x_test, filepath, tensorflow_model_dataset):
        if False:
            i = 10
            return i + 15
        "Test TensorflowModelDataset cannot save subclassed user models in HDF5 format\n\n        Subclassed model\n\n        From TF docs\n        First of all, a subclassed model that has never been used cannot be saved.\n        That's because a subclassed model needs to be called on some data in order to\n        create its weights.\n        "
        hdf5_data_set = tensorflow_model_dataset(filepath=filepath, save_args={'save_format': 'h5'})
        dummy_tf_subclassed_model.fit(dummy_x_train, dummy_y_train, batch_size=64, epochs=1)
        dummy_tf_subclassed_model.predict(dummy_x_test)
        pattern = 'Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. It does not work for subclassed models, because such models are defined via the body of a Python method, which isn\\\'t safely serializable. Consider saving to the Tensorflow SavedModel format \\(by setting save_format=\\"tf\\"\\) or using `save_weights`.'
        with pytest.raises(DatasetError, match=pattern):
            hdf5_data_set.save(dummy_tf_subclassed_model)

    @pytest.mark.parametrize('filepath,instance_type', [('s3://bucket/test_tf', S3FileSystem), ('file:///tmp/test_tf', LocalFileSystem), ('/tmp/test_tf', LocalFileSystem), ('gcs://bucket/test_tf', GCSFileSystem), ('https://example.com/test_tf', HTTPFileSystem)])
    def test_protocol_usage(self, filepath, instance_type, tensorflow_model_dataset):
        if False:
            i = 10
            return i + 15
        'Test that can be instantiated with mocked arbitrary file systems.'
        data_set = tensorflow_model_dataset(filepath=filepath)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    @pytest.mark.parametrize('load_args', [{'k1': 'v1', 'compile': False}], indirect=True)
    def test_load_extra_params(self, tf_model_dataset, load_args):
        if False:
            return 10
        'Test overriding the default load arguments.'
        for (key, value) in load_args.items():
            assert tf_model_dataset._load_args[key] == value

    def test_catalog_release(self, mocker, tensorflow_model_dataset):
        if False:
            i = 10
            return i + 15
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.tf'
        data_set = tensorflow_model_dataset(filepath=filepath)
        assert data_set._version_cache.currsize == 0
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)
        assert data_set._version_cache.currsize == 0

    @pytest.mark.parametrize('fs_args', [{'storage_option': 'value'}])
    def test_fs_args(self, fs_args, mocker, tensorflow_model_dataset):
        if False:
            print('Hello World!')
        fs_mock = mocker.patch('fsspec.filesystem')
        tensorflow_model_dataset('test.tf', fs_args=fs_args)
        fs_mock.assert_called_once_with('file', auto_mkdir=True, storage_option='value')

    def test_exists_with_exception(self, tf_model_dataset, mocker):
        if False:
            for i in range(10):
                print('nop')
        'Test `exists` method invocation when `get_filepath_str` raises an exception.'
        mocker.patch('kedro.io.core.get_filepath_str', side_effect=DatasetError)
        assert not tf_model_dataset.exists()

    def test_save_and_overwrite_existing_model(self, tf_model_dataset, dummy_tf_base_model, dummy_tf_base_model_new):
        if False:
            print('Hello World!')
        'Test models are correcty overwritten.'
        tf_model_dataset.save(dummy_tf_base_model)
        tf_model_dataset.save(dummy_tf_base_model_new)
        reloaded = tf_model_dataset.load()
        assert len(dummy_tf_base_model.layers) != len(reloaded.layers)
        assert len(dummy_tf_base_model_new.layers) == len(reloaded.layers)

class TestTensorFlowModelDatasetVersioned:
    """Test suite with versioning argument passed into TensorFlowModelDataset creator"""

    @pytest.mark.parametrize('load_version,save_version', [('2019-01-01T23.59.59.999Z', '2019-01-01T23.59.59.999Z'), (None, None)], indirect=True)
    def test_save_and_load(self, dummy_tf_base_model, versioned_tf_model_dataset, dummy_x_test, load_version, save_version):
        if False:
            for i in range(10):
                print('nop')
        'Test saving and reloading the versioned data set.'
        predictions = dummy_tf_base_model.predict(dummy_x_test)
        versioned_tf_model_dataset.save(dummy_tf_base_model)
        reloaded = versioned_tf_model_dataset.load()
        new_predictions = reloaded.predict(dummy_x_test)
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-06, atol=1e-06)

    def test_hdf5_save_format(self, dummy_tf_base_model, dummy_x_test, filepath, tensorflow_model_dataset, load_version, save_version):
        if False:
            while True:
                i = 10
        'Test versioned TensorflowModelDataset can save TF graph models in\n        HDF5 format'
        hdf5_dataset = tensorflow_model_dataset(filepath=filepath, save_args={'save_format': 'h5'}, version=Version(load_version, save_version))
        predictions = dummy_tf_base_model.predict(dummy_x_test)
        hdf5_dataset.save(dummy_tf_base_model)
        reloaded = hdf5_dataset.load()
        new_predictions = reloaded.predict(dummy_x_test)
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-06, atol=1e-06)

    def test_prevent_overwrite(self, dummy_tf_base_model, versioned_tf_model_dataset):
        if False:
            return 10
        'Check the error when attempting to override the data set if the\n        corresponding file for a given save version already exists.'
        versioned_tf_model_dataset.save(dummy_tf_base_model)
        pattern = "Save path \\'.+\\' for TensorFlowModelDataset\\(.+\\) must not exist if versioning is enabled\\."
        with pytest.raises(DatasetError, match=pattern):
            versioned_tf_model_dataset.save(dummy_tf_base_model)

    @pytest.mark.parametrize('load_version,save_version', [('2019-01-01T23.59.59.999Z', '2019-01-02T00.00.00.000Z')], indirect=True)
    def test_save_version_warning(self, versioned_tf_model_dataset, load_version, save_version, dummy_tf_base_model):
        if False:
            print('Hello World!')
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for TensorFlowModelDataset\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            versioned_tf_model_dataset.save(dummy_tf_base_model)

    def test_http_filesystem_no_versioning(self, tensorflow_model_dataset):
        if False:
            return 10
        pattern = 'Versioning is not supported for HTTP protocols.'
        with pytest.raises(DatasetError, match=pattern):
            tensorflow_model_dataset(filepath='https://example.com/file.tf', version=Version(None, None))

    def test_exists(self, versioned_tf_model_dataset, dummy_tf_base_model):
        if False:
            while True:
                i = 10
        'Test `exists` method invocation for versioned data set.'
        assert not versioned_tf_model_dataset.exists()
        versioned_tf_model_dataset.save(dummy_tf_base_model)
        assert versioned_tf_model_dataset.exists()

    def test_no_versions(self, versioned_tf_model_dataset):
        if False:
            for i in range(10):
                print('nop')
        'Check the error if no versions are available for load.'
        pattern = 'Did not find any versions for TensorFlowModelDataset\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            versioned_tf_model_dataset.load()

    def test_version_str_repr(self, tf_model_dataset, versioned_tf_model_dataset):
        if False:
            i = 10
            return i + 15
        'Test that version is in string representation of the class instance\n        when applicable.'
        assert str(tf_model_dataset._filepath) in str(tf_model_dataset)
        assert 'version=' not in str(tf_model_dataset)
        assert 'protocol' in str(tf_model_dataset)
        assert 'save_args' in str(tf_model_dataset)
        assert str(versioned_tf_model_dataset._filepath) in str(versioned_tf_model_dataset)
        ver_str = f'version={versioned_tf_model_dataset._version}'
        assert ver_str in str(versioned_tf_model_dataset)
        assert 'protocol' in str(versioned_tf_model_dataset)
        assert 'save_args' in str(versioned_tf_model_dataset)

    def test_versioning_existing_dataset(self, tf_model_dataset, versioned_tf_model_dataset, dummy_tf_base_model):
        if False:
            print('Hello World!')
        'Check behavior when attempting to save a versioned dataset on top of an\n        already existing (non-versioned) dataset. Note: because TensorFlowModelDataset\n        saves to a directory even if non-versioned, an error is not expected.'
        tf_model_dataset.save(dummy_tf_base_model)
        assert tf_model_dataset.exists()
        assert tf_model_dataset._filepath == versioned_tf_model_dataset._filepath
        versioned_tf_model_dataset.save(dummy_tf_base_model)
        assert versioned_tf_model_dataset.exists()

    def test_save_and_load_with_device(self, dummy_tf_base_model, dummy_x_test, filepath, tensorflow_model_dataset, load_version, save_version):
        if False:
            for i in range(10):
                print('nop')
        'Test versioned TensorflowModelDataset can load models using an explicit tf_device'
        hdf5_dataset = tensorflow_model_dataset(filepath=filepath, load_args={'tf_device': '/CPU:0'}, version=Version(load_version, save_version))
        predictions = dummy_tf_base_model.predict(dummy_x_test)
        hdf5_dataset.save(dummy_tf_base_model)
        reloaded = hdf5_dataset.load()
        new_predictions = reloaded.predict(dummy_x_test)
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-06, atol=1e-06)