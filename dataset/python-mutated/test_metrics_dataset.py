import json
from pathlib import Path, PurePosixPath
import pytest
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs.core import S3FileSystem
from kedro.extras.datasets.tracking import MetricsDataSet
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER, Version

@pytest.fixture
def filepath_json(tmp_path):
    if False:
        print('Hello World!')
    return (tmp_path / 'test.json').as_posix()

@pytest.fixture
def metrics_dataset(filepath_json, save_args, fs_args):
    if False:
        for i in range(10):
            print('nop')
    return MetricsDataSet(filepath=filepath_json, save_args=save_args, fs_args=fs_args)

@pytest.fixture
def explicit_versioned_metrics_dataset(filepath_json, load_version, save_version):
    if False:
        print('Hello World!')
    return MetricsDataSet(filepath=filepath_json, version=Version(load_version, save_version))

@pytest.fixture
def dummy_data():
    if False:
        return 10
    return {'col1': 1, 'col2': 2, 'col3': 3}

class TestMetricsDataSet:

    def test_save_data(self, dummy_data, tmp_path, filepath_json, save_version):
        if False:
            print('Hello World!')
        'Test saving and reloading the data set.'
        metrics_dataset = MetricsDataSet(filepath=filepath_json, version=Version(None, save_version))
        metrics_dataset.save(dummy_data)
        actual_filepath = Path(metrics_dataset._filepath.as_posix())
        test_filepath = tmp_path / 'locally_saved.json'
        test_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(test_filepath, 'w', encoding='utf-8') as file:
            json.dump(dummy_data, file)
        with open(test_filepath, encoding='utf-8') as file:
            test_data = json.load(file)
        with open(actual_filepath / save_version / 'test.json', encoding='utf-8') as actual_file:
            actual_data = json.load(actual_file)
        assert actual_data == test_data
        assert metrics_dataset._fs_open_args_load == {}
        assert metrics_dataset._fs_open_args_save == {'mode': 'w'}

    def test_load_fail(self, metrics_dataset, dummy_data):
        if False:
            i = 10
            return i + 15
        metrics_dataset.save(dummy_data)
        pattern = "Loading not supported for 'MetricsDataSet'"
        with pytest.raises(DatasetError, match=pattern):
            metrics_dataset.load()

    def test_exists(self, metrics_dataset, dummy_data):
        if False:
            i = 10
            return i + 15
        'Test `exists` method invocation for both existing and\n        nonexistent data set.'
        assert not metrics_dataset.exists()
        metrics_dataset.save(dummy_data)
        assert metrics_dataset.exists()

    @pytest.mark.parametrize('save_args', [{'k1': 'v1', 'index': 'value'}], indirect=True)
    def test_save_extra_params(self, metrics_dataset, save_args):
        if False:
            i = 10
            return i + 15
        'Test overriding the default save arguments.'
        for (key, value) in save_args.items():
            assert metrics_dataset._save_args[key] == value

    @pytest.mark.parametrize('fs_args', [{'open_args_load': {'mode': 'rb', 'compression': 'gzip'}}], indirect=True)
    def test_open_extra_args(self, metrics_dataset, fs_args):
        if False:
            return 10
        assert metrics_dataset._fs_open_args_load == fs_args['open_args_load']
        assert metrics_dataset._fs_open_args_save == {'mode': 'w'}

    @pytest.mark.parametrize('filepath,instance_type', [('s3://bucket/file.json', S3FileSystem), ('file:///tmp/test.json', LocalFileSystem), ('/tmp/test.json', LocalFileSystem), ('gcs://bucket/file.json', GCSFileSystem)])
    def test_protocol_usage(self, filepath, instance_type):
        if False:
            i = 10
            return i + 15
        data_set = MetricsDataSet(filepath=filepath)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    def test_catalog_release(self, mocker):
        if False:
            i = 10
            return i + 15
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.json'
        data_set = MetricsDataSet(filepath=filepath)
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

    def test_fail_on_saving_non_numeric_value(self, metrics_dataset):
        if False:
            return 10
        data = {'col1': 1, 'col2': 2, 'col3': 'hello'}
        pattern = 'The MetricsDataSet expects only numeric values.'
        with pytest.raises(DatasetError, match=pattern):
            metrics_dataset.save(data)

    def test_not_version_str_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that version is not in string representation of the class instance.'
        filepath = 'test.json'
        ds = MetricsDataSet(filepath=filepath)
        assert filepath in str(ds)
        assert 'version' not in str(ds)
        assert 'MetricsDataSet' in str(ds)
        assert 'protocol' in str(ds)
        assert "save_args={'indent': 2}" in str(ds)

    def test_version_str_repr(self, load_version, save_version):
        if False:
            print('Hello World!')
        'Test that version is in string representation of the class instance.'
        filepath = 'test.json'
        ds_versioned = MetricsDataSet(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert 'MetricsDataSet' in str(ds_versioned)
        assert 'protocol' in str(ds_versioned)
        assert "save_args={'indent': 2}" in str(ds_versioned)

    def test_prevent_overwrite(self, explicit_versioned_metrics_dataset, dummy_data):
        if False:
            return 10
        'Check the error when attempting to override the data set if the\n        corresponding json file for a given save version already exists.'
        explicit_versioned_metrics_dataset.save(dummy_data)
        pattern = "Save path \\'.+\\' for MetricsDataSet\\(.+\\) must not exist if versioning is enabled\\."
        with pytest.raises(DatasetError, match=pattern):
            explicit_versioned_metrics_dataset.save(dummy_data)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, explicit_versioned_metrics_dataset, load_version, save_version, dummy_data):
        if False:
            i = 10
            return i + 15
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for MetricsDataSet\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            explicit_versioned_metrics_dataset.save(dummy_data)

    def test_http_filesystem_no_versioning(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = 'Versioning is not supported for HTTP protocols.'
        with pytest.raises(DatasetError, match=pattern):
            MetricsDataSet(filepath='https://example.com/file.json', version=Version(None, None))