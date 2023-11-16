from pathlib import Path, PurePosixPath
import pandas as pd
import pytest
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from pandas.testing import assert_frame_equal
from s3fs.core import S3FileSystem
from kedro.extras.datasets.yaml import YAMLDataSet
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER, Version

@pytest.fixture
def filepath_yaml(tmp_path):
    if False:
        print('Hello World!')
    return (tmp_path / 'test.yaml').as_posix()

@pytest.fixture
def yaml_data_set(filepath_yaml, save_args, fs_args):
    if False:
        return 10
    return YAMLDataSet(filepath=filepath_yaml, save_args=save_args, fs_args=fs_args)

@pytest.fixture
def versioned_yaml_data_set(filepath_yaml, load_version, save_version):
    if False:
        return 10
    return YAMLDataSet(filepath=filepath_yaml, version=Version(load_version, save_version))

@pytest.fixture
def dummy_data():
    if False:
        print('Hello World!')
    return {'col1': 1, 'col2': 2, 'col3': 3}

class TestYAMLDataSet:

    def test_save_and_load(self, yaml_data_set, dummy_data):
        if False:
            for i in range(10):
                print('nop')
        'Test saving and reloading the data set.'
        yaml_data_set.save(dummy_data)
        reloaded = yaml_data_set.load()
        assert dummy_data == reloaded
        assert yaml_data_set._fs_open_args_load == {}
        assert yaml_data_set._fs_open_args_save == {'mode': 'w'}

    def test_exists(self, yaml_data_set, dummy_data):
        if False:
            return 10
        'Test `exists` method invocation for both existing and\n        nonexistent data set.'
        assert not yaml_data_set.exists()
        yaml_data_set.save(dummy_data)
        assert yaml_data_set.exists()

    @pytest.mark.parametrize('save_args', [{'k1': 'v1', 'index': 'value'}], indirect=True)
    def test_save_extra_params(self, yaml_data_set, save_args):
        if False:
            while True:
                i = 10
        'Test overriding the default save arguments.'
        for (key, value) in save_args.items():
            assert yaml_data_set._save_args[key] == value

    @pytest.mark.parametrize('fs_args', [{'open_args_load': {'mode': 'rb', 'compression': 'gzip'}}], indirect=True)
    def test_open_extra_args(self, yaml_data_set, fs_args):
        if False:
            return 10
        assert yaml_data_set._fs_open_args_load == fs_args['open_args_load']
        assert yaml_data_set._fs_open_args_save == {'mode': 'w'}

    def test_load_missing_file(self, yaml_data_set):
        if False:
            i = 10
            return i + 15
        'Check the error when trying to load missing file.'
        pattern = 'Failed while loading data from data set YAMLDataSet\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            yaml_data_set.load()

    @pytest.mark.parametrize('filepath,instance_type', [('s3://bucket/file.yaml', S3FileSystem), ('file:///tmp/test.yaml', LocalFileSystem), ('/tmp/test.yaml', LocalFileSystem), ('gcs://bucket/file.yaml', GCSFileSystem), ('https://example.com/file.yaml', HTTPFileSystem)])
    def test_protocol_usage(self, filepath, instance_type):
        if False:
            i = 10
            return i + 15
        data_set = YAMLDataSet(filepath=filepath)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    def test_catalog_release(self, mocker):
        if False:
            while True:
                i = 10
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.yaml'
        data_set = YAMLDataSet(filepath=filepath)
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

    def test_dataframe_support(self, yaml_data_set):
        if False:
            i = 10
            return i + 15
        data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5]})
        yaml_data_set.save(data.to_dict())
        reloaded = yaml_data_set.load()
        assert isinstance(reloaded, dict)
        data_df = pd.DataFrame.from_dict(reloaded)
        assert_frame_equal(data, data_df)

class TestYAMLDataSetVersioned:

    def test_version_str_repr(self, load_version, save_version):
        if False:
            return 10
        'Test that version is in string representation of the class instance\n        when applicable.'
        filepath = 'test.yaml'
        ds = YAMLDataSet(filepath=filepath)
        ds_versioned = YAMLDataSet(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds)
        assert 'version' not in str(ds)
        assert filepath in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert 'YAMLDataSet' in str(ds_versioned)
        assert 'YAMLDataSet' in str(ds)
        assert 'protocol' in str(ds_versioned)
        assert 'protocol' in str(ds)
        assert "save_args={'default_flow_style': False}" in str(ds)
        assert "save_args={'default_flow_style': False}" in str(ds_versioned)

    def test_save_and_load(self, versioned_yaml_data_set, dummy_data):
        if False:
            return 10
        'Test that saved and reloaded data matches the original one for\n        the versioned data set.'
        versioned_yaml_data_set.save(dummy_data)
        reloaded = versioned_yaml_data_set.load()
        assert dummy_data == reloaded

    def test_no_versions(self, versioned_yaml_data_set):
        if False:
            return 10
        'Check the error if no versions are available for load.'
        pattern = 'Did not find any versions for YAMLDataSet\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            versioned_yaml_data_set.load()

    def test_exists(self, versioned_yaml_data_set, dummy_data):
        if False:
            i = 10
            return i + 15
        'Test `exists` method invocation for versioned data set.'
        assert not versioned_yaml_data_set.exists()
        versioned_yaml_data_set.save(dummy_data)
        assert versioned_yaml_data_set.exists()

    def test_prevent_overwrite(self, versioned_yaml_data_set, dummy_data):
        if False:
            for i in range(10):
                print('nop')
        'Check the error when attempting to override the data set if the\n        corresponding yaml file for a given save version already exists.'
        versioned_yaml_data_set.save(dummy_data)
        pattern = "Save path \\'.+\\' for YAMLDataSet\\(.+\\) must not exist if versioning is enabled\\."
        with pytest.raises(DatasetError, match=pattern):
            versioned_yaml_data_set.save(dummy_data)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, versioned_yaml_data_set, load_version, save_version, dummy_data):
        if False:
            print('Hello World!')
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for YAMLDataSet\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            versioned_yaml_data_set.save(dummy_data)

    def test_http_filesystem_no_versioning(self):
        if False:
            print('Hello World!')
        pattern = 'Versioning is not supported for HTTP protocols.'
        with pytest.raises(DatasetError, match=pattern):
            YAMLDataSet(filepath='https://example.com/file.yaml', version=Version(None, None))

    def test_versioning_existing_dataset(self, yaml_data_set, versioned_yaml_data_set, dummy_data):
        if False:
            for i in range(10):
                print('nop')
        'Check the error when attempting to save a versioned dataset on top of an\n        already existing (non-versioned) dataset.'
        yaml_data_set.save(dummy_data)
        assert yaml_data_set.exists()
        assert yaml_data_set._filepath == versioned_yaml_data_set._filepath
        pattern = f'(?=.*file with the same name already exists in the directory)(?=.*{versioned_yaml_data_set._filepath.parent.as_posix()})'
        with pytest.raises(DatasetError, match=pattern):
            versioned_yaml_data_set.save(dummy_data)
        Path(yaml_data_set._filepath.as_posix()).unlink()
        versioned_yaml_data_set.save(dummy_data)
        assert versioned_yaml_data_set.exists()