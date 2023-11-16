from pathlib import Path, PurePosixPath
import pandas as pd
import pytest
from adlfs import AzureBlobFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from pandas.testing import assert_frame_equal
from s3fs.core import S3FileSystem
from kedro.extras.datasets.pandas import JSONDataSet
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER, Version

@pytest.fixture
def filepath_json(tmp_path):
    if False:
        while True:
            i = 10
    return (tmp_path / 'test.json').as_posix()

@pytest.fixture
def json_data_set(filepath_json, load_args, save_args, fs_args):
    if False:
        return 10
    return JSONDataSet(filepath=filepath_json, load_args=load_args, save_args=save_args, fs_args=fs_args)

@pytest.fixture
def versioned_json_data_set(filepath_json, load_version, save_version):
    if False:
        return 10
    return JSONDataSet(filepath=filepath_json, version=Version(load_version, save_version))

@pytest.fixture
def dummy_dataframe():
    if False:
        return 10
    return pd.DataFrame({'col1': [1, 2], 'col2': [4, 5], 'col3': [5, 6]})

class TestJSONDataSet:

    def test_save_and_load(self, json_data_set, dummy_dataframe):
        if False:
            while True:
                i = 10
        'Test saving and reloading the data set.'
        json_data_set.save(dummy_dataframe)
        reloaded = json_data_set.load()
        assert_frame_equal(dummy_dataframe, reloaded)

    def test_exists(self, json_data_set, dummy_dataframe):
        if False:
            for i in range(10):
                print('nop')
        'Test `exists` method invocation for both existing and\n        nonexistent data set.'
        assert not json_data_set.exists()
        json_data_set.save(dummy_dataframe)
        assert json_data_set.exists()

    @pytest.mark.parametrize('load_args', [{'k1': 'v1', 'index': 'value'}], indirect=True)
    def test_load_extra_params(self, json_data_set, load_args):
        if False:
            for i in range(10):
                print('nop')
        'Test overriding the default load arguments.'
        for (key, value) in load_args.items():
            assert json_data_set._load_args[key] == value

    @pytest.mark.parametrize('save_args', [{'k1': 'v1', 'index': 'value'}], indirect=True)
    def test_save_extra_params(self, json_data_set, save_args):
        if False:
            i = 10
            return i + 15
        'Test overriding the default save arguments.'
        for (key, value) in save_args.items():
            assert json_data_set._save_args[key] == value

    @pytest.mark.parametrize('load_args,save_args', [({'storage_options': {'a': 'b'}}, {}), ({}, {'storage_options': {'a': 'b'}}), ({'storage_options': {'a': 'b'}}, {'storage_options': {'x': 'y'}})])
    def test_storage_options_dropped(self, load_args, save_args, caplog, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        filepath = str(tmp_path / 'test.csv')
        ds = JSONDataSet(filepath=filepath, load_args=load_args, save_args=save_args)
        records = [r for r in caplog.records if r.levelname == 'WARNING']
        expected_log_message = f"Dropping 'storage_options' for {filepath}, please specify them under 'fs_args' or 'credentials'."
        assert records[0].getMessage() == expected_log_message
        assert 'storage_options' not in ds._save_args
        assert 'storage_options' not in ds._load_args

    def test_load_missing_file(self, json_data_set):
        if False:
            i = 10
            return i + 15
        'Check the error when trying to load missing file.'
        pattern = 'Failed while loading data from data set JSONDataSet\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            json_data_set.load()

    @pytest.mark.parametrize('filepath,instance_type,credentials,load_path', [('s3://bucket/file.json', S3FileSystem, {}, 's3://bucket/file.json'), ('file:///tmp/test.json', LocalFileSystem, {}, '/tmp/test.json'), ('/tmp/test.json', LocalFileSystem, {}, '/tmp/test.json'), ('gcs://bucket/file.json', GCSFileSystem, {}, 'gcs://bucket/file.json'), ('https://example.com/file.json', HTTPFileSystem, {}, 'https://example.com/file.json'), ('abfs://bucket/file.csv', AzureBlobFileSystem, {'account_name': 'test', 'account_key': 'test'}, 'abfs://bucket/file.csv')])
    def test_protocol_usage(self, filepath, instance_type, credentials, load_path, mocker):
        if False:
            for i in range(10):
                print('nop')
        data_set = JSONDataSet(filepath=filepath, credentials=credentials)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)
        mock_pandas_call = mocker.patch('pandas.read_json')
        data_set.load()
        assert mock_pandas_call.call_count == 1
        assert mock_pandas_call.call_args_list[0][0][0] == load_path

    def test_catalog_release(self, mocker):
        if False:
            return 10
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.json'
        data_set = JSONDataSet(filepath=filepath)
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

class TestJSONDataSetVersioned:

    def test_version_str_repr(self, load_version, save_version):
        if False:
            i = 10
            return i + 15
        'Test that version is in string representation of the class instance\n        when applicable.'
        filepath = 'test.json'
        ds = JSONDataSet(filepath=filepath)
        ds_versioned = JSONDataSet(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds)
        assert 'version' not in str(ds)
        assert filepath in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert 'JSONDataSet' in str(ds_versioned)
        assert 'JSONDataSet' in str(ds)
        assert 'protocol' in str(ds_versioned)
        assert 'protocol' in str(ds)

    def test_save_and_load(self, versioned_json_data_set, dummy_dataframe):
        if False:
            print('Hello World!')
        'Test that saved and reloaded data matches the original one for\n        the versioned data set.'
        versioned_json_data_set.save(dummy_dataframe)
        reloaded_df = versioned_json_data_set.load()
        assert_frame_equal(dummy_dataframe, reloaded_df)

    def test_no_versions(self, versioned_json_data_set):
        if False:
            return 10
        'Check the error if no versions are available for load.'
        pattern = 'Did not find any versions for JSONDataSet\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            versioned_json_data_set.load()

    def test_exists(self, versioned_json_data_set, dummy_dataframe):
        if False:
            i = 10
            return i + 15
        'Test `exists` method invocation for versioned data set.'
        assert not versioned_json_data_set.exists()
        versioned_json_data_set.save(dummy_dataframe)
        assert versioned_json_data_set.exists()

    def test_prevent_overwrite(self, versioned_json_data_set, dummy_dataframe):
        if False:
            print('Hello World!')
        'Check the error when attempting to override the data set if the\n        corresponding hdf file for a given save version already exists.'
        versioned_json_data_set.save(dummy_dataframe)
        pattern = "Save path \\'.+\\' for JSONDataSet\\(.+\\) must not exist if versioning is enabled\\."
        with pytest.raises(DatasetError, match=pattern):
            versioned_json_data_set.save(dummy_dataframe)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, versioned_json_data_set, load_version, save_version, dummy_dataframe):
        if False:
            while True:
                i = 10
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for JSONDataSet\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            versioned_json_data_set.save(dummy_dataframe)

    def test_http_filesystem_no_versioning(self):
        if False:
            return 10
        pattern = 'Versioning is not supported for HTTP protocols.'
        with pytest.raises(DatasetError, match=pattern):
            JSONDataSet(filepath='https://example.com/file.json', version=Version(None, None))

    def test_versioning_existing_dataset(self, json_data_set, versioned_json_data_set, dummy_dataframe):
        if False:
            for i in range(10):
                print('nop')
        'Check the error when attempting to save a versioned dataset on top of an\n        already existing (non-versioned) dataset.'
        json_data_set.save(dummy_dataframe)
        assert json_data_set.exists()
        assert json_data_set._filepath == versioned_json_data_set._filepath
        pattern = f'(?=.*file with the same name already exists in the directory)(?=.*{versioned_json_data_set._filepath.parent.as_posix()})'
        with pytest.raises(DatasetError, match=pattern):
            versioned_json_data_set.save(dummy_dataframe)
        Path(json_data_set._filepath.as_posix()).unlink()
        versioned_json_data_set.save(dummy_dataframe)
        assert versioned_json_data_set.exists()