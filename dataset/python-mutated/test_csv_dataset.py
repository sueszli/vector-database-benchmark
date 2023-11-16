from pathlib import Path, PurePosixPath
from time import sleep
import pandas as pd
import pytest
from adlfs import AzureBlobFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from pandas.testing import assert_frame_equal
from s3fs.core import S3FileSystem
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER, Version, generate_timestamp

@pytest.fixture
def filepath_csv(tmp_path):
    if False:
        print('Hello World!')
    return (tmp_path / 'test.csv').as_posix()

@pytest.fixture
def csv_data_set(filepath_csv, load_args, save_args, fs_args):
    if False:
        while True:
            i = 10
    return CSVDataSet(filepath=filepath_csv, load_args=load_args, save_args=save_args, fs_args=fs_args)

@pytest.fixture
def versioned_csv_data_set(filepath_csv, load_version, save_version):
    if False:
        i = 10
        return i + 15
    return CSVDataSet(filepath=filepath_csv, version=Version(load_version, save_version))

@pytest.fixture
def dummy_dataframe():
    if False:
        return 10
    return pd.DataFrame({'col1': [1, 2], 'col2': [4, 5], 'col3': [5, 6]})

class TestCSVDataSet:

    def test_save_and_load(self, csv_data_set, dummy_dataframe):
        if False:
            print('Hello World!')
        'Test saving and reloading the data set.'
        csv_data_set.save(dummy_dataframe)
        reloaded = csv_data_set.load()
        assert_frame_equal(dummy_dataframe, reloaded)

    def test_exists(self, csv_data_set, dummy_dataframe):
        if False:
            while True:
                i = 10
        'Test `exists` method invocation for both existing and\n        nonexistent data set.'
        assert not csv_data_set.exists()
        csv_data_set.save(dummy_dataframe)
        assert csv_data_set.exists()

    @pytest.mark.parametrize('load_args', [{'k1': 'v1', 'index': 'value'}], indirect=True)
    def test_load_extra_params(self, csv_data_set, load_args):
        if False:
            print('Hello World!')
        'Test overriding the default load arguments.'
        for (key, value) in load_args.items():
            assert csv_data_set._load_args[key] == value

    @pytest.mark.parametrize('save_args', [{'k1': 'v1', 'index': 'value'}], indirect=True)
    def test_save_extra_params(self, csv_data_set, save_args):
        if False:
            i = 10
            return i + 15
        'Test overriding the default save arguments.'
        for (key, value) in save_args.items():
            assert csv_data_set._save_args[key] == value

    @pytest.mark.parametrize('load_args,save_args', [({'storage_options': {'a': 'b'}}, {}), ({}, {'storage_options': {'a': 'b'}}), ({'storage_options': {'a': 'b'}}, {'storage_options': {'x': 'y'}})])
    def test_storage_options_dropped(self, load_args, save_args, caplog, tmp_path):
        if False:
            return 10
        filepath = str(tmp_path / 'test.csv')
        ds = CSVDataSet(filepath=filepath, load_args=load_args, save_args=save_args)
        records = [r for r in caplog.records if r.levelname == 'WARNING']
        expected_log_message = f"Dropping 'storage_options' for {filepath}, please specify them under 'fs_args' or 'credentials'."
        assert records[0].getMessage() == expected_log_message
        assert 'storage_options' not in ds._save_args
        assert 'storage_options' not in ds._load_args

    def test_load_missing_file(self, csv_data_set):
        if False:
            print('Hello World!')
        'Check the error when trying to load missing file.'
        pattern = 'Failed while loading data from data set CSVDataSet\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            csv_data_set.load()

    @pytest.mark.parametrize('filepath,instance_type,credentials', [('s3://bucket/file.csv', S3FileSystem, {}), ('file:///tmp/test.csv', LocalFileSystem, {}), ('/tmp/test.csv', LocalFileSystem, {}), ('gcs://bucket/file.csv', GCSFileSystem, {}), ('https://example.com/file.csv', HTTPFileSystem, {}), ('abfs://bucket/file.csv', AzureBlobFileSystem, {'account_name': 'test', 'account_key': 'test'})])
    def test_protocol_usage(self, filepath, instance_type, credentials):
        if False:
            for i in range(10):
                print('nop')
        data_set = CSVDataSet(filepath=filepath, credentials=credentials)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    def test_catalog_release(self, mocker):
        if False:
            print('Hello World!')
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.csv'
        data_set = CSVDataSet(filepath=filepath)
        assert data_set._version_cache.currsize == 0
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)
        assert data_set._version_cache.currsize == 0

class TestCSVDataSetVersioned:

    def test_version_str_repr(self, load_version, save_version):
        if False:
            print('Hello World!')
        'Test that version is in string representation of the class instance\n        when applicable.'
        filepath = 'test.csv'
        ds = CSVDataSet(filepath=filepath)
        ds_versioned = CSVDataSet(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds)
        assert 'version' not in str(ds)
        assert filepath in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert 'CSVDataSet' in str(ds_versioned)
        assert 'CSVDataSet' in str(ds)
        assert 'protocol' in str(ds_versioned)
        assert 'protocol' in str(ds)
        assert "save_args={'index': False}" in str(ds)
        assert "save_args={'index': False}" in str(ds_versioned)

    def test_save_and_load(self, versioned_csv_data_set, dummy_dataframe):
        if False:
            for i in range(10):
                print('nop')
        'Test that saved and reloaded data matches the original one for\n        the versioned data set.'
        versioned_csv_data_set.save(dummy_dataframe)
        reloaded_df = versioned_csv_data_set.load()
        assert_frame_equal(dummy_dataframe, reloaded_df)

    def test_multiple_loads(self, versioned_csv_data_set, dummy_dataframe, filepath_csv):
        if False:
            for i in range(10):
                print('nop')
        "Test that if a new version is created mid-run, by an\n        external system, it won't be loaded in the current run."
        versioned_csv_data_set.save(dummy_dataframe)
        versioned_csv_data_set.load()
        v1 = versioned_csv_data_set.resolve_load_version()
        sleep(0.5)
        v_new = generate_timestamp()
        CSVDataSet(filepath=filepath_csv, version=Version(v_new, v_new)).save(dummy_dataframe)
        versioned_csv_data_set.load()
        v2 = versioned_csv_data_set.resolve_load_version()
        assert v2 == v1
        ds_new = CSVDataSet(filepath=filepath_csv, version=Version(None, None))
        assert ds_new.resolve_load_version() == v_new

    def test_multiple_saves(self, dummy_dataframe, filepath_csv):
        if False:
            print('Hello World!')
        'Test multiple cycles of save followed by load for the same dataset'
        ds_versioned = CSVDataSet(filepath=filepath_csv, version=Version(None, None))
        ds_versioned.save(dummy_dataframe)
        first_save_version = ds_versioned.resolve_save_version()
        first_load_version = ds_versioned.resolve_load_version()
        assert first_load_version == first_save_version
        sleep(0.5)
        ds_versioned.save(dummy_dataframe)
        second_save_version = ds_versioned.resolve_save_version()
        second_load_version = ds_versioned.resolve_load_version()
        assert second_load_version == second_save_version
        assert second_load_version > first_load_version
        ds_new = CSVDataSet(filepath=filepath_csv, version=Version(None, None))
        assert ds_new.resolve_load_version() == second_load_version

    def test_release_instance_cache(self, dummy_dataframe, filepath_csv):
        if False:
            return 10
        'Test that cache invalidation does not affect other instances'
        ds_a = CSVDataSet(filepath=filepath_csv, version=Version(None, None))
        assert ds_a._version_cache.currsize == 0
        ds_a.save(dummy_dataframe)
        assert ds_a._version_cache.currsize == 2
        ds_b = CSVDataSet(filepath=filepath_csv, version=Version(None, None))
        assert ds_b._version_cache.currsize == 0
        ds_b.resolve_save_version()
        assert ds_b._version_cache.currsize == 1
        ds_b.resolve_load_version()
        assert ds_b._version_cache.currsize == 2
        ds_a.release()
        assert ds_a._version_cache.currsize == 0
        assert ds_b._version_cache.currsize == 2

    def test_no_versions(self, versioned_csv_data_set):
        if False:
            print('Hello World!')
        'Check the error if no versions are available for load.'
        pattern = 'Did not find any versions for CSVDataSet\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            versioned_csv_data_set.load()

    def test_exists(self, versioned_csv_data_set, dummy_dataframe):
        if False:
            while True:
                i = 10
        'Test `exists` method invocation for versioned data set.'
        assert not versioned_csv_data_set.exists()
        versioned_csv_data_set.save(dummy_dataframe)
        assert versioned_csv_data_set.exists()

    def test_prevent_overwrite(self, versioned_csv_data_set, dummy_dataframe):
        if False:
            i = 10
            return i + 15
        'Check the error when attempting to override the data set if the\n        corresponding CSV file for a given save version already exists.'
        versioned_csv_data_set.save(dummy_dataframe)
        pattern = "Save path \\'.+\\' for CSVDataSet\\(.+\\) must not exist if versioning is enabled\\."
        with pytest.raises(DatasetError, match=pattern):
            versioned_csv_data_set.save(dummy_dataframe)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, versioned_csv_data_set, load_version, save_version, dummy_dataframe):
        if False:
            print('Hello World!')
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for CSVDataSet\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            versioned_csv_data_set.save(dummy_dataframe)

    def test_http_filesystem_no_versioning(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = 'Versioning is not supported for HTTP protocols.'
        with pytest.raises(DatasetError, match=pattern):
            CSVDataSet(filepath='https://example.com/file.csv', version=Version(None, None))

    def test_versioning_existing_dataset(self, csv_data_set, versioned_csv_data_set, dummy_dataframe):
        if False:
            print('Hello World!')
        'Check the error when attempting to save a versioned dataset on top of an\n        already existing (non-versioned) dataset.'
        csv_data_set.save(dummy_dataframe)
        assert csv_data_set.exists()
        assert csv_data_set._filepath == versioned_csv_data_set._filepath
        pattern = f'(?=.*file with the same name already exists in the directory)(?=.*{versioned_csv_data_set._filepath.parent.as_posix()})'
        with pytest.raises(DatasetError, match=pattern):
            versioned_csv_data_set.save(dummy_dataframe)
        Path(csv_data_set._filepath.as_posix()).unlink()
        versioned_csv_data_set.save(dummy_dataframe)
        assert versioned_csv_data_set.exists()