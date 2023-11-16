from pathlib import Path, PurePosixPath
import numpy as np
import pytest
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs.core import S3FileSystem
from kedro.extras.datasets.svmlight import SVMLightDataSet
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER, Version

@pytest.fixture
def filepath_svm(tmp_path):
    if False:
        return 10
    return (tmp_path / 'test.svm').as_posix()

@pytest.fixture
def svm_data_set(filepath_svm, save_args, load_args, fs_args):
    if False:
        i = 10
        return i + 15
    return SVMLightDataSet(filepath=filepath_svm, save_args=save_args, load_args=load_args, fs_args=fs_args)

@pytest.fixture
def versioned_svm_data_set(filepath_svm, load_version, save_version):
    if False:
        i = 10
        return i + 15
    return SVMLightDataSet(filepath=filepath_svm, version=Version(load_version, save_version))

@pytest.fixture
def dummy_data():
    if False:
        return 10
    features = np.array([[1, 2, 10], [1, 0.4, 3.2], [0, 0, 0]])
    label = np.array([1, 0, 3])
    return (features, label)

class TestSVMLightDataSet:

    def test_save_and_load(self, svm_data_set, dummy_data):
        if False:
            for i in range(10):
                print('nop')
        'Test saving and reloading the data set.'
        svm_data_set.save(dummy_data)
        (reloaded_features, reloaded_label) = svm_data_set.load()
        (original_features, original_label) = dummy_data
        assert (original_features == reloaded_features).all()
        assert (original_label == reloaded_label).all()
        assert svm_data_set._fs_open_args_load == {'mode': 'rb'}
        assert svm_data_set._fs_open_args_save == {'mode': 'wb'}

    def test_exists(self, svm_data_set, dummy_data):
        if False:
            while True:
                i = 10
        'Test `exists` method invocation for both existing and\n        nonexistent data set.'
        assert not svm_data_set.exists()
        svm_data_set.save(dummy_data)
        assert svm_data_set.exists()

    @pytest.mark.parametrize('save_args', [{'zero_based': False, 'comment': 'comment'}], indirect=True)
    def test_save_extra_save_args(self, svm_data_set, save_args):
        if False:
            i = 10
            return i + 15
        'Test overriding the default save arguments.'
        for (key, value) in save_args.items():
            assert svm_data_set._save_args[key] == value

    @pytest.mark.parametrize('load_args', [{'zero_based': False, 'n_features': 3}], indirect=True)
    def test_save_extra_load_args(self, svm_data_set, load_args):
        if False:
            for i in range(10):
                print('nop')
        'Test overriding the default load arguments.'
        for (key, value) in load_args.items():
            assert svm_data_set._load_args[key] == value

    @pytest.mark.parametrize('fs_args', [{'open_args_load': {'mode': 'rb', 'compression': 'gzip'}}], indirect=True)
    def test_open_extra_args(self, svm_data_set, fs_args):
        if False:
            for i in range(10):
                print('nop')
        assert svm_data_set._fs_open_args_load == fs_args['open_args_load']
        assert svm_data_set._fs_open_args_save == {'mode': 'wb'}

    def test_load_missing_file(self, svm_data_set):
        if False:
            while True:
                i = 10
        'Check the error when trying to load missing file.'
        pattern = 'Failed while loading data from data set SVMLightDataSet\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            svm_data_set.load()

    @pytest.mark.parametrize('filepath,instance_type', [('s3://bucket/file.svm', S3FileSystem), ('file:///tmp/test.svm', LocalFileSystem), ('/tmp/test.svm', LocalFileSystem), ('gcs://bucket/file.svm', GCSFileSystem), ('https://example.com/file.svm', HTTPFileSystem)])
    def test_protocol_usage(self, filepath, instance_type):
        if False:
            i = 10
            return i + 15
        data_set = SVMLightDataSet(filepath=filepath)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    def test_catalog_release(self, mocker):
        if False:
            return 10
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.svm'
        data_set = SVMLightDataSet(filepath=filepath)
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

class TestSVMLightDataSetVersioned:

    def test_version_str_repr(self, load_version, save_version):
        if False:
            while True:
                i = 10
        'Test that version is in string representation of the class instance\n        when applicable.'
        filepath = 'test.svm'
        ds = SVMLightDataSet(filepath=filepath)
        ds_versioned = SVMLightDataSet(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds)
        assert 'version' not in str(ds)
        assert filepath in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert 'SVMLightDataSet' in str(ds_versioned)
        assert 'SVMLightDataSet' in str(ds)
        assert 'protocol' in str(ds_versioned)
        assert 'protocol' in str(ds)

    def test_save_and_load(self, versioned_svm_data_set, dummy_data):
        if False:
            print('Hello World!')
        'Test that saved and reloaded data matches the original one for\n        the versioned data set.'
        versioned_svm_data_set.save(dummy_data)
        (reloaded_features, reloaded_label) = versioned_svm_data_set.load()
        (original_features, original_label) = dummy_data
        assert (original_features == reloaded_features).all()
        assert (original_label == reloaded_label).all()

    def test_no_versions(self, versioned_svm_data_set):
        if False:
            return 10
        'Check the error if no versions are available for load.'
        pattern = 'Did not find any versions for SVMLightDataSet\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            versioned_svm_data_set.load()

    def test_exists(self, versioned_svm_data_set, dummy_data):
        if False:
            for i in range(10):
                print('nop')
        'Test `exists` method invocation for versioned data set.'
        assert not versioned_svm_data_set.exists()
        versioned_svm_data_set.save(dummy_data)
        assert versioned_svm_data_set.exists()

    def test_prevent_overwrite(self, versioned_svm_data_set, dummy_data):
        if False:
            for i in range(10):
                print('nop')
        'Check the error when attempting to override the data set if the\n        corresponding json file for a given save version already exists.'
        versioned_svm_data_set.save(dummy_data)
        pattern = "Save path \\'.+\\' for SVMLightDataSet\\(.+\\) must not exist if versioning is enabled\\."
        with pytest.raises(DatasetError, match=pattern):
            versioned_svm_data_set.save(dummy_data)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, versioned_svm_data_set, load_version, save_version, dummy_data):
        if False:
            while True:
                i = 10
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for SVMLightDataSet\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            versioned_svm_data_set.save(dummy_data)

    def test_http_filesystem_no_versioning(self):
        if False:
            return 10
        pattern = 'Versioning is not supported for HTTP protocols.'
        with pytest.raises(DatasetError, match=pattern):
            SVMLightDataSet(filepath='https://example.com/file.svm', version=Version(None, None))

    def test_versioning_existing_dataset(self, svm_data_set, versioned_svm_data_set, dummy_data):
        if False:
            i = 10
            return i + 15
        'Check the error when attempting to save a versioned dataset on top of an\n        already existing (non-versioned) dataset.'
        svm_data_set.save(dummy_data)
        assert svm_data_set.exists()
        assert svm_data_set._filepath == versioned_svm_data_set._filepath
        pattern = f'(?=.*file with the same name already exists in the directory)(?=.*{versioned_svm_data_set._filepath.parent.as_posix()})'
        with pytest.raises(DatasetError, match=pattern):
            versioned_svm_data_set.save(dummy_data)
        Path(svm_data_set._filepath.as_posix()).unlink()
        versioned_svm_data_set.save(dummy_data)
        assert versioned_svm_data_set.exists()