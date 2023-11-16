from pathlib import Path, PurePosixPath
import networkx
import pytest
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs.core import S3FileSystem
from kedro.extras.datasets.networkx import GraphMLDataSet
from kedro.io import DatasetError, Version
from kedro.io.core import PROTOCOL_DELIMITER
ATTRS = {'source': 'from', 'target': 'to', 'name': 'fake_id', 'key': 'fake_key', 'link': 'fake_link'}

@pytest.fixture
def filepath_graphml(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    return (tmp_path / 'some_dir' / 'test.graphml').as_posix()

@pytest.fixture
def graphml_data_set(filepath_graphml):
    if False:
        while True:
            i = 10
    return GraphMLDataSet(filepath=filepath_graphml, load_args={'node_type': int}, save_args={})

@pytest.fixture
def versioned_graphml_data_set(filepath_graphml, load_version, save_version):
    if False:
        print('Hello World!')
    return GraphMLDataSet(filepath=filepath_graphml, version=Version(load_version, save_version), load_args={'node_type': int}, save_args={})

@pytest.fixture()
def dummy_graph_data():
    if False:
        print('Hello World!')
    return networkx.complete_graph(3)

class TestGraphMLDataSet:

    def test_save_and_load(self, graphml_data_set, dummy_graph_data):
        if False:
            print('Hello World!')
        'Test saving and reloading the data set.'
        graphml_data_set.save(dummy_graph_data)
        reloaded = graphml_data_set.load()
        assert dummy_graph_data.nodes(data=True) == reloaded.nodes(data=True)
        assert graphml_data_set._fs_open_args_load == {'mode': 'rb'}
        assert graphml_data_set._fs_open_args_save == {'mode': 'wb'}

    def test_load_missing_file(self, graphml_data_set):
        if False:
            i = 10
            return i + 15
        'Check the error when trying to load missing file.'
        pattern = 'Failed while loading data from data set GraphMLDataSet\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            assert graphml_data_set.load()

    def test_exists(self, graphml_data_set, dummy_graph_data):
        if False:
            return 10
        'Test `exists` method invocation.'
        assert not graphml_data_set.exists()
        graphml_data_set.save(dummy_graph_data)
        assert graphml_data_set.exists()

    @pytest.mark.parametrize('filepath,instance_type', [('s3://bucket/file.graphml', S3FileSystem), ('file:///tmp/test.graphml', LocalFileSystem), ('/tmp/test.graphml', LocalFileSystem), ('gcs://bucket/file.graphml', GCSFileSystem), ('https://example.com/file.graphml', HTTPFileSystem)])
    def test_protocol_usage(self, filepath, instance_type):
        if False:
            print('Hello World!')
        data_set = GraphMLDataSet(filepath=filepath)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    def test_catalog_release(self, mocker):
        if False:
            while True:
                i = 10
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.graphml'
        data_set = GraphMLDataSet(filepath=filepath)
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

class TestGraphMLDataSetVersioned:

    def test_save_and_load(self, versioned_graphml_data_set, dummy_graph_data):
        if False:
            print('Hello World!')
        'Test that saved and reloaded data matches the original one for\n        the versioned data set.'
        versioned_graphml_data_set.save(dummy_graph_data)
        reloaded = versioned_graphml_data_set.load()
        assert dummy_graph_data.nodes(data=True) == reloaded.nodes(data=True)
        assert versioned_graphml_data_set._fs_open_args_load == {'mode': 'rb'}
        assert versioned_graphml_data_set._fs_open_args_save == {'mode': 'wb'}

    def test_no_versions(self, versioned_graphml_data_set):
        if False:
            while True:
                i = 10
        'Check the error if no versions are available for load.'
        pattern = 'Did not find any versions for GraphMLDataSet\\(.+\\)'
        with pytest.raises(DatasetError, match=pattern):
            versioned_graphml_data_set.load()

    def test_exists(self, versioned_graphml_data_set, dummy_graph_data):
        if False:
            for i in range(10):
                print('nop')
        'Test `exists` method invocation for versioned data set.'
        assert not versioned_graphml_data_set.exists()
        versioned_graphml_data_set.save(dummy_graph_data)
        assert versioned_graphml_data_set.exists()

    def test_prevent_override(self, versioned_graphml_data_set, dummy_graph_data):
        if False:
            while True:
                i = 10
        'Check the error when attempt to override the same data set\n        version.'
        versioned_graphml_data_set.save(dummy_graph_data)
        pattern = "Save path \\'.+\\' for GraphMLDataSet\\(.+\\) must not exist if versioning is enabled"
        with pytest.raises(DatasetError, match=pattern):
            versioned_graphml_data_set.save(dummy_graph_data)

    @pytest.mark.parametrize('load_version', ['2019-01-01T23.59.59.999Z'], indirect=True)
    @pytest.mark.parametrize('save_version', ['2019-01-02T00.00.00.000Z'], indirect=True)
    def test_save_version_warning(self, versioned_graphml_data_set, load_version, save_version, dummy_graph_data):
        if False:
            print('Hello World!')
        'Check the warning when saving to the path that differs from\n        the subsequent load path.'
        pattern = f"Save version '{save_version}' did not match load version '{load_version}' for GraphMLDataSet\\(.+\\)"
        with pytest.warns(UserWarning, match=pattern):
            versioned_graphml_data_set.save(dummy_graph_data)

    def test_version_str_repr(self, load_version, save_version):
        if False:
            print('Hello World!')
        'Test that version is in string representation of the class instance\n        when applicable.'
        filepath = 'test.graphml'
        ds = GraphMLDataSet(filepath=filepath)
        ds_versioned = GraphMLDataSet(filepath=filepath, version=Version(load_version, save_version))
        assert filepath in str(ds)
        assert 'version' not in str(ds)
        assert filepath in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert 'GraphMLDataSet' in str(ds_versioned)
        assert 'GraphMLDataSet' in str(ds)
        assert 'protocol' in str(ds_versioned)
        assert 'protocol' in str(ds)

    def test_versioning_existing_dataset(self, graphml_data_set, versioned_graphml_data_set, dummy_graph_data):
        if False:
            return 10
        'Check the error when attempting to save a versioned dataset on top of an\n        already existing (non-versioned) dataset.'
        graphml_data_set.save(dummy_graph_data)
        assert graphml_data_set.exists()
        assert graphml_data_set._filepath == versioned_graphml_data_set._filepath
        pattern = f'(?=.*file with the same name already exists in the directory)(?=.*{versioned_graphml_data_set._filepath.parent.as_posix()})'
        with pytest.raises(DatasetError, match=pattern):
            versioned_graphml_data_set.save(dummy_graph_data)
        Path(graphml_data_set._filepath.as_posix()).unlink()
        versioned_graphml_data_set.save(dummy_graph_data)
        assert versioned_graphml_data_set.exists()