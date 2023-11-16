from pathlib import PurePosixPath
import pandas as pd
import pytest
from adlfs import AzureBlobFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from plotly import graph_objects
from plotly.graph_objs import Scatter
from s3fs.core import S3FileSystem
from kedro.extras.datasets.plotly import PlotlyDataSet
from kedro.io import DatasetError
from kedro.io.core import PROTOCOL_DELIMITER

@pytest.fixture
def filepath_json(tmp_path):
    if False:
        while True:
            i = 10
    return (tmp_path / 'test.json').as_posix()

@pytest.fixture
def plotly_data_set(filepath_json, load_args, save_args, fs_args, plotly_args):
    if False:
        i = 10
        return i + 15
    return PlotlyDataSet(filepath=filepath_json, load_args=load_args, save_args=save_args, fs_args=fs_args, plotly_args=plotly_args)

@pytest.fixture
def plotly_args():
    if False:
        return 10
    return {'fig': {'orientation': 'h', 'x': 'col1', 'y': 'col2'}, 'layout': {'title': 'Test', 'xaxis_title': 'x', 'yaxis_title': 'y'}, 'type': 'scatter'}

@pytest.fixture
def dummy_dataframe():
    if False:
        while True:
            i = 10
    return pd.DataFrame({'col1': [1, 2], 'col2': [4, 5], 'col3': [5, 6]})

class TestPlotlyDataSet:

    def test_save_and_load(self, plotly_data_set, dummy_dataframe):
        if False:
            return 10
        'Test saving and reloading the data set.'
        plotly_data_set.save(dummy_dataframe)
        reloaded = plotly_data_set.load()
        assert isinstance(reloaded, graph_objects.Figure)
        assert 'Test' in str(reloaded['layout']['title'])
        assert isinstance(reloaded['data'][0], Scatter)

    def test_exists(self, plotly_data_set, dummy_dataframe):
        if False:
            return 10
        'Test `exists` method invocation for both existing and\n        nonexistent data set.'
        assert not plotly_data_set.exists()
        plotly_data_set.save(dummy_dataframe)
        assert plotly_data_set.exists()

    def test_load_missing_file(self, plotly_data_set):
        if False:
            for i in range(10):
                print('nop')
        'Check the error when trying to load missing file.'
        pattern = 'Failed while loading data from data set PlotlyDataSet\\(.*\\)'
        with pytest.raises(DatasetError, match=pattern):
            plotly_data_set.load()

    @pytest.mark.parametrize('filepath,instance_type,credentials', [('s3://bucket/file.json', S3FileSystem, {}), ('file:///tmp/test.json', LocalFileSystem, {}), ('/tmp/test.json', LocalFileSystem, {}), ('gcs://bucket/file.json', GCSFileSystem, {}), ('https://example.com/file.json', HTTPFileSystem, {}), ('abfs://bucket/file.csv', AzureBlobFileSystem, {'account_name': 'test', 'account_key': 'test'})])
    def test_protocol_usage(self, filepath, instance_type, credentials, plotly_args):
        if False:
            for i in range(10):
                print('nop')
        data_set = PlotlyDataSet(filepath=filepath, credentials=credentials, plotly_args=plotly_args)
        assert isinstance(data_set._fs, instance_type)
        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]
        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

    def test_catalog_release(self, mocker, plotly_args):
        if False:
            print('Hello World!')
        fs_mock = mocker.patch('fsspec.filesystem').return_value
        filepath = 'test.json'
        data_set = PlotlyDataSet(filepath=filepath, plotly_args=plotly_args)
        data_set.release()
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

    def test_fail_if_invalid_plotly_args_provided(self):
        if False:
            while True:
                i = 10
        plotly_args = []
        filepath = 'test.json'
        data_set = PlotlyDataSet(filepath=filepath, plotly_args=plotly_args)
        with pytest.raises(DatasetError):
            data_set.save(dummy_dataframe)