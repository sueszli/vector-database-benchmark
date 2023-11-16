import os
import shutil
from unittest import mock
import pandas as pd
import pytest
ray = pytest.importorskip('ray')
dask = pytest.importorskip('dask')
from ludwig.data.dataset.ray import RayDatasetBatcher, read_remote_parquet
pytestmark = pytest.mark.distributed

def test_async_reader_error():
    if False:
        for i in range(10):
            print('nop')
    pipeline = mock.Mock()
    features = {'num1': {'name': 'num1', 'type': 'number'}, 'bin1': {'name': 'bin1', 'type': 'binary'}}
    training_set_metadata = {'num1': {}, 'bin1': {}}
    with pytest.raises(TypeError, match="'Mock' object is not iterable"):
        RayDatasetBatcher(dataset_epoch_iterator=iter([pipeline]), features=features, training_set_metadata=training_set_metadata, batch_size=64, samples_per_epoch=100, ignore_last=False)

@pytest.fixture(scope='module')
def parquet_file(ray_cluster_2cpu) -> str:
    if False:
        while True:
            i = 10
    'Write a multi-file parquet dataset to the cwd.\n\n    Returns:\n        The path to the parquet dataset.\n    '
    df = pd.DataFrame({'col1': list(range(1000)), 'col2': list(range(1000))})
    df = dask.dataframe.from_pandas(df, chunksize=100)
    cwd = os.getcwd()
    filepath = os.path.join(cwd, 'data.training.parquet')
    df.to_parquet(filepath, engine='pyarrow')
    yield filepath
    shutil.rmtree(filepath)

@pytest.fixture(scope='module', params=['absolute', 'relative'])
def parquet_filepath(parquet_file: str, request: 'pytest.FixtureRequest') -> str:
    if False:
        print('Hello World!')
    'Convert a filepath in the CWD to either an absolute or relative path.\n\n    Args:\n        parquet_file: Absolute path to a parquet file in the CWD\n        request: pytest request fixture with the fixture parameters\n\n    Returns:\n        Either the absolute or relative path of the parquet file.\n    '
    filepath_type = request.param
    return parquet_file if filepath_type == 'absolute' else os.path.basename(parquet_file)

def test_read_remote_parquet(parquet_filepath: str):
    if False:
        while True:
            i = 10
    'Test for the fix to https://github.com/ludwig-ai/ludwig/issues/3440.\n\n    Parquet file reads will fail with `pyarrow.lib.ArrowInvalid` under the following conditions:\n        1) The Parquet data is in multi-file format\n        2) A relative filepath is passed to the read function\n        3) A filesystem object is passed to the read function\n\n    The issue can be resolved by either:\n        1) Passing an absolute filepath\n        2) Not passing a filesystem object\n    '
    read_remote_parquet(parquet_filepath)