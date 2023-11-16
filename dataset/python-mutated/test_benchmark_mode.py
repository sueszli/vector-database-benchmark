import unittest.mock as mock
import pytest
import modin.pandas as pd
from modin.config import Engine
engine = Engine.get()
if engine == 'Ray':
    wait_method = 'modin.core.execution.ray.implementations.' + 'pandas_on_ray.partitioning.' + 'PandasOnRayDataframePartitionManager.wait_partitions'
elif engine == 'Dask':
    wait_method = 'modin.core.execution.dask.implementations.' + 'pandas_on_dask.partitioning.' + 'PandasOnDaskDataframePartitionManager.wait_partitions'
elif engine == 'Unidist':
    wait_method = 'modin.core.execution.unidist.implementations.' + 'pandas_on_unidist.partitioning.' + 'PandasOnUnidistDataframePartitionManager.wait_partitions'
else:
    wait_method = 'modin.core.dataframe.pandas.partitioning.' + 'partition_manager.PandasDataframePartitionManager.wait_partitions'

@pytest.mark.parametrize('set_benchmark_mode', [False], indirect=True)
def test_turn_off(set_benchmark_mode):
    if False:
        print('Hello World!')
    df = pd.DataFrame([0])
    with mock.patch(wait_method) as wait:
        df.dropna()
    wait.assert_not_called()

@pytest.mark.parametrize('set_benchmark_mode', [True], indirect=True)
def test_turn_on(set_benchmark_mode):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame([0])
    with mock.patch(wait_method) as wait:
        df.dropna()
    wait.assert_called()