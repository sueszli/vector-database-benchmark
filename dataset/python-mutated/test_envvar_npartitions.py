import numpy as np
import pytest
import modin.pandas as pd
from modin.config import NPartitions

@pytest.mark.parametrize('num_partitions', [2, 4, 6, 8, 10])
def test_set_npartitions(num_partitions):
    if False:
        i = 10
        return i + 15
    NPartitions.put(num_partitions)
    data = np.random.randint(0, 100, size=(2 ** 16, 2 ** 8))
    df = pd.DataFrame(data)
    part_shape = df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == num_partitions and part_shape[1] == min(num_partitions, 8)

@pytest.mark.parametrize('left_num_partitions', [2, 4, 6, 8, 10])
@pytest.mark.parametrize('right_num_partitions', [2, 4, 6, 8, 10])
def test_runtime_change_npartitions(left_num_partitions, right_num_partitions):
    if False:
        while True:
            i = 10
    NPartitions.put(left_num_partitions)
    data = np.random.randint(0, 100, size=(2 ** 16, 2 ** 8))
    left_df = pd.DataFrame(data)
    part_shape = left_df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == left_num_partitions and part_shape[1] == min(left_num_partitions, 8)
    NPartitions.put(right_num_partitions)
    right_df = pd.DataFrame(data)
    part_shape = right_df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == right_num_partitions and part_shape[1] == min(right_num_partitions, 8)