import pytest
import modin.pandas as pd
from modin.config import NPartitions
NPartitions.put(4)

@pytest.mark.parametrize('axis', [0, 1, None])
@pytest.mark.parametrize('dtype', ['DataFrame', 'Series'])
def test_repartition(axis, dtype):
    if False:
        return 10
    if axis in (1, None) and dtype == 'Series':
        return
    df = pd.DataFrame({'col1': [1, 2], 'col2': [5, 6]})
    df2 = pd.DataFrame({'col3': [9, 4]})
    df = pd.concat([df, df2], axis=1)
    df = pd.concat([df, df], axis=0)
    obj = df if dtype == 'DataFrame' else df['col1']
    source_shapes = {'DataFrame': (2, 2), 'Series': (2, 1)}
    assert obj._query_compiler._modin_frame._partitions.shape == source_shapes[dtype]
    kwargs = {'axis': axis} if dtype == 'DataFrame' else {}
    obj = obj._repartition(**kwargs)
    if dtype == 'DataFrame':
        results = {None: (1, 1), 0: (1, 2), 1: (2, 1)}
    else:
        results = {None: (1, 1), 0: (1, 1), 1: (2, 1)}
    assert obj._query_compiler._modin_frame._partitions.shape == results[axis]