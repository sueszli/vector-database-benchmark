"""The module defines interface for an axis partition with PyArrow storage format and Ray engine."""
import pyarrow
import ray
from modin.core.dataframe.pandas.partitioning.axis_partition import BaseDataframeAxisPartition
from .partition import PyarrowOnRayDataframePartition

class PyarrowOnRayDataframeAxisPartition(BaseDataframeAxisPartition):
    """
    Class defines axis partition interface with PyArrow storage format and Ray engine.

    Inherits functionality from ``BaseDataframeAxisPartition`` class.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition for.
    """

    def __init__(self, list_of_blocks):
        if False:
            print('Hello World!')
        assert all([len(partition.list_of_blocks) == 1 for partition in list_of_blocks]), 'Implementation assumes that each partition contains a signle block.'
        self.list_of_blocks = [obj.list_of_blocks[0] for obj in list_of_blocks]

    def apply(self, func, *args, num_splits=None, other_axis_partition=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Apply func to the object in the Plasma store.\n\n        Parameters\n        ----------\n        func : callable or ray.ObjectRef\n            The function to apply.\n        *args : iterable\n            Positional arguments to pass with `func`.\n        num_splits : int, optional\n            The number of times to split the resulting object.\n        other_axis_partition : PyarrowOnRayDataframeAxisPartition, optional\n            Another ``PyarrowOnRayDataframeAxisPartition`` object to apply to\n            `func` with this one.\n        **kwargs : dict\n            Additional keyward arguments to pass with `func`.\n\n        Returns\n        -------\n        list\n            List with ``PyarrowOnRayDataframePartition`` objects.\n\n        Notes\n        -----\n        See notes in Parent class about this method.\n        '
        if num_splits is None:
            num_splits = len(self.list_of_blocks)
        if other_axis_partition is not None:
            return [PyarrowOnRayDataframePartition(obj) for obj in deploy_ray_func_between_two_axis_partitions.options(num_returns=num_splits).remote(self.axis, func, args, kwargs, num_splits, len(self.list_of_blocks), *self.list_of_blocks + other_axis_partition.list_of_blocks)]
        return [PyarrowOnRayDataframePartition(obj) for obj in deploy_ray_axis_func.options(num_returns=num_splits).remote(self.axis, func, args, kwargs, num_splits, *self.list_of_blocks)]

class PyarrowOnRayDataframeColumnPartition(PyarrowOnRayDataframeAxisPartition):
    """
    The column partition implementation for PyArrow storage format and Ray engine.

    All of the implementation for this class is in the ``PyarrowOnRayDataframeAxisPartition``
    parent class, and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition.
    """
    axis = 0

class PyarrowOnRayDataframeRowPartition(PyarrowOnRayDataframeAxisPartition):
    """
    The row partition implementation for PyArrow storage format and Ray engine.

    All of the implementation for this class is in the ``PyarrowOnRayDataframeAxisPartition``
    parent class, and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition.
    """
    axis = 1

def concat_arrow_table_partitions(axis, partitions):
    if False:
        while True:
            i = 10
    '\n    Concatenate given `partitions` in a single table.\n\n    Parameters\n    ----------\n    axis : {0, 1}\n        The axis to concatenate over.\n    partitions : array-like\n        Array with partitions for concatenating.\n\n    Returns\n    -------\n    pyarrow.Table\n        ``pyarrow.Table`` constructed from the passed partitions.\n    '
    if axis == 0:
        table = pyarrow.Table.from_batches([part.to_batches(part.num_rows)[0] for part in partitions])
    else:
        table = partitions[0].drop([partitions[0].columns[-1].name])
        for obj in partitions[1:]:
            i = 0
            for col in obj.itercolumns():
                if i < obj.num_columns - 1:
                    table = table.append_column(col)
                i += 1
        table = table.append_column(partitions[0].columns[-1])
    return table

def split_arrow_table_result(axis, result, num_partitions, num_splits, metadata):
    if False:
        print('Hello World!')
    '\n    Split ``pyarrow.Table`` according to the passed parameters.\n\n    Parameters\n    ----------\n    axis : {0, 1}\n        The axis to perform the function along.\n    result : pyarrow.Table\n        Resulting table to split.\n    num_partitions : int\n        Number of partitions that `result` was constructed from.\n    num_splits : int\n        The number of splits to return.\n    metadata : dict\n        Dictionary with ``pyarrow.Table`` metadata.\n\n    Returns\n    -------\n    list\n        List of PyArrow Tables.\n    '
    chunksize = num_splits // num_partitions if num_splits % num_partitions == 0 else num_splits // num_partitions + 1
    if axis == 0:
        return [pyarrow.Table.from_batches([part]) for part in result.to_batches(chunksize)]
    else:
        return [result.drop([result.columns[i].name for i in range(result.num_columns) if i >= n * chunksize or i < (n - 1) * chunksize]).append_column(result.columns[-1]).replace_schema_metadata(metadata=metadata) for n in range(1, num_splits)] + [result.drop([result.columns[i].name for i in range(result.num_columns) if i < (num_splits - 1) * chunksize]).replace_schema_metadata(metadata=metadata)]

@ray.remote
def deploy_ray_axis_func(axis, func, f_args, f_kwargs, num_splits, *partitions):
    if False:
        i = 10
        return i + 15
    '\n    Deploy a function along a full axis in Ray.\n\n    Parameters\n    ----------\n    axis : {0, 1}\n        The axis to perform the function along.\n    func : callable\n            The function to perform.\n    f_args : list or tuple\n        Positional arguments to pass to ``func``.\n    f_kwargs : dict\n        Keyword arguments to pass to ``func``.\n    num_splits : int\n        The number of splits to return.\n    *partitions : array-like\n        All partitions that make up the full axis (row or column).\n\n    Returns\n    -------\n    list\n        List of PyArrow Tables.\n    '
    table = concat_arrow_table_partitions(axis, partitions)
    try:
        result = func(table, *f_args, **f_kwargs)
    except Exception:
        result = pyarrow.Table.from_pandas(func(table.to_pandas(), *f_args, **f_kwargs))
    return split_arrow_table_result(axis, result, len(partitions), num_splits, table.schema.metadata)

@ray.remote
def deploy_ray_func_between_two_axis_partitions(axis, func, f_args, f_kwargs, num_splits, len_of_left, *partitions):
    if False:
        return 10
    '\n    Deploy a function along a full axis between two data sets in Ray.\n\n    Parameters\n    ----------\n    axis : {0, 1}\n        The axis to perform the function along.\n    func : callable\n        The function to perform.\n    f_args : list or tuple\n        Positional arguments to pass to ``func``.\n    f_kwargs : dict\n        Keyword arguments to pass to ``func``.\n    num_splits : int\n        The number of splits to return.\n    len_of_left : int\n        The number of values in `partitions` that belong to the left data set.\n    *partitions : array-like\n        All partitions that make up the full axis (row or column)\n        for both data sets.\n\n    Returns\n    -------\n    list\n        List of PyArrow Tables.\n    '
    lt_table = concat_arrow_table_partitions(axis, partitions[:len_of_left])
    rt_table = concat_arrow_table_partitions(axis, partitions[len_of_left:])
    try:
        result = func(lt_table, rt_table, *f_args, **f_kwargs)
    except Exception:
        lt_frame = lt_table.from_pandas()
        rt_frame = rt_table.from_pandas()
        result = pyarrow.Table.from_pandas(func(lt_frame, rt_frame, *f_args, **f_kwargs))
    return split_arrow_table_result(axis, result, len(result.num_rows), num_splits, result.schema.metadata)