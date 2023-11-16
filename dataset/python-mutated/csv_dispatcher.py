"""Module holds ``cuDFCSVDispatcher`` that is implemented using cuDF-entities."""
from typing import Tuple
import numpy as np
from modin.core.execution.ray.implementations.cudf_on_ray.partitioning.partition_manager import GPU_MANAGERS
from modin.core.io import CSVDispatcher

class cuDFCSVDispatcher(CSVDispatcher):
    """
    The class implements ``CSVDispatcher`` using cuDF storage format.

    This class handles utils for reading `.csv` files.
    """

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        if False:
            print('Hello World!')
        '\n        Build array with partitions of `cls.frame_partition_cls` class.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n        row_lengths : list\n            Partitions rows lengths.\n        column_widths : list\n            Number of columns in each partition.\n\n        Returns\n        -------\n        np.ndarray\n            Array with shape equals to the shape of `partition_ids` and\n            filed with partitions objects.\n        '

        def create_partition(i, j):
            if False:
                print('Hello World!')
            return cls.frame_partition_cls(GPU_MANAGERS[i], partition_ids[i][j], length=row_lengths[i], width=column_widths[j])
        return np.array([[create_partition(i, j) for j in range(len(partition_ids[i]))] for i in range(len(partition_ids))])

    @classmethod
    def _launch_tasks(cls, splits: list, **partition_kwargs) -> Tuple[list, list, list]:
        if False:
            print('Hello World!')
        '\n        Launch tasks to read partitions.\n\n        Parameters\n        ----------\n        splits : list\n            List of tuples with partitions data, which defines\n            parser task (start/end read bytes and etc).\n        **partition_kwargs : dict\n            Dictionary with keyword args that will be passed to the parser function.\n\n        Returns\n        -------\n        partition_ids : list\n            List with references to the partitions data.\n        index_ids : list\n            List with references to the partitions index objects.\n        dtypes_ids : list\n            List with references to the partitions dtypes objects.\n        '
        partition_ids = [None] * len(splits)
        index_ids = [None] * len(splits)
        dtypes_ids = [None] * len(splits)
        gpu_manager = 0
        for (idx, (start, end)) in enumerate(splits):
            partition_kwargs.update({'start': start, 'end': end, 'gpu': gpu_manager})
            (*partition_ids[idx], index_ids[idx], dtypes_ids[idx]) = cls.deploy(func=cls.parse, f_kwargs=partition_kwargs, num_returns=partition_kwargs.get('num_splits') + 2)
            gpu_manager += 1
        return (partition_ids, index_ids, dtypes_ids)