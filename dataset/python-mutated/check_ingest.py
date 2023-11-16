import sys
import time
from typing import Optional
import numpy as np
import ray
from ray import train
from ray.air.config import DatasetConfig, ScalingConfig
from ray.data import Dataset, DataIterator, Preprocessor
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train import DataConfig
from ray.util.annotations import Deprecated, DeveloperAPI
MAKE_LOCAL_DATA_ITERATOR_DEPRECATION_MSG = '\nmake_local_dataset_iterator is deprecated. Call ``iterator()`` directly on your dataset instead to create a local DataIterator.\n'

@DeveloperAPI
class DummyTrainer(DataParallelTrainer):
    """A Trainer that does nothing except read the data for a given number of epochs.

    It prints out as much debugging statistics as possible.

    This is useful for debugging data ingest problem. This trainer supports normal
    scaling options same as any other Trainer (e.g., num_workers, use_gpu).

    Args:
        scaling_config: Configuration for how to scale training. This is the same
            as for :class:`~ray.train.base_trainer.BaseTrainer`.
        num_epochs: How many many times to iterate through the datasets for.
        prefetch_batches: The number of batches to prefetch ahead of the
            current block during the scan. This is the same as
            :meth:`~ray.data.Dataset.iter_batches`
    """

    def __init__(self, *args, scaling_config: Optional[ScalingConfig]=None, num_epochs: int=1, prefetch_batches: int=1, batch_size: Optional[int]=4096, prefetch_blocks: int=0, **kwargs):
        if False:
            i = 10
            return i + 15
        if not scaling_config:
            scaling_config = ScalingConfig(num_workers=1)
        super().__init__(*args, train_loop_per_worker=DummyTrainer.make_train_loop(num_epochs, prefetch_batches, prefetch_blocks, batch_size), scaling_config=scaling_config, **kwargs)

    @staticmethod
    def make_train_loop(num_epochs: int, prefetch_batches: int, prefetch_blocks: int, batch_size: Optional[int]):
        if False:
            return 10
        'Make a debug train loop that runs for the given amount of epochs.'

        def train_loop_per_worker():
            if False:
                print('Hello World!')
            import pandas as pd
            rank = train.get_context().get_world_rank()
            data_shard = train.get_dataset_shard('train')
            start = time.perf_counter()
            (epochs_read, batches_read, bytes_read) = (0, 0, 0)
            batch_delays = []
            print('Starting train loop on worker', rank)
            for epoch in range(num_epochs):
                epochs_read += 1
                batch_start = time.perf_counter()
                for batch in data_shard.iter_batches(prefetch_batches=prefetch_batches, prefetch_blocks=prefetch_blocks, batch_size=batch_size):
                    batch_delay = time.perf_counter() - batch_start
                    batch_delays.append(batch_delay)
                    batches_read += 1
                    if isinstance(batch, pd.DataFrame):
                        bytes_read += int(batch.memory_usage(index=True, deep=True).sum())
                    elif isinstance(batch, np.ndarray):
                        bytes_read += batch.nbytes
                    elif isinstance(batch, dict):
                        for arr in batch.values():
                            bytes_read += arr.nbytes
                    else:
                        bytes_read += sys.getsizeof(batch)
                    train.report(dict(bytes_read=bytes_read, batches_read=batches_read, epochs_read=epochs_read, batch_delay=batch_delay))
                    batch_start = time.perf_counter()
            delta = time.perf_counter() - start
            print('Time to read all data', delta, 'seconds')
            print('P50/P95/Max batch delay (s)', np.quantile(batch_delays, 0.5), np.quantile(batch_delays, 0.95), np.max(batch_delays))
            print('Num epochs read', epochs_read)
            print('Num batches read', batches_read)
            print('Num bytes read', round(bytes_read / (1024 * 1024), 2), 'MiB')
            print('Mean throughput', round(bytes_read / (1024 * 1024) / delta, 2), 'MiB/s')
            if rank == 0:
                print('Ingest stats from rank=0:\n\n{}'.format(data_shard.stats()))
        return train_loop_per_worker

@Deprecated(MAKE_LOCAL_DATA_ITERATOR_DEPRECATION_MSG)
def make_local_dataset_iterator(dataset: Dataset, preprocessor: Preprocessor, dataset_config: DatasetConfig) -> DataIterator:
    if False:
        return 10
    'A helper function to create a local\n    :py:class:`DataIterator <ray.data.DataIterator>`,\n    like the one returned by :meth:`~ray.train.get_dataset_shard`.\n\n    This function should only be used for development and debugging. It will\n    raise an exception if called by a worker instead of the driver.\n\n    Args:\n        dataset: The input Dataset.\n        preprocessor: The preprocessor that will be applied to the input dataset.\n        dataset_config: The dataset config normally passed to the trainer.\n    '
    raise DeprecationWarning(MAKE_LOCAL_DATA_ITERATOR_DEPRECATION_MSG)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', '-e', type=int, default=1, help='Number of epochs to read.')
    parser.add_argument('--prefetch-batches', '-b', type=int, default=1, help='Number of batches to prefetch when reading data.')
    args = parser.parse_args()
    ds = ray.data.range_tensor(50000, shape=(80, 80, 4), parallelism=100)
    ds = ds.map_batches(lambda df: df * 2, batch_format='pandas')
    ds = ds.map_batches(lambda df: df * 2, batch_format='pandas')
    trainer = DummyTrainer(scaling_config=ScalingConfig(num_workers=1, use_gpu=False), datasets={'train': ds}, num_epochs=args.num_epochs, prefetch_batches=args.prefetch_batches, dataset_config=DataConfig(), batch_size=None)
    print('Dataset config', trainer.get_dataset_config())
    trainer.fit()
    try:
        print('Memory stats at end of ingest:\n\n{}'.format(ray._private.internal_api.memory_summary(stats_only=True)))
    except Exception:
        print('Error getting Ray memory stats')