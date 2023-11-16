from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
from ray.actor import ActorHandle
from ray.data import Dataset

@dataclass
class RayDatasetSpec:
    """Configuration for Datasets to pass to the training workers.

    dataset_or_dict: An optional Dataset or a dictionary of
        datasets to be sharded across all the training workers, which can be accessed
        from the training function via ``ray.train.get_dataset_shard()``. Multiple
        Datasets can be passed in as a dictionary that maps each name key to a
        Dataset value, and each Dataset can be accessed from the training function
        by passing in a `dataset_name` argument to ``ray.train.get_dataset_shard()``.
    dataset_split_fn: An optional callable to specify how the provided ``dataset``
        should be split across the training workers. It is expected to take in two
        arguments. The first one is the ``dataset``, just as is passed in to the
        ``_RayDatasetSpec``. The second argument is a list of the ActorHandles of the
        training workers (to use as locality hints). The Callable is expected to
        return a list of Datasets or a list of dictionaries of Datasets,
        with the length of the list equal to the length of the list of actor handles.
        If None is provided, the provided Dataset(s) will be equally split.

    """
    dataset_or_dict: Optional[Union[Dataset, Dict[str, Dataset]]]
    dataset_split_fn: Optional[Callable[[Union[Dataset, Dict[str, Dataset]], List[ActorHandle]], List[Union[Dataset, Dict[str, Dataset]]]]] = None

    def _default_split_fn(self, training_worker_handles: List[ActorHandle]) -> List[Optional[Union[Dataset, Dict[str, Dataset]]]]:
        if False:
            print('Hello World!')

        def split_dataset(dataset_or_pipeline):
            if False:
                for i in range(10):
                    print('nop')
            return dataset_or_pipeline.split(len(training_worker_handles), equal=True, locality_hints=training_worker_handles)
        if isinstance(self.dataset_or_dict, dict):
            dataset_shards = [{} for _ in range(len(training_worker_handles))]
            for (key, dataset) in self.dataset_or_dict.items():
                split_datasets = split_dataset(dataset)
                assert len(split_datasets) == len(training_worker_handles)
                for i in range(len(split_datasets)):
                    dataset_shards[i][key] = split_datasets[i]
            return dataset_shards
        else:
            return split_dataset(self.dataset_or_dict)

    def get_dataset_shards(self, training_worker_handles: List[ActorHandle]) -> List[Optional[Union[Dataset, Dict[str, Dataset]]]]:
        if False:
            i = 10
            return i + 15
        'Returns Dataset splits based off the spec and the given training workers\n\n        Args:\n            training_worker_handles: A list of the training worker actor handles.\n\n        Returns:\n            A list of Dataset shards or list of dictionaries of Dataset shards,\n                one for each training worker.\n\n        '
        if not self.dataset_or_dict:
            return [None] * len(training_worker_handles)
        if self.dataset_split_fn is None:
            return self._default_split_fn(training_worker_handles)
        else:
            splits = self.dataset_split_fn(self.dataset_or_dict, training_worker_handles)
            if not len(splits) == len(training_worker_handles):
                raise RuntimeError(f'The list of Datasets returned by the `dataset_split_fn`: {len(splits)} does not match the number of training workers: {len(training_worker_handles)}')
            return splits