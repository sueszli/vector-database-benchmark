import asyncio
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np
from fairseq.data import data_utils
from . import FairseqDataset
logger = logging.getLogger(__name__)

class MultiCorpusDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together.
    Unless batch_sample=True, requires each instance
    to be the same dataset, as the collate method needs to work on batches with
    samples from each dataset.

    Allows specifying a distribution over the datasets to use. Note that unlike
    MultiCorpusSampledDataset, this distribution allows sampling for each item,
    rather than on a batch level. Note that datasets with sampling probabilty
    of 0 will be skipped.

    Each time ordered_indices() is called, a new sample is generated with
    the specified distribution.

    Args:
        datasets: a OrderedDict of FairseqDataset instances.
        distribution: a List containing the probability of getting an utterance from
                        corresponding dataset
        seed: random seed for sampling the datsets
        sort_indices: if true, will sort the ordered indices by size
        batch_sample: if true, will ensure each batch is from a single dataset
    """

    def __init__(self, datasets: Dict[str, FairseqDataset], distribution: List[float], seed: int, sort_indices: bool=False, batch_sample: bool=False, distributed_rank: Optional[int]=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        assert len(datasets) == len(distribution)
        assert sum(distribution) == 1
        self.datasets = datasets
        self.distribution = distribution
        self.seed = seed
        self.sort_indices = sort_indices
        self.batch_sample = batch_sample
        self.distributed_rank = distributed_rank
        self.dataset_list = list(datasets.values())
        self.total_num_instances = 0
        first_dataset = self.dataset_list[0]
        self.num_instances_per_dataset = []
        self.dataset_offsets = []
        for (i, dataset) in enumerate(self.dataset_list):
            assert isinstance(dataset, FairseqDataset)
            assert type(dataset) is type(first_dataset)
            self.num_instances_per_dataset.append(0 if self.distribution[i] == 0 else len(dataset))
            self.dataset_offsets.append(self.total_num_instances)
            self.total_num_instances += self.num_instances_per_dataset[i]

    def ordered_indices(self):
        if False:
            print('Hello World!')
        start = time.time()
        with data_utils.numpy_seed(self.seed, self.epoch):
            logger.info(f'sampling new dataset with seed {self.seed} epoch {self.epoch}')
            sampled_indices = []
            num_selected_instances = 0
            for (i, key) in enumerate(self.datasets):
                if self.distribution[i] == 0:
                    continue
                if i < len(self.datasets) - 1:
                    num_instances = int(self.distribution[i] * self.total_num_instances)
                    high = self.dataset_offsets[i + 1]
                else:
                    num_instances = self.total_num_instances - num_selected_instances
                    high = self.total_num_instances
                logger.info(f'sampling {num_instances} from {key} dataset')
                num_selected_instances += num_instances
                dataset_size = len(self.datasets[key])
                num_copies = num_instances // dataset_size
                dataset_indices = (np.random.permutation(high - self.dataset_offsets[i]) + self.dataset_offsets[i])[:num_instances - num_copies * dataset_size]
                if num_copies > 0:
                    sampled_indices += list(np.concatenate((np.repeat(np.arange(self.dataset_offsets[i], high), num_copies), dataset_indices)))
                else:
                    sampled_indices += list(dataset_indices)
            assert len(sampled_indices) == self.total_num_instances, f'{len(sampled_indices)} vs {self.total_num_instances}'
            np.random.shuffle(sampled_indices)
            if self.sort_indices:
                sampled_indices.sort(key=lambda i: self.num_tokens(i))
            logger.info('multi_corpus_dataset ordered_indices took {}s'.format(time.time() - start))
            return np.array(sampled_indices, dtype=np.int64)

    def _map_index(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        If dataset A has length N and dataset B has length M\n        then index 1 maps to index 1 of dataset A, and index N + 1\n        maps to index 1 of B.\n        '
        counter = 0
        for (num_instances, key) in zip(self.num_instances_per_dataset, self.datasets):
            if index < counter + num_instances:
                return (index - counter, key)
            counter += num_instances
        raise ValueError('Invalid index: {}, max: {}'.format(index, self.total_num_instances))

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Length of this dataset is the sum of individual datasets\n        '
        return self.total_num_instances

    async def getitem(self, index):
        (new_index, key) = self._map_index(index)
        try:
            if hasattr(self.datasets[key], 'getitem'):
                item = await self.datasets[key].getitem(new_index)
            else:
                item = self.datasets[key][new_index]
            item['full_id'] = index
            return item
        except Exception as e:
            e.args = (f'Error from {key} dataset', *e.args)
            raise

    def __getitem__(self, index):
        if False:
            return 10
        return asyncio.run(self.getitem(index))

    async def getitems(self, indices):
        max_concurrency = 32
        sem = asyncio.Semaphore(max_concurrency)

        async def controlled_getitem(index):
            async with sem:
                return await self.getitem(index)
        coroutines = []
        for index in indices:
            coroutines.append(controlled_getitem(index))
        results = await asyncio.gather(*coroutines)
        return results

    def __getitems__(self, indices):
        if False:
            print('Hello World!')
        return asyncio.run(self.getitems(indices))

    def collater(self, samples):
        if False:
            i = 10
            return i + 15
        '\n        If we are doing batch sampling, then pick the right collater to use.\n\n        Otherwise we assume all collaters are the same.\n        '
        if len(samples) == 0:
            return None
        if 'full_id' in samples[0]:
            (_, key) = self._map_index(samples[0]['full_id'])
            try:
                batch = self.datasets[key].collater(samples)
            except Exception:
                print(f'Collating failed for key {key}', flush=True)
                raise
            return batch
        else:
            return list(self.datasets.values())[0].collater(samples)

    def num_tokens(self, index: int):
        if False:
            i = 10
            return i + 15
        (index, key) = self._map_index(index)
        return self.datasets[key].num_tokens(index)

    def size(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        (index, key) = self._map_index(index)
        return self.datasets[key].size(index)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        if False:
            while True:
                i = 10
        return False

    def set_epoch(self, epoch, **unused):
        if False:
            return 10
        super().set_epoch(epoch)
        logger.info(f'setting epoch of multi_corpus_dataset to {epoch}')
        self.epoch = epoch

    @property
    def supports_prefetch(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def supports_fetch_outside_dataloader(self):
        if False:
            i = 10
            return i + 15
        return all((self.datasets[key].supports_fetch_outside_dataloader for key in self.datasets))

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        if False:
            i = 10
            return i + 15
        if not self.batch_sample:
            return super().batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)
        dataset_indices = {key: [] for key in self.datasets}
        for i in indices:
            (_, key) = self._map_index(i)
            dataset_indices[key].append(i)
        batches = []
        for key in dataset_indices:
            cur_batches = super().batch_by_size(np.array(dataset_indices[key], dtype=np.int64), max_tokens, max_sentences, required_batch_size_multiple)
            logger.info(f'Created {len(cur_batches)} batches for dataset {key}')
            batches += cur_batches
        if self.distributed_rank is not None:
            with data_utils.numpy_seed(self.seed, self.epoch, self.distributed_rank):
                np.random.shuffle(batches)
        return batches