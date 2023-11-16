import logging
import math
from typing import List, Optional, NamedTuple
import numpy as np
import torch
from fairseq.data import ConcatDataset, LanguagePairDataset, FileAudioDataset, data_utils
from fairseq.data import FairseqDataset
logger = logging.getLogger(__name__)

class ModalityDatasetItem(NamedTuple):
    datasetname: str
    dataset: any
    max_positions: List[int]
    max_tokens: Optional[int] = None
    max_sentences: Optional[int] = None

class MultiModalityDataset(ConcatDataset):

    def __init__(self, datasets: List[ModalityDatasetItem]):
        if False:
            print('Hello World!')
        id_to_mode = []
        dsets = []
        max_tokens = []
        max_sentences = []
        max_positions = []
        for dset in datasets:
            id_to_mode.append(dset.datasetname)
            dsets.append(dset.dataset)
            max_tokens.append(dset.max_tokens)
            max_positions.append(dset.max_positions)
            max_sentences.append(dset.max_sentences)
        weights = [1.0 for s in dsets]
        super().__init__(dsets, weights)
        self.max_tokens = max_tokens
        self.max_positions = max_positions
        self.max_sentences = max_sentences
        self.id_to_mode = id_to_mode
        self.raw_sub_batch_samplers = []
        self._cur_epoch = 0

    def set_epoch(self, epoch):
        if False:
            while True:
                i = 10
        super().set_epoch(epoch)
        self._cur_epoch = epoch

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        (dataset_idx, sample_idx) = self._get_dataset_and_sample_index(idx)
        sample = self.datasets[dataset_idx][sample_idx]
        return (dataset_idx, sample)

    def collater(self, samples):
        if False:
            for i in range(10):
                print('nop')
        if len(samples) == 0:
            return {}
        dataset_idx = samples[0][0]
        assert sum([0 if dataset_idx == s[0] else 1 for s in samples]) == 0
        samples = self.datasets[dataset_idx].collater([x[1] for x in samples])
        samples['net_input']['mode'] = self.id_to_mode[dataset_idx]
        return samples

    def size(self, index: int):
        if False:
            while True:
                i = 10
        if len(self.datasets) == 1:
            return self.datasets[0].size(index)
        return super().size(index)

    @property
    def sizes(self):
        if False:
            i = 10
            return i + 15
        if len(self.datasets) == 1:
            return self.datasets[0].sizes
        return super().sizes

    def ordered_indices(self):
        if False:
            while True:
                i = 10
        '\n        Returns indices sorted by length. So less padding is needed.\n        '
        if len(self.datasets) == 1:
            return [self.datasets[0].ordered_indices()]
        indices_group = []
        for (d_idx, ds) in enumerate(self.datasets):
            sample_num = self.cumulative_sizes[d_idx]
            if d_idx > 0:
                sample_num = sample_num - self.cumulative_sizes[d_idx - 1]
            assert sample_num == len(ds)
            indices_group.append(ds.ordered_indices())
        return indices_group

    def get_raw_batch_samplers(self, required_batch_size_multiple, seed):
        if False:
            return 10
        if len(self.raw_sub_batch_samplers) > 0:
            logger.info(' raw_sub_batch_samplers exists. No action is taken')
            return
        with data_utils.numpy_seed(seed):
            indices = self.ordered_indices()
        for (i, ds) in enumerate(self.datasets):
            indices[i] = ds.filter_indices_by_size(indices[i], self.max_positions[i])[0]
            sub_batch_sampler = ds.batch_by_size(indices[i], max_tokens=self.max_tokens[i], max_sentences=self.max_sentences[i], required_batch_size_multiple=required_batch_size_multiple)
            self.raw_sub_batch_samplers.append(sub_batch_sampler)

    def get_batch_samplers(self, mult_ratios, required_batch_size_multiple, seed):
        if False:
            for i in range(10):
                print('nop')
        self.get_raw_batch_samplers(required_batch_size_multiple, seed)
        batch_samplers = []
        for (i, _) in enumerate(self.datasets):
            if i > 0:
                sub_batch_sampler = [[y + self.cumulative_sizes[i - 1] for y in x] for x in self.raw_sub_batch_samplers[i]]
            else:
                sub_batch_sampler = list(self.raw_sub_batch_samplers[i])
            smp_r = mult_ratios[i]
            if smp_r != 1:
                is_increase = 'increased' if smp_r > 1 else 'decreased'
                logger.info('number of batch for the dataset {} is {} from {} to {}'.format(self.id_to_mode[i], is_increase, len(sub_batch_sampler), int(len(sub_batch_sampler) * smp_r)))
                mul_samplers = []
                for _ in range(math.floor(smp_r)):
                    mul_samplers = mul_samplers + sub_batch_sampler
                if math.floor(smp_r) != smp_r:
                    with data_utils.numpy_seed(seed + self._cur_epoch):
                        np.random.shuffle(sub_batch_sampler)
                        smp_num = int((smp_r - math.floor(smp_r)) * len(sub_batch_sampler))
                    mul_samplers = mul_samplers + sub_batch_sampler[:smp_num]
                sub_batch_sampler = mul_samplers
            else:
                logger.info('dataset {} batch number is {} '.format(self.id_to_mode[i], len(sub_batch_sampler)))
            batch_samplers.append(sub_batch_sampler)
        return batch_samplers

class LangPairMaskDataset(FairseqDataset):

    def __init__(self, dataset: LanguagePairDataset, src_eos: int, src_bos: Optional[int]=None, noise_id: Optional[int]=-1, mask_ratio: Optional[float]=0, mask_type: Optional[str]='random'):
        if False:
            for i in range(10):
                print('nop')
        self.dataset = dataset
        self.src_eos = src_eos
        self.src_bos = src_bos
        self.noise_id = noise_id
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        assert mask_type in ('random', 'tail')

    @property
    def src_sizes(self):
        if False:
            return 10
        return self.dataset.src_sizes

    @property
    def tgt_sizes(self):
        if False:
            return 10
        return self.dataset.tgt_sizes

    @property
    def sizes(self):
        if False:
            return 10
        return self.dataset.sizes

    def get_batch_shapes(self):
        if False:
            print('Hello World!')
        if hasattr(self.dataset, 'get_batch_shapes'):
            return self.dataset.get_batch_shapes()
        return self.dataset.buckets

    def num_tokens_vec(self, indices):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset.num_tokens_vec(indices)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.dataset)

    def num_tokens(self, index):
        if False:
            i = 10
            return i + 15
        return self.dataset.num_tokens(index)

    def size(self, index):
        if False:
            print('Hello World!')
        return self.dataset.size(index)

    def ordered_indices(self):
        if False:
            return 10
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        if False:
            return 10
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset.prefetch(indices)

    def mask_src_tokens(self, sample):
        if False:
            i = 10
            return i + 15
        src_item = sample['source']
        mask = None
        if self.mask_type == 'random':
            mask = torch.rand(len(src_item)).le(self.mask_ratio)
        else:
            mask = torch.ones(len(src_item))
            mask[:int(len(src_item) * (1 - self.mask_ratio))] = 0
            mask = mask.eq(1)
        if src_item[0] == self.src_bos:
            mask[0] = False
        if src_item[-1] == self.src_eos:
            mask[-1] = False
        mask_src_item = src_item.masked_fill(mask, self.noise_id)
        smp = {'id': sample['id'], 'source': mask_src_item, 'target': sample['target']}
        return smp

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        sample = self.dataset[index]
        if self.mask_ratio > 0:
            sample = self.mask_src_tokens(sample)
        return sample

    def collater(self, samples, pad_to_length=None):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset.collater(samples, pad_to_length)

class FileAudioDatasetWrapper(FileAudioDataset):

    def collater(self, samples):
        if False:
            return 10
        samples = super().collater(samples)
        if len(samples) == 0:
            return {}
        samples['net_input']['src_tokens'] = samples['net_input']['source']
        samples['net_input']['prev_output_tokens'] = None
        del samples['net_input']['source']
        samples['net_input']['src_lengths'] = None
        samples['net_input']['alignment'] = None
        return samples