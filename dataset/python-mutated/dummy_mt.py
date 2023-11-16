import logging
import numpy as np
import torch
from fairseq.data import Dictionary, FairseqDataset
from fairseq.tasks import LegacyFairseqTask, register_task
logger = logging.getLogger(__name__)

@register_task('dummy_mt')
class DummyMTTask(LegacyFairseqTask):

    @staticmethod
    def add_args(parser):
        if False:
            i = 10
            return i + 15
        'Add task-specific arguments to the parser.'
        parser.add_argument('--dict-size', default=49996, type=int)
        parser.add_argument('--dataset-size', default=100000, type=int)
        parser.add_argument('--src-len', default=30, type=int)
        parser.add_argument('--tgt-len', default=30, type=int)

    def __init__(self, args, dictionary):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        dictionary.pad_to_multiple_(8)
        self.dummy_src = torch.arange(args.src_len + 1) + dictionary.pad() + 1
        self.dummy_tgt = torch.arange(args.tgt_len + 1) + dictionary.pad() + 1

    @classmethod
    def setup_task(cls, args, **kwargs):
        if False:
            while True:
                i = 10
        'Setup the task.'
        dictionary = Dictionary()
        for i in range(args.dict_size):
            dictionary.add_symbol('word{}'.format(i))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        args.max_source_positions = args.src_len + dictionary.pad() + 2
        args.max_target_positions = args.tgt_len + dictionary.pad() + 2
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if False:
            print('Hello World!')
        'Load a given dataset split.\n        Args:\n            split (str): name of the split (e.g., train, valid, test)\n        '
        item_size = max(self.args.src_len, self.args.tgt_len)
        if self.args.batch_size is not None:
            bsz = self.args.batch_size
        else:
            bsz = max(1, self.args.max_tokens // item_size)
        tgt = torch.stack([self.dummy_tgt for _ in range(bsz)])
        self.datasets[split] = DummyDataset({'id': 1, 'net_input': {'src_tokens': torch.stack([self.dummy_src for _ in range(bsz)]), 'src_lengths': torch.full((bsz,), self.args.src_len, dtype=torch.long), 'prev_output_tokens': tgt.clone()}, 'target': tgt, 'nsentences': bsz, 'ntokens': bsz * self.args.tgt_len}, num_items=self.args.dataset_size, item_size=item_size)

    @property
    def source_dictionary(self):
        if False:
            print('Hello World!')
        return self.dictionary

    @property
    def target_dictionary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dictionary

class DummyDataset(FairseqDataset):

    def __init__(self, batch, num_items, item_size):
        if False:
            print('Hello World!')
        super().__init__()
        self.batch = batch
        self.num_items = num_items
        self.item_size = item_size

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return index

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_items

    def collater(self, samples):
        if False:
            print('Hello World!')
        return self.batch

    @property
    def sizes(self):
        if False:
            return 10
        return np.array([self.item_size] * self.num_items)

    def num_tokens(self, index):
        if False:
            while True:
                i = 10
        return self.item_size

    def size(self, index):
        if False:
            return 10
        return self.item_size

    def ordered_indices(self):
        if False:
            return 10
        return np.arange(self.num_items)

    @property
    def supports_prefetch(self):
        if False:
            print('Hello World!')
        return False