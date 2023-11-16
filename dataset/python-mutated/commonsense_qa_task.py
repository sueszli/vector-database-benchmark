import json
import os
import numpy as np
import torch
from fairseq.data import Dictionary, IdDataset, ListDataset, NestedDictionaryDataset, NumelDataset, NumSamplesDataset, RawLabelDataset, RightPadDataset, SortDataset, data_utils, encoders
from fairseq.tasks import LegacyFairseqTask, register_task

@register_task('commonsense_qa')
class CommonsenseQATask(LegacyFairseqTask):
    """Task to finetune RoBERTa for Commonsense QA."""

    @staticmethod
    def add_args(parser):
        if False:
            i = 10
            return i + 15
        'Add task-specific arguments to the parser.'
        parser.add_argument('data', metavar='DIR', help='path to data directory; we load <split>.jsonl')
        parser.add_argument('--init-token', type=int, default=None, help='add token at the beginning of each batch item')
        parser.add_argument('--num-classes', type=int, default=5)

    def __init__(self, args, vocab):
        if False:
            i = 10
            return i + 15
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol('<mask>')
        self.bpe = encoders.build_bpe(args)

    @classmethod
    def load_dictionary(cls, filename):
        if False:
            while True:
                i = 10
        'Load the dictionary from the filename\n\n        Args:\n            filename (str): the filename\n        '
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        if False:
            i = 10
            return i + 15
        assert args.criterion == 'sentence_ranking', 'Must set --criterion=sentence_ranking'
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))
        return cls(args, vocab)

    def load_dataset(self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs):
        if False:
            while True:
                i = 10
        'Load a given dataset split.\n\n        Args:\n            split (str): name of the split (e.g., train, valid, test)\n        '

        def binarize(s, append_bos=False):
            if False:
                print('Hello World!')
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(s, append_eos=True, add_if_not_exist=False).long()
            if append_bos and self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens
        if data_path is None:
            data_path = os.path.join(self.args.data, split + '.jsonl')
        if not os.path.exists(data_path):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path))
        src_tokens = [[] for i in range(self.args.num_classes)]
        src_lengths = [[] for i in range(self.args.num_classes)]
        labels = []
        with open(data_path) as h:
            for line in h:
                example = json.loads(line.strip())
                if 'answerKey' in example:
                    label = ord(example['answerKey']) - ord('A')
                    labels.append(label)
                question = example['question']['stem']
                assert len(example['question']['choices']) == self.args.num_classes
                question = 'Q: ' + question
                question_toks = binarize(question, append_bos=True)
                for (i, choice) in enumerate(example['question']['choices']):
                    src = 'A: ' + choice['text']
                    src_bin = torch.cat([question_toks, binarize(src)])
                    src_tokens[i].append(src_bin)
                    src_lengths[i].append(len(src_bin))
        assert all((len(src_tokens[0]) == len(src_tokens[i]) for i in range(self.args.num_classes)))
        assert len(src_tokens[0]) == len(src_lengths[0])
        assert len(labels) == 0 or len(labels) == len(src_tokens[0])
        for i in range(self.args.num_classes):
            src_lengths[i] = np.array(src_lengths[i])
            src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
            src_lengths[i] = ListDataset(src_lengths[i])
        dataset = {'id': IdDataset(), 'nsentences': NumSamplesDataset(), 'ntokens': NumelDataset(src_tokens[0], reduce=True)}
        for i in range(self.args.num_classes):
            dataset.update({'net_input{}'.format(i + 1): {'src_tokens': RightPadDataset(src_tokens[i], pad_idx=self.source_dictionary.pad()), 'src_lengths': src_lengths[i]}})
        if len(labels) > 0:
            dataset.update({'target': RawLabelDataset(labels)})
        dataset = NestedDictionaryDataset(dataset, sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])])
        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(dataset, sort_order=[np.random.permutation(len(dataset))])
        print('| Loaded {} with {} samples'.format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args, from_checkpoint=False):
        if False:
            print('Hello World!')
        from fairseq import models
        model = models.build_model(args, self)
        model.register_classification_head('sentence_classification_head', num_classes=1)
        return model

    @property
    def source_dictionary(self):
        if False:
            return 10
        return self.vocab

    @property
    def target_dictionary(self):
        if False:
            return 10
        return self.vocab