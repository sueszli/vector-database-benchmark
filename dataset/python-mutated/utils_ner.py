""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

class Split(Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'

class TokenClassificationTask:

    @staticmethod
    def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if False:
            return 10
        raise NotImplementedError

    @staticmethod
    def get_labels(path: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @staticmethod
    def convert_examples_to_features(examples: List[InputExample], label_list: List[str], max_seq_length: int, tokenizer: PreTrainedTokenizer, cls_token_at_end=False, cls_token='[CLS]', cls_token_segment_id=1, sep_token='[SEP]', sep_token_extra=False, pad_on_left=False, pad_token=0, pad_token_segment_id=0, pad_token_label_id=-100, sequence_a_segment_id=0, mask_padding_with_zero=True) -> List[InputFeatures]:
        if False:
            for i in range(10):
                print('nop')
        'Loads a data file into a list of `InputFeatures`\n        `cls_token_at_end` define the location of the CLS token:\n            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]\n            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]\n        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)\n        '
        label_map = {label: i for (i, label) in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info('Writing example %d of %d', ex_index, len(examples))
            tokens = []
            label_ids = []
            for (word, label) in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:max_seq_length - special_tokens_count]
                label_ids = label_ids[:max_seq_length - special_tokens_count]
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = [pad_token] * padding_length + input_ids
                input_mask = [0 if mask_padding_with_zero else 1] * padding_length + input_mask
                segment_ids = [pad_token_segment_id] * padding_length + segment_ids
                label_ids = [pad_token_label_id] * padding_length + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            if ex_index < 5:
                logger.info('*** Example ***')
                logger.info('guid: %s', example.guid)
                logger.info('tokens: %s', ' '.join([str(x) for x in tokens]))
                logger.info('input_ids: %s', ' '.join([str(x) for x in input_ids]))
                logger.info('input_mask: %s', ' '.join([str(x) for x in input_mask]))
                logger.info('segment_ids: %s', ' '.join([str(x) for x in segment_ids]))
                logger.info('label_ids: %s', ' '.join([str(x) for x in label_ids]))
            if 'token_type_ids' not in tokenizer.model_input_names:
                segment_ids = None
            features.append(InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids))
        return features
if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    class TokenClassificationDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """
        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

        def __init__(self, token_classification_task: TokenClassificationTask, data_dir: str, tokenizer: PreTrainedTokenizer, labels: List[str], model_type: str, max_seq_length: Optional[int]=None, overwrite_cache=False, mode: Split=Split.train):
            if False:
                i = 10
                return i + 15
            cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}'.format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)))
            lock_path = cached_features_file + '.lock'
            with FileLock(lock_path):
                if os.path.exists(cached_features_file) and (not overwrite_cache):
                    logger.info(f'Loading features from cached file {cached_features_file}')
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f'Creating features from dataset file at {data_dir}')
                    examples = token_classification_task.read_examples_from_file(data_dir, mode)
                    self.features = token_classification_task.convert_examples_to_features(examples, labels, max_seq_length, tokenizer, cls_token_at_end=bool(model_type in ['xlnet']), cls_token=tokenizer.cls_token, cls_token_segment_id=2 if model_type in ['xlnet'] else 0, sep_token=tokenizer.sep_token, sep_token_extra=False, pad_on_left=bool(tokenizer.padding_side == 'left'), pad_token=tokenizer.pad_token_id, pad_token_segment_id=tokenizer.pad_token_type_id, pad_token_label_id=self.pad_token_label_id)
                    logger.info(f'Saving features into cached file {cached_features_file}')
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            if False:
                for i in range(10):
                    print('nop')
            return self.features[i]
if is_tf_available():
    import tensorflow as tf

    class TFTokenClassificationDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """
        features: List[InputFeatures]
        pad_token_label_id: int = -100

        def __init__(self, token_classification_task: TokenClassificationTask, data_dir: str, tokenizer: PreTrainedTokenizer, labels: List[str], model_type: str, max_seq_length: Optional[int]=None, overwrite_cache=False, mode: Split=Split.train):
            if False:
                while True:
                    i = 10
            examples = token_classification_task.read_examples_from_file(data_dir, mode)
            self.features = token_classification_task.convert_examples_to_features(examples, labels, max_seq_length, tokenizer, cls_token_at_end=bool(model_type in ['xlnet']), cls_token=tokenizer.cls_token, cls_token_segment_id=2 if model_type in ['xlnet'] else 0, sep_token=tokenizer.sep_token, sep_token_extra=False, pad_on_left=bool(tokenizer.padding_side == 'left'), pad_token=tokenizer.pad_token_id, pad_token_segment_id=tokenizer.pad_token_type_id, pad_token_label_id=self.pad_token_label_id)

            def gen():
                if False:
                    return 10
                for ex in self.features:
                    if ex.token_type_ids is None:
                        yield ({'input_ids': ex.input_ids, 'attention_mask': ex.attention_mask}, ex.label_ids)
                    else:
                        yield ({'input_ids': ex.input_ids, 'attention_mask': ex.attention_mask, 'token_type_ids': ex.token_type_ids}, ex.label_ids)
            if 'token_type_ids' not in tokenizer.model_input_names:
                self.dataset = tf.data.Dataset.from_generator(gen, ({'input_ids': tf.int32, 'attention_mask': tf.int32}, tf.int64), ({'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None])}, tf.TensorShape([None])))
            else:
                self.dataset = tf.data.Dataset.from_generator(gen, ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int64), ({'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 'token_type_ids': tf.TensorShape([None])}, tf.TensorShape([None])))

        def get_dataset(self):
            if False:
                print('Hello World!')
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))
            return self.dataset

        def __len__(self):
            if False:
                return 10
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            if False:
                i = 10
                return i + 15
            return self.features[i]