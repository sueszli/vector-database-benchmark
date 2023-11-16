"""Utilities for pre-processing classification data."""
from absl import logging
from official.nlp.xlnet import data_utils
SEG_ID_A = 0
SEG_ID_B = 1

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        if False:
            while True:
                i = 10
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    if False:
        return 10
    'Truncates a sequence pair in place to the maximum length.'
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(example_index, example, label_list, max_seq_length, tokenize_fn, use_bert_format):
    if False:
        i = 10
        return i + 15
    'Converts a single `InputExample` into a single `InputFeatures`.'
    if isinstance(example, PaddingInputExample):
        return InputFeatures(input_ids=[0] * max_seq_length, input_mask=[1] * max_seq_length, segment_ids=[0] * max_seq_length, label_id=0, is_real_example=False)
    if label_list is not None:
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
    tokens_a = tokenize_fn(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenize_fn(example.text_b)
    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    elif len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:max_seq_length - 2]
    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
    tokens.append(data_utils.SEP_ID)
    segment_ids.append(SEG_ID_A)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(SEG_ID_B)
        tokens.append(data_utils.SEP_ID)
        segment_ids.append(SEG_ID_B)
    if use_bert_format:
        tokens.insert(0, data_utils.CLS_ID)
        segment_ids.insert(0, data_utils.SEG_ID_CLS)
    else:
        tokens.append(data_utils.CLS_ID)
        segment_ids.append(data_utils.SEG_ID_CLS)
    input_ids = tokens
    input_mask = [0] * len(input_ids)
    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        if use_bert_format:
            input_ids = input_ids + [0] * delta_len
            input_mask = input_mask + [1] * delta_len
            segment_ids = segment_ids + [data_utils.SEG_ID_PAD] * delta_len
        else:
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [data_utils.SEG_ID_PAD] * delta_len + segment_ids
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    if label_list is not None:
        label_id = label_map[example.label]
    else:
        label_id = example.label
    if example_index < 5:
        logging.info('*** Example ***')
        logging.info('guid: %s', example.guid)
        logging.info('input_ids: %s', ' '.join([str(x) for x in input_ids]))
        logging.info('input_mask: %s', ' '.join([str(x) for x in input_mask]))
        logging.info('segment_ids: %s', ' '.join([str(x) for x in segment_ids]))
        logging.info('label: %d (id = %d)', example.label, label_id)
    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)
    return feature