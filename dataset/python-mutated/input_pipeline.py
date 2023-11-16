"""BERT model input pipelines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def decode_record(record, name_to_features):
    if False:
        print('Hello World!')
    'Decodes a record to a TensorFlow example.'
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example

def single_file_dataset(input_file, name_to_features):
    if False:
        while True:
            i = 10
    'Creates a single-file dataset to be passed for BERT custom training.'
    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: decode_record(record, name_to_features))
    if isinstance(input_file, str) or len(input_file) == 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        d = d.with_options(options)
    return d

def create_pretrain_dataset(input_patterns, seq_length, max_predictions_per_seq, batch_size, is_training=True, input_pipeline_context=None):
    if False:
        return 10
    'Creates input dataset from (tf)records files for pretraining.'
    name_to_features = {'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64), 'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64), 'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64), 'masked_lm_positions': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64), 'masked_lm_ids': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64), 'masked_lm_weights': tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32), 'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64)}
    dataset = tf.data.Dataset.list_files(input_patterns, shuffle=is_training)
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines, input_pipeline_context.input_pipeline_id)
    dataset = dataset.repeat()
    input_files = []
    for input_pattern in input_patterns:
        input_files.extend(tf.io.gfile.glob(input_pattern))
    dataset = dataset.shuffle(len(input_files))
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    decode_fn = lambda record: decode_record(record, name_to_features)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _select_data_from_record(record):
        if False:
            i = 10
            return i + 15
        'Filter out features to use for pretraining.'
        x = {'input_word_ids': record['input_ids'], 'input_mask': record['input_mask'], 'input_type_ids': record['segment_ids'], 'masked_lm_positions': record['masked_lm_positions'], 'masked_lm_ids': record['masked_lm_ids'], 'masked_lm_weights': record['masked_lm_weights'], 'next_sentence_labels': record['next_sentence_labels']}
        y = record['masked_lm_weights']
        return (x, y)
    dataset = dataset.map(_select_data_from_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1024)
    return dataset

def create_classifier_dataset(file_path, seq_length, batch_size, is_training=True, input_pipeline_context=None):
    if False:
        i = 10
        return i + 15
    'Creates input dataset from (tf)records files for train/eval.'
    name_to_features = {'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64), 'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64), 'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64), 'label_ids': tf.io.FixedLenFeature([], tf.int64), 'is_real_example': tf.io.FixedLenFeature([], tf.int64)}
    dataset = single_file_dataset(file_path, name_to_features)
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines, input_pipeline_context.input_pipeline_id)

    def _select_data_from_record(record):
        if False:
            return 10
        x = {'input_word_ids': record['input_ids'], 'input_mask': record['input_mask'], 'input_type_ids': record['segment_ids']}
        y = record['label_ids']
        return (x, y)
    dataset = dataset.map(_select_data_from_record)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(1024)
    return dataset

def create_squad_dataset(file_path, seq_length, batch_size, is_training=True, input_pipeline_context=None):
    if False:
        return 10
    'Creates input dataset from (tf)records files for train/eval.'
    name_to_features = {'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64), 'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64), 'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64)}
    if is_training:
        name_to_features['start_positions'] = tf.io.FixedLenFeature([], tf.int64)
        name_to_features['end_positions'] = tf.io.FixedLenFeature([], tf.int64)
    else:
        name_to_features['unique_ids'] = tf.io.FixedLenFeature([], tf.int64)
    dataset = single_file_dataset(file_path, name_to_features)
    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines, input_pipeline_context.input_pipeline_id)

    def _select_data_from_record(record):
        if False:
            print('Hello World!')
        'Dispatches record to features and labels.'
        (x, y) = ({}, {})
        for (name, tensor) in record.items():
            if name in ('start_positions', 'end_positions'):
                y[name] = tensor
            elif name == 'input_ids':
                x['input_word_ids'] = tensor
            elif name == 'segment_ids':
                x['input_type_ids'] = tensor
            else:
                x[name] = tensor
        return (x, y)
    dataset = dataset.map(_select_data_from_record)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1024)
    return dataset