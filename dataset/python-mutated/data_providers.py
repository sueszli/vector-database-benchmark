"""Defines data providers used in training and evaluating TCNs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import random
import numpy as np
import preprocessing
import tensorflow as tf

def record_dataset(filename):
    if False:
        print('Hello World!')
    'Generate a TFRecordDataset from a `filename`.'
    return tf.data.TFRecordDataset(filename)

def full_sequence_provider(file_list, num_views):
    if False:
        for i in range(10):
            print('nop')
    'Provides full preprocessed image sequences.\n\n  Args:\n    file_list: List of strings, paths to TFRecords to preprocess.\n    num_views: Int, the number of simultaneous viewpoints at each timestep in\n      the dataset.\n  Returns:\n    preprocessed: A 4-D float32 `Tensor` holding a sequence of preprocessed\n      images.\n    raw_image_strings: A 2-D string `Tensor` holding a sequence of raw\n      jpeg-encoded image strings.\n    task: String, the name of the sequence.\n    seq_len: Int, the number of timesteps in the sequence.\n  '

    def _parse_sequence(x):
        if False:
            i = 10
            return i + 15
        (context, views, seq_len) = parse_sequence_example(x, num_views)
        task = context['task']
        return (views, task, seq_len)
    data_files = tf.contrib.slim.parallel_reader.get_data_files(file_list)
    dataset = tf.data.Dataset.from_tensor_slices(data_files)
    dataset = dataset.repeat(1)
    dataset = dataset.flat_map(record_dataset)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(12)
    dataset = dataset.map(_parse_sequence, num_parallel_calls=12)
    dataset = dataset.prefetch(12)
    dataset = dataset.make_one_shot_iterator()
    (views, task, seq_len) = dataset.get_next()
    return (views, task, seq_len)

def parse_labeled_example(example_proto, view_index, preprocess_fn, image_attr_keys, label_attr_keys):
    if False:
        return 10
    "Parses a labeled test example from a specified view.\n\n  Args:\n    example_proto: A scalar string Tensor.\n    view_index: Int, index on which view to parse.\n    preprocess_fn: A function with the signature (raw_images, is_training) ->\n      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`\n      of raw images, is_training is a Boolean describing if we're in training,\n      and preprocessed_images is a 4-D float32 image `Tensor` holding\n      preprocessed images.\n    image_attr_keys: List of Strings, names for image keys.\n    label_attr_keys: List of Strings, names for label attributes.\n  Returns:\n    data: A tuple of images, attributes and tasks `Tensors`.\n  "
    features = {}
    for attr_key in image_attr_keys:
        features[attr_key] = tf.FixedLenFeature((), tf.string)
    for attr_key in label_attr_keys:
        features[attr_key] = tf.FixedLenFeature((), tf.int64)
    parsed_features = tf.parse_single_example(example_proto, features)
    image_only_keys = [i for i in image_attr_keys if 'image' in i]
    view_image_key = image_only_keys[view_index]
    image = preprocessing.decode_image(parsed_features[view_image_key])
    preprocessed = preprocess_fn(image, is_training=False)
    attributes = [parsed_features[k] for k in label_attr_keys]
    task = parsed_features['task']
    return tuple([preprocessed] + attributes + [task])

def labeled_data_provider(filenames, preprocess_fn, view_index, image_attr_keys, label_attr_keys, batch_size=32, num_epochs=1):
    if False:
        print('Hello World!')
    "Gets a batched dataset iterator over annotated test images + labels.\n\n  Provides a single view, specifed in `view_index`.\n\n  Args:\n    filenames: List of Strings, paths to tfrecords on disk.\n    preprocess_fn: A function with the signature (raw_images, is_training) ->\n      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`\n      of raw images, is_training is a Boolean describing if we're in training,\n      and preprocessed_images is a 4-D float32 image `Tensor` holding\n      preprocessed images.\n    view_index: Int, the index of the view to embed.\n    image_attr_keys: List of Strings, names for image keys.\n    label_attr_keys: List of Strings, names for label attributes.\n    batch_size: Int, size of the batch.\n    num_epochs: Int, number of epochs over the classification dataset.\n  Returns:\n    batch_images: 4-d float `Tensor` holding the batch images for the view.\n    labels: K-d int `Tensor` holding the K label attributes.\n    tasks: 1-D String `Tensor`, holding the task names for each batch element.\n  "
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda p: parse_labeled_example(p, view_index, preprocess_fn, image_attr_keys, label_attr_keys))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    data_iterator = dataset.make_one_shot_iterator()
    batch_data = data_iterator.get_next()
    batch_images = batch_data[0]
    batch_labels = tf.stack(batch_data[1:-1], 1)
    batch_tasks = batch_data[-1]
    batch_images = set_image_tensor_batch_dim(batch_images, batch_size)
    batch_labels.set_shape([batch_size, len(label_attr_keys)])
    batch_tasks.set_shape([batch_size])
    return (batch_images, batch_labels, batch_tasks)

def parse_sequence_example(serialized_example, num_views):
    if False:
        i = 10
        return i + 15
    'Parses a serialized sequence example into views, sequence length data.'
    context_features = {'task': tf.FixedLenFeature(shape=[], dtype=tf.string), 'len': tf.FixedLenFeature(shape=[], dtype=tf.int64)}
    view_names = ['view%d' % i for i in range(num_views)]
    fixed_features = [tf.FixedLenSequenceFeature(shape=[], dtype=tf.string) for _ in range(len(view_names))]
    sequence_features = dict(zip(view_names, fixed_features))
    (context_parse, sequence_parse) = tf.parse_single_sequence_example(serialized=serialized_example, context_features=context_features, sequence_features=sequence_features)
    views = tf.stack([sequence_parse[v] for v in view_names])
    lens = [sequence_parse[v].get_shape().as_list()[0] for v in view_names]
    assert len(set(lens)) == 1
    seq_len = tf.shape(sequence_parse[view_names[-1]])[0]
    return (context_parse, views, seq_len)

def get_shuffled_input_records(file_list):
    if False:
        print('Hello World!')
    'Build a tf.data.Dataset of shuffled input TFRecords that repeats.'
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.shuffle(len(file_list))
    dataset = dataset.repeat()
    dataset = dataset.flat_map(record_dataset)
    dataset = dataset.repeat()
    return dataset

def get_tcn_anchor_pos_indices(seq_len, num_views, num_pairs, window):
    if False:
        i = 10
        return i + 15
    'Gets batch TCN anchor positive timestep and view indices.\n\n  This gets random (anchor, positive) timesteps from a sequence, and chooses\n  2 random differing viewpoints for each anchor positive pair.\n\n  Args:\n    seq_len: Int, the size of the batch sequence in timesteps.\n    num_views: Int, the number of simultaneous viewpoints at each timestep.\n    num_pairs: Int, the number of pairs to build.\n    window: Int, the window (in frames) from which to take anchor, positive\n      and negative indices.\n  Returns:\n    ap_time_indices: 1-D Int `Tensor` with size [num_pairs], holding the\n      timestep for each (anchor,pos) pair.\n    a_view_indices: 1-D Int `Tensor` with size [num_pairs], holding the\n      view index for each anchor.\n    p_view_indices: 1-D Int `Tensor` with size [num_pairs], holding the\n      view index for each positive.\n  '

    def f1():
        if False:
            print('Hello World!')
        range_min = tf.random_shuffle(tf.range(seq_len - window))[0]
        range_max = range_min + window
        return tf.range(range_min, range_max)

    def f2():
        if False:
            print('Hello World!')
        return tf.range(seq_len)
    time_indices = tf.cond(tf.greater(seq_len, window), f1, f2)
    shuffled_indices = tf.random_shuffle(time_indices)
    num_pairs = tf.minimum(seq_len, num_pairs)
    ap_time_indices = shuffled_indices[:num_pairs]
    view_indices = tf.tile(tf.expand_dims(tf.range(num_views), 0), (num_pairs, 1))
    shuffled_view_indices = tf.map_fn(tf.random_shuffle, view_indices)
    a_view_indices = shuffled_view_indices[:, 0]
    p_view_indices = shuffled_view_indices[:, 1]
    return (ap_time_indices, a_view_indices, p_view_indices)

def set_image_tensor_batch_dim(tensor, batch_dim):
    if False:
        i = 10
        return i + 15
    'Sets the batch dimension on an image tensor.'
    shape = tensor.get_shape()
    tensor.set_shape([batch_dim, shape[1], shape[2], shape[3]])
    return tensor

def parse_sequence_to_pairs_batch(serialized_example, preprocess_fn, is_training, num_views, batch_size, window):
    if False:
        print('Hello World!')
    "Parses a serialized sequence example into a batch of preprocessed data.\n\n  Args:\n    serialized_example: A serialized SequenceExample.\n    preprocess_fn: A function with the signature (raw_images, is_training) ->\n      preprocessed_images.\n    is_training: Boolean, whether or not we're in training.\n    num_views: Int, the number of simultaneous viewpoints at each timestep in\n      the dataset.\n    batch_size: Int, size of the batch to get.\n    window: Int, only take pairs from a maximium window of this size.\n  Returns:\n    preprocessed: A 4-D float32 `Tensor` holding preprocessed images.\n    anchor_images: A 4-D float32 `Tensor` holding raw anchor images.\n    pos_images: A 4-D float32 `Tensor` holding raw positive images.\n  "
    (_, views, seq_len) = parse_sequence_example(serialized_example, num_views)
    num_pairs = batch_size // 2
    (ap_time_indices, a_view_indices, p_view_indices) = get_tcn_anchor_pos_indices(seq_len, num_views, num_pairs, window)
    combined_anchor_indices = tf.concat([tf.expand_dims(a_view_indices, 1), tf.expand_dims(ap_time_indices, 1)], 1)
    combined_pos_indices = tf.concat([tf.expand_dims(p_view_indices, 1), tf.expand_dims(ap_time_indices, 1)], 1)
    anchor_images = tf.gather_nd(views, combined_anchor_indices)
    pos_images = tf.gather_nd(views, combined_pos_indices)
    anchor_images = tf.map_fn(preprocessing.decode_image, anchor_images, dtype=tf.float32)
    pos_images = tf.map_fn(preprocessing.decode_image, pos_images, dtype=tf.float32)
    concatenated = tf.concat([anchor_images, pos_images], 0)
    preprocessed = preprocess_fn(concatenated, is_training)
    (anchor_prepro, positive_prepro) = tf.split(preprocessed, num_or_size_splits=2, axis=0)
    ims = [anchor_prepro, positive_prepro, anchor_images, pos_images]
    ims = [set_image_tensor_batch_dim(i, num_pairs) for i in ims]
    [anchor_prepro, positive_prepro, anchor_images, pos_images] = ims
    anchor_labels = tf.range(1, num_pairs + 1)
    positive_labels = tf.range(1, num_pairs + 1)
    return (anchor_prepro, positive_prepro, anchor_images, pos_images, anchor_labels, positive_labels, seq_len)

def multiview_pairs_provider(file_list, preprocess_fn, num_views, window, is_training, batch_size, examples_per_seq=2, num_parallel_calls=12, sequence_prefetch_size=12, batch_prefetch_size=12):
    if False:
        while True:
            i = 10
    "Provides multi-view TCN anchor-positive image pairs.\n\n  Returns batches of Multi-view TCN pairs, where each pair consists of an\n  anchor and a positive coming from different views from the same timestep.\n  Batches are filled one entire sequence at a time until\n  batch_size is exhausted. Pairs are chosen randomly without replacement\n  within a sequence.\n\n  Used by:\n    * triplet semihard loss.\n    * clustering loss.\n    * npairs loss.\n    * lifted struct loss.\n    * contrastive loss.\n\n  Args:\n    file_list: List of Strings, paths to tfrecords.\n    preprocess_fn: A function with the signature (raw_images, is_training) ->\n      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`\n      of raw images, is_training is a Boolean describing if we're in training,\n      and preprocessed_images is a 4-D float32 image `Tensor` holding\n      preprocessed images.\n    num_views: Int, the number of simultaneous viewpoints at each timestep.\n    window: Int, size of the window (in frames) from which to draw batch ids.\n    is_training: Boolean, whether or not we're in training.\n    batch_size: Int, how many examples in the batch (num pairs * 2).\n    examples_per_seq: Int, how many examples to take per sequence.\n    num_parallel_calls: Int, the number of elements to process in parallel by\n      mapper.\n    sequence_prefetch_size: Int, size of the buffer used to prefetch sequences.\n    batch_prefetch_size: Int, size of the buffer used to prefetch batches.\n  Returns:\n    batch_images: A 4-D float32 `Tensor` holding preprocessed batch images.\n    anchor_labels: A 1-D int32 `Tensor` holding anchor image labels.\n    anchor_images: A 4-D float32 `Tensor` holding raw anchor images.\n    positive_labels: A 1-D int32 `Tensor` holding positive image labels.\n    pos_images: A 4-D float32 `Tensor` holding raw positive images.\n  "

    def _parse_sequence(x):
        if False:
            for i in range(10):
                print('nop')
        return parse_sequence_to_pairs_batch(x, preprocess_fn, is_training, num_views, examples_per_seq, window)
    dataset = get_shuffled_input_records(file_list)
    dataset = dataset.prefetch(sequence_prefetch_size)
    dataset = dataset.map(_parse_sequence, num_parallel_calls=num_parallel_calls)

    def seq_greater_than_min(seqlen, maximum):
        if False:
            print('Hello World!')
        return seqlen >= maximum
    filter_fn = functools.partial(seq_greater_than_min, maximum=examples_per_seq)
    dataset = dataset.filter(lambda a, b, c, d, e, f, seqlen: filter_fn(seqlen))
    assert batch_size % examples_per_seq == 0
    sequences_per_batch = batch_size // examples_per_seq
    dataset = dataset.batch(sequences_per_batch)
    dataset = dataset.prefetch(batch_prefetch_size)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    ims = list(data[:4])
    (anchor_labels, positive_labels) = data[4:6]
    anchor_labels.set_shape([sequences_per_batch, None])
    positive_labels.set_shape([sequences_per_batch, None])

    def _reshape_to_batchsize(im):
        if False:
            while True:
                i = 10
        '[num_sequences, num_per_seq, ...] images to [batch_size, ...].'
        sequence_ims = tf.split(im, num_or_size_splits=sequences_per_batch, axis=0)
        sequence_ims = [tf.squeeze(i) for i in sequence_ims]
        return tf.concat(sequence_ims, axis=0)
    anchor_labels = _reshape_to_batchsize(anchor_labels)
    positive_labels = _reshape_to_batchsize(positive_labels)

    def _set_shape(im):
        if False:
            for i in range(10):
                print('nop')
        'Sets a static shape for an image tensor of [sequences_per_batch,...] .'
        shape = im.get_shape()
        im.set_shape([sequences_per_batch, shape[1], shape[2], shape[3], shape[4]])
        return im
    ims = [_set_shape(im) for im in ims]
    ims = [_reshape_to_batchsize(im) for im in ims]
    (anchor_prepro, positive_prepro, anchor_images, pos_images) = ims
    batch_images = tf.concat([anchor_prepro, positive_prepro], axis=0)
    return (batch_images, anchor_labels, positive_labels, anchor_images, pos_images)

def get_svtcn_indices(seq_len, batch_size, num_views):
    if False:
        return 10
    'Gets a random window of contiguous time indices from a sequence.\n\n  Args:\n    seq_len: Int, number of timesteps in the image sequence.\n    batch_size: Int, size of the batch to construct.\n    num_views: Int, the number of simultaneous viewpoints at each\n      timestep in the dataset.\n\n  Returns:\n    time_indices: 1-D Int `Tensor` with size [batch_size], holding the\n      timestep for each batch image.\n    view_indices: 1-D Int `Tensor` with size [batch_size], holding the\n      view for each batch image. This is consistent across the batch.\n  '

    def f1():
        if False:
            print('Hello World!')
        range_min = tf.random_shuffle(tf.range(seq_len - batch_size))[0]
        range_max = range_min + batch_size
        return tf.range(range_min, range_max)

    def f2():
        if False:
            while True:
                i = 10
        return tf.range(seq_len)
    time_indices = tf.cond(tf.greater(seq_len, batch_size), f1, f2)
    random_view = tf.random_shuffle(tf.range(num_views))[0]
    view_indices = tf.tile([random_view], (batch_size,))
    return (time_indices, view_indices)

def parse_sequence_to_svtcn_batch(serialized_example, preprocess_fn, is_training, num_views, batch_size):
    if False:
        for i in range(10):
            print('nop')
    'Parses a serialized sequence example into a batch of SVTCN data.'
    (_, views, seq_len) = parse_sequence_example(serialized_example, num_views)
    (time_indices, view_indices) = get_svtcn_indices(seq_len, batch_size, num_views)
    combined_indices = tf.concat([tf.expand_dims(view_indices, 1), tf.expand_dims(time_indices, 1)], 1)
    images = tf.gather_nd(views, combined_indices)
    images = tf.map_fn(preprocessing.decode_image, images, dtype=tf.float32)
    preprocessed = preprocess_fn(images, is_training)
    return (preprocessed, images, time_indices)

def singleview_tcn_provider(file_list, preprocess_fn, num_views, is_training, batch_size, num_parallel_calls=12, sequence_prefetch_size=12, batch_prefetch_size=12):
    if False:
        print('Hello World!')
    "Provides data to train singleview TCNs.\n\n  Args:\n    file_list: List of Strings, paths to tfrecords.\n    preprocess_fn: A function with the signature (raw_images, is_training) ->\n      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`\n      of raw images, is_training is a Boolean describing if we're in training,\n      and preprocessed_images is a 4-D float32 image `Tensor` holding\n      preprocessed images.\n    num_views: Int, the number of simultaneous viewpoints at each timestep.\n    is_training: Boolean, whether or not we're in training.\n    batch_size: Int, how many examples in the batch.\n    num_parallel_calls: Int, the number of elements to process in parallel by\n      mapper.\n    sequence_prefetch_size: Int, size of the buffer used to prefetch sequences.\n    batch_prefetch_size: Int, size of the buffer used to prefetch batches.\n\n  Returns:\n    batch_images: A 4-D float32 `Tensor` of preprocessed images.\n    raw_images: A 4-D float32 `Tensor` of raw images.\n    timesteps: A 1-D int32 `Tensor` of timesteps associated with each image.\n  "

    def _parse_sequence(x):
        if False:
            print('Hello World!')
        return parse_sequence_to_svtcn_batch(x, preprocess_fn, is_training, num_views, batch_size)
    dataset = get_shuffled_input_records(file_list)
    dataset = dataset.prefetch(sequence_prefetch_size)
    dataset = dataset.map(_parse_sequence, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(batch_prefetch_size)
    dataset = dataset.make_one_shot_iterator()
    (batch_images, raw_images, timesteps) = dataset.get_next()
    return (batch_images, raw_images, timesteps)