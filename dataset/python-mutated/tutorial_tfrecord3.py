"""
You will learn.

1. How to save time-series data (e.g. sentence) into TFRecord format file.
2. How to read time-series data from TFRecord format file.
3. How to create inputs, targets and mask.

Reference
----------
1. Google's im2txt - MSCOCO Image Captioning example
2. TFRecord in http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
3. Batching and Padding data in http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

"""
import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorlayer as tl

def _int64_feature(value):
    if False:
        print('Hello World!')
    'Wrapper for inserting an int64 Feature into a SequenceExample proto,\n    e.g, An integer label.\n    '
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if False:
        print('Hello World!')
    'Wrapper for inserting a bytes Feature into a SequenceExample proto,\n    e.g, an image in byte\n    '
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature_list(values):
    if False:
        while True:
            i = 10
    'Wrapper for inserting an int64 FeatureList into a SequenceExample proto,\n    e.g, sentence in list of ints\n    '
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    if False:
        for i in range(10):
            print('nop')
    'Wrapper for inserting a bytes FeatureList into a SequenceExample proto,\n    e.g, sentence in list of bytes\n    '
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
cwd = os.getcwd()
IMG_DIR = cwd + '/data/cat/'
SEQ_FIR = cwd + '/data/cat_caption.json'
VOC_FIR = cwd + '/vocab.txt'
with tf.gfile.FastGFile(SEQ_FIR, 'r') as f:
    caption_data = json.loads(str(f.read()))
(processed_capts, img_capts) = ([], [])
for idx in range(len(caption_data['images'])):
    img_capt = caption_data['images'][idx]['caption']
    img_capts.append(img_capt)
    processed_capts.append(tl.nlp.process_sentence(img_capt, start_word='<S>', end_word='</S>'))
print('Original Captions: %s' % img_capts)
print('Processed Captions: %s\n' % processed_capts)
_ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
vocab = tl.nlp.Vocabulary(VOC_FIR, start_word='<S>', end_word='</S>', unk_word='<UNK>')
writer = tf.python_io.TFRecordWriter('train.cat_caption')
for idx in range(len(caption_data['images'])):
    img_name = caption_data['images'][idx]['file_name']
    img_capt = '<S> ' + caption_data['images'][idx]['caption'] + ' </S>'
    img_capt_ids = [vocab.word_to_id(word) for word in img_capt.split(' ')]
    print('%s : %s : %s' % (img_name, img_capt, img_capt_ids))
    img = Image.open(IMG_DIR + img_name)
    img = img.resize((299, 299))
    img_raw = img.tobytes()
    img_capt_b = [v.encode() for v in img_capt.split(' ')]
    context = tf.train.Features(feature={'image/img_raw': _bytes_feature(img_raw)})
    feature_lists = tf.train.FeatureLists(feature_list={'image/caption': _bytes_feature_list(img_capt_b), 'image/caption_ids': _int64_feature_list(img_capt_ids)})
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    writer.write(sequence_example.SerializeToString())
writer.close()
filename_queue = tf.train.string_input_producer(['train.cat_caption'])
reader = tf.TFRecordReader()
(_, serialized_example) = reader.read(filename_queue)
(features, sequence_features) = tf.parse_single_sequence_example(serialized_example, context_features={'image/img_raw': tf.FixedLenFeature([], tf.string)}, sequence_features={'image/caption': tf.FixedLenSequenceFeature([], dtype=tf.string), 'image/caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64)})
c = tf.contrib.learn.run_n(features, n=1, feed_dict=None)
im = Image.frombytes('RGB', (299, 299), c[0]['image/img_raw'])
tl.visualize.frame(np.asarray(im), second=1, saveable=False, name='frame', fig_idx=1236)
c = tf.contrib.learn.run_n(sequence_features, n=1, feed_dict=None)
print(c[0])

def distort_image(image, thread_id):
    if False:
        print('Hello World!')
    'Perform random distortions on an image.\n    Args:\n        image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).\n        thread_id: Preprocessing thread id used to select the ordering of color\n        distortions. There should be a multiple of 2 preprocessing threads.\n    Returns:````\n        distorted_image: A float32 Tensor of shape [height, width, 3] with values in\n        [0, 1].\n    '
    with tf.name_scope('flip_horizontal'):
        image = tf.image.random_flip_left_right(image)
    color_ordering = thread_id % 2
    with tf.name_scope('distort_color'):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def prefetch_input_data(reader, file_pattern, is_training, batch_size, values_per_shard, input_queue_capacity_factor=16, num_reader_threads=1, shard_queue_name='filename_queue', value_queue_name='input_queue'):
    if False:
        i = 10
        return i + 15
    'Prefetches string values from disk into an input queue.\n\n    In training the capacity of the queue is important because a larger queue\n    means better mixing of training examples between shards. The minimum number of\n    values kept in the queue is values_per_shard * input_queue_capacity_factor,\n    where input_queue_memory factor should be chosen to trade-off better mixing\n    with memory usage.\n\n    Args:\n        reader: Instance of tf.ReaderBase.\n        file_pattern: Comma-separated list of file patterns (e.g.\n            /tmp/train_data-?????-of-00100).\n        is_training: Boolean; whether prefetching for training or eval.\n        batch_size: Model batch size used to determine queue capacity.\n        values_per_shard: Approximate number of values per shard.\n        input_queue_capacity_factor: Minimum number of values to keep in the queue\n        in multiples of values_per_shard. See comments above.\n        num_reader_threads: Number of reader threads to fill the queue.\n        shard_queue_name: Name for the shards filename queue.\n        value_queue_name: Name for the values input queue.\n\n    Returns:\n        A Queue containing prefetched string values.\n    '
    data_files = []
    for pattern in file_pattern.split(','):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tl.logging.fatal('Found no input files matching %s', file_pattern)
    else:
        tl.logging.info('Prefetching values from %d files matching %s', len(data_files), file_pattern)
    if is_training:
        print('   is_training == True : RandomShuffleQueue')
        filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_queue_examples, dtypes=[tf.string], name='random_' + value_queue_name)
    else:
        print('   is_training == False : FIFOQueue')
        filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string], name='fifo_' + value_queue_name)
    enqueue_ops = []
    for _ in range(num_reader_threads):
        (_, value) = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    tf.summary.scalar('queue/%s/fraction_of_%d_full' % (values_queue.name, capacity), tf.cast(values_queue.size(), tf.float32) * (1.0 / capacity))
    return values_queue
is_training = True
resize_height = resize_width = 346
height = width = 299
reader = tf.TFRecordReader()
input_queue = prefetch_input_data(reader, file_pattern='train.cat_caption', is_training=is_training, batch_size=4, values_per_shard=2300, input_queue_capacity_factor=2, num_reader_threads=1)
serialized_sequence_example = input_queue.dequeue()
(context, sequence) = tf.parse_single_sequence_example(serialized=serialized_sequence_example, context_features={'image/img_raw': tf.FixedLenFeature([], dtype=tf.string)}, sequence_features={'image/caption': tf.FixedLenSequenceFeature([], dtype=tf.string), 'image/caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64)})
img = tf.decode_raw(context['image/img_raw'], tf.uint8)
img = tf.reshape(img, [height, width, 3])
img = tf.image.convert_image_dtype(img, dtype=tf.float32)
try:
    img = tf.image.resize_images(img, size=(resize_height, resize_width), method=tf.image.ResizeMethod.BILINEAR)
except Exception:
    img = tf.image.resize_images(img, new_height=resize_height, new_width=resize_width, method=tf.image.ResizeMethod.BILINEAR)
if is_training:
    img = tf.random_crop(img, [height, width, 3])
else:
    img = tf.image.resize_image_with_crop_or_pad(img, height, width)
if is_training:
    img = distort_image(img, thread_id=0)
img = tf.subtract(img, 0.5)
img = tf.multiply(img, 2.0)
img_cap = sequence['image/caption']
img_cap_ids = sequence['image/caption_ids']
(img_batch, img_cap_batch, img_cap_ids_batch) = tf.train.batch([img, img_cap, img_cap_ids], batch_size=4, capacity=50000, dynamic_pad=True, num_threads=4)
sess = tf.Session()
tl.layers.initialize_global_variables(sess)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for _ in range(3):
    print('Step %s' % _)
    (imgs, caps, caps_id) = sess.run([img_batch, img_cap_batch, img_cap_ids_batch])
    print(caps)
    print(caps_id)
    tl.visualize.images2d((imgs + 1) / 2, second=1, saveable=False, name='batch', dtype=None, fig_idx=202025)
coord.request_stop()
coord.join(threads)
sess.close()

def batch_with_dynamic_pad(images_and_captions, batch_size, queue_capacity, add_summaries=True):
    if False:
        i = 10
        return i + 15
    "Batches input images and captions.\n\n    This function splits the caption into an input sequence and a target sequence,\n    where the target sequence is the input sequence right-shifted by 1. Input and\n    target sequences are batched and padded up to the maximum length of sequences\n    in the batch. A mask is created to distinguish real words from padding words.\n\n    Example:\n        Actual captions in the batch ('-' denotes padded character):\n        [\n            [ 1 2 5 4 5 ],\n            [ 1 2 3 4 - ],\n            [ 1 2 3 - - ],\n        ]\n\n        input_seqs:\n        [\n            [ 1 2 3 4 ],\n            [ 1 2 3 - ],\n            [ 1 2 - - ],\n        ]\n\n        target_seqs:\n        [\n            [ 2 3 4 5 ],\n            [ 2 3 4 - ],\n            [ 2 3 - - ],\n        ]\n\n        mask:\n        [\n            [ 1 1 1 1 ],\n            [ 1 1 1 0 ],\n            [ 1 1 0 0 ],\n        ]\n\n    Args:\n        images_and_captions: A list of pairs [image, caption], where image is a\n        Tensor of shape [height, width, channels] and caption is a 1-D Tensor of\n        any length. Each pair will be processed and added to the queue in a\n        separate thread.\n        batch_size: Batch size.\n        queue_capacity: Queue capacity.\n        add_summaries: If true, add caption length summaries.\n\n    Returns:\n        images: A Tensor of shape [batch_size, height, width, channels].\n        input_seqs: An int32 Tensor of shape [batch_size, padded_length].\n        target_seqs: An int32 Tensor of shape [batch_size, padded_length].\n        mask: An int32 0/1 Tensor of shape [batch_size, padded_length].\n    "
    enqueue_list = []
    for (image, caption) in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([image, input_seq, target_seq, indicator])
    (images, input_seqs, target_seqs, mask) = tf.train.batch_join(enqueue_list, batch_size=batch_size, capacity=queue_capacity, dynamic_pad=True, name='batch_and_pad')
    if add_summaries:
        lengths = tf.add(tf.reduce_sum(mask, 1), 1)
        tf.summary.scalar('caption_length/batch_min', tf.reduce_min(lengths))
        tf.summary.scalar('caption_length/batch_max', tf.reduce_max(lengths))
        tf.summary.scalar('caption_length/batch_mean', tf.reduce_mean(lengths))
    return (images, input_seqs, target_seqs, mask)
(images, input_seqs, target_seqs, input_mask) = batch_with_dynamic_pad(images_and_captions=[[img, img_cap]], batch_size=4, queue_capacity=50000)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for _ in range(3):
    print('Step %s' % _)
    (imgs, inputs, targets, masks) = sess.run([images, input_seqs, target_seqs, input_mask])
    print(inputs)
    print(targets)
    print(masks)
    tl.visualize.images2d((imgs + 1) / 2, second=1, saveable=False, name='batch', dtype=None, fig_idx=202025)
coord.request_stop()
coord.join(threads)
sess.close()