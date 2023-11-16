"""CIFAR dataset input module.
"""
import tensorflow as tf

def build_input(dataset, data_path, batch_size, mode):
    if False:
        i = 10
        return i + 15
    "Build CIFAR image and labels.\n\n  Args:\n    dataset: Either 'cifar10' or 'cifar100'.\n    data_path: Filename for data.\n    batch_size: Input batch size.\n    mode: Either 'train' or 'eval'.\n  Returns:\n    images: Batches of images. [batch_size, image_size, image_size, 3]\n    labels: Batches of labels. [batch_size, num_classes]\n  Raises:\n    ValueError: when the specified dataset is not supported.\n  "
    image_size = 32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    else:
        raise ValueError('Not supported dataset %s', dataset)
    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes
    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    (_, value) = reader.read(file_queue)
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]), [depth, image_size, image_size])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        example_queue = tf.RandomShuffleQueue(capacity=16 * batch_size, min_after_dequeue=8 * batch_size, dtypes=[tf.float32, tf.int32], shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)
        example_queue = tf.FIFOQueue(3 * batch_size, dtypes=[tf.float32, tf.int32], shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1
    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * num_threads))
    (images, labels) = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(tf.concat(values=[indices, labels], axis=1), [batch_size, num_classes], 1.0, 0.0)
    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes
    tf.summary.image('images', images)
    return (images, labels)