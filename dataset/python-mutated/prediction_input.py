"""Code for building the input for the prediction model."""
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
FLAGS = flags.FLAGS
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3
IMG_WIDTH = 64
IMG_HEIGHT = 64
STATE_DIM = 5

def build_tfrecord_input(training=True):
    if False:
        while True:
            i = 10
    'Create input tfrecord tensors.\n\n  Args:\n    training: training or validation data.\n  Returns:\n    list of tensors corresponding to images, actions, and states. The images\n    tensor is 5D, batch x time x height x width x channels. The state and\n    action tensors are 3D, batch x time x dimension.\n  Raises:\n    RuntimeError: if no files found.\n  '
    filenames = gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')
    index = int(np.floor(FLAGS.train_val_split * len(filenames)))
    if training:
        filenames = filenames[:index]
    else:
        filenames = filenames[index:]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    reader = tf.TFRecordReader()
    (_, serialized_example) = reader.read(filename_queue)
    (image_seq, state_seq, action_seq) = ([], [], [])
    for i in range(FLAGS.sequence_length):
        image_name = 'move/' + str(i) + '/image/encoded'
        action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
        state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'
        if FLAGS.use_state:
            features = {image_name: tf.FixedLenFeature([1], tf.string), action_name: tf.FixedLenFeature([STATE_DIM], tf.float32), state_name: tf.FixedLenFeature([STATE_DIM], tf.float32)}
        else:
            features = {image_name: tf.FixedLenFeature([1], tf.string)}
        features = tf.parse_single_example(serialized_example, features=features)
        image_buffer = tf.reshape(features[image_name], shape=[])
        image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
        image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        if IMG_HEIGHT != IMG_WIDTH:
            raise ValueError('Unequal height and width unsupported')
        crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
        image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0
        image_seq.append(image)
        if FLAGS.use_state:
            state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
            state_seq.append(state)
            action = tf.reshape(features[action_name], shape=[1, STATE_DIM])
            action_seq.append(action)
    image_seq = tf.concat(axis=0, values=image_seq)
    if FLAGS.use_state:
        state_seq = tf.concat(axis=0, values=state_seq)
        action_seq = tf.concat(axis=0, values=action_seq)
        [image_batch, action_batch, state_batch] = tf.train.batch([image_seq, action_seq, state_seq], FLAGS.batch_size, num_threads=FLAGS.batch_size, capacity=100 * FLAGS.batch_size)
        return (image_batch, action_batch, state_batch)
    else:
        image_batch = tf.train.batch([image_seq], FLAGS.batch_size, num_threads=FLAGS.batch_size, capacity=100 * FLAGS.batch_size)
        zeros_batch = tf.zeros([FLAGS.batch_size, FLAGS.sequence_length, STATE_DIM])
        return (image_batch, zeros_batch, zeros_batch)