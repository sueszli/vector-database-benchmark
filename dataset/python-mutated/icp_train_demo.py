"""Verify the op's ability to discover a hidden transformation and residual."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import random
import time
import google3
from absl import app
from absl import flags
from absl import logging
import icp_grad
from icp_op import icp
import icp_util
import numpy as np
import tensorflow as tf
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', '/tmp/icp_train_demo', 'Directory to save event files for TensorBoard.')
SECRET_EGO_MOTION = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
RES_CENTER = [0.103, 1.954, 0]
RES_RADIUS = 10.0
SECRET_RES_HEIGHT = 0.1

class DataProducer(object):
    """Generates training data."""

    def __init__(self):
        if False:
            return 10
        pass

    @classmethod
    def setup(cls):
        if False:
            while True:
                i = 10
        'Open a KITTI video and read its point clouds.'
        lidar_cloud_path = os.path.join(FLAGS.test_srcdir, icp_util.LIDAR_CLOUD_PATH)
        cls.sample_cloud = np.load(lidar_cloud_path)
        logging.info('sample_cloud: %s', cls.sample_cloud)
        x_min = np.min(cls.sample_cloud[:, 0])
        x_max = np.max(cls.sample_cloud[:, 0])
        y_min = np.min(cls.sample_cloud[:, 1])
        y_max = np.max(cls.sample_cloud[:, 1])
        z_min = np.min(cls.sample_cloud[:, 2])
        z_max = np.max(cls.sample_cloud[:, 2])
        logging.info('x: %s - %s', x_min, x_max)
        logging.info('y: %s - %s', y_min, y_max)
        logging.info('z: %s - %s', z_min, z_max)

    @classmethod
    def random_transform(cls):
        if False:
            print('Hello World!')
        tx = random.uniform(-0.2, 0.2)
        ty = random.uniform(-0.2, 0.2)
        tz = random.uniform(-0.9, 0.9)
        rx = random.uniform(-0.2, 0.2) * np.pi
        ry = random.uniform(-0.2, 0.2) * np.pi
        rz = random.uniform(-0.2, 0.2) * np.pi
        transform = [tx, ty, tz, rx, ry, rz]
        return transform

    @classmethod
    def next_batch(cls, batch_size):
        if False:
            for i in range(10):
                print('nop')
        'Returns a training batch.'
        source_items = []
        target_items = []
        for _ in range(batch_size):
            source_cloud = icp_util.np_transform_cloud_xyz(cls.sample_cloud, cls.random_transform())
            source_items.append(source_cloud)
            dist_to_center = np.linalg.norm((source_cloud - RES_CENTER)[:, :2], axis=1, keepdims=True)
            res = np.maximum(RES_RADIUS - dist_to_center, 0.0) / RES_RADIUS
            res *= SECRET_RES_HEIGHT
            res = np.concatenate((np.zeros_like(res), np.zeros_like(res), res), axis=1)
            target_cloud = icp_util.np_transform_cloud_xyz(source_cloud + res, SECRET_EGO_MOTION)
            target_items.append(target_cloud)
        return (np.stack(source_items), np.stack(target_items))

def placeholder_inputs(batch_size):
    if False:
        for i in range(10):
            print('nop')
    cloud_shape = (batch_size, DataProducer.sample_cloud.shape[0], 3)
    source_placeholder = tf.placeholder(tf.float32, shape=cloud_shape)
    target_placeholder = tf.placeholder(tf.float32, shape=cloud_shape)
    return (source_placeholder, target_placeholder)

def fill_feed_dict(source_placeholder, target_placeholder):
    if False:
        i = 10
        return i + 15
    (source_feed, target_feed) = DataProducer.next_batch(FLAGS.batch_size)
    feed_dict = {source_placeholder: source_feed, target_placeholder: target_feed}
    return feed_dict

def run_training():
    if False:
        while True:
            i = 10
    'Train model for a number of steps.'
    with tf.Graph().as_default():
        DataProducer.setup()
        (source_placeholder, target_placeholder) = placeholder_inputs(FLAGS.batch_size)
        (transform, residual) = inference(source_placeholder, target_placeholder)
        loss = loss_func(transform, residual)
        train_op = training(loss, FLAGS.learning_rate)
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            sess.run(init)
            for step in range(FLAGS.max_steps):
                start_time = time.time()
                feed_dict = fill_feed_dict(source_placeholder, target_placeholder)
                (_, loss_value) = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Step %d: loss = %f (%.2f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

def inference(source, target):
    if False:
        while True:
            i = 10
    'Builds model.'
    ego_motion = tf.Variable(tf.zeros([6]), name='ego_motion')
    res_height = tf.Variable(tf.fill([1], 0.0), name='res_height')
    tf.summary.scalar('tx', ego_motion[0])
    tf.summary.scalar('ty', ego_motion[1])
    tf.summary.scalar('tz', ego_motion[2])
    tf.summary.scalar('rx', ego_motion[3])
    tf.summary.scalar('ry', ego_motion[4])
    tf.summary.scalar('rz', ego_motion[5])
    tf.summary.scalar('res_height', res_height[0])
    dist_to_center = tf.norm((source - RES_CENTER)[:, :, :2], axis=2, keep_dims=True)
    res = tf.maximum(RES_RADIUS - dist_to_center, 0.0) / RES_RADIUS
    res *= res_height
    res = tf.concat([tf.zeros_like(res), tf.zeros_like(res), res], axis=2)
    shifted_source = source + res
    ego_motion = tf.stack([ego_motion] * FLAGS.batch_size)
    (transform, residual) = icp(shifted_source, ego_motion, target)
    return (transform, residual)

def loss_func(transform, residual):
    if False:
        while True:
            i = 10
    return tf.reduce_mean(tf.square(transform), name='transform_mean') + tf.reduce_mean(tf.square(residual), name='residual_mean')

def training(loss, learning_rate):
    if False:
        for i in range(10):
            print('nop')
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def main(_):
    if False:
        i = 10
        return i + 15
    run_training()
if __name__ == '__main__':
    app.run(main)