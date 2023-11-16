from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer

  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def load_images(path):
    print("loading...")
    td = []
    tl = []
    ed = []
    el = []
    for filename in os.listdir(path):
        if filename.endswith(".bmp"):
            i = cv2.imread(path + filename)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            i = cv2.resize(i, (28, 28))
            td.append(i)
            if "pos" in filename:
                tl.append(1)
            else:
                tl.append(0)
    for filename in os.listdir(path+"eval/"):
        if filename.endswith(".bmp"):
            i = cv2.imread(path + "eval/" + filename)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            i = cv2.resize(i, (28, 28))
            ed.append(i)
            if "pos" in filename:
                el.append(1)
            else:
                el.append(0)
    train_data = np.stack(td)
    train_labels = np.stack(tl)
    eval_data = np.stack(ed)
    eval_labels = np.stack(el)

    print(len(train_data))
    print(len(train_labels))

    return (train_data, train_labels), (eval_data, eval_labels)
