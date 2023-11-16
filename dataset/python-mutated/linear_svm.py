import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import random
import sys
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of samples per batch.')
tf.app.flags.DEFINE_integer('num_steps', 500, 'Number of steps for training.')
tf.app.flags.DEFINE_boolean('is_evaluation', True, 'Whether or not the model should be evaluated.')
tf.app.flags.DEFINE_float('C_param', 0.1, 'penalty parameter of the error term.')
tf.app.flags.DEFINE_float('Reg_param', 1.0, 'penalty parameter of the error term.')
tf.app.flags.DEFINE_float('delta', 1.0, 'The parameter set for margin.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, 'The initial learning rate for optimization.')
FLAGS = tf.app.flags.FLAGS

def loss_fn(W, b, x_data, y_target):
    if False:
        print('Hello World!')
    logits = tf.subtract(tf.matmul(x_data, W), b)
    norm_term = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W), W)), 2)
    classification_loss = tf.reduce_mean(tf.maximum(0.0, tf.subtract(FLAGS.delta, tf.multiply(logits, y_target))))
    total_loss = tf.add(tf.multiply(FLAGS.C_param, classification_loss), tf.multiply(FLAGS.Reg_param, norm_term))
    return total_loss

def inference_fn(W, b, x_data, y_target):
    if False:
        return 10
    prediction = tf.sign(tf.subtract(tf.matmul(x_data, W), b))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
    return accuracy

def next_batch_fn(x_train, y_train, num_samples=FLAGS.batch_size):
    if False:
        for i in range(10):
            print('nop')
    index = np.random.choice(len(x_train), size=num_samples)
    X_batch = x_train[index]
    y_batch = np.transpose([y_train[index]])
    return (X_batch, y_batch)
iris = datasets.load_iris()
X = iris.data[:, :2]
y = np.array([1 if label == 0 else -1 for label in iris.target])
my_randoms = np.random.choice(X.shape[0], X.shape[0], replace=False)
train_indices = my_randoms[0:int(0.5 * X.shape[0])]
test_indices = my_randoms[int(0.5 * X.shape[0]):]
x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]
x_data = tf.placeholder(shape=[None, X.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[X.shape[1], 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
total_loss = loss_fn(W, b, x_data, y_target)
accuracy = inference_fn(W, b, x_data, y_target)
train_op = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate).minimize(total_loss)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for step_idx in range(FLAGS.num_steps):
    (X_batch, y_batch) = next_batch_fn(x_train, y_train, num_samples=FLAGS.batch_size)
    sess.run(train_op, feed_dict={x_data: X_batch, y_target: y_batch})
    loss_step = sess.run(total_loss, feed_dict={x_data: X_batch, y_target: y_batch})
    train_acc_step = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
    test_acc_step = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})
    if step_idx % 100 == 0:
        print('Step #%d, training accuracy= %% %.2f, testing accuracy= %% %.2f ' % (step_idx, float(100 * train_acc_step), float(100 * test_acc_step)))
if FLAGS.is_evaluation:
    [[w1], [w2]] = sess.run(W)
    [[bias]] = sess.run(b)
    x_line = [data[1] for data in X]
    line = []
    line = [-w2 / w1 * i + bias / w1 for i in x_line]
    for (index, data) in enumerate(X):
        if y[index] == 1:
            positive_X = data[1]
            positive_y = data[0]
        elif y[index] == -1:
            negative_X = data[1]
            negative_y = data[0]
        else:
            sys.exit('Invalid label!')