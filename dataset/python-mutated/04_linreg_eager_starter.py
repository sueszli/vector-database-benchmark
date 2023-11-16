""" Starter code for a simple regression example using eager execution.
Created by Akshay Agrawal (akshayka@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 04
"""
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import utils
DATA_FILE = 'data/birth_life_2010.txt'
(data, n_samples) = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
w = None
b = None

def prediction(x):
    if False:
        while True:
            i = 10
    pass

def squared_loss(y, y_predicted):
    if False:
        i = 10
        return i + 15
    pass

def huber_loss(y, y_predicted):
    if False:
        print('Hello World!')
    'Huber loss with `m` set to `1.0`.'
    pass

def train(loss_fn):
    if False:
        while True:
            i = 10
    'Train a regression model evaluated using `loss_fn`.'
    print('Training; loss function: ' + loss_fn.__name__)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    def loss_for_example(x, y):
        if False:
            i = 10
            return i + 15
        pass
    grad_fn = None
    start = time.time()
    for epoch in range(100):
        total_loss = 0.0
        for (x_i, y_i) in tfe.Iterator(dataset):
            optimizer.apply_gradients(gradients)
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
    print('Took: %f seconds' % (time.time() - start))
    print('Eager execution exhibits significant overhead per operation. As you increase your batch size, the impact of the overhead will become less noticeable. Eager execution is under active development: expect performance to increase substantially in the near future!')
train(huber_loss)
plt.plot(data[:, 0], data[:, 1], 'bo')
plt.plot(data[:, 0], data[:, 0] * w.numpy() + b.numpy(), 'r', label='huber regression')
plt.legend()
plt.show()