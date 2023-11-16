import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Input, LayerList
from tensorlayer.models import Model
tl.logging.set_verbosity(tl.logging.DEBUG)
(X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=(-1, 784))

class CustomModelHidden(Model):

    def __init__(self):
        if False:
            return 10
        super(CustomModelHidden, self).__init__()
        self.dropout1 = Dropout(keep=0.8)
        self.seq = LayerList([Dense(n_units=800, act=tf.nn.relu, in_channels=784), Dropout(keep=0.8), Dense(n_units=800, act=tf.nn.relu, in_channels=800)])
        self.dropout3 = Dropout(keep=0.8)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        z = self.dropout1(x)
        z = self.seq(z)
        z = self.dropout3(z)
        return z

class CustomModelOut(Model):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CustomModelOut, self).__init__()
        self.dense3 = Dense(n_units=10, act=tf.nn.relu, in_channels=800)

    def forward(self, x, foo=None):
        if False:
            while True:
                i = 10
        out = self.dense3(x)
        if foo is not None:
            out = tf.nn.relu(out)
        return out
MLP1 = CustomModelHidden()
MLP2 = CustomModelOut()
n_epoch = 500
batch_size = 500
print_freq = 5
train_weights = MLP1.trainable_weights + MLP2.trainable_weights
optimizer = tf.optimizers.Adam(learning_rate=0.0001)
for epoch in range(n_epoch):
    start_time = time.time()
    for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP1.train()
        MLP2.train()
        with tf.GradientTape() as tape:
            _hidden = MLP1(X_batch)
            _logits = MLP2(_hidden, foo=1)
            _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        MLP1.eval()
        MLP2.eval()
        print('Epoch {} of {} took {}'.format(epoch + 1, n_epoch, time.time() - start_time))
        (train_loss, train_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            _hidden = MLP1(X_batch)
            _logits = MLP2(_hidden, foo=1)
            train_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   train foo=1 loss: {}'.format(train_loss / n_iter))
        print('   train foo=1 acc:  {}'.format(train_acc / n_iter))
        (val_loss, val_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _hidden = MLP1(X_batch)
            _logits = MLP2(_hidden, foo=1)
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   val foo=1 loss: {}'.format(val_loss / n_iter))
        print('   val foo=1 acc:  {}'.format(val_acc / n_iter))
        (val_loss, val_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _hidden = MLP1(X_batch)
            _logits = MLP2(_hidden, foo=0)
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   val foo=0 loss: {}'.format(val_loss / n_iter))
        print('   val foo=0 acc:  {}'.format(val_acc / n_iter))
MLP1.eval()
MLP2.eval()
(test_loss, test_acc, n_iter) = (0, 0, 0)
for (X_batch, y_batch) in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
    _hidden = MLP1(X_batch)
    _logits = MLP2(_hidden, foo=0)
    test_loss += tl.cost.cross_entropy(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print('   test foo=1 loss: {}'.format(val_loss / n_iter))
print('   test foo=1 acc:  {}'.format(val_acc / n_iter))