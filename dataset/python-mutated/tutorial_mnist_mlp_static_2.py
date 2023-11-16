import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.models import Model
tl.logging.set_verbosity(tl.logging.DEBUG)
(X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=(-1, 784))

def hidden_model(inputs_shape):
    if False:
        for i in range(10):
            print('nop')
    ni = Input(inputs_shape)
    nn = Dropout(keep=0.8)(ni)
    nn = Dense(n_units=800, act=tf.nn.relu)(nn)
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=800, act=tf.nn.relu)(nn)
    return Model(inputs=ni, outputs=nn, name='mlp_hidden')

def get_model(inputs_shape, hmodel):
    if False:
        i = 10
        return i + 15
    hidden = hmodel.as_layer()
    ni = Input(inputs_shape)
    nn = hidden(ni)
    nn = Dropout(keep=0.8)(nn)
    nn = Dense(n_units=10, act=tf.nn.relu)(nn)
    return Model(inputs=ni, outputs=nn, name='mlp')
MLP_hidden = hidden_model([None, 784])
MLP = get_model([None, 784], MLP_hidden)
n_epoch = 500
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
optimizer = tf.optimizers.Adam(lr=0.0001)
for epoch in range(n_epoch):
    start_time = time.time()
    for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.train()
        with tf.GradientTape() as tape:
            _logits = MLP(X_batch)
            _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        MLP.eval()
        print('Epoch {} of {} took {}'.format(epoch + 1, n_epoch, time.time() - start_time))
        (train_loss, train_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            _logits = MLP(X_batch)
            train_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   train loss: {}'.format(train_loss / n_iter))
        print('   train acc:  {}'.format(train_acc / n_iter))
        (val_loss, val_acc, n_iter) = (0, 0, 0)
        for (X_batch, y_batch) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = MLP(X_batch)
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print('   val loss: {}'.format(val_loss / n_iter))
        print('   val acc:  {}'.format(val_acc / n_iter))
MLP.eval()
(test_loss, test_acc, n_iter) = (0, 0, 0)
for (X_batch, y_batch) in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
    _logits = MLP(X_batch)
    test_loss += tl.cost.cross_entropy(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print('   test loss: {}'.format(test_loss / n_iter))
print('   test acc:  {}'.format(test_acc / n_iter))