import sys
import gzip
import shutil
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
if sys.version_info > (3, 0):
    writemode = 'wb'
else:
    writemode = 'w'
zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read())

def load_mnist(path, kind='train'):
    if False:
        i = 10
        return i + 15
    'Load MNIST data from `path`'
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        (magic, n) = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        (magic, num, rows, cols) = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = (images / 255.0 - 0.5) * 2
    return (images, labels)
(X_train, y_train) = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
(X_test, y_test) = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
(fig, ax) = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
(fig, ax) = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
np.savez_compressed('mnist_scaled.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
mnist = np.load('mnist_scaled.npz')
mnist.files
(X_train, y_train, X_test, y_test) = [mnist[f] for f in ['X_train', 'y_train', 'X_test', 'y_test']]
del mnist
X_train.shape

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """

    def __init__(self, n_hidden=30, l2=0.0, epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        if False:
            return 10
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        if False:
            i = 10
            return i + 15
        'Encode labels into one-hot representation\n\n        Parameters\n        ------------\n        y : array, shape = [n_samples]\n            Target values.\n\n        Returns\n        -----------\n        onehot : array, shape = (n_samples, n_labels)\n\n        '
        onehot = np.zeros((n_classes, y.shape[0]))
        for (idx, val) in enumerate(y.astype(int)):
            onehot[val, idx] = 1.0
        return onehot.T

    def _sigmoid(self, z):
        if False:
            return 10
        'Compute logistic function (sigmoid)'
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        if False:
            i = 10
            return i + 15
        'Compute forward propagation step'
        z_h = np.dot(X, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h)
        z_out = np.dot(a_h, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)
        return (z_h, a_h, z_out, a_out)

    def _compute_cost(self, y_enc, output):
        if False:
            print('Hello World!')
        'Compute cost function.\n\n        Parameters\n        ----------\n        y_enc : array, shape = (n_samples, n_labels)\n            one-hot encoded class labels.\n        output : array, shape = [n_samples, n_output_units]\n            Activation of the output layer (forward propagation)\n\n        Returns\n        ---------\n        cost : float\n            Regularized cost\n\n        '
        L2_term = self.l2 * (np.sum(self.w_h ** 2.0) + np.sum(self.w_out ** 2.0))
        term1 = -y_enc * np.log(output)
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict class labels\n\n        Parameters\n        -----------\n        X : array, shape = [n_samples, n_features]\n            Input layer with original features.\n\n        Returns:\n        ----------\n        y_pred : array, shape = [n_samples]\n            Predicted class labels.\n\n        '
        (z_h, a_h, z_out, a_out) = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        if False:
            return 10
        ' Learn weights from training data.\n\n        Parameters\n        -----------\n        X_train : array, shape = [n_samples, n_features]\n            Input layer with original features.\n        y_train : array, shape = [n_samples]\n            Target class labels.\n        X_valid : array, shape = [n_samples, n_features]\n            Sample features for validation during training\n        y_valid : array, shape = [n_samples]\n            Sample labels for validation during training\n\n        Returns:\n        ----------\n        self\n\n        '
        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))
        epoch_strlen = len(str(self.epochs))
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
        y_train_enc = self._onehot(y_train, n_output)
        for i in range(self.epochs):
            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                self.random.shuffle(indices)
            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                (z_h, a_h, z_out, a_out) = self._forward(X_train[batch_idx])
                sigma_out = a_out - y_train_enc[batch_idx]
                sigmoid_derivative_h = a_h * (1.0 - a_h)
                sigma_h = np.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)
                delta_w_h = grad_w_h + self.l2 * self.w_h
                delta_b_h = grad_b_h
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                delta_w_out = grad_w_out + self.l2 * self.w_out
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
            (z_h, a_h, z_out, a_out) = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = np.sum(y_train == y_train_pred).astype(np.float) / X_train.shape[0]
            valid_acc = np.sum(y_valid == y_valid_pred).astype(np.float) / X_valid.shape[0]
            sys.stderr.write('\r%0*d/%d | Cost: %.2f | Train/Valid Acc.: %.2f%%/%.2f%% ' % (epoch_strlen, i + 1, self.epochs, cost, train_acc * 100, valid_acc * 100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
        return self
n_epochs = 200
if 'TRAVIS' in os.environ:
    n_epochs = 20
nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=n_epochs, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)
nn.fit(X_train=X_train[:55000], y_train=y_train[:55000], X_valid=X_train[55000:], y_valid=y_train[55000:])
a = np.arange(5)
b = a
print('a & b', np.may_share_memory(a, b))
a = np.arange(5)
print('a & b', np.may_share_memory(a, b))
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (acc * 100))
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
(fig, ax) = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()