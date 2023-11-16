import logging
import numpy as np
from autograd import elementwise_grad
from mla.base import BaseEstimator
from mla.metrics.metrics import get_metric
from mla.neuralnet.layers import PhaseMixin
from mla.neuralnet.loss import get_loss
from mla.utils import batch_iterator
np.random.seed(9999)
'\nArchitecture inspired from:\n\n    https://github.com/fchollet/keras\n    https://github.com/andersbll/deeppy\n'

class NeuralNet(BaseEstimator):
    fit_required = False

    def __init__(self, layers, optimizer, loss, max_epochs=10, batch_size=64, metric='mse', shuffle=False, verbose=True):
        if False:
            for i in range(10):
                print('nop')
        self.verbose = verbose
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.loss = get_loss(loss)
        if loss == 'categorical_crossentropy':
            self.loss_grad = lambda actual, predicted: -(actual - predicted)
        else:
            self.loss_grad = elementwise_grad(self.loss, 1)
        self.metric = get_metric(metric)
        self.layers = layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._n_layers = 0
        self.log_metric = True if loss != metric else False
        self.metric_name = metric
        self.bprop_entry = self._find_bprop_entry()
        self.training = False
        self._initialized = False

    def _setup_layers(self, x_shape):
        if False:
            print('Hello World!')
        "Initialize model's layers."
        x_shape = list(x_shape)
        x_shape[0] = self.batch_size
        for layer in self.layers:
            layer.setup(x_shape)
            x_shape = layer.shape(x_shape)
        self._n_layers = len(self.layers)
        self.optimizer.setup(self)
        self._initialized = True
        logging.info('Total parameters: %s' % self.n_params)

    def _find_bprop_entry(self):
        if False:
            for i in range(10):
                print('nop')
        'Find entry layer for back propagation.'
        if len(self.layers) > 0 and (not hasattr(self.layers[-1], 'parameters')):
            return -1
        return len(self.layers)

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        if not self._initialized:
            self._setup_layers(X.shape)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        self._setup_input(X, y)
        self.is_training = True
        self.optimizer.optimize(self)
        self.is_training = False

    def update(self, X, y):
        if False:
            i = 10
            return i + 15
        y_pred = self.fprop(X)
        grad = self.loss_grad(y, y_pred)
        for layer in reversed(self.layers[:self.bprop_entry]):
            grad = layer.backward_pass(grad)
        return self.loss(y, y_pred)

    def fprop(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Forward propagation.'
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def _predict(self, X=None):
        if False:
            return 10
        if not self._initialized:
            self._setup_layers(X.shape)
        y = []
        X_batch = batch_iterator(X, self.batch_size)
        for Xb in X_batch:
            y.append(self.fprop(Xb))
        return np.concatenate(y)

    @property
    def parametric_layers(self):
        if False:
            for i in range(10):
                print('nop')
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                yield layer

    @property
    def parameters(self):
        if False:
            while True:
                i = 10
        'Returns a list of all parameters.'
        params = []
        for layer in self.parametric_layers:
            params.append(layer.parameters)
        return params

    def error(self, X=None, y=None):
        if False:
            i = 10
            return i + 15
        'Calculate an error for given examples.'
        training_phase = self.is_training
        if training_phase:
            self.is_training = False
        if X is None and y is None:
            y_pred = self._predict(self.X)
            score = self.metric(self.y, y_pred)
        else:
            y_pred = self._predict(X)
            score = self.metric(y, y_pred)
        if training_phase:
            self.is_training = True
        return score

    @property
    def is_training(self):
        if False:
            i = 10
            return i + 15
        return self.training

    @is_training.setter
    def is_training(self, train):
        if False:
            while True:
                i = 10
        self.training = train
        for layer in self.layers:
            if isinstance(layer, PhaseMixin):
                layer.is_training = train

    def shuffle_dataset(self):
        if False:
            i = 10
            return i + 15
        'Shuffle rows in the dataset.'
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        self.X = self.X.take(indices, axis=0)
        self.y = self.y.take(indices, axis=0)

    @property
    def n_layers(self):
        if False:
            while True:
                i = 10
        'Returns the number of layers.'
        return self._n_layers

    @property
    def n_params(self):
        if False:
            while True:
                i = 10
        'Return the number of trainable parameters.'
        return sum([layer.parameters.n_params for layer in self.parametric_layers])

    def reset(self):
        if False:
            print('Hello World!')
        self._initialized = False