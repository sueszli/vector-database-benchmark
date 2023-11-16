"""This file is only here to speed up the execution of notebooks.

It contains a subset of the code defined in simple_rnn.ipynb and
lstm_text.ipynb, in particular the code compiling Theano function.
Executing this script first will populate the cache of compiled C code,
which will make subsequent compilations faster.

The use case is to run this script in the background when a demo VM
such as the one for NVIDIA's qwikLABS, so that the compilation phase
started from the notebooks is faster.

"""
import numpy
import theano
import theano.tensor as T
from theano import config
from theano.tensor.nnet import categorical_crossentropy
floatX = theano.config.floatX

class SimpleRNN(object):

    def __init__(self, input_dim, recurrent_dim):
        if False:
            i = 10
            return i + 15
        w_xh = numpy.random.normal(0, 0.01, (input_dim, recurrent_dim))
        w_hh = numpy.random.normal(0, 0.02, (recurrent_dim, recurrent_dim))
        self.w_xh = theano.shared(numpy.asarray(w_xh, dtype=floatX), name='w_xh')
        self.w_hh = theano.shared(numpy.asarray(w_hh, dtype=floatX), name='w_hh')
        self.b_h = theano.shared(numpy.zeros((recurrent_dim,), dtype=floatX), name='b_h')
        self.parameters = [self.w_xh, self.w_hh, self.b_h]

    def _step(self, input_t, previous):
        if False:
            for i in range(10):
                print('nop')
        return T.tanh(T.dot(previous, self.w_hh) + input_t)

    def __call__(self, x):
        if False:
            i = 10
            return i + 15
        x_w_xh = T.dot(x, self.w_xh) + self.b_h
        (result, updates) = theano.scan(self._step, sequences=[x_w_xh], outputs_info=[T.zeros_like(self.b_h)])
        return result
w_ho_np = numpy.random.normal(0, 0.01, (15, 1))
w_ho = theano.shared(numpy.asarray(w_ho_np, dtype=floatX), name='w_ho')
b_o = theano.shared(numpy.zeros((1,), dtype=floatX), name='b_o')
x = T.matrix('x')
my_rnn = SimpleRNN(1, 15)
hidden = my_rnn(x)
prediction = T.dot(hidden, w_ho) + b_o
parameters = my_rnn.parameters + [w_ho, b_o]
l2 = sum(((p ** 2).sum() for p in parameters))
mse = T.mean((prediction[:-1] - x[1:]) ** 2)
cost = mse + 0.0001 * l2
gradient = T.grad(cost, wrt=parameters)
lr = 0.3
updates = [(par, par - lr * gra) for (par, gra) in zip(parameters, gradient)]
update_model = theano.function([x], cost, updates=updates)
get_cost = theano.function([x], mse)
predict = theano.function([x], prediction)
get_hidden = theano.function([x], hidden)
get_gradient = theano.function([x], gradient)
predict = theano.function([x], prediction)
x_t = T.vector()
h_p = T.vector()
preactivation = T.dot(x_t, my_rnn.w_xh) + my_rnn.b_h
h_t = my_rnn._step(preactivation, h_p)
o_t = T.dot(h_t, w_ho) + b_o
single_step = theano.function([x_t, h_p], [o_t, h_t])

def gauss_weight(rng, ndim_in, ndim_out=None, sd=0.005):
    if False:
        while True:
            i = 10
    if ndim_out is None:
        ndim_out = ndim_in
    W = rng.randn(ndim_in, ndim_out) * sd
    return numpy.asarray(W, dtype=config.floatX)

def index_dot(indices, w):
    if False:
        return 10
    return w[indices.flatten()]

class LstmLayer:

    def __init__(self, rng, input, mask, n_in, n_h):
        if False:
            for i in range(10):
                print('nop')
        self.W_i = theano.shared(gauss_weight(rng, n_in, n_h), 'W_i', borrow=True)
        self.W_f = theano.shared(gauss_weight(rng, n_in, n_h), 'W_f', borrow=True)
        self.W_c = theano.shared(gauss_weight(rng, n_in, n_h), 'W_c', borrow=True)
        self.W_o = theano.shared(gauss_weight(rng, n_in, n_h), 'W_o', borrow=True)
        self.U_i = theano.shared(gauss_weight(rng, n_h), 'U_i', borrow=True)
        self.U_f = theano.shared(gauss_weight(rng, n_h), 'U_f', borrow=True)
        self.U_c = theano.shared(gauss_weight(rng, n_h), 'U_c', borrow=True)
        self.U_o = theano.shared(gauss_weight(rng, n_h), 'U_o', borrow=True)
        self.b_i = theano.shared(numpy.zeros((n_h,), dtype=config.floatX), 'b_i', borrow=True)
        self.b_f = theano.shared(numpy.zeros((n_h,), dtype=config.floatX), 'b_f', borrow=True)
        self.b_c = theano.shared(numpy.zeros((n_h,), dtype=config.floatX), 'b_c', borrow=True)
        self.b_o = theano.shared(numpy.zeros((n_h,), dtype=config.floatX), 'b_o', borrow=True)
        self.params = [self.W_i, self.W_f, self.W_c, self.W_o, self.U_i, self.U_f, self.U_c, self.U_o, self.b_i, self.b_f, self.b_c, self.b_o]
        outputs_info = [T.zeros((input.shape[1], n_h)), T.zeros((input.shape[1], n_h))]
        (rval, updates) = theano.scan(self._step, sequences=[mask, input], outputs_info=outputs_info)
        self.output = rval[0]

    def _step(self, m_, x_, h_, c_):
        if False:
            for i in range(10):
                print('nop')
        i_preact = index_dot(x_, self.W_i) + T.dot(h_, self.U_i) + self.b_i
        i = T.nnet.sigmoid(i_preact)
        f_preact = index_dot(x_, self.W_f) + T.dot(h_, self.U_f) + self.b_f
        f = T.nnet.sigmoid(f_preact)
        o_preact = index_dot(x_, self.W_o) + T.dot(h_, self.U_o) + self.b_o
        o = T.nnet.sigmoid(o_preact)
        c_preact = index_dot(x_, self.W_c) + T.dot(h_, self.U_c) + self.b_c
        c = T.tanh(c_preact)
        c = f * c_ + i * c
        c = m_[:, None] * c + (1.0 - m_)[:, None] * c_
        h = o * T.tanh(c)
        h = m_[:, None] * h + (1.0 - m_)[:, None] * h_
        return (h, c)

def sequence_categorical_crossentropy(prediction, targets, mask):
    if False:
        for i in range(10):
            print('nop')
    prediction_flat = prediction.reshape((prediction.shape[0] * prediction.shape[1], prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)

class LogisticRegression(object):

    def __init__(self, rng, input, n_in, n_out):
        if False:
            for i in range(10):
                print('nop')
        W = gauss_weight(rng, n_in, n_out)
        self.W = theano.shared(value=numpy.asarray(W, dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        energy = T.dot(input, self.W) + self.b
        energy_exp = T.exp(energy - T.max(energy, axis=2, keepdims=True))
        pmf = energy_exp / energy_exp.sum(axis=2, keepdims=True)
        self.p_y_given_x = pmf
        self.params = [self.W, self.b]
batch_size = 100
n_h = 50
rng = numpy.random.RandomState(12345)
x = T.lmatrix('x')
mask = T.matrix('mask')
recurrent_layer = LstmLayer(rng=rng, input=x, mask=mask, n_in=111, n_h=n_h)
logreg_layer = LogisticRegression(rng=rng, input=recurrent_layer.output[:-1], n_in=n_h, n_out=111)
cost = sequence_categorical_crossentropy(logreg_layer.p_y_given_x, x[1:], mask[1:]) / batch_size
params = logreg_layer.params + recurrent_layer.params
grads = T.grad(cost, params)
learning_rate = 0.1
updates = [(param_i, param_i - learning_rate * grad_i) for (param_i, grad_i) in zip(params, grads)]
update_model = theano.function([x, mask], cost, updates=updates)
evaluate_model = theano.function([x, mask], cost)
x_t = T.iscalar()
h_p = T.vector()
c_p = T.vector()
(h_t, c_t) = recurrent_layer._step(T.ones(1), x_t, h_p, c_p)
energy = T.dot(h_t, logreg_layer.W) + logreg_layer.b
energy_exp = T.exp(energy - T.max(energy, axis=1, keepdims=True))
output = energy_exp / energy_exp.sum(axis=1, keepdims=True)
single_step = theano.function([x_t, h_p, c_p], [output, h_t, c_t])