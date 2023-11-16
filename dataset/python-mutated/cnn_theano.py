from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime
from benchmark import get_data, error_rate

def relu(a):
    if False:
        print('Hello World!')
    return a * (a > 0)

def convpool(X, W, b, poolsize=(2, 2)):
    if False:
        for i in range(10):
            print('nop')
    conv_out = conv2d(input=X, filters=W)
    pooled_out = pool.pool_2d(input=conv_out, ws=poolsize, ignore_border=True)
    return relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

def init_filter(shape, poolsz):
    if False:
        for i in range(10):
            print('nop')
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[1:]))
    return w.astype(np.float32)

def rearrange(X):
    if False:
        for i in range(10):
            print('nop')
    return (X.transpose(3, 2, 0, 1) / 255).astype(np.float32)

def main():
    if False:
        print('Hello World!')
    (train, test) = get_data()
    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    del train
    (Xtrain, Ytrain) = shuffle(Xtrain, Ytrain)
    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten() - 1
    del test
    max_iter = 6
    print_period = 10
    lr = np.float32(0.001)
    mu = np.float32(0.9)
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz
    M = 500
    K = 10
    poolsz = (2, 2)
    W1_shape = (20, 3, 5, 5)
    W1_init = init_filter(W1_shape, poolsz)
    b1_init = np.zeros(W1_shape[0], dtype=np.float32)
    W2_shape = (50, 20, 5, 5)
    W2_init = init_filter(W2_shape, poolsz)
    b2_init = np.zeros(W2_shape[0], dtype=np.float32)
    W3_init = np.random.randn(W2_shape[0] * 5 * 5, M) / np.sqrt(W2_shape[0] * 5 * 5 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)
    X = T.tensor4('X', dtype='float32')
    Y = T.ivector('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
    W3 = theano.shared(W3_init.astype(np.float32), 'W3')
    b3 = theano.shared(b3_init, 'b3')
    W4 = theano.shared(W4_init.astype(np.float32), 'W4')
    b4 = theano.shared(b4_init, 'b4')
    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)
    pY = T.nnet.softmax(Z3.dot(W4) + b4)
    cost = -T.log(pY[T.arange(Y.shape[0]), Y]).mean()
    prediction = T.argmax(pY, axis=1)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    dparams = [theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in params]
    updates = []
    grads = T.grad(cost, params)
    for (p, dp, g) in zip(params, dparams, grads):
        dp_update = mu * dp - lr * g
        p_update = p + dp_update
        updates.append((dp, dp_update))
        updates.append((p, p_update))
    train = theano.function(inputs=[X, Y], updates=updates)
    get_prediction = theano.function(inputs=[X, Y], outputs=[cost, prediction])
    t0 = datetime.now()
    costs = []
    for i in range(max_iter):
        (Xtrain, Ytrain) = shuffle(Xtrain, Ytrain)
        for j in range(n_batches):
            Xbatch = Xtrain[j * batch_sz:j * batch_sz + batch_sz,]
            Ybatch = Ytrain[j * batch_sz:j * batch_sz + batch_sz,]
            train(Xbatch, Ybatch)
            if j % print_period == 0:
                (cost_val, prediction_val) = get_prediction(Xtest, Ytest)
                err = error_rate(prediction_val, Ytest)
                print('Cost / err at iteration i=%d, j=%d: %.3f / %.3f' % (i, j, cost_val, err))
                costs.append(cost_val)
    print('Elapsed time:', datetime.now() - t0)
    plt.plot(costs)
    plt.show()
if __name__ == '__main__':
    main()