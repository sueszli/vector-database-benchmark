import numpy as np
import mxnet as mx
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
np.random.seed(777)
mx.random.seed(777)
timesteps = seq_length = 7
batch_size = 32
data_dim = 5

def build_sym(seq_len, use_cudnn=False):
    if False:
        print('Hello World!')
    'Build the symbol for stock-price prediction\n\n    Parameters\n    ----------\n    seq_len : int\n    use_cudnn : bool, optional\n        Whether to use the LSTM implemented in cudnn, will be faster than the original version\n\n    Returns\n    -------\n    pred : mx.sym.Symbol\n        The prediction result\n    '
    data = mx.sym.var('data')
    target = mx.sym.var('target')
    data = mx.sym.transpose(data, axes=(1, 0, 2))
    if use_cudnn:
        lstm1 = mx.rnn.FusedRNNCell(num_hidden=5, mode='lstm', prefix='lstm1_')
        lstm2 = mx.rnn.FusedRNNCell(num_hidden=10, mode='lstm', prefix='lstm2_', get_next_state=True)
    else:
        lstm1 = mx.rnn.LSTMCell(num_hidden=5, prefix='lstm1_')
        lstm2 = mx.rnn.LSTMCell(num_hidden=10, prefix='lstm2_')
    (L1, _) = lstm1.unroll(length=seq_len, inputs=data, merge_outputs=True, layout='TNC')
    L1 = mx.sym.Dropout(L1, p=0.2)
    (_, L2_states) = lstm2.unroll(length=seq_len, inputs=L1, merge_outputs=True, layout='TNC')
    L2 = mx.sym.reshape(L2_states[0], shape=(-1, 0), reverse=True)
    pred = mx.sym.FullyConnected(L2, num_hidden=1, name='pred')
    pred = mx.sym.LinearRegressionOutput(data=pred, label=target)
    return pred
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)
x = xy
y = xy[:, [-1]]
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    print(_x, '->', _y)
    dataX.append(_x)
    dataY.append(_y)
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
(trainX, testX) = (np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)]))
(trainY, testY) = (np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)]))

def train_eval_net(use_cudnn):
    if False:
        i = 10
        return i + 15
    pred = build_sym(seq_len=seq_length, use_cudnn=use_cudnn)
    net = mx.mod.Module(symbol=pred, data_names=['data'], label_names=['target'], context=mx.gpu())
    train_iter = mx.io.NDArrayIter(data=trainX, label=trainY, data_name='data', label_name='target', batch_size=batch_size, shuffle=True)
    test_iter = mx.io.NDArrayIter(data=testX, label=testY, data_name='data', label_name='target', batch_size=batch_size)
    net.fit(train_data=train_iter, eval_data=test_iter, initializer=mx.init.Xavier(rnd_type='gaussian', magnitude=1), optimizer='adam', optimizer_params={'learning_rate': 0.001}, eval_metric='mse', num_epoch=200)
    testPredict = net.predict(test_iter).asnumpy()
    mse = np.mean((testPredict - testY) ** 2)
    return (testPredict, mse)
import time
print('Begin to train LSTM with CUDNN acceleration...')
begin = time.time()
(cudnn_pred, cudnn_mse) = train_eval_net(use_cudnn=True)
end = time.time()
cudnn_time_spent = end - begin
print('Done!')
print('Begin to train LSTM without CUDNN acceleration...')
begin = time.time()
(normal_pred, normal_mse) = train_eval_net(use_cudnn=False)
end = time.time()
normal_time_spent = end - begin
print('Done!')
print('CUDNN time spent: %g, test mse: %g' % (cudnn_time_spent, cudnn_mse))
print('NoCUDNN time spent: %g, test mse: %g' % (normal_time_spent, normal_mse))
plt.close('all')
fig = plt.figure()
plt.plot(testY, label='Groud Truth')
plt.plot(cudnn_pred, label='With cuDNN')
plt.plot(normal_pred, label='Without cuDNN')
plt.legend()
plt.show()
'\nCUDNN time spent: 10.0955, test mse: 0.00721571\nNoCUDNN time spent: 38.9882, test mse: 0.00565724\n'