from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import SGD
batch_size = 128
epochs = 20
reg = 0.0001
A = load_npz('Atrain.npz')
A_test = load_npz('Atest.npz')
mask = (A > 0) * 1.0
mask_test = (A_test > 0) * 1.0
A_copy = A.copy()
mask_copy = mask.copy()
A_test_copy = A_test.copy()
mask_test_copy = mask_test.copy()
(N, M) = A.shape
print('N:', N, 'M:', M)
print('N // batch_size:', N // batch_size)
mu = A.sum() / mask.sum()
print('mu:', mu)
i = Input(shape=(M,))
x = Dropout(0.7)(i)
x = Dense(700, activation='tanh', kernel_regularizer=l2(reg))(x)
x = Dense(M, kernel_regularizer=l2(reg))(x)

def custom_loss(y_true, y_pred):
    if False:
        i = 10
        return i + 15
    mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
    diff = y_pred - y_true
    sqdiff = diff * diff * mask
    sse = K.sum(K.sum(sqdiff))
    n = K.sum(K.sum(mask))
    return sse / n

def generator(A, M):
    if False:
        print('Hello World!')
    while True:
        (A, M) = shuffle(A, M)
        for i in range(A.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, A.shape[0])
            a = A[i * batch_size:upper].toarray()
            m = M[i * batch_size:upper].toarray()
            a = a - mu * m
            noisy = a
            yield (noisy, a)

def test_generator(A, M, A_test, M_test):
    if False:
        for i in range(10):
            print('nop')
    while True:
        for i in range(A.shape[0] // batch_size + 1):
            upper = min((i + 1) * batch_size, A.shape[0])
            a = A[i * batch_size:upper].toarray()
            m = M[i * batch_size:upper].toarray()
            at = A_test[i * batch_size:upper].toarray()
            mt = M_test[i * batch_size:upper].toarray()
            a = a - mu * m
            at = at - mu * mt
            yield (a, at)
model = Model(i, x)
model.compile(loss=custom_loss, optimizer=SGD(lr=0.08, momentum=0.9), metrics=[custom_loss])
r = model.fit(generator(A, mask), validation_data=test_generator(A_copy, mask_copy, A_test_copy, mask_test_copy), epochs=epochs, steps_per_epoch=A.shape[0] // batch_size + 1, validation_steps=A_test.shape[0] // batch_size + 1)
print(r.history.keys())
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test loss')
plt.legend()
plt.show()
plt.plot(r.history['custom_loss'], label='train mse')
plt.plot(r.history['val_custom_loss'], label='test mse')
plt.legend()
plt.show()