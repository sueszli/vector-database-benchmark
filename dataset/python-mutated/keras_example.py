from __future__ import print_function, division
from builtins import range
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.io import loadmat
from sklearn.utils import shuffle
from benchmark import get_data, error_rate

def rearrange(X):
    if False:
        for i in range(10):
            print('nop')
    return (X.transpose(3, 0, 1, 2) / 255.0).astype(np.float32)
(train, test) = get_data()
Xtrain = rearrange(train['X'])
Ytrain = train['y'].flatten() - 1
del train
Xtest = rearrange(test['X'])
Ytest = test['y'].flatten() - 1
del test
K = len(set(Ytrain))
i = Input(shape=Xtrain.shape[1:])
x = Conv2D(filters=20, kernel_size=(5, 5))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(filters=50, kernel_size=(5, 5))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(units=500)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)
model = Model(inputs=i, outputs=x)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=32)
print('Returned:', r)
print(r.history.keys())
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()