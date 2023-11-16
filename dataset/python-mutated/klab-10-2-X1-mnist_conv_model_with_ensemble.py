from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
import numpy as np
from keras.utils import to_categorical
(train_data, test_data) = mnist.load_data()
x_train = train_data[0]
x_train = np.asarray([np.reshape(data, (28, 28, 1)) for data in x_train])
y_train = to_categorical(train_data[1])
x_test = test_data[0]
x_test = np.asarray([np.reshape(data, (28, 28, 1)) for data in x_test])
y_test_pure = test_data[1]
y_test = to_categorical(test_data[1])

def make_conv_model_list():
    if False:
        for i in range(10):
            print('nop')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(2, 2))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(64, (3, 3), use_bias=True, padding='same', activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model
ensemble_model_list = []
ensemble_size = 5
for i in range(ensemble_size):
    this_model = make_conv_model_list()
    print('i_th model training')
    this_model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(x_test, y_test))
    ensemble_model_list.append(this_model)
all_predicted = np.zeros(y_test.shape)
for one in ensemble_model_list:
    all_predicted = all_predicted + one.predict(x_test)
final_res = []
for one in all_predicted:
    final_res.append(np.argmax(one))
final_res = np.asarray(final_res)
entire = y_test_pure.shape[0]
ele = 0
for one in range(entire):
    if y_test_pure[one] == final_res[one]:
        ele = ele + 1
res_acc = ele / entire
print('fianl result test accuracy:', res_acc)