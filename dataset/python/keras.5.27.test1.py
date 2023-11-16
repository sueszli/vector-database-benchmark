import tensorflow as tf
import numpy as np
from ipywidgets import Box
from tensorflow.keras import layers
from tensorflow.python.keras.layers import MaxPooling2D, Convolution2D, LeakyReLU, Dense
from tensorflow.python.pywrap_tensorflow_internal import Flatten

print(tf.VERSION)
print(tf.keras.__version__)
model=tf.keras.Sequential()
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
# Create a sigmoid layer:
#layers.Dense(64, activation='sigmoid')
# Or:
#layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
#layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
#layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
#layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
#layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#一千个浮点数，0-32随机选择
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

#model.fit(data, labels, epochs=10, batch_size=32,
          #validation_data=(val_data, val_labels))

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
#model.fit(dataset, epochs=10, steps_per_epoch=30)
model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
result = model.predict(data, batch_size=32)
print(result)
