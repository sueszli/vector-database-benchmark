"""
Title: Gradient Centralization for Better Training Performance
Author: [Rishit Dagli](https://github.com/Rishit-dagli)
Converted to Keras 3 by: [Muhammad Anas Raza](https://anasrz.com)
Date created: 06/18/21
Last modified: 07/25/23
Description: Implement Gradient Centralization to improve training performance of DNNs.
Accelerator: GPU
"""
"\n## Introduction\n\nThis example implements [Gradient Centralization](https://arxiv.org/abs/2004.01461), a\nnew optimization technique for Deep Neural Networks by Yong et al., and demonstrates it\non Laurence Moroney's [Horses or Humans\nDataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans). Gradient\nCentralization can both speedup training process and improve the final generalization\nperformance of DNNs. It operates directly on gradients by centralizing the gradient\nvectors to have zero mean. Gradient Centralization morever improves the Lipschitzness of\nthe loss function and its gradient so that the training process becomes more efficient\nand stable.\n\nThis example requires `tensorflow_datasets` which can\nbe installed with this command:\n\n```\npip install tensorflow-datasets\n```\n"
'\n## Setup\n'
from time import time
import keras
from keras import layers
from keras.optimizers import RMSprop
from keras import ops
from tensorflow import data as tf_data
import tensorflow_datasets as tfds
'\n## Prepare the data\n\nFor this example, we will be using the [Horses or Humans\ndataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans).\n'
num_classes = 2
input_shape = (300, 300, 3)
dataset_name = 'horses_or_humans'
batch_size = 128
AUTOTUNE = tf_data.AUTOTUNE
((train_ds, test_ds), metadata) = tfds.load(name=dataset_name, split=[tfds.Split.TRAIN, tfds.Split.TEST], with_info=True, as_supervised=True)
print(f"Image shape: {metadata.features['image'].shape}")
print(f"Training images: {metadata.splits['train'].num_examples}")
print(f"Test images: {metadata.splits['test'].num_examples}")
'\n## Use Data Augmentation\n\nWe will rescale the data to `[0, 1]` and perform simple augmentations to our data.\n'
rescale = layers.Rescaling(1.0 / 255)
data_augmentation = [layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.3), layers.RandomZoom(0.2)]

def apply_aug(x):
    if False:
        for i in range(10):
            print('nop')
    for aug in data_augmentation:
        x = aug(x)
    return x

def prepare(ds, shuffle=False, augment=False):
    if False:
        for i in range(10):
            print('nop')
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(lambda x, y: (apply_aug(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)
'\nRescale and augment the data\n'
train_ds = prepare(train_ds, shuffle=True, augment=True)
test_ds = prepare(test_ds)
'\n## Define a model\n\nIn this section we will define a Convolutional neural network.\n'
model = keras.Sequential([layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)), layers.MaxPooling2D(2, 2), layers.Conv2D(32, (3, 3), activation='relu'), layers.Dropout(0.5), layers.MaxPooling2D(2, 2), layers.Conv2D(64, (3, 3), activation='relu'), layers.Dropout(0.5), layers.MaxPooling2D(2, 2), layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D(2, 2), layers.Conv2D(64, (3, 3), activation='relu'), layers.MaxPooling2D(2, 2), layers.Flatten(), layers.Dropout(0.5), layers.Dense(512, activation='relu'), layers.Dense(1, activation='sigmoid')])
'\n## Implement Gradient Centralization\n\nWe will now\nsubclass the `RMSProp` optimizer class modifying the\n`keras.optimizers.Optimizer.get_gradients()` method where we now implement Gradient\nCentralization. On a high level the idea is that let us say we obtain our gradients\nthrough back propogation for a Dense or Convolution layer we then compute the mean of the\ncolumn vectors of the weight matrix, and then remove the mean from each column vector.\n\nThe experiments in [this paper](https://arxiv.org/abs/2004.01461) on various\napplications, including general image classification, fine-grained image classification,\ndetection and segmentation and Person ReID demonstrate that GC can consistently improve\nthe performance of DNN learning.\n\nAlso, for simplicity at the moment we are not implementing gradient cliiping functionality,\nhowever this quite easy to implement.\n\nAt the moment we are just creating a subclass for the `RMSProp` optimizer\nhowever you could easily reproduce this for any other optimizer or on a custom\noptimizer in the same way. We will be using this class in the later section when\nwe train a model with Gradient Centralization.\n'

class GCRMSprop(RMSprop):

    def get_gradients(self, loss, params):
        if False:
            return 10
        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= ops.mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)
        return grads
optimizer = GCRMSprop(learning_rate=0.0001)
'\n## Training utilities\n\nWe will also create a callback which allows us to easily measure the total training time\nand the time taken for each epoch since we are interested in comparing the effect of\nGradient Centralization on the model we built above.\n'

class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        if False:
            print('Hello World!')
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        if False:
            print('Hello World!')
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        if False:
            return 10
        self.times.append(time() - self.epoch_time_start)
'\n## Train the model without GC\n\nWe now train the model we built earlier without Gradient Centralization which we can\ncompare to the training performance of the model trained with Gradient Centralization.\n'
time_callback_no_gc = TimeHistory()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])
model.summary()
'\nWe also save the history since we later want to compare our model trained with and not\ntrained with Gradient Centralization\n'
history_no_gc = model.fit(train_ds, epochs=10, verbose=1, callbacks=[time_callback_no_gc])
'\n## Train the model with GC\n\nWe will now train the same model, this time using Gradient Centralization,\nnotice our optimizer is the one using Gradient Centralization this time.\n'
time_callback_gc = TimeHistory()
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
history_gc = model.fit(train_ds, epochs=10, verbose=1, callbacks=[time_callback_gc])
'\n## Comparing performance\n'
print('Not using Gradient Centralization')
print(f"Loss: {history_no_gc.history['loss'][-1]}")
print(f"Accuracy: {history_no_gc.history['accuracy'][-1]}")
print(f'Training Time: {sum(time_callback_no_gc.times)}')
print('Using Gradient Centralization')
print(f"Loss: {history_gc.history['loss'][-1]}")
print(f"Accuracy: {history_gc.history['accuracy'][-1]}")
print(f'Training Time: {sum(time_callback_gc.times)}')
"\nReaders are encouraged to try out Gradient Centralization on different datasets from\ndifferent domains and experiment with it's effect. You are strongly advised to check out\nthe [original paper](https://arxiv.org/abs/2004.01461) as well - the authors present\nseveral studies on Gradient Centralization showing how it can improve general\nperformance, generalization, training time as well as more efficient.\n\nMany thanks to [Ali Mustufa Shaikh](https://github.com/ialimustufa) for reviewing this\nimplementation.\n"