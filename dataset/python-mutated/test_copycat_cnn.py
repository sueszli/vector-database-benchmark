from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import keras
import keras.backend as k
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from art.attacks.extraction.copycat_cnn import CopycatCNN
from art.estimators.classification.keras import KerasClassifier
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowClassifier
from tests.utils import TestBase, get_image_classifier_kr, get_image_classifier_pt, get_image_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt, get_tabular_classifier_tf, master_seed
logger = logging.getLogger(__name__)
NB_EPOCHS = 20
NB_STOLEN = 100

class TestCopycatCNN(TestBase):
    """
    A unittest class for testing the CopycatCNN attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        master_seed(seed=1234)
        super().setUpClass()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2.')
    def test_tensorflow_classifier(self):
        if False:
            return 10
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        (victim_tfc, sess) = get_image_classifier_tf()
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])
        conv = tf.layers.conv2d(input_ph, 1, 7, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 4, 4)
        flattened = tf.layers.flatten(conv)
        logits = tf.layers.dense(flattened, 10)
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        thieved_tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph, train=train, loss=loss, learning=None, sess=sess)
        copycat_cnn = CopycatCNN(classifier=victim_tfc, batch_size_query=self.batch_size, batch_size_fit=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_tfc = copycat_cnn.extract(x=self.x_train_mnist, thieved_classifier=thieved_tfc)
        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train_mnist[:100]), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train_mnist[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        if sess is not None:
            sess.close()
            tf.reset_default_graph()

    def test_keras_classifier(self):
        if False:
            i = 10
            return i + 15
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        victim_krc = get_image_classifier_kr()
        model = Sequential()
        model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        loss = keras.losses.categorical_crossentropy
        try:
            from keras.optimizers import Adam
            optimizer = Adam(lr=0.001)
        except ImportError:
            from keras.optimizers import adam_v2
            optimizer = adam_v2.Adam(lr=0.001)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        thieved_krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        copycat_cnn = CopycatCNN(classifier=victim_krc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_krc = copycat_cnn.extract(x=self.x_train_mnist, thieved_classifier=thieved_krc)
        victim_preds = np.argmax(victim_krc.predict(x=self.x_train_mnist[:100]), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train_mnist[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        k.clear_session()

    def test_pytorch_classifier(self):
        if False:
            return 10
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        x_train = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        victim_ptc = get_image_classifier_pt()

        class Model(nn.Module):
            """
            Create model for pytorch.
            """

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Model, self).__init__()
                self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
                self.pool = nn.MaxPool2d(4, 4)
                self.fullyconnected = nn.Linear(25, 10)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                '\n                Forward function to evaluate the model\n\n                :param x: Input to the model\n                :return: Prediction of the model\n                '
                x = self.conv(x)
                x = torch.nn.functional.relu(x)
                x = self.pool(x)
                x = x.reshape(-1, 25)
                x = self.fullyconnected(x)
                x = torch.nn.functional.softmax(x, dim=1)
                return x
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        thieved_ptc = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1))
        copycat_cnn = CopycatCNN(classifier=victim_ptc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_ptc = copycat_cnn.extract(x=x_train, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        with self.assertRaises(ValueError):
            _ = CopycatCNN(classifier=victim_ptc, batch_size_fit=-1, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        with self.assertRaises(ValueError):
            _ = CopycatCNN(classifier=victim_ptc, batch_size_fit=self.batch_size, batch_size_query=-1, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        with self.assertRaises(ValueError):
            _ = CopycatCNN(classifier=victim_ptc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=-1, nb_stolen=NB_STOLEN)
        with self.assertRaises(ValueError):
            _ = CopycatCNN(classifier=victim_ptc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=-1)
        with self.assertRaises(ValueError):
            _ = CopycatCNN(classifier=victim_ptc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN, use_probability='True')

class TestCopycatCNNVectors(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2.')
    def test_tensorflow_iris(self):
        if False:
            return 10
        '\n        First test for TensorFlow.\n        :return:\n        '
        (victim_tfc, sess) = get_tabular_classifier_tf()
        input_ph = tf.placeholder(tf.float32, shape=[None, 4])
        output_ph = tf.placeholder(tf.int32, shape=[None, 3])
        dense1 = tf.layers.dense(input_ph, 10)
        dense2 = tf.layers.dense(dense1, 10)
        logits = tf.layers.dense(dense2, 3)
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        thieved_tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph, train=train, loss=loss, learning=None, sess=sess, channels_first=True)
        copycat_cnn = CopycatCNN(classifier=victim_tfc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_tfc = copycat_cnn.extract(x=self.x_train_iris, thieved_classifier=thieved_tfc)
        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train_iris[:100]), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train_iris[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        if sess is not None:
            sess.close()
            tf.reset_default_graph()

    def test_keras_iris(self):
        if False:
            while True:
                i = 10
        '\n        Second test for Keras.\n        :return:\n        '
        victim_krc = get_tabular_classifier_kr()
        model = Sequential()
        model.add(Dense(10, input_shape=(4,), activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        try:
            from keras.optimizers import Adam
            optimizer = Adam(lr=0.001)
        except ImportError:
            from keras.optimizers import adam_v2
            optimizer = adam_v2.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        thieved_krc = KerasClassifier(model, clip_values=(0, 1), use_logits=False, channels_first=True)
        copycat_cnn = CopycatCNN(classifier=victim_krc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_krc = copycat_cnn.extract(x=self.x_train_iris, thieved_classifier=thieved_krc)
        victim_preds = np.argmax(victim_krc.predict(x=self.x_train_iris[:100]), axis=1)
        thieved_preds = np.argmax(thieved_krc.predict(x=self.x_train_iris[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
        k.clear_session()

    def test_pytorch_iris(self):
        if False:
            print('Hello World!')
        '\n        Third test for PyTorch.\n        :return:\n        '
        victim_ptc = get_tabular_classifier_pt()

        class Model(nn.Module):
            """
            Create Iris model for PyTorch.
            """

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Model, self).__init__()
                self.fully_connected1 = nn.Linear(4, 10)
                self.fully_connected2 = nn.Linear(10, 10)
                self.fully_connected3 = nn.Linear(10, 3)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.fully_connected1(x)
                x = self.fully_connected2(x)
                logit_output = self.fully_connected3(x)
                return logit_output
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        thieved_ptc = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(4,), nb_classes=3, clip_values=(0, 1), channels_first=True)
        copycat_cnn = CopycatCNN(classifier=victim_ptc, batch_size_fit=self.batch_size, batch_size_query=self.batch_size, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_ptc = copycat_cnn.extract(x=self.x_train_iris, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_iris[:100]), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_iris[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        self.assertGreater(acc, 0.3)
if __name__ == '__main__':
    unittest.main()