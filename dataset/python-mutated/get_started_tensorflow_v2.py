"""
The script demonstrates a simple example of using ART with TensorFlow v2.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
((x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value) = load_mnist()
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

class TensorFlowModel(Model):
    """
    Standard TensorFlow model for unit testing.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(TensorFlowModel, self).__init__()
        self.conv1 = Conv2D(filters=4, kernel_size=5, activation='relu')
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation='relu')
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation='relu')
        self.logits = Dense(10, activation='linear')

    def call(self, x):
        if False:
            while True:
                i = 10
        '\n        Call function to evaluate the model.\n\n        :param x: Input to the model\n        :return: Prediction of the model\n        '
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.logits(x)
        return x
model = TensorFlowModel()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
classifier = TensorFlowV2Classifier(model=model, loss_object=loss_object, optimizer=optimizer, nb_classes=10, input_shape=(28, 28, 1), clip_values=(0, 1))
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on benign test examples: {}%'.format(accuracy * 100))
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=x_test)
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))