"""
Title: Knowledge Distillation
Author: [Kenneth Borup](https://twitter.com/Kennethborup)
Converted to Keras 3 by: [Md Awsafur Rahman](https://awsaf49.github.io)
Date created: 2020/09/01
Last modified: 2020/09/01
Description: Implementation of classical Knowledge Distillation.
Accelerator: GPU
"""
'\n## Introduction to Knowledge Distillation\n\nKnowledge Distillation is a procedure for model\ncompression, in which a small (student) model is trained to match a large pre-trained\n(teacher) model. Knowledge is transferred from the teacher model to the student\nby minimizing a loss function, aimed at matching softened teacher logits as well as\nground-truth labels.\n\nThe logits are softened by applying a "temperature" scaling function in the softmax,\neffectively smoothing out the probability distribution and revealing\ninter-class relationships learned by the teacher.\n\n**Reference:**\n\n- [Hinton et al. (2015)](https://arxiv.org/abs/1503.02531)\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import layers
from keras import ops
import tensorflow as tf
import numpy as np
'\n## Construct `Distiller()` class\n\nThe custom `Distiller()` class, overrides the `Model` methods `compile`, `compute_loss`,\nand `call`. In order to use the distiller, we need:\n\n- A trained teacher model\n- A student model to train\n- A student loss function on the difference between student predictions and ground-truth\n- A distillation loss function, along with a `temperature`, on the difference between the\nsoft student predictions and the soft teacher labels\n- An `alpha` factor to weight the student and distillation loss\n- An optimizer for the student and (optional) metrics to evaluate performance\n\nIn the `compute_loss` method, we perform a forward pass of both the teacher and student,\ncalculate the loss with weighting of the `student_loss` and `distillation_loss` by `alpha`\nand `1 - alpha`, respectively. Note: only the student weights are updated.\n'

class Distiller(keras.Model):

    def __init__(self, student, teacher):
        if False:
            return 10
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        if False:
            i = 10
            return i + 15
        'Configure the distiller.\n\n        Args:\n            optimizer: Keras optimizer for the student weights\n            metrics: Keras metrics for evaluation\n            student_loss_fn: Loss function of difference between student\n                predictions and ground-truth\n            distillation_loss_fn: Loss function of difference between soft\n                student predictions and soft teacher predictions\n            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn\n            temperature: Temperature for softening probability distributions.\n                Larger temperature gives softer distributions.\n        '
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        if False:
            print('Hello World!')
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        distillation_loss = self.distillation_loss_fn(ops.softmax(teacher_pred / self.temperature, axis=1), ops.softmax(y_pred / self.temperature, axis=1)) * self.temperature ** 2
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        if False:
            print('Hello World!')
        return self.student(x)
'\n## Create student and teacher models\n\nInitialy, we create a teacher model and a smaller student model. Both models are\nconvolutional neural networks and created using `Sequential()`,\nbut could be any Keras model.\n'
teacher = keras.Sequential([keras.Input(shape=(28, 28, 1)), layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'), layers.LeakyReLU(negative_slope=0.2), layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'), layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'), layers.Flatten(), layers.Dense(10)], name='teacher')
student = keras.Sequential([keras.Input(shape=(28, 28, 1)), layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'), layers.LeakyReLU(negative_slope=0.2), layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'), layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'), layers.Flatten(), layers.Dense(10)], name='student')
student_scratch = keras.models.clone_model(student)
'\n## Prepare the dataset\n\nThe dataset used for training the teacher and distilling the teacher is\n[MNIST](https://keras.io/api/datasets/mnist/), and the procedure would be equivalent for\nany other\ndataset, e.g. [CIFAR-10](https://keras.io/api/datasets/cifar10/), with a suitable choice\nof models. Both the student and teacher are trained on the training set and evaluated on\nthe test set.\n'
batch_size = 64
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = x_test.astype('float32') / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
'\n## Train the teacher\n\nIn knowledge distillation we assume that the teacher is trained and fixed. Thus, we start\nby training the teacher model on the training set in the usual way.\n'
teacher.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy()])
teacher.fit(x_train, y_train, epochs=5)
teacher.evaluate(x_test, y_test)
'\n## Distill teacher to student\n\nWe have already trained the teacher model, and we only need to initialize a\n`Distiller(student, teacher)` instance, `compile()` it with the desired losses,\nhyperparameters and optimizer, and distill the teacher to the student.\n'
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.SparseCategoricalAccuracy()], student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True), distillation_loss_fn=keras.losses.KLDivergence(), alpha=0.1, temperature=10)
distiller.fit(x_train, y_train, epochs=3)
distiller.evaluate(x_test, y_test)
'\n## Train student from scratch for comparison\n\nWe can also train an equivalent student model from scratch without the teacher, in order\nto evaluate the performance gain obtained by knowledge distillation.\n'
student_scratch.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy()])
student_scratch.fit(x_train, y_train, epochs=3)
student_scratch.evaluate(x_test, y_test)
'\nIf the teacher is trained for 5 full epochs and the student is distilled on this teacher\nfor 3 full epochs, you should in this example experience a performance boost compared to\ntraining the same student model from scratch, and even compared to the teacher itself.\nYou should expect the teacher to have accuracy around 97.6%, the student trained from\nscratch should be around 97.6%, and the distilled student should be around 98.1%. Remove\nor try out different seeds to use different weight initializations.\n'