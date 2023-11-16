"""
Title: Supervised Contrastive Learning
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2020/11/30
Last modified: 2020/11/30
Description: Using supervised contrastive learning for image classification.
Accelerator: GPU
"""
'\n## Introduction\n\n[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)\n(Prannay Khosla et al.) is a training methodology that outperforms\nsupervised training with crossentropy on classification tasks.\n\nEssentially, training an image classification model with Supervised Contrastive\nLearning is performed in two phases:\n\n1. Training an encoder to learn to produce vector representations of input images such\nthat representations of images in the same class will be more similar compared to\nrepresentations of images in different classes.\n2. Training a classifier on top of the frozen encoder.\n'
'\n## Setup\n'
import keras
from keras import ops
from keras import layers
from keras.applications.resnet_v2 import ResNet50V2
'\n## Prepare the data\n'
num_classes = 10
input_shape = (32, 32, 3)
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')
'\n## Using image data augmentation\n'
data_augmentation = keras.Sequential([layers.Normalization(), layers.RandomFlip('horizontal'), layers.RandomRotation(0.02)])
data_augmentation.layers[0].adapt(x_train)
'\n## Build the encoder model\n\nThe encoder model takes the image as input and turns it into a 2048-dimensional\nfeature vector.\n'

def create_encoder():
    if False:
        for i in range(10):
            print('nop')
    resnet = ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name='cifar10-encoder')
    return model
encoder = create_encoder()
encoder.summary()
learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05
'\n## Build the classification model\n\nThe classification model adds a fully-connected layer on top of the encoder,\nplus a softmax layer with the target classes.\n'

def create_classifier(encoder, trainable=True):
    if False:
        for i in range(10):
            print('nop')
    for layer in encoder.layers:
        layer.trainable = trainable
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation='relu')(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name='cifar10-classifier')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model
'\n## Define npairs loss function\n'

def npairs_loss(y_true, y_pred):
    if False:
        for i in range(10):
            print('nop')
    'Computes the npairs loss between `y_true` and `y_pred`.\n\n    Npairs loss expects paired data where a pair is composed of samples from\n    the same labels and each pairs in the minibatch have different labels.\n    The loss takes each row of the pair-wise similarity matrix, `y_pred`,\n    as logits and the remapped multi-class labels, `y_true`, as labels.\n\n\n    See:\n    http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf\n\n    Args:\n      y_true: Ground truth values, of shape `[batch_size]` of multi-class\n        labels.\n      y_pred: Predicted values of shape `[batch_size, batch_size]` of\n        similarity matrix between embedding matrices.\n\n    Returns:\n      npairs_loss: float scalar.\n    '
    y_pred = ops.cast(y_pred, 'float32')
    y_true = ops.cast(y_true, y_pred.dtype)
    y_true = ops.cast(ops.equal(y_true, ops.transpose(y_true)), y_pred.dtype)
    y_true /= ops.sum(y_true, 1, keepdims=True)
    loss = ops.categorical_crossentropy(y_true, y_pred, from_logits=True)
    return ops.mean(loss)
'\n## Experiment 1: Train the baseline classification model\n\nIn this experiment, a baseline classifier is trained as usual, i.e., the\nencoder and the classifier parts are trained together as a single model\nto minimize the crossentropy loss.\n'
encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()
history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)
accuracy = classifier.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {round(accuracy * 100, 2)}%')
'\n## Experiment 2: Use supervised contrastive learning\n\nIn this experiment, the model is trained in two phases. In the first phase,\nthe encoder is pretrained to optimize the supervised contrastive loss,\ndescribed in [Prannay Khosla et al.](https://arxiv.org/abs/2004.11362).\n\nIn the second phase, the classifier is trained using the trained encoder with\nits weights freezed; only the weights of fully-connected layers with the\nsoftmax are optimized.\n\n### 1. Supervised contrastive learning loss function\n'

class SupervisedContrastiveLoss(keras.losses.Loss):

    def __init__(self, temperature=1, name=None):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        if False:
            print('Hello World!')
        feature_vectors_normalized = keras.utils.normalize(feature_vectors, axis=1, order=2)
        logits = ops.divide(ops.matmul(feature_vectors_normalized, ops.transpose(feature_vectors_normalized)), self.temperature)
        return npairs_loss(ops.squeeze(labels), logits)

def add_projection_head(encoder):
    if False:
        return 10
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation='relu')(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name='cifar-encoder_with_projection-head')
    return model
'\n### 2. Pretrain the encoder\n'
encoder = create_encoder()
encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=SupervisedContrastiveLoss(temperature))
encoder_with_projection_head.summary()
history = encoder_with_projection_head.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)
'\n### 3. Train the classifier with the frozen encoder\n'
classifier = create_classifier(encoder, trainable=False)
history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs)
accuracy = classifier.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {round(accuracy * 100, 2)}%')
'\nWe get to an improved test accuracy.\n'
'\n## Conclusion\n\nAs shown in the experiments, using the supervised contrastive learning technique\noutperformed the conventional technique in terms of the test accuracy. Note that\nthe same training budget (i.e., number of epochs) was given to each technique.\nSupervised contrastive learning pays off when the encoder involves a complex\narchitecture, like ResNet, and multi-class problems with many labels.\nIn addition, large batch sizes and multi-layer projection heads\nimprove its effectiveness. See the [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)\npaper for more details.\n\n'