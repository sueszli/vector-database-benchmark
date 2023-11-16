from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import logging
from art.defences.preprocessor import Mixup, Cutout
from art.defences.trainer import DPInstaHideTrainer
from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
from tests.utils import ARTTestException
from tests.utils import get_image_classifier_hf
logger = logging.getLogger(__name__)

@pytest.fixture()
def get_mnist_classifier(framework):
    if False:
        print('Hello World!')

    def _get_classifier():
        if False:
            while True:
                i = 10
        if framework == 'pytorch':
            import torch
            model = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7), torch.nn.ReLU(), torch.nn.MaxPool2d(4, 4), torch.nn.Flatten(), torch.nn.Linear(25, 10))
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            classifier = PyTorchClassifier(model, loss=criterion, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10)
        elif framework == 'tensorflow2':
            import tensorflow as tf
            from tensorflow.keras import layers, Sequential
            model = Sequential()
            model.add(layers.Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D(pool_size=(4, 4)))
            model.add(layers.Flatten())
            model.add(layers.Dense(10))
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            classifier = TensorFlowV2Classifier(model, nb_classes=10, input_shape=(28, 28, 1), loss_object=loss_object, optimizer=optimizer)
        elif framework in ('keras', 'kerastf'):
            import tensorflow as tf
            from tensorflow.keras import layers, Sequential
            if tf.__version__[0] == '2':
                tf.compat.v1.disable_eager_execution()
            model = Sequential()
            model.add(layers.Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D(pool_size=(4, 4)))
            model.add(layers.Flatten())
            model.add(layers.Dense(10))
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=True)
        elif framework == 'huggingface':
            classifier = get_image_classifier_hf(from_logits=True)
        else:
            classifier = None
        return classifier
    return _get_classifier

@pytest.mark.only_with_platform('pytorch', 'tensorflow2', 'keras', 'kerastf', 'huggingface')
@pytest.mark.parametrize('noise', ['gaussian', 'laplacian', 'exponential'])
def test_dp_instahide_single_aug(art_warning, get_mnist_classifier, get_default_mnist_subset, get_default_cifar10_subset, noise, framework):
    if False:
        return 10
    classifier = get_mnist_classifier()
    ((x_train, y_train), (_, _)) = get_default_mnist_subset
    mixup = Mixup(num_classes=10)
    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=mixup, noise=noise, loc=0, scale=0.1)
        trainer.fit(x_train, y_train, nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2', 'keras', 'kerastf', 'huggingface')
@pytest.mark.parametrize('noise', ['gaussian', 'laplacian', 'exponential'])
def test_dp_instahide_multiple_aug(art_warning, get_mnist_classifier, get_default_mnist_subset, get_default_cifar10_subset, noise, framework):
    if False:
        print('Hello World!')
    classifier = get_mnist_classifier()
    ((x_train, y_train), (_, _)) = get_default_mnist_subset
    mixup = Mixup(num_classes=10)
    cutout = Cutout(length=8, channels_first=False)
    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=[mixup, cutout], noise=noise, loc=0, scale=0.1)
        trainer.fit(x_train, y_train, nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2', 'keras', 'kerastf', 'huggingface')
@pytest.mark.parametrize('noise', ['gaussian', 'laplacian', 'exponential'])
def test_dp_instahide_validation_data(art_warning, get_mnist_classifier, get_default_mnist_subset, get_default_cifar10_subset, noise, framework):
    if False:
        return 10
    classifier = get_mnist_classifier()
    ((x_train, y_train), (x_test, y_test)) = get_default_mnist_subset
    mixup = Mixup(num_classes=10)
    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=mixup, noise=noise, loc=0, scale=0.1)
        trainer.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2', 'keras', 'kerastf', 'huggingface')
@pytest.mark.parametrize('noise', ['gaussian', 'laplacian', 'exponential'])
def test_dp_instahide_generator(art_warning, get_mnist_classifier, get_default_mnist_subset, get_default_cifar10_subset, noise, framework):
    if False:
        for i in range(10):
            print('nop')
    from art.data_generators import NumpyDataGenerator
    classifier = get_mnist_classifier()
    ((x_train, y_train), (_, _)) = get_default_mnist_subset
    mixup = Mixup(num_classes=10)
    generator = NumpyDataGenerator(x_train, y_train, batch_size=len(x_train))
    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=mixup, noise=noise, loc=0, scale=0.1)
        trainer.fit_generator(generator, nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)