from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import logging
import numpy as np
from art.defences.trainer import AdversarialTrainerFBFPyTorch

@pytest.fixture()
def get_adv_trainer(framework, image_dl_estimator):
    if False:
        i = 10
        return i + 15

    def _get_adv_trainer():
        if False:
            while True:
                i = 10
        if framework == 'keras':
            trainer = None
        if framework in ['tensorflow', 'tensorflow2v1']:
            trainer = None
        if framework in ['pytorch', 'huggingface']:
            (classifier, _) = image_dl_estimator()
            trainer = AdversarialTrainerFBFPyTorch(classifier, eps=0.05)
        if framework == 'scikitlearn':
            trainer = None
        return trainer
    return _get_adv_trainer

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        while True:
            i = 10
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.skip_framework('tensorflow', 'keras', 'scikitlearn', 'mxnet', 'kerastf', 'huggingface')
def test_adversarial_trainer_fbf_pytorch_fit_and_predict(get_adv_trainer, fix_get_mnist_subset):
    if False:
        i = 10
        return i + 15
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()
    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return
    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(float(np.mean(x_test_mnist_original - x_test_mnist)), 0.0, decimal=4)
    assert accuracy == 0.32
    assert accuracy_new == 0.63
    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))

@pytest.mark.skip_framework('tensorflow', 'keras', 'scikitlearn', 'mxnet', 'kerastf', 'huggingface')
def test_adversarial_trainer_fbf_pytorch_fit_generator_and_predict(get_adv_trainer, fix_get_mnist_subset, image_data_generator):
    if False:
        i = 10
        return i + 15
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()
    generator = image_data_generator()
    trainer = get_adv_trainer()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return
    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    trainer.fit_generator(generator=generator, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(float(np.mean(x_test_mnist_original - x_test_mnist)), 0.0, decimal=4)
    assert accuracy == 0.32
    assert accuracy_new > 0.2