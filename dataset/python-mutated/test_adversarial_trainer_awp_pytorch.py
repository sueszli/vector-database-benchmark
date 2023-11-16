from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import logging
import numpy as np
from art.defences.trainer import AdversarialTrainerAWPPyTorch
from art.attacks.evasion import ProjectedGradientDescent

@pytest.fixture()
def get_adv_trainer_awppgd(framework, image_dl_estimator):
    if False:
        while True:
            i = 10

    def _get_adv_trainer_awppgd():
        if False:
            print('Hello World!')
        if framework == 'keras':
            trainer = None
        if framework in ['tensorflow', 'tensorflow2v1']:
            trainer = None
        if framework == 'pytorch':
            (classifier, _) = image_dl_estimator(from_logits=True)
            (proxy_classifier, _) = image_dl_estimator(from_logits=True)
            attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.2, eps_step=0.02, max_iter=20, targeted=False, num_random_init=1, batch_size=128, verbose=False)
            trainer = AdversarialTrainerAWPPyTorch(classifier, proxy_classifier, attack, mode='PGD', gamma=0.01, beta=6.0, warmup=0)
        if framework == 'scikitlearn':
            trainer = None
        return trainer
    return _get_adv_trainer_awppgd

@pytest.fixture()
def get_adv_trainer_awptrades(framework, image_dl_estimator):
    if False:
        print('Hello World!')

    def _get_adv_trainer_awptrades():
        if False:
            i = 10
            return i + 15
        if framework == 'keras':
            trainer = None
        if framework in ['tensorflow', 'tensorflow2v1']:
            trainer = None
        if framework == 'pytorch':
            (classifier, _) = image_dl_estimator(from_logits=True)
            (proxy_classifier, _) = image_dl_estimator(from_logits=True)
            attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.2, eps_step=0.02, max_iter=20, targeted=False, num_random_init=1, batch_size=128, verbose=False)
            trainer = AdversarialTrainerAWPPyTorch(classifier, proxy_classifier, attack, mode='TRADES', gamma=0.01, beta=6.0, warmup=0)
        if framework == 'scikitlearn':
            trainer = None
        return trainer
    return _get_adv_trainer_awptrades

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        return 10
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.only_with_platform('pytorch')
@pytest.mark.parametrize('label_format', ['one_hot', 'numerical'])
def test_adversarial_trainer_awppgd_pytorch_fit_and_predict(get_adv_trainer_awppgd, fix_get_mnist_subset, label_format):
    if False:
        while True:
            i = 10
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()
    if label_format == 'one_hot':
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == 'numerical':
        y_train_mnist = np.argmax(y_train_mnist, axis=1)
        y_test_mnist = np.argmax(y_test_mnist, axis=1)
    trainer = get_adv_trainer_awppgd()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return
    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]
    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test_mnist) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(float(np.mean(x_test_mnist_original - x_test_mnist)), 0.0, decimal=4)
    assert accuracy == 0.32
    assert accuracy_new > 0.32
    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))

@pytest.mark.only_with_platform('pytorch')
@pytest.mark.parametrize('label_format', ['one_hot', 'numerical'])
def test_adversarial_trainer_awptrades_pytorch_fit_and_predict(get_adv_trainer_awptrades, fix_get_mnist_subset, label_format):
    if False:
        i = 10
        return i + 15
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()
    if label_format == 'one_hot':
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == 'numerical':
        y_train_mnist = np.argmax(y_train_mnist, axis=1)
        y_test_mnist = np.argmax(y_test_mnist, axis=1)
    trainer = get_adv_trainer_awptrades()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return
    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]
    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test_mnist) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(float(np.mean(x_test_mnist_original - x_test_mnist)), 0.0, decimal=4)
    assert accuracy == 0.32
    assert accuracy_new > 0.32
    trainer.fit(x_train_mnist, y_train_mnist, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))

@pytest.mark.only_with_platform('pytorch')
@pytest.mark.parametrize('label_format', ['one_hot', 'numerical'])
def test_adversarial_trainer_awppgd_pytorch_fit_generator_and_predict(get_adv_trainer_awppgd, fix_get_mnist_subset, image_data_generator, label_format):
    if False:
        return 10
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()
    if label_format == 'one_hot':
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == 'numerical':
        y_test_mnist = np.argmax(y_test_mnist, axis=1)
    generator = image_data_generator()
    trainer = get_adv_trainer_awppgd()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return
    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]
    trainer.fit_generator(generator=generator, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test_mnist) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(float(np.mean(x_test_mnist_original - x_test_mnist)), 0.0, decimal=4)
    assert accuracy == 0.32
    assert accuracy_new > 0.32
    trainer.fit_generator(generator=generator, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))

@pytest.mark.only_with_platform('pytorch')
@pytest.mark.parametrize('label_format', ['one_hot', 'numerical'])
def test_adversarial_trainer_awptrades_pytorch_fit_generator_and_predict(get_adv_trainer_awptrades, fix_get_mnist_subset, image_data_generator, label_format):
    if False:
        for i in range(10):
            print('nop')
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    x_test_mnist_original = x_test_mnist.copy()
    if label_format == 'one_hot':
        assert y_train_mnist.shape[-1] == 10
        assert y_test_mnist.shape[-1] == 10
    if label_format == 'numerical':
        y_test_mnist = np.argmax(y_test_mnist, axis=1)
    generator = image_data_generator()
    trainer = get_adv_trainer_awptrades()
    if trainer is None:
        logging.warning("Couldn't perform  this test because no trainer is defined for this framework configuration")
        return
    predictions = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy = np.sum(predictions == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy = np.sum(predictions == y_test_mnist) / x_test_mnist.shape[0]
    trainer.fit_generator(generator=generator, nb_epochs=20)
    predictions_new = np.argmax(trainer.predict(x_test_mnist), axis=1)
    if label_format == 'one_hot':
        accuracy_new = np.sum(predictions_new == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    else:
        accuracy_new = np.sum(predictions_new == y_test_mnist) / x_test_mnist.shape[0]
    np.testing.assert_array_almost_equal(float(np.mean(x_test_mnist_original - x_test_mnist)), 0.0, decimal=4)
    assert accuracy == 0.32
    assert accuracy_new > 0.32
    trainer.fit_generator(generator=generator, nb_epochs=20, validation_data=(x_train_mnist, y_train_mnist))