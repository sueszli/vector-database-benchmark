from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.defences.trainer.adversarial_trainer_madry_pgd import AdversarialTrainerMadryPGD
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        i = 10
        return i + 15
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.only_with_platform('pytorch', 'tensorflow2', 'huggingface', 'tensorflow1', 'tensorflow2v1')
def test_fit_predict(art_warning, image_dl_estimator, fix_get_mnist_subset):
    if False:
        print('Hello World!')
    (classifier, _) = image_dl_estimator()
    (x_train, y_train, x_test, y_test) = fix_get_mnist_subset
    x_test_original = x_test.copy()
    adv_trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=1, batch_size=128)
    adv_trainer.fit(x_train, y_train)
    predictions_new = np.argmax(adv_trainer.trainer.get_classifier().predict(x_test), axis=1)
    accuracy_new = np.mean(predictions_new == np.argmax(y_test, axis=1))
    assert accuracy_new == pytest.approx(0.375, abs=0.05)
    assert np.allclose(x_test_original, x_test)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2', 'tensorflow1', 'huggingface', 'tensorflow2v1')
def test_get_classifier(art_warning, image_dl_estimator):
    if False:
        while True:
            i = 10
    (classifier, _) = image_dl_estimator()
    adv_trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=1, batch_size=128)
    _ = adv_trainer.get_classifier()