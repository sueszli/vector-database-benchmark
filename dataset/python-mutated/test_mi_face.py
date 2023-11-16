from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
import numpy as np
from art.attacks.inference.model_inversion import MIFace
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, ClassGradientsMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        while True:
            i = 10
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

def backend_check_inferred_values(attack, mnist_dataset, classifier):
    if False:
        return 10
    x_train_infer_from_zero = attack.infer(None, y=np.arange(10))
    preds = np.argmax(classifier.predict(x_train_infer_from_zero), axis=1)
    np.testing.assert_array_equal(preds, np.arange(10))
    (x_train_mnist, y_train_mnist, _, _) = mnist_dataset
    x_original = x_train_mnist[:10]
    x_noisy = np.clip(x_original + np.random.uniform(-0.01, 0.01, x_original.shape), 0, 1)
    x_train_infer_from_noisy = attack.infer(x_noisy, y=y_train_mnist[:10])
    diff_noisy = np.mean(np.reshape(np.abs(x_original - x_noisy), (len(x_original), -1)), axis=1)
    diff_inferred = np.mean(np.reshape(np.abs(x_original - x_train_infer_from_noisy), (len(x_original), -1)), axis=1)
    np.testing.assert_array_less(diff_noisy, diff_inferred)

@pytest.mark.framework_agnostic
def test_miface(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    if False:
        for i in range(10):
            print('nop')
    try:
        classifier = image_dl_estimator_for_attack(MIFace)
        attack = MIFace(classifier, max_iter=150, batch_size=3)
        backend_check_inferred_values(attack, fix_get_mnist_subset, classifier)
    except ARTTestException as e:
        art_warning(e)

def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        print('Hello World!')
    try:
        classifier = image_dl_estimator_for_attack(MIFace)
        with pytest.raises(ValueError):
            _ = MIFace(classifier, max_iter=-0.5)
        with pytest.raises(ValueError):
            _ = MIFace(classifier, window_length=-0.5)
        with pytest.raises(ValueError):
            _ = MIFace(classifier, threshold=-0.5)
        with pytest.raises(ValueError):
            _ = MIFace(classifier, learning_rate=-0.5)
        with pytest.raises(ValueError):
            _ = MIFace(classifier, batch_size=-0.5)
        with pytest.raises(ValueError):
            _ = MIFace(classifier, verbose=-0.5)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        print('Hello World!')
    try:
        backend_test_classifier_type_check_fail(MIFace, [BaseEstimator, ClassifierMixin, ClassGradientsMixin])
    except ARTTestException as e:
        art_warning(e)