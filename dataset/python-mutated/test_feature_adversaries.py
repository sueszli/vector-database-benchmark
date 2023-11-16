import logging
import numpy as np
import pytest
from art.attacks.evasion import FeatureAdversariesNumpy
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        print('Hello World!')
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.skip_framework('pytorch')
@pytest.mark.framework_agnostic
def test_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    if False:
        for i in range(10):
            print('nop')
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        classifier = image_dl_estimator_for_attack(FeatureAdversariesNumpy)
        attack = FeatureAdversariesNumpy(classifier, delta=0.2, layer=1, batch_size=32)
        x_train_mnist_adv = attack.generate(x=x_train_mnist[0:3], y=x_test_mnist[0:3], maxiter=1)
        assert np.mean(x_train_mnist[0:3]) == pytest.approx(0.13015706282513004, 0.01)
        assert np.mean(x_train_mnist_adv) == pytest.approx(0.1592448561261751, 0.01)
        with pytest.raises(ValueError):
            attack.generate(x=x_train_mnist[0:3], y=None)
        with pytest.raises(ValueError):
            attack.generate(x=x_train_mnist[0:3], y=x_test_mnist[0:2])
        with pytest.raises(ValueError):
            attack.generate(x=x_train_mnist[0:3, 0:5, 0:5, :], y=x_test_mnist[0:3])
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        i = 10
        return i + 15
    try:
        classifier = image_dl_estimator_for_attack(FeatureAdversariesNumpy)
        with pytest.raises(ValueError):
            _ = FeatureAdversariesNumpy(classifier, delta=None)
        with pytest.raises(ValueError):
            _ = FeatureAdversariesNumpy(classifier, delta=-1.0)
        with pytest.raises(ValueError):
            _ = FeatureAdversariesNumpy(classifier, delta=1.0, layer=1.0)
        with pytest.raises(ValueError):
            _ = FeatureAdversariesNumpy(classifier, delta=1.0, layer=1, batch_size=-1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        return 10
    try:
        backend_test_classifier_type_check_fail(FeatureAdversariesNumpy, [BaseEstimator, NeuralNetworkMixin])
    except ARTTestException as e:
        art_warning(e)