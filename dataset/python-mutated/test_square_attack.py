import logging
import pytest
import numpy as np
from art.attacks.evasion import SquareAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        i = 10
        return i + 15
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.skip_framework('keras', 'scikitlearn', 'mxnet', 'kerastf')
@pytest.mark.parametrize('norm', [2, 'inf'])
def test_generate(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, norm):
    if False:
        for i in range(10):
            print('nop')
    try:
        classifier = image_dl_estimator_for_attack(SquareAttack)
        attack = SquareAttack(estimator=classifier, norm=norm, max_iter=5, eps=0.3, p_init=0.8, nb_restarts=1, verbose=False)
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
        if norm == 'inf':
            expected_mean = 0.053533513
            expected_max = 0.3
        elif norm == 2:
            expected_mean = 0.00073682
            expected_max = 0.25
        else:
            raise ValueError('Value of `norm` not recognized.')
        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(expected_mean, abs=0.025)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(expected_max, abs=0.05)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        for i in range(10):
            print('nop')
    try:
        classifier = image_dl_estimator_for_attack(SquareAttack)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, norm=0)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, max_iter=-1)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, eps='1.0')
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, eps=-1)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, p_init='1.0')
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, p_init=-1)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, nb_restarts=1.0)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, nb_restarts=-1)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, batch_size=1.0)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, batch_size=-1)
        with pytest.raises(ValueError):
            _ = SquareAttack(classifier, verbose='true')
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        while True:
            i = 10
    try:
        backend_test_classifier_type_check_fail(SquareAttack, [BaseEstimator, NeuralNetworkMixin])
    except ARTTestException as e:
        art_warning(e)