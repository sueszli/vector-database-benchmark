import logging
import pytest
import numpy as np
from art.attacks.evasion import AutoProjectedGradientDescent
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        print('Hello World!')
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.parametrize('loss_type', ['cross_entropy', 'difference_logits_ratio'])
@pytest.mark.parametrize('norm', ['inf', np.inf, 1, 2])
@pytest.mark.skip_framework('keras', 'non_dl_frameworks', 'mxnet', 'kerastf', 'tensorflow1', 'tensorflow2v1')
def test_generate(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, framework, loss_type, norm):
    if False:
        while True:
            i = 10
    print('test_generate')
    try:
        classifier = image_dl_estimator_for_attack(AutoProjectedGradientDescent, from_logits=True)
        print('framework', framework)
        if framework in ['tensorflow1', 'tensorflow2v1'] and loss_type == 'difference_logits_ratio':
            with pytest.raises(ValueError):
                _ = AutoProjectedGradientDescent(estimator=classifier, norm=norm, eps=0.3, eps_step=0.1, max_iter=5, targeted=False, nb_random_init=1, batch_size=32, loss_type=loss_type, verbose=False)
        else:
            attack = AutoProjectedGradientDescent(estimator=classifier, norm=norm, eps=0.3, eps_step=0.1, max_iter=5, targeted=False, nb_random_init=1, batch_size=32, loss_type=loss_type, verbose=False)
            (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
            x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
            assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) > 0.0
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        return 10
    try:
        classifier = image_dl_estimator_for_attack(AutoProjectedGradientDescent, from_logits=True)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, norm=0)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, eps='1')
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, eps=-1.0)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, eps_step='1')
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, eps_step=-1.0)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, max_iter=-1)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, targeted='true')
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, nb_random_init=1.0)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, nb_random_init=-1)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, batch_size=1.0)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, batch_size=-1)
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, loss_type='test')
        with pytest.raises(ValueError):
            _ = AutoProjectedGradientDescent(classifier, verbose='true')
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        return 10
    try:
        backend_test_classifier_type_check_fail(AutoProjectedGradientDescent, [BaseEstimator, LossGradientsMixin, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)