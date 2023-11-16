import logging
import pytest
import numpy as np
from art.attacks.evasion import AutoAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttack
from art.estimators.estimator import BaseEstimator
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

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_generate_default(art_warning, fix_get_mnist_subset, image_dl_estimator):
    if False:
        print('Hello World!')
    try:
        (classifier, _) = image_dl_estimator(from_logits=True)
        attack = AutoAttack(estimator=classifier, norm=np.inf, eps=0.3, eps_step=0.1, attacks=None, batch_size=32, estimator_orig=None)
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.0292, abs=0.105)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.3, abs=0.05)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_generate_attacks_and_targeted(art_warning, fix_get_mnist_subset, image_dl_estimator):
    if False:
        print('Hello World!')
    try:
        (classifier, _) = image_dl_estimator(from_logits=True)
        norm = np.inf
        eps = 0.3
        eps_step = 0.1
        batch_size = 32
        attacks = list()
        attacks.append(AutoProjectedGradientDescent(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, max_iter=100, targeted=True, nb_random_init=5, batch_size=batch_size, loss_type='cross_entropy'))
        attacks.append(AutoProjectedGradientDescent(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, max_iter=100, targeted=False, nb_random_init=5, batch_size=batch_size, loss_type='difference_logits_ratio'))
        attacks.append(DeepFool(classifier=classifier, max_iter=100, epsilon=1e-06, nb_grads=3, batch_size=batch_size))
        attacks.append(SquareAttack(estimator=classifier, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5))
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        attack = AutoAttack(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, attacks=attacks, batch_size=batch_size, estimator_orig=None, targeted=False)
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.0182, abs=0.105)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.3, abs=0.05)
        attack = AutoAttack(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, attacks=attacks, batch_size=batch_size, estimator_orig=None, targeted=True)
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
        assert np.mean(x_train_mnist_adv - x_train_mnist) == pytest.approx(0.0179, abs=0.0075)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(eps, abs=0.005)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_attack_if_targeted_not_supported(art_warning, fix_get_mnist_subset, image_dl_estimator):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError) as excinfo:
        (classifier, _) = image_dl_estimator(from_logits=True)
        attack = SquareAttack(estimator=classifier, norm=np.inf, max_iter=5000, eps=0.3, p_init=0.8, nb_restarts=5)
        attack.set_params(targeted=True)
    assert str(excinfo.value) == 'The attribute "targeted" cannot be set for this attack.'

@pytest.mark.skip_framework('tensorflow1', 'keras', 'pytorch', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        for i in range(10):
            print('nop')
    try:
        from art.attacks.evasion import FastGradientMethod
        classifier = image_dl_estimator_for_attack(AutoAttack)
        attacks = [FastGradientMethod(estimator=classifier)]
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, norm=0)
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, eps='1')
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, eps=-1.0)
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, eps_step='1')
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, eps_step=-1.0)
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, batch_size=1.0)
        with pytest.raises(ValueError):
            _ = AutoAttack(classifier, attacks=attacks, batch_size=-1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        return 10
    try:
        backend_test_classifier_type_check_fail(AutoAttack, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_generate_parallel(art_warning, fix_get_mnist_subset, image_dl_estimator):
    if False:
        i = 10
        return i + 15
    try:
        (classifier, _) = image_dl_estimator(from_logits=True)
        norm = np.inf
        eps = 0.3
        eps_step = 0.1
        batch_size = 32
        attacks = list()
        attacks.append(AutoProjectedGradientDescent(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, max_iter=100, targeted=True, nb_random_init=5, batch_size=batch_size, loss_type='cross_entropy', verbose=False))
        attacks.append(AutoProjectedGradientDescent(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, max_iter=100, targeted=False, nb_random_init=5, batch_size=batch_size, loss_type='difference_logits_ratio', verbose=False))
        attacks.append(DeepFool(classifier=classifier, max_iter=100, epsilon=1e-06, nb_grads=3, batch_size=batch_size, verbose=False))
        attacks.append(SquareAttack(estimator=classifier, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5, verbose=False))
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        attack = AutoAttack(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, attacks=attacks, batch_size=batch_size, estimator_orig=None, targeted=False, parallel=True)
        attack_noparallel = AutoAttack(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, attacks=attacks, batch_size=batch_size, estimator_orig=None, targeted=False, parallel=False)
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
        x_train_mnist_adv_nop = attack_noparallel.generate(x=x_train_mnist, y=y_train_mnist)
        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.0182, abs=0.105)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.3, abs=0.05)
        noparallel_perturbation = np.linalg.norm(x_train_mnist[[2]] - x_train_mnist_adv_nop[[2]])
        parallel_perturbation = np.linalg.norm(x_train_mnist[[2]] - x_train_mnist_adv[[2]])
        assert parallel_perturbation < noparallel_perturbation
        attack = AutoAttack(estimator=classifier, norm=norm, eps=eps, eps_step=eps_step, attacks=attacks, batch_size=batch_size, estimator_orig=None, targeted=True, parallel=True)
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)
        assert np.mean(x_train_mnist_adv - x_train_mnist) == pytest.approx(0.0, abs=0.0075)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(eps, abs=0.005)
    except ARTTestException as e:
        art_warning(e)