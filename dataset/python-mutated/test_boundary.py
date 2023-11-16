import pytest
import logging
from art.attacks.evasion import BoundaryAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from tests.attacks.utils import backend_targeted_tabular, backend_untargeted_tabular, backend_targeted_images
from tests.attacks.utils import back_end_untargeted_images, backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    if False:
        for i in range(10):
            print('nop')
    ((x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist)) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('clipped_classifier, targeted', [(True, True), (True, False), (False, True), (False, False)])
def test_tabular(art_warning, tabular_dl_estimator, framework, get_iris_dataset, clipped_classifier, targeted):
    if False:
        while True:
            i = 10
    try:
        classifier = tabular_dl_estimator(clipped=clipped_classifier)
        attack = BoundaryAttack(classifier, targeted=targeted, max_iter=10, verbose=False)
        if targeted:
            backend_targeted_tabular(attack, get_iris_dataset)
        else:
            backend_untargeted_tabular(attack, get_iris_dataset, clipped=clipped_classifier)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('targeted', [True, False])
def test_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, framework, targeted):
    if False:
        for i in range(10):
            print('nop')
    try:
        classifier = image_dl_estimator_for_attack(BoundaryAttack)
        attack = BoundaryAttack(estimator=classifier, targeted=targeted, max_iter=20, verbose=False)
        if targeted:
            backend_targeted_images(attack, fix_get_mnist_subset)
        else:
            back_end_untargeted_images(attack, fix_get_mnist_subset, framework)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        while True:
            i = 10
    try:
        classifier = image_dl_estimator_for_attack(BoundaryAttack)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, max_iter=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, num_trial=1.0)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, num_trial=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, sample_size=1.0)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, sample_size=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, init_size=1.0)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, init_size=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, epsilon=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, delta=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, step_adapt=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, min_epsilon='1.0')
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, min_epsilon=-1)
        with pytest.raises(ValueError):
            _ = BoundaryAttack(classifier, verbose='true')
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        backend_test_classifier_type_check_fail(BoundaryAttack, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)