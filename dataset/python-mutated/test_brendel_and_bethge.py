import logging
import numpy as np
import pytest
from art.attacks.evasion import BrendelBethgeAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_framework('tensorflow1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
@pytest.mark.parametrize('targeted', [True, False])
@pytest.mark.parametrize('norm', [0, 1, 2, np.inf, 'inf'])
def test_generate(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack, targeted, norm):
    if False:
        return 10
    try:
        ((_, _), (x_test_mnist, y_test_mnist)) = get_default_mnist_subset
        classifier = image_dl_estimator_for_attack(BrendelBethgeAttack, defended=False, from_logits=True)
        attack = BrendelBethgeAttack(estimator=classifier, norm=norm, targeted=targeted, overshoot=1.1, steps=1, lr=0.001, lr_decay=0.5, lr_num_decay=20, momentum=0.8, binary_search_steps=1, init_size=5, batch_size=32)
        attack.generate(x=x_test_mnist[0:1].astype(np.float32), y=y_test_mnist[0:1].astype(int))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    if False:
        i = 10
        return i + 15
    try:
        classifier = image_dl_estimator_for_attack(BrendelBethgeAttack, from_logits=True)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, norm=3)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, targeted='true')
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, overshoot=1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, overshoot=0.9)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, steps=1.0)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, steps=0)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, lr=1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, lr=-1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, lr_decay=1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, lr_decay=-1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, lr_num_decay=1.0)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, lr_num_decay=-1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, momentum=1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, momentum=-1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, binary_search_steps=1.0)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, binary_search_steps=-1)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, init_size=1.0)
        with pytest.raises(ValueError):
            _ = BrendelBethgeAttack(classifier, init_size=-1)
    except ARTTestException as e:
        art_warning(e)

def test_classifier_type_check_fail():
    if False:
        return 10
    backend_test_classifier_type_check_fail(BrendelBethgeAttack, [BaseEstimator, LossGradientsMixin, ClassifierMixin])