from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.attacks.poisoning import PoisoningAttackCleanLabelBackdoor, PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import to_categorical
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_framework('non_dl_frameworks', 'mxnet')
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator, framework):
    if False:
        return 10
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator()
        target = to_categorical([9], 10)[0]
        print(x_train.shape)

        def mod(x):
            if False:
                i = 10
                return i + 15
            return add_pattern_bd(x, channels_first=classifier.channels_first)
        backdoor = PoisoningAttackBackdoor(mod)
        attack = PoisoningAttackCleanLabelBackdoor(backdoor, classifier, target)
        (poison_data, poison_labels) = attack.poison(x_train, y_train)
        np.testing.assert_equal(poison_data.shape, x_train.shape)
        np.testing.assert_equal(poison_labels.shape, y_train.shape)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.parametrize('params', [dict(pp_poison=-0.2), dict(pp_poison=1.2)])
@pytest.mark.skip_framework('non_dl_frameworks', 'mxnet')
def test_failure_modes(art_warning, image_dl_estimator, params):
    if False:
        i = 10
        return i + 15
    try:
        (classifier, _) = image_dl_estimator()
        target = to_categorical([9], 10)[0]
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        with pytest.raises(ValueError):
            _ = PoisoningAttackCleanLabelBackdoor(backdoor, classifier, target, **params)
    except ARTTestException as e:
        art_warning(e)