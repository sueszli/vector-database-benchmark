from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.attacks.poisoning import GradientMatchingAttack
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2')
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        print('Hello World!')
    try:
        ((x_train, y_train), (x_test, y_test)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator()
        class_source = 0
        class_target = 1
        epsilon = 0.3
        percent_poison = 0.01
        index_target = np.where(y_test.argmax(axis=1) == class_source)[0][5]
        x_trigger = x_test[index_target:index_target + 1]
        (x_train, y_train) = (x_train[:1000], y_train[:1000])
        y_train = np.argmax(y_train, axis=-1)
        attack = GradientMatchingAttack(classifier, epsilon=epsilon, percent_poison=percent_poison, max_trials=1, max_epochs=1, verbose=False)
        (x_poison, y_poison) = attack.poison(x_trigger, [class_target], x_train, y_train)
        np.testing.assert_(np.all(np.sum(np.reshape((x_poison - x_train) ** 2, [x_poison.shape[0], -1]), axis=1) < epsilon))
        np.testing.assert_(np.sum(np.sum(np.reshape((x_poison - x_train) ** 2, [x_poison.shape[0], -1]), axis=1) > 0) <= percent_poison * x_train.shape[0])
        np.testing.assert_equal(np.shape(x_poison), np.shape(x_train))
        np.testing.assert_equal(np.shape(y_poison), np.shape(y_train))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2')
def test_check_params(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        return 10
    try:
        (classifier, _) = image_dl_estimator(functional=True)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=1.2)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, max_epochs=0)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, max_trials=0)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, clip_values=1)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, epsilon=-1)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, batch_size=0)
        with pytest.raises(ValueError):
            _ = GradientMatchingAttack(classifier, percent_poison=0.01, verbose=1.1)
    except ARTTestException as e:
        art_warning(e)