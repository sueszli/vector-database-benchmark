from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from PIL import Image
from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2')
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator, framework):
    if False:
        return 10
    try:
        ((x_train, y_train), (x_test, y_test)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator(functional=True)
        patch_size = 4
        img = Image.open('notebooks/trigger_10.png').convert('L')
        img = np.asarray(img.resize((patch_size, patch_size)))
        if classifier.channels_first:
            patch = np.asarray(img).reshape(1, patch_size, patch_size)
        else:
            patch = np.asarray(img).reshape(patch_size, patch_size, 1)
        (x_train, y_train) = (x_train[:1000], y_train[:1000])
        class_source = 0
        class_target = 1
        K = 1
        x_train_ = np.copy(x_train)
        index_source = np.where(y_train == class_source)[0][0:K]
        index_target = np.where(y_train == class_target)[0]
        x_trigger = x_train_[index_source]
        epsilon = 16 * 255
        percent_poison = 0.01
        attack = SleeperAgentAttack(classifier, percent_poison=percent_poison, max_trials=1, max_epochs=10, clip_values=(0, 1), epsilon=epsilon, batch_size=500, verbose=1, indices_target=index_target, patching_strategy='random', selection_strategy='random', patch=patch, retraining_factor=4, model_retrain=False, model_retraining_epoch=40, class_source=class_source, class_target=class_target)
        (x_poison, y_poison) = attack.poison(x_trigger, [class_target], x_train, y_train, x_test, y_test)
        np.testing.assert_(np.all(np.sum(np.reshape((x_poison - x_train) ** 2, [x_poison.shape[0], -1]), axis=1) < epsilon))
        np.testing.assert_(np.sum(np.sum(np.reshape((x_poison - x_train) ** 2, [x_poison.shape[0], -1]), axis=1) > 0) <= percent_poison * x_train.shape[0])
        np.testing.assert_equal(np.shape(x_poison), np.shape(x_train))
        np.testing.assert_equal(np.shape(y_poison), np.shape(y_train))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch', 'tensorflow2')
def test_check_params(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        while True:
            i = 10
    try:
        (classifier, _) = image_dl_estimator(functional=True)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3], indices_target=[0], patch=[1])
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3], indices_target=0, patch=np.ones((1, 8, 8)))
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3], indices_target=[0], patch=np.ones((1, 8, 8)), patching_strategy=1)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3], indices_target=[0], patch=np.ones((1, 8, 8)), selection_strategy=1)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3], indices_target=[0], patch=np.ones((1, 8, 8)), selection_strategy=1, class_source=0, class_target=0)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=1.1, learning_rate_schedule=[0.1, 0.2, 0.3], indices_target=[0], patch=np.ones((1, 8, 8)), selection_strategy=1)
    except ARTTestException as e:
        art_warning(e)