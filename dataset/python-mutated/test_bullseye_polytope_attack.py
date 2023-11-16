from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.attacks.poisoning import BullseyePolytopeAttackPyTorch
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_framework('non_dl_frameworks', 'tensorflow', 'mxnet', 'keras', 'kerastf', 'huggingface')
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        for i in range(10):
            print('nop')
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        attack = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, net_repeat=2)
        (poison_data, poison_labels) = attack.poison(x_train[5:10], y_train[5:10])
        np.testing.assert_equal(poison_data.shape, x_train[5:10].shape)
        np.testing.assert_equal(poison_labels.shape, y_train[5:10].shape)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[5:10])
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('non_dl_frameworks', 'tensorflow', 'mxnet', 'keras', 'kerastf', 'huggingface')
def test_poison_end2end(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        return 10
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        attack = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, net_repeat=2, endtoend=True)
        (poison_data, poison_labels) = attack.poison(x_train[5:10], y_train[5:10])
        np.testing.assert_equal(poison_data.shape, x_train[5:10].shape)
        np.testing.assert_equal(poison_labels.shape, y_train[5:10].shape)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[5:10])
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('non_dl_frameworks', 'tensorflow', 'mxnet', 'keras', 'kerastf')
def test_poison_multiple_layers(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        return 10
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        num_layers = len(classifier.layer_names)
        attack = BullseyePolytopeAttackPyTorch(classifier, target, [num_layers - 2, num_layers - 3])
        (poison_data, poison_labels) = attack.poison(x_train[5:10], y_train[5:10])
        np.testing.assert_equal(poison_data.shape, x_train[5:10].shape)
        np.testing.assert_equal(poison_labels.shape, y_train[5:10].shape)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[5:10])
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('non_dl_frameworks', 'tensorflow', 'mxnet', 'keras', 'kerastf')
def test_check_params(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        print('Hello World!')
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, learning_rate=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, max_iter=-1)
        with pytest.raises(TypeError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, 2.5)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, opt='new optimizer')
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, momentum=1.2)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, decay_iter=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, epsilon=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, dropout=2)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, net_repeat=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, -1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, decay_coeff=2)
    except ARTTestException as e:
        art_warning(e)