from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.attacks.poisoning import HiddenTriggerBackdoor
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.hugging_face import HuggingFaceClassifierPyTorch
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_framework('non_dl_frameworks', 'tensorflow1', 'tensorflow2v1', 'mxnet')
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator, framework):
    if False:
        while True:
            i = 10
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        functional = True
        if framework == 'huggingface':
            functional = False
        (classifier, _) = image_dl_estimator(functional=functional)
        if isinstance(classifier, (PyTorchClassifier, HuggingFaceClassifierPyTorch)):

            def mod(x):
                if False:
                    i = 10
                    return i + 15
                original_dtype = x.dtype
                x = np.transpose(x, (0, 2, 3, 1)).astype(np.float32)
                x = add_pattern_bd(x)
                x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
                return x.astype(original_dtype)
        else:

            def mod(x):
                if False:
                    i = 10
                    return i + 15
                original_dtype = x.dtype
                x = add_pattern_bd(x)
                return x.astype(original_dtype)
        backdoor = PoisoningAttackBackdoor(mod)
        target = y_train[0]
        diff_index = list(set(np.arange(len(y_train))) - set(np.where(np.all(y_train == target, axis=1))[0]))[0]
        source = y_train[diff_index]
        attack = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1)
        (poison_data, poison_inds) = attack.poison(x_train, y_train)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[poison_inds])
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('non_dl_frameworks', 'tensorflow1', 'tensorflow2v1', 'mxnet')
def test_check_params(art_warning, get_default_mnist_subset, image_dl_estimator, framework):
    if False:
        i = 10
        return i + 15
    try:
        ((x_train, y_train), (_, _)) = get_default_mnist_subset
        functional = True
        if framework == 'huggingface':
            functional = False
        (classifier, _) = image_dl_estimator(functional=functional)
        if isinstance(classifier, PyTorchClassifier):

            def mod(x):
                if False:
                    i = 10
                    return i + 15
                original_dtype = x.dtype
                x = np.transpose(x, (0, 2, 3, 1)).astype(np.float32)
                x = add_pattern_bd(x)
                x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
                return x.astype(original_dtype)
        else:

            def mod(x):
                if False:
                    while True:
                        i = 10
                original_dtype = x.dtype
                x = add_pattern_bd(x)
                return x.astype(original_dtype)
        backdoor = PoisoningAttackBackdoor(mod)
        target = y_train[0]
        diff_index = list(set(np.arange(len(y_train))) - set(np.where(np.all(y_train == target, axis=1))[0]))[0]
        source = y_train[diff_index]
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=0, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1, learning_rate=-1)
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1, learning_rate=-1)
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=source, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1)
        with pytest.raises(TypeError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=source, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1)
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=-1, target=target, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1)
        with pytest.raises(TypeError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=2.5, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1)
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=-1, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1)
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=-1, decay_iter=1, max_iter=2, batch_size=1, poison_percent=0.1)
        with pytest.raises(ValueError):
            _ = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=len(classifier.layer_names) - 2, backdoor=backdoor, decay_coeff=0.95, decay_iter=1, max_iter=2, batch_size=1, poison_percent=1.1)
    except ARTTestException as e:
        art_warning(e)