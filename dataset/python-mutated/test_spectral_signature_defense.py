from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
import numpy as np
from art.defences.detector.poison import SpectralSignatureDefense
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)
(NB_TRAIN, NB_TEST, BATCH_SIZE, EPS_MULTIPLIER, UB_PCT_POISON) = (300, 10, 128, 1.5, 0.2)

@pytest.mark.parametrize('params', [dict(batch_size=-1), dict(eps_multiplier=-1.0), dict(expected_pp_poison=2.0)])
def test_wrong_parameters(params, art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        for i in range(10):
            print('nop')
    try:
        ((x_train_mnist, y_train_mnist), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator()
        classifier.fit(x_train_mnist[:NB_TRAIN], y_train_mnist[:NB_TRAIN], nb_epochs=1)
        with pytest.raises(ValueError):
            _ = SpectralSignatureDefense(classifier, x_train_mnist[:NB_TRAIN], y_train_mnist[:NB_TRAIN], **params)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('non_dl_frameworks', 'mxnet')
def test_detect_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        for i in range(10):
            print('nop')
    try:
        ((x_train_mnist, y_train_mnist), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator()
        classifier.fit(x_train_mnist[:NB_TRAIN], y_train_mnist[:NB_TRAIN], nb_epochs=1)
        defence = SpectralSignatureDefense(classifier, x_train_mnist[:NB_TRAIN], y_train_mnist[:NB_TRAIN], batch_size=BATCH_SIZE, eps_multiplier=EPS_MULTIPLIER, expected_pp_poison=UB_PCT_POISON)
        (report, is_clean) = defence.detect_poison()
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('non_dl_frameworks', 'mxnet')
def test_evaluate_defense(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        while True:
            i = 10
    try:
        ((x_train_mnist, y_train_mnist), (_, _)) = get_default_mnist_subset
        (classifier, _) = image_dl_estimator()
        classifier.fit(x_train_mnist[:NB_TRAIN], y_train_mnist[:NB_TRAIN], nb_epochs=1)
        defence = SpectralSignatureDefense(classifier, x_train_mnist[:NB_TRAIN], y_train_mnist[:NB_TRAIN], batch_size=BATCH_SIZE, eps_multiplier=EPS_MULTIPLIER, expected_pp_poison=UB_PCT_POISON)
        _ = defence.evaluate_defence(np.zeros(NB_TRAIN))
    except ARTTestException as e:
        art_warning(e)