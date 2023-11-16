"""
A unittest class for testing the subset scanning detector.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
import numpy as np
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.defences.detector.evasion import SubsetScanningDetector
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.only_with_platform('keras', 'kerastf', 'tensorflow2', 'pytorch')
def test_subsetscannning_detector_scan_clean(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        print('Hello World!')
    ((x_train, _), (x_test, _)) = get_default_mnist_subset
    (classifier, _) = image_dl_estimator()
    bgd_data = x_train
    clean_data = x_test
    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        (_, _, dpwr) = detector.scan(clean_x=clean_data, adv_x=clean_data)
        assert dpwr > 0
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('keras', 'kerastf', 'tensorflow2', 'pytorch')
def test_subsetscannning_detector_scan_adv(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        i = 10
        return i + 15
    ((x_train, _), (x_test, _)) = get_default_mnist_subset
    (classifier, _) = image_dl_estimator()
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)
    bgd_data = x_train
    clean_data = x_test
    adv_data = x_test_adv
    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        (_, _, dpwr) = detector.scan(clean_x=clean_data, adv_x=adv_data)
        assert dpwr > 0
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('keras', 'kerastf', 'tensorflow2', 'pytorch')
def test_subsetscannning_detector_scan_size(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        return 10
    ((x_train, _), (x_test, _)) = get_default_mnist_subset
    (classifier, _) = image_dl_estimator()
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)
    bgd_data = x_train
    clean_data = x_test
    adv_data = np.concatenate((x_test, x_test_adv), axis=0)
    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        (_, _, dpwr) = detector.scan(clean_x=clean_data, adv_x=adv_data, clean_size=85, adv_size=15)
        assert dpwr > 0
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('keras', 'kerastf', 'tensorflow2', 'pytorch')
def test_subsetscannning_detector_detect(art_warning, get_default_mnist_subset, image_dl_estimator):
    if False:
        while True:
            i = 10
    ((x_train, _), (x_test, _)) = get_default_mnist_subset
    (classifier, _) = image_dl_estimator()
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)
    bgd_data = x_train
    adv_data = x_test_adv
    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        (_, is_adversarial) = detector.detect(adv_data)
        assert len(is_adversarial) == len(adv_data)
    except ARTTestException as e:
        art_warning(e)