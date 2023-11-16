from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.preprocessing.standardisation_mean_std import StandardisationMeanStd, StandardisationMeanStdPyTorch, StandardisationMeanStdTensorFlow
from art.preprocessing.standardisation_mean_std.utils import broadcastable_mean_std
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture(params=[True, False], ids=['channels_first', 'channels_last'])
def image_batch(request):
    if False:
        while True:
            i = 10
    '\n    Create image fixture of shape NFHWC and NCFHW.\n    '
    channels_first = request.param
    test_input = np.ones((2, 3, 32, 32))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 1))
    test_mean = [0] * 3
    test_std = [1] * 3
    test_output = test_input.copy()
    return (test_input, test_output, test_mean, test_std)

@pytest.mark.framework_agnostic
def test_broadcastable_mean_std(art_warning):
    if False:
        while True:
            i = 10
    try:
        (mean, std) = broadcastable_mean_std(np.ones((1, 3, 20, 20)), np.ones(3), np.ones(3))
        assert mean.shape == std.shape == (1, 3, 1, 1)
        (mean, std) = broadcastable_mean_std(np.ones((1, 3, 20, 20)), np.ones((1, 3, 1, 1)), np.ones((1, 3, 1, 1)))
        assert mean.shape == std.shape == (1, 3, 1, 1)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_standardisation_mean_std(art_warning, image_batch):
    if False:
        return 10
    try:
        (x, x_expected, mean, std) = image_batch
        standard = StandardisationMeanStd(mean=mean, std=std)
        (x_preprocessed, _) = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, x_expected)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('pytorch')
def test_standardisation_mean_std_pytorch(art_warning, image_batch):
    if False:
        i = 10
        return i + 15
    try:
        (x, x_expected, mean, std) = image_batch
        standard = StandardisationMeanStdPyTorch(mean=mean, std=std)
        (x_preprocessed, _) = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, x_expected)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.only_with_platform('tensorflow2')
def test_standardisation_mean_std_tensorflow_v2(art_warning, image_batch):
    if False:
        while True:
            i = 10
    try:
        (x, x_expected, mean, std) = image_batch
        standard = StandardisationMeanStdTensorFlow(mean=mean, std=std)
        (x_preprocessed, _) = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, x_expected)
    except ARTTestException as e:
        art_warning(e)