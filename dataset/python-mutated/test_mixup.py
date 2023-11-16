from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import Mixup
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.fixture(params=[1, 3], ids=['grayscale', 'RGB'])
def image_batch(request):
    if False:
        while True:
            i = 10
    '\n    Image fixtures of shape NHWC.\n    '
    channels = request.param
    data_shape = (4, 8, 12, channels)
    return (0.5 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)

@pytest.fixture(params=[1, 3], ids=['grayscale', 'RGB'])
def empty_image(request):
    if False:
        for i in range(10):
            print('nop')
    '\n    Empty image fixtures of shape NHWC.\n    '
    channels = request.param
    data_shape = (4, 8, 12, channels)
    return np.zeros(data_shape).astype(ART_NUMPY_DTYPE)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('alpha', [1.0, 2.5])
@pytest.mark.parametrize('num_mix', [2, 3])
def test_mixup_image_data(art_warning, image_batch, alpha, num_mix):
    if False:
        i = 10
        return i + 15
    try:
        mixup = Mixup(num_classes=10, alpha=alpha, num_mix=num_mix)
        (x, y) = mixup(image_batch, np.arange(len(image_batch)))
        assert_array_almost_equal(x, image_batch)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(image_batch)))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
@pytest.mark.parametrize('alpha', [1.0])
@pytest.mark.parametrize('num_mix', [2])
def test_mixup_empty_data(art_warning, empty_image, alpha, num_mix):
    if False:
        while True:
            i = 10
    try:
        mixup = Mixup(num_classes=10, alpha=alpha, num_mix=num_mix)
        (x, y) = mixup(empty_image, np.arange(len(empty_image)))
        assert_array_equal(x, empty_image)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(empty_image)))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_missing_labels_error(art_warning, tabular_batch):
    if False:
        i = 10
        return i + 15
    try:
        test_input = tabular_batch
        mixup = Mixup(num_classes=10)
        exc_msg = 'Labels `y` cannot be None.'
        with pytest.raises(ValueError, match=exc_msg):
            mixup(test_input)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_check_params(art_warning):
    if False:
        while True:
            i = 10
    try:
        with pytest.raises(ValueError):
            _ = Mixup(num_classes=0)
        with pytest.raises(ValueError):
            _ = Mixup(num_classes=10, alpha=0)
        with pytest.raises(ValueError):
            _ = Mixup(num_classes=10, alpha=-1)
        with pytest.raises(ValueError):
            _ = Mixup(num_classes=10, num_mix=1)
        with pytest.raises(ValueError):
            _ = Mixup(num_classes=10, num_mix=-1)
    except ARTTestException as e:
        art_warning(e)