from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.preprocessing.audio import LFilterPyTorch
from art.config import ART_NUMPY_DTYPE
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_module('torchaudio')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
@pytest.mark.parametrize('fir_filter', [False, True])
def test_audio_filter(fir_filter, art_warning, expected_values):
    if False:
        i = 10
        return i + 15
    try:
        expected_data = expected_values()
        x1 = expected_data[0]
        x2 = expected_data[1]
        x3 = expected_data[2]
        result_0 = expected_data[3]
        result_1 = expected_data[4]
        result_2 = expected_data[5]
        x = np.array([np.array(x1 * 2, dtype=ART_NUMPY_DTYPE), np.array(x2 * 2, dtype=ART_NUMPY_DTYPE), np.array(x3 * 2, dtype=ART_NUMPY_DTYPE)], dtype=object)
        numerator_coef = np.array([0.1, 0.2, -0.1, -0.2], dtype=ART_NUMPY_DTYPE)
        if fir_filter:
            denominator_coef = np.array([1.0, 0.0, 0.0, 0.0], dtype=ART_NUMPY_DTYPE)
        else:
            denominator_coef = np.array([1.0, 0.1, 0.3, 0.4], dtype=ART_NUMPY_DTYPE)
        audio_filter = LFilterPyTorch(numerator_coef=numerator_coef, denominator_coef=denominator_coef)
        result = audio_filter(x)
        assert result[1] is None
        np.testing.assert_array_almost_equal(result_0, result[0][0], decimal=0)
        np.testing.assert_array_almost_equal(result_1, result[0][1], decimal=0)
        np.testing.assert_array_almost_equal(result_2, result[0][2], decimal=0)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_module('torchaudio')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_default(art_warning):
    if False:
        return 10
    try:
        x = np.array([[0.37, 0.68, 0.63, 0.48, 0.48, 0.18, 0.19]], dtype=ART_NUMPY_DTYPE)
        audio_filter = LFilterPyTorch()
        result = audio_filter(x)
        assert result[1] is None
        np.testing.assert_array_almost_equal(x, result[0], decimal=0)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_module('torchaudio')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_clip_values(art_warning):
    if False:
        return 10
    try:
        x = np.array([[0.37, 0.68, 0.63, 0.48, 0.48, 0.18, 0.19]], dtype=ART_NUMPY_DTYPE)
        audio_filter = LFilterPyTorch(clip_values=(0, 1))
        result = audio_filter(x)
        assert result[1] is None
        np.testing.assert_array_almost_equal(x, result[0], decimal=0)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_module('torchaudio')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_triple_clip_values_error(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        exc_msg = '`clip_values` should be a tuple of 2 floats containing the allowed data range.'
        with pytest.raises(ValueError, match=exc_msg):
            LFilterPyTorch(numerator_coef=np.array([0.1, 0.2, 0.3]), denominator_coef=np.array([0.1, 0.2, 0.3]), clip_values=(0, 1, 2))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_module('torchaudio')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_relation_clip_values_error(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        exc_msg = 'Invalid `clip_values`: min >= max.'
        with pytest.raises(ValueError, match=exc_msg):
            LFilterPyTorch(numerator_coef=np.array([0.1, 0.2, 0.3]), denominator_coef=np.array([0.1, 0.2, 0.3]), clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_check_params(art_warning):
    if False:
        while True:
            i = 10
    try:
        with pytest.raises(ValueError):
            _ = LFilterPyTorch(numerator_coef=np.array([0.1, 0.2, -0.1, -0.2]), denominator_coef=[0.0, 0.1, 0.3, 0.4])
        with pytest.raises(ValueError):
            _ = LFilterPyTorch(numerator_coef=np.array([0.1, 0.2, -0.1, -0.2]), denominator_coef=np.array([0.0, 0.1, 0.3, 0.4]))
        with pytest.raises(ValueError):
            _ = LFilterPyTorch(numerator_coef=[0.1, 0.2, -0.1, -0.2], denominator_coef=np.array([1.0, 0.1, 0.3, 0.4]))
        with pytest.raises(ValueError):
            _ = LFilterPyTorch(numerator_coef=np.array([0.1, 0.2, -0.1, -0.2]), denominator_coef=np.array([1.0, 0.1, 0.3, 0.4, 0.2]))
        with pytest.raises(ValueError):
            _ = LFilterPyTorch(numerator_coef=np.array([0.1, 0.2, -0.1, -0.2]), denominator_coef=np.array([1.0, 0.1, 0.3, 0.4]), verbose='True')
    except ARTTestException as e:
        art_warning(e)