from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.config import ART_NUMPY_DTYPE
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_module('deepspeech_pytorch')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
@pytest.mark.parametrize('use_amp', [False, True])
@pytest.mark.parametrize('device_type', ['cpu', 'gpu'])
def test_pytorch_deep_speech(art_warning, expected_values, use_amp, device_type):
    if False:
        for i in range(10):
            print('nop')
    import torch
    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
    try:
        if use_amp and (not torch.cuda.is_available()):
            return
        speech_recognizer = PyTorchDeepSpeech(pretrained_model='librispeech', device_type=device_type, use_amp=use_amp)
        version = 'v{}'.format(speech_recognizer._version)
        expected_data = expected_values()
        x1 = expected_data['x_1']
        x2 = expected_data['x_2']
        x3 = expected_data['x_3']
        expected_sizes = expected_data['expected_sizes']
        expected_transcriptions1 = expected_data['expected_transcriptions_1']
        expected_transcriptions2 = expected_data['expected_transcriptions_2']
        expected_probs = expected_data['expected_probs'][version]
        expected_gradients1 = expected_data['expected_gradients_1'][version]
        expected_gradients2 = expected_data['expected_gradients_2'][version]
        expected_gradients3 = expected_data['expected_gradients_3'][version]
        x = np.array([np.array(x1 * 100, dtype=ART_NUMPY_DTYPE), np.array(x2 * 100, dtype=ART_NUMPY_DTYPE), np.array(x3 * 100, dtype=ART_NUMPY_DTYPE)])
        y = np.array(['SIX', 'HI', 'GOOD'])
        (probs, sizes) = speech_recognizer.predict(x, batch_size=2, transcription_output=False)
        np.testing.assert_array_almost_equal(probs[1][1], expected_probs, decimal=3)
        np.testing.assert_array_almost_equal(sizes, expected_sizes)
        transcriptions = speech_recognizer.predict(x, batch_size=2, transcription_output=True)
        assert (expected_transcriptions1 == transcriptions).all()
        transcriptions = speech_recognizer.predict(np.array([x[0]]), batch_size=2, transcription_output=True)
        assert (expected_transcriptions2 == transcriptions).all()
        grads = speech_recognizer.loss_gradient(x, y)
        assert grads[0].shape == (1300,)
        assert grads[1].shape == (1500,)
        assert grads[2].shape == (1400,)
        np.testing.assert_array_almost_equal(grads[0][:20], expected_gradients1, decimal=-2)
        np.testing.assert_array_almost_equal(grads[1][:20], expected_gradients2, decimal=-2)
        np.testing.assert_array_almost_equal(grads[2][:20], expected_gradients3, decimal=-2)
        parameters = speech_recognizer.model.parameters()
        speech_recognizer._optimizer = torch.optim.SGD(parameters, lr=0.01)
        transcriptions1 = speech_recognizer.predict(x, batch_size=2, transcription_output=True)
        speech_recognizer.fit(x=x, y=y, batch_size=2, nb_epochs=5)
        transcriptions2 = speech_recognizer.predict(x, batch_size=2, transcription_output=True)
        assert not (transcriptions1 == transcriptions2).all()
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_module('deepspeech_pytorch')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
def test_pytorch_deep_speech_preprocessor(art_warning, expected_values):
    if False:
        i = 10
        return i + 15
    import torch
    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
    from art.defences.preprocessor.mp3_compression_pytorch import Mp3CompressionPyTorch
    try:
        defense = Mp3CompressionPyTorch(sample_rate=16000, channels_first=True)
        speech_recognizer = PyTorchDeepSpeech(pretrained_model='librispeech', preprocessing_defences=[defense], device_type='cpu', use_amp=False)
        version = 'v{}'.format(speech_recognizer._version)
        expected_data = expected_values()
        x1 = expected_data['x_preprocessor_1']
        expected_sizes = expected_data['expected_sizes_preprocessor']
        expected_transcriptions1 = expected_data['expected_transcriptions_preprocessor_1']
        expected_transcriptions2 = expected_data['expected_transcriptions_preprocessor_2']
        expected_probs = expected_data['expected_probs_preprocessor'][version]
        expected_gradients1 = expected_data['expected_gradients_preprocessor_1'][version]
        expected_gradients2 = expected_data['expected_gradients_preprocessor_2'][version]
        expected_gradients3 = expected_data['expected_gradients_preprocessor_3'][version]
        x = np.array([x1 * 100, x1 * 100, x1 * 100], dtype=ART_NUMPY_DTYPE)
        y = np.array(['SIX', 'HI', 'GOOD'])
        (probs, sizes) = speech_recognizer.predict(x, batch_size=1, transcription_output=False)
        np.testing.assert_array_almost_equal(probs[1][1], expected_probs, decimal=3)
        np.testing.assert_array_almost_equal(sizes, expected_sizes)
        transcriptions = speech_recognizer.predict(x, batch_size=2, transcription_output=True)
        assert (expected_transcriptions1 == transcriptions).all()
        transcriptions = speech_recognizer.predict(x[[0]], batch_size=1, transcription_output=True)
        assert (expected_transcriptions2 == transcriptions).all()
        grads = speech_recognizer.loss_gradient(x, y)
        assert grads[0].shape == (1300,)
        assert grads[1].shape == (1300,)
        assert grads[2].shape == (1300,)
        np.testing.assert_array_almost_equal(grads[0][:20], expected_gradients1, decimal=-2)
        np.testing.assert_array_almost_equal(grads[1][:20], expected_gradients2, decimal=-2)
        np.testing.assert_array_almost_equal(grads[2][:20], expected_gradients3, decimal=-2)
        parameters = speech_recognizer.model.parameters()
        speech_recognizer._optimizer = torch.optim.SGD(parameters, lr=0.01)
        _ = speech_recognizer.predict(x, batch_size=2, transcription_output=True)
        speech_recognizer.fit(x=x, y=y, batch_size=2, nb_epochs=10)
        _ = speech_recognizer.predict(x, batch_size=2, transcription_output=True)
    except ARTTestException as e:
        art_warning(e)