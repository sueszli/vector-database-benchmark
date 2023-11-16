import logging
import numpy as np
import pytest
from art.config import ART_NUMPY_DTYPE
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_module('espresso')
@pytest.mark.skip_framework('tensorflow', 'tensorflow2v1', 'keras', 'kerastf', 'mxnet', 'non_dl_frameworks')
@pytest.mark.parametrize('device_type', ['cpu'])
def test_pytorch_espresso(art_warning, expected_values, device_type):
    if False:
        while True:
            i = 10
    import torch
    from art.estimators.speech_recognition.pytorch_espresso import PyTorchEspresso
    try:
        speech_recognizer = PyTorchEspresso(model='librispeech_transformer', espresso_config_filepath=None, device_type=device_type)
        expected_data = expected_values()
        x1 = expected_data['x1']
        x2 = expected_data['x2']
        x3 = expected_data['x3']
        expected_transcriptions1 = expected_data['expected_transcriptions1']
        expected_transcriptions2 = expected_data['expected_transcriptions2']
        expected_gradients1 = expected_data['expected_gradients1']
        expected_gradients2 = expected_data['expected_gradients2']
        expected_gradients3 = expected_data['expected_gradients3']
        x = np.array([np.array(x1 * 100, dtype=ART_NUMPY_DTYPE), np.array(x2 * 100, dtype=ART_NUMPY_DTYPE), np.array(x3 * 100, dtype=ART_NUMPY_DTYPE)])
        y = np.array(['SIX', 'HI', 'GOOD'])
        _ = speech_recognizer.predict(x[[0]], batch_size=2)
        transcriptions = speech_recognizer.predict(x, batch_size=2)
        assert (expected_transcriptions1 == transcriptions).all()
        transcriptions = speech_recognizer.predict(np.array([x[0]]), batch_size=2)
        assert (expected_transcriptions2 == transcriptions).all()
        grads = speech_recognizer.loss_gradient(x, y)
        assert grads[0].shape == (1300,)
        assert grads[1].shape == (1500,)
        assert grads[2].shape == (1400,)
        np.testing.assert_array_almost_equal(grads[0][:20], expected_gradients1, decimal=-2)
        np.testing.assert_array_almost_equal(grads[1][:20], expected_gradients2, decimal=-2)
        np.testing.assert_array_almost_equal(grads[2][:20], expected_gradients3, decimal=-2)
        with pytest.raises(NotImplementedError):
            speech_recognizer.fit(x=x, y=y, batch_size=2, nb_epochs=5)
        local_batch_size = len(x)
        real_lengths = np.array([x_.shape[0] for x_ in x])
        local_max_length = np.max(real_lengths)
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float64)
        original_input = np.zeros([local_batch_size, local_max_length], dtype=np.float64)
        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, :len(x[local_batch_size_idx])] = 1
            original_input[local_batch_size_idx, :len(x[local_batch_size_idx])] = x[local_batch_size_idx]
        (loss, decoded_output) = speech_recognizer.compute_loss_and_decoded_output(masked_adv_input=torch.tensor(original_input), original_output=y)
        assert loss.detach().numpy() == pytest.approx(46.3156, abs=20.0)
        assert all(decoded_output == ['EH', 'EH', 'EH'])
    except ARTTestException as e:
        art_warning(e)