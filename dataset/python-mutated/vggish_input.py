"""Compute input examples for VGGish from audio waveform."""
from . import mel_features
from . import vggish_params
from turicreate._deps.minimal_package import _minimal_package_import_check
import numpy as np

def waveform_to_examples(data, sample_rate):
    if False:
        for i in range(10):
            print('nop')
    'Converts audio waveform into an array of examples for VGGish.\n\n  Args:\n    data: np.array of either one dimension (mono) or two dimensions\n      (multi-channel, with the outer dimension representing channels).\n      Each sample is generally expected to lie in the range [-1.0, +1.0],\n      although this is not required.\n    sample_rate: Sample rate of data.\n\n  Returns:\n    3-D np.array of shape [num_examples, num_frames, num_bands] which represents\n    a sequence of examples, each of which contains a patch of log mel\n    spectrogram, covering num_frames frames of audio and num_bands mel frequency\n    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.\n  '
    resampy = _minimal_package_import_check('resampy')
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)
    log_mel = mel_features.log_mel_spectrogram(data, audio_sample_rate=vggish_params.SAMPLE_RATE, log_offset=vggish_params.LOG_OFFSET, window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS, hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS, num_mel_bins=vggish_params.NUM_MEL_BINS, lower_edge_hertz=vggish_params.MEL_MIN_HZ, upper_edge_hertz=vggish_params.MEL_MAX_HZ)
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(log_mel, window_length=example_window_length, hop_length=example_hop_length)
    return log_mel_examples

def wavfile_to_examples(wav_file):
    if False:
        return 10
    'Convenience wrapper around waveform_to_examples() for a common WAV format.\n\n  Args:\n    wav_file: String path to a file, or a file-like object. The file\n    is assumed to contain WAV audio data with signed 16-bit PCM samples.\n\n  Returns:\n    See waveform_to_examples.\n  '
    from scipy.io import wavfile
    (sr, wav_data) = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0
    return waveform_to_examples(samples, sr)