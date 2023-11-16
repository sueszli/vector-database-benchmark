"""Mel-Frequency Cepstral Coefficients (MFCCs) ops."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('signal.mfccs_from_log_mel_spectrograms')
@dispatch.add_dispatch_support
def mfccs_from_log_mel_spectrograms(log_mel_spectrograms, name=None):
    if False:
        print('Hello World!')
    "Computes [MFCCs][mfcc] of `log_mel_spectrograms`.\n\n  Implemented with GPU-compatible ops and supports gradients.\n\n  [Mel-Frequency Cepstral Coefficient (MFCC)][mfcc] calculation consists of\n  taking the DCT-II of a log-magnitude mel-scale spectrogram. [HTK][htk]'s MFCCs\n  use a particular scaling of the DCT-II which is almost orthogonal\n  normalization. We follow this convention.\n\n  All `num_mel_bins` MFCCs are returned and it is up to the caller to select\n  a subset of the MFCCs based on their application. For example, it is typical\n  to only use the first few for speech recognition, as this results in\n  an approximately pitch-invariant representation of the signal.\n\n  For example:\n\n  ```python\n  batch_size, num_samples, sample_rate = 32, 32000, 16000.0\n  # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].\n  pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)\n\n  # A 1024-point STFT with frames of 64 ms and 75% overlap.\n  stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256,\n                         fft_length=1024)\n  spectrograms = tf.abs(stfts)\n\n  # Warp the linear scale spectrograms into the mel-scale.\n  num_spectrogram_bins = stfts.shape[-1].value\n  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80\n  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(\n    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,\n    upper_edge_hertz)\n  mel_spectrograms = tf.tensordot(\n    spectrograms, linear_to_mel_weight_matrix, 1)\n  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(\n    linear_to_mel_weight_matrix.shape[-1:]))\n\n  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.\n  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)\n\n  # Compute MFCCs from log_mel_spectrograms and take the first 13.\n  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(\n    log_mel_spectrograms)[..., :13]\n  ```\n\n  Args:\n    log_mel_spectrograms: A `[..., num_mel_bins]` `float32`/`float64` `Tensor`\n      of log-magnitude mel-scale spectrograms.\n    name: An optional name for the operation.\n  Returns:\n    A `[..., num_mel_bins]` `float32`/`float64` `Tensor` of the MFCCs of\n    `log_mel_spectrograms`.\n\n  Raises:\n    ValueError: If `num_mel_bins` is not positive.\n\n  [mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum\n  [htk]: https://en.wikipedia.org/wiki/HTK_(software)\n  "
    with ops.name_scope(name, 'mfccs_from_log_mel_spectrograms', [log_mel_spectrograms]):
        log_mel_spectrograms = ops.convert_to_tensor(log_mel_spectrograms)
        if log_mel_spectrograms.shape.ndims and log_mel_spectrograms.shape.dims[-1].value is not None:
            num_mel_bins = log_mel_spectrograms.shape.dims[-1].value
            if num_mel_bins == 0:
                raise ValueError('num_mel_bins must be positive. Got: %s' % log_mel_spectrograms)
        else:
            num_mel_bins = array_ops.shape(log_mel_spectrograms)[-1]
        dct2 = dct_ops.dct(log_mel_spectrograms, type=2)
        return dct2 * math_ops.rsqrt(math_ops.cast(num_mel_bins, dct2.dtype) * 2.0)