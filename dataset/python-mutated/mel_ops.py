"""mel conversion ops."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def _mel_to_hertz(mel_values, name=None):
    if False:
        while True:
            i = 10
    'Converts frequencies in `mel_values` from the mel scale to linear scale.\n\n  Args:\n    mel_values: A `Tensor` of frequencies in the mel scale.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of the same shape and type as `mel_values` containing linear\n    scale frequencies in Hertz.\n  '
    with ops.name_scope(name, 'mel_to_hertz', [mel_values]):
        mel_values = ops.convert_to_tensor(mel_values)
        return _MEL_BREAK_FREQUENCY_HERTZ * (math_ops.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0)

def _hertz_to_mel(frequencies_hertz, name=None):
    if False:
        i = 10
        return i + 15
    'Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.\n\n  Args:\n    frequencies_hertz: A `Tensor` of frequencies in Hertz.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of the same shape and type of `frequencies_hertz` containing\n    frequencies in the mel scale.\n  '
    with ops.name_scope(name, 'hertz_to_mel', [frequencies_hertz]):
        frequencies_hertz = ops.convert_to_tensor(frequencies_hertz)
        return _MEL_HIGH_FREQUENCY_Q * math_ops.log(1.0 + frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)

def _validate_arguments(num_mel_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, dtype):
    if False:
        print('Hello World!')
    'Checks the inputs to linear_to_mel_weight_matrix.'
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
    if lower_edge_hertz < 0.0:
        raise ValueError('lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' % (lower_edge_hertz, upper_edge_hertz))
    if not isinstance(sample_rate, tensor.Tensor):
        if sample_rate <= 0.0:
            raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
        if upper_edge_hertz > sample_rate / 2:
            raise ValueError('upper_edge_hertz must not be larger than the Nyquist frequency (sample_rate / 2). Got %s for sample_rate: %s' % (upper_edge_hertz, sample_rate))
    if not dtype.is_floating:
        raise ValueError('dtype must be a floating point type. Got: %s' % dtype)

@tf_export('signal.linear_to_mel_weight_matrix')
@dispatch.add_dispatch_support
def linear_to_mel_weight_matrix(num_mel_bins=20, num_spectrogram_bins=129, sample_rate=8000, lower_edge_hertz=125.0, upper_edge_hertz=3800.0, dtype=dtypes.float32, name=None):
    if False:
        return 10
    'Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].\n\n  Returns a weight matrix that can be used to re-weight a `Tensor` containing\n  `num_spectrogram_bins` linearly sampled frequency information from\n  `[0, sample_rate / 2]` into `num_mel_bins` frequency information from\n  `[lower_edge_hertz, upper_edge_hertz]` on the [mel scale][mel].\n\n  This function follows the [Hidden Markov Model Toolkit\n  (HTK)](http://htk.eng.cam.ac.uk/) convention, defining the mel scale in\n  terms of a frequency in hertz according to the following formula:\n\n      $$\\textrm{mel}(f) = 2595 * \\textrm{log}_{10}(1 + \\frac{f}{700})$$\n\n  In the returned matrix, all the triangles (filterbanks) have a peak value\n  of 1.0.\n\n  For example, the returned matrix `A` can be used to right-multiply a\n  spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear\n  scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram"\n  `M` of shape `[frames, num_mel_bins]`.\n\n      # `S` has shape [frames, num_spectrogram_bins]\n      # `M` has shape [frames, num_mel_bins]\n      M = tf.matmul(S, A)\n\n  The matrix can be used with `tf.tensordot` to convert an arbitrary rank\n  `Tensor` of linear-scale spectral bins into the mel scale.\n\n      # S has shape [..., num_spectrogram_bins].\n      # M has shape [..., num_mel_bins].\n      M = tf.tensordot(S, A, 1)\n\n  Args:\n    num_mel_bins: Python int. How many bands in the resulting mel spectrum.\n    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the\n      source spectrogram data, which is understood to be `fft_size // 2 + 1`,\n      i.e. the spectrogram only contains the nonredundant FFT bins.\n    sample_rate: An integer or float `Tensor`. Samples per second of the input\n      signal used to create the spectrogram. Used to figure out the frequencies\n      corresponding to each spectrogram bin, which dictates how they are mapped\n      into the mel scale.\n    lower_edge_hertz: Python float. Lower bound on the frequencies to be\n      included in the mel spectrum. This corresponds to the lower edge of the\n      lowest triangular band.\n    upper_edge_hertz: Python float. The desired top edge of the highest\n      frequency band.\n    dtype: The `DType` of the result matrix. Must be a floating point type.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.\n\n  Raises:\n    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not\n      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly\n      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.\n\n  [mel]: https://en.wikipedia.org/wiki/Mel_scale\n  '
    with ops.name_scope(name, 'linear_to_mel_weight_matrix') as name:
        if isinstance(sample_rate, tensor.Tensor):
            maybe_const_val = tensor_util.constant_value(sample_rate)
            if maybe_const_val is not None:
                sample_rate = maybe_const_val
        _validate_arguments(num_mel_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, dtype)
        sample_rate = math_ops.cast(sample_rate, dtype, name='sample_rate')
        lower_edge_hertz = ops.convert_to_tensor(lower_edge_hertz, dtype, name='lower_edge_hertz')
        upper_edge_hertz = ops.convert_to_tensor(upper_edge_hertz, dtype, name='upper_edge_hertz')
        zero = ops.convert_to_tensor(0.0, dtype)
        bands_to_zero = 1
        nyquist_hertz = sample_rate / 2.0
        linear_frequencies = math_ops.linspace(zero, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
        spectrogram_bins_mel = array_ops.expand_dims(_hertz_to_mel(linear_frequencies), 1)
        band_edges_mel = shape_ops.frame(math_ops.linspace(_hertz_to_mel(lower_edge_hertz), _hertz_to_mel(upper_edge_hertz), num_mel_bins + 2), frame_length=3, frame_step=1)
        (lower_edge_mel, center_mel, upper_edge_mel) = tuple((array_ops.reshape(t, [1, num_mel_bins]) for t in array_ops.split(band_edges_mel, 3, axis=1)))
        lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
        mel_weights_matrix = math_ops.maximum(zero, math_ops.minimum(lower_slopes, upper_slopes))
        return array_ops.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]], name=name)