"""Defines CalibrationAlgorithm for calculating min and max values calculated by calibration method."""
import abc
import itertools
import logging
import numpy as np
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stats_pb2
_CalibrationMethod = quant_opts_pb2.CalibrationOptions.CalibrationMethod
_REGISTRY = {}

def _implements(calib_method: _CalibrationMethod):
    if False:
        i = 10
        return i + 15

    def decorator(cls):
        if False:
            print('Hello World!')
        assert calib_method not in _REGISTRY
        _REGISTRY[calib_method] = cls
        return cls
    return decorator

class _CalibrationAlgorithmBase(abc.ABC):
    """Abstract base class for calibration algorithm."""

    def __init__(self, statistics: calib_stats_pb2.CalibrationStatistics, calib_opts: quant_opts_pb2.CalibrationOptions):
        if False:
            print('Hello World!')
        self._statistics = statistics
        self._calib_opts = calib_opts

    @abc.abstractmethod
    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            print('Hello World!')
        pass

class _HistogramCalibrationAlgorithmBase(_CalibrationAlgorithmBase):
    """Base class for histogram calibrators."""

    def __init__(self, statistics: calib_stats_pb2.CalibrationStatistics, calib_opts: quant_opts_pb2.CalibrationOptions):
        if False:
            while True:
                i = 10
        'Builds histogram using statistics.histogram_statistics.\n\n    lower_bound                                    hist_mid\n         v                                            v\n         |=========|=========|=========|=========|=========|\n                    bin width\n\n    Args:\n      statistics: Collected calibration statistics.\n      calib_opts: Calibration options used for calculating min and max.\n    '
        super().__init__(statistics, calib_opts)
        hist_stats = statistics.histogram_statistics
        self._bin_width = hist_stats.bin_width
        self._lower_bound = hist_stats.lower_bound
        self._hist_freq = np.array(hist_stats.hist_freq)
        self._num_bins = len(self._hist_freq)
        self._num_bits = 8
        first_mid = self._lower_bound + self._bin_width / 2
        last_mid = first_mid + (self._num_bins - 1) * self._bin_width
        self._hist_mids = np.linspace(first_mid, last_mid, self._num_bins)

    def _get_dequantized_hist_mids_after_quantize(self, quant_min: float, quant_max: float) -> np.ndarray:
        if False:
            print('Hello World!')
        'Quantizes and dequantizes hist_mids using quant_min and quant_max.\n\n    Quantization converts the range of numbers from [quant_min, quant_max] to\n    [0, 2^num_bits - 1]. Values less than quant_min are converted to 0, and\n    values greater than quant_max are converted to 2^num_bits - 1.\n\n    The histogram represents the distribution of the data, and our goal is to\n    find the quant_min and quant_max that best describe this distribution. To do\n    this, we quantize hist_mids using quant_min and quant_max and dequantize\n    them again. Then the difference between hist_mids and dequantized hist_mids\n    equates to quantization error when using quant_min and quant_max.\n\n\n    Args:\n      quant_min: The minimum real value that can be represented by a quantized\n        value.\n      quant_max: The maximum real value that can be represented by a quantized\n        value.\n\n    Returns:\n      dequantized hist_mids after quantizing by quant_min and quant_max\n    '
        maxbound = 2 ** self._num_bits - 1
        minbound = 0
        scale = (quant_max - quant_min) / maxbound
        zero_point = -quant_min / scale
        if abs(zero_point) > 9000000000.0:
            zero_point = 9000000000.0
        if abs(scale) < 1e-09:
            scale = 1e-09
        zero_point = round(zero_point)
        quantized_hist_mids = np.clip(np.round(self._hist_mids / scale) + zero_point, minbound, maxbound)
        dequantized_hist_mids = scale * (quantized_hist_mids - zero_point)
        return dequantized_hist_mids

    def _get_weighted_mean_squared_error(self, quant_min, quant_max) -> tuple[float, float, float]:
        if False:
            return 10
        'Gets mean squared error between hist_mids and dequantized hist_mids.\n\n    Quantization converts the range of numbers from [quant_min, quant_max] to\n    [0, 2^num_bits - 1]. Values less than quant_min are converted to 0, and\n    values greater than quant_max are converted to 2^num_bits - 1.\n\n    Args:\n      quant_min: The minimum real value that can be represented by a quantized\n        value.\n      quant_max: The maximum real value that can be represented by a quantized\n        value.\n\n    Returns:\n      (error, quant_min, quant_max): Tuple of weighted mean squared error.\n      error = (hist_mids - dequantized_hist_mids)**2 * hist_freq\n    '
        dequantized_hist_mids = self._get_dequantized_hist_mids_after_quantize(quant_min, quant_max)
        squared_error = (self._hist_mids - dequantized_hist_mids) ** 2
        weighted_error = np.sum(squared_error * self._hist_freq)
        return (weighted_error, quant_min, quant_max)

    def _get_min_max_value_by_expanding_range(self, start_idx: int) -> tuple[float, float]:
        if False:
            for i in range(10):
                print('nop')
        'Starting from start_idx, expand left and right alternately to find the min value of mse loss.\n\n    Args:\n      start_idx: Index to start quantization.\n\n    Returns:\n      (min_value, max_value): Min and max calculated.\n    '
        mse_min = (float('inf'), float('inf'), float('inf'))
        (left, right) = (start_idx, start_idx)
        move_left = True
        while not (left == 0 and right == self._num_bins - 1):
            if move_left and left > 0 or right == self._num_bins - 1:
                left = max(left - 1, 0)
            else:
                right = min(right + 1, self._num_bins - 1)
            move_left = not move_left
            (quant_min, quant_max) = (self._hist_mids[left], self._hist_mids[right])
            mse_tuple = self._get_weighted_mean_squared_error(quant_min, quant_max)
            mse_min = min(mse_tuple, mse_min)
        (min_value, max_value) = (mse_min[1], mse_min[2])
        return (min_value, max_value)

@_implements(_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX)
class _MinMax(_CalibrationAlgorithmBase):
    """MinMaxCalibrationAlgorithm for calculating min and max values of calibration result.

  MinMax calibration calculates the global min and global max values.

  global min = min of given sample inputs
  global max = max of given sample inputs
  """

    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            print('Hello World!')
        'Calculates the global min and max values.\n\n    Returns:\n      (min_value, max_value): Min and max calculated using MinMax\n    '
        return (self._statistics.min_max_statistics.global_min, self._statistics.min_max_statistics.global_max)

@_implements(_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX)
class _AverageMinMax(_CalibrationAlgorithmBase):
    """AverageMinMaxCalibrationAlgorithm for calculating min and max values of calibration result.

  AverageMinMax calibration calculates the average of min and max values.
  average of min = sum of min values / number of samples
  average of max = sum of max values / number of samples
  """

    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            for i in range(10):
                print('nop')
        'Calculates the average of min and max values.\n\n    Returns:\n      (min_value, max_value): Min and max calculated using AverageMinMax\n\n    Raises:\n      ValueError: num_samples is 0.\n    '
        average_min_max_statistics = self._statistics.average_min_max_statistics
        num_samples = average_min_max_statistics.num_samples
        if num_samples == 0:
            raise ValueError(f'num_samples must not be 0 when calibration method is AverageMinMax: {self._calib_opts}')
        (min_value, max_value) = (average_min_max_statistics.min_sum / num_samples, average_min_max_statistics.max_sum / num_samples)
        return (min_value, max_value)

@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE)
class _HistogramPercentile(_HistogramCalibrationAlgorithmBase):
    """HistogramPercentile for calculating min and max values of calibration result."""

    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            i = 10
            return i + 15
        'Calculates min and max from statistics using calibration options.\n\n    A "percentile" is a statistical concept that represents the value below\n    which a given percentage of data falls in a dataset. It involves sorting the\n    data from smallest to largest and then finding the value at a specified\n    percentage position. For example, the 0.01 percentile represents the value\n    in a given data set that corresponds to the lowest 0.01% of the data.\n\n    HistogramPercentile calibration uses min_percentile and max_percentile to\n    find min and max.\n\n    min_percentile and max_percentile must be in range [0, 100].\n    min_percentile is 0.001 by default.\n    max_percentile is 99.999 by default.\n\n    Returns:\n      (min_value, max_value): Min and max calculated using HistogramPercentile\n    '
        total_freq = sum(self._hist_freq)
        hist_freq_cumsum = np.cumsum(self._hist_freq) / total_freq
        (min_quantile, max_quantile) = (self._calib_opts.calibration_parameters.min_percentile / 100.0, self._calib_opts.calibration_parameters.max_percentile / 100.0)
        (min_quantile_idx, max_quantile_idx) = (np.searchsorted(hist_freq_cumsum, min_quantile, side='right'), np.searchsorted(hist_freq_cumsum, max_quantile, side='left'))
        (min_value, max_value) = (self._hist_mids[min_quantile_idx], self._hist_mids[max_quantile_idx])
        return (min_value, max_value)

@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE)
class _HistogramMseBruteforce(_HistogramCalibrationAlgorithmBase):
    """HistogramMseBruteforce for calculating min and max values of calibration result."""

    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            while True:
                i = 10
        'Finds the optimal quant_min and quant_max by testing all possible cases.\n\n    It guarantees optimal quant_min and quant_max for the representative\n    dataset, but not for the test dataset.\n\n    Returns:\n      (min_value, max_value): Min and max calculated using\n      HistogramMseBruteforce.\n    '
        if self._num_bins > 512:
            logging.warning('num_bins=%d is too large. The HISTOGRAM_MSE_BRUTEFORCE method tests all histogram mid value pairs, so it may take a long time.', self._num_bins)
        mse_min = (float('inf'), float('inf'), float('inf'))
        for (left, right) in itertools.combinations(range(self._num_bins), 2):
            (quant_min, quant_max) = (self._hist_mids[left], self._hist_mids[right])
            mse_tuple = self._get_weighted_mean_squared_error(quant_min, quant_max)
            mse_min = min(mse_tuple, mse_min)
        (min_value, max_value) = (mse_min[1], mse_min[2])
        return (min_value, max_value)

@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY)
class _HistogramMseMaxFrequency(_HistogramCalibrationAlgorithmBase):
    """HistogramMseMaxFrequency for calculating min and max values of calibration result."""

    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            print('Hello World!')
        'Finds min and max starting from the index of the max frequency.\n\n     The HistogramMseMaxFrequency method starts from the bin with the highest\n     frequency and expands the range to both sides. This performs well when data\n     is well spread on both sides of the max frequency.\n\n    Returns:\n      (min_value, max_value): Min and max calculated using method to expand the\n      range based on max frequency.\n    '
        freq_max_idx = np.argmax(self._hist_freq)
        return self._get_min_max_value_by_expanding_range(freq_max_idx)

@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC)
class _HistogramMseSymmetric(_HistogramCalibrationAlgorithmBase):
    """HistogramMseSymmetric for calculating min and max values of calibration result."""

    def get_min_max_value(self) -> tuple[float, float]:
        if False:
            for i in range(10):
                print('nop')
        'Finds min and max starting from the center index.\n\n    The HistogramMseSymmetric method starts from the center bin and expands the\n    range to both sides. This works better when the data is well-centered.\n\n    Returns:\n      (min_value, max_value): Min and max calculated using the method starting\n      from center and expanding.\n    '
        return self._get_min_max_value_by_expanding_range(self._num_bins // 2)

def get_min_max_value(statistics: calib_stats_pb2.CalibrationStatistics, calib_opts: quant_opts_pb2.CalibrationOptions) -> tuple[float, float]:
    if False:
        return 10
    'Calculates min and max from statistics using calibration options.\n\n  Args:\n    statistics: Collected calibration statistics.\n    calib_opts: Calibration options used for calculating min and max.\n\n  Returns:\n    (min_value, max_value): Min and max calculated using calib_opts.\n\n  Raises:\n    ValueError: Unsupported calibration method is given.\n  '
    calib_method = calib_opts.calibration_method
    if calib_method not in _REGISTRY:
        raise ValueError(f'Unsupported calibration method: {calib_method}')
    calibration_algorithm = _REGISTRY[calib_method](statistics, calib_opts)
    return calibration_algorithm.get_min_max_value()