import math
from enum import Enum
import numpy as np
from urh import settings
from urh.cythonext import signal_functions
from urh.util import util
from urh.util.Logger import logger

class FilterType(Enum):
    moving_average = 'moving average'
    dc_correction = 'DC correction'
    custom = 'custom'

class Filter(object):
    BANDWIDTHS = {'Very Narrow': 0.001, 'Narrow': 0.01, 'Medium': 0.08, 'Wide': 0.1, 'Very Wide': 0.42}

    def __init__(self, taps: list, filter_type: FilterType=FilterType.custom):
        if False:
            print('Hello World!')
        self.filter_type = filter_type
        self.taps = taps

    def work(self, input_signal: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        if self.filter_type == FilterType.dc_correction:
            return input_signal - np.mean(input_signal, axis=0)
        else:
            return self.apply_fir_filter(input_signal.flatten())

    def apply_fir_filter(self, input_signal: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        if input_signal.dtype != np.complex64:
            tmp = np.empty(len(input_signal) // 2, dtype=np.complex64)
            tmp.real = input_signal[0::2]
            tmp.imag = input_signal[1::2]
            input_signal = tmp
        return signal_functions.fir_filter(input_signal, np.array(self.taps, dtype=np.complex64))

    @staticmethod
    def read_configured_filter_bw() -> float:
        if False:
            return 10
        bw_type = settings.read('bandpass_filter_bw_type', 'Medium', str)
        if bw_type in Filter.BANDWIDTHS:
            return Filter.BANDWIDTHS[bw_type]
        if bw_type.lower() == 'custom':
            return settings.read('bandpass_filter_custom_bw', 0.1, float)
        return 0.08

    @staticmethod
    def get_bandwidth_from_filter_length(N):
        if False:
            for i in range(10):
                print('nop')
        return 4 / N

    @staticmethod
    def get_filter_length_from_bandwidth(bw):
        if False:
            for i in range(10):
                print('nop')
        N = int(math.ceil(4 / bw))
        return N + 1 if N % 2 == 0 else N

    @staticmethod
    def fft_convolve_1d(x: np.ndarray, h: np.ndarray):
        if False:
            print('Hello World!')
        n = len(x) + len(h) - 1
        n_opt = 1 << (n - 1).bit_length()
        if np.issubdtype(x.dtype, np.complexfloating) or np.issubdtype(h.dtype, np.complexfloating):
            (fft, ifft) = (np.fft.fft, np.fft.ifft)
        else:
            (fft, ifft) = (np.fft.rfft, np.fft.irfft)
        result = ifft(fft(x, n_opt) * fft(h, n_opt), n_opt)[0:n]
        too_much = (len(result) - len(x)) // 2
        return result[too_much:-too_much]

    @staticmethod
    def apply_bandpass_filter(data, f_low, f_high, filter_bw=0.08):
        if False:
            for i in range(10):
                print('nop')
        if f_low > f_high:
            (f_low, f_high) = (f_high, f_low)
        f_low = util.clip(f_low, -0.5, 0.5)
        f_high = util.clip(f_high, -0.5, 0.5)
        h = Filter.design_windowed_sinc_bandpass(f_low, f_high, filter_bw)
        if len(h) < 8 * math.log(math.sqrt(len(data))):
            logger.debug('Use normal convolve')
            return np.convolve(data, h, 'same')
        else:
            logger.debug('Use FFT convolve')
            return Filter.fft_convolve_1d(data, h)

    @staticmethod
    def design_windowed_sinc_lpf(fc, bw):
        if False:
            print('Hello World!')
        N = Filter.get_filter_length_from_bandwidth(bw)
        h = np.sinc(2 * fc * (np.arange(N) - (N - 1) / 2.0))
        w = np.blackman(N)
        h = h * w
        h_unity = h / np.sum(h)
        return h_unity

    @staticmethod
    def design_windowed_sinc_bandpass(f_low, f_high, bw):
        if False:
            return 10
        f_shift = (f_low + f_high) / 2
        f_c = (f_high - f_low) / 2
        N = Filter.get_filter_length_from_bandwidth(bw)
        return Filter.design_windowed_sinc_lpf(f_c, bw=bw) * np.exp(complex(0, 1) * np.pi * 2 * f_shift * np.arange(0, N, dtype=complex))