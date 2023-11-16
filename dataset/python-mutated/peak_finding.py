"""Benchmarks for peak finding related functions."""
from .common import Benchmark, safe_import
with safe_import():
    from scipy.signal import find_peaks, peak_prominences, peak_widths
    from scipy.datasets import electrocardiogram

class FindPeaks(Benchmark):
    """Benchmark `scipy.signal.find_peaks`.

    Notes
    -----
    The first value of `distance` is None in which case the benchmark shows
    the actual speed of the underlying maxima finding function.
    """
    param_names = ['distance']
    params = [[None, 8, 64, 512, 4096]]

    def setup(self, distance):
        if False:
            print('Hello World!')
        self.x = electrocardiogram()

    def time_find_peaks(self, distance):
        if False:
            print('Hello World!')
        find_peaks(self.x, distance=distance)

class PeakProminences(Benchmark):
    """Benchmark `scipy.signal.peak_prominences`."""
    param_names = ['wlen']
    params = [[None, 8, 64, 512, 4096]]

    def setup(self, wlen):
        if False:
            i = 10
            return i + 15
        self.x = electrocardiogram()
        self.peaks = find_peaks(self.x)[0]

    def time_peak_prominences(self, wlen):
        if False:
            while True:
                i = 10
        peak_prominences(self.x, self.peaks, wlen)

class PeakWidths(Benchmark):
    """Benchmark `scipy.signal.peak_widths`."""
    param_names = ['rel_height']
    params = [[0, 0.25, 0.5, 0.75, 1]]

    def setup(self, rel_height):
        if False:
            return 10
        self.x = electrocardiogram()
        self.peaks = find_peaks(self.x)[0]
        self.prominence_data = peak_prominences(self.x, self.peaks)

    def time_peak_widths(self, rel_height):
        if False:
            while True:
                i = 10
        peak_widths(self.x, self.peaks, rel_height, self.prominence_data)