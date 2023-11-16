import math
import cmath
from gnuradio import gr
from gnuradio.blocks import rotator_cc
from .filter_python import fft_filter_ccc
__all__ = ['freq_xlating_fft_filter_ccc']

class freq_xlating_fft_filter_ccc(gr.hier_block2):

    def __init__(self, decim, taps, center_freq, samp_rate):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'freq_xlating_fft_filter_ccc', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.decim = decim
        self.taps = taps
        self.center_freq = center_freq
        self.samp_rate = samp_rate
        self._filter = fft_filter_ccc(decim, taps)
        self._rotator = rotator_cc(0.0)
        self.connect(self, self._filter, self._rotator, self)
        self._refresh()

    def _rotate_taps(self, taps, phase_inc):
        if False:
            i = 10
            return i + 15
        return [x * cmath.exp(i * phase_inc * 1j) for (i, x) in enumerate(taps)]

    def _refresh(self):
        if False:
            i = 10
            return i + 15
        phase_inc = 2.0 * math.pi * self.center_freq / self.samp_rate
        rtaps = self._rotate_taps(self.taps, phase_inc)
        self._filter.set_taps(rtaps)
        self._rotator.set_phase_inc(-self.decim * phase_inc)

    def set_taps(self, taps):
        if False:
            return 10
        self.taps = taps
        self._refresh()

    def set_center_freq(self, center_freq):
        if False:
            print('Hello World!')
        self.center_freq = center_freq
        self._refresh()

    def set_nthreads(self, nthreads):
        if False:
            while True:
                i = 10
        self._filter.set_nthreads(nthreads)

    def declare_sample_delay(self, samp_delay):
        if False:
            return 10
        self._filter.declare_sample_delay(samp_delay)