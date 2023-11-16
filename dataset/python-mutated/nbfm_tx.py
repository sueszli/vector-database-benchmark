import math
from gnuradio import gr, filter
from .fm_emph import fm_preemph
from . import analog_python as analog

class nbfm_tx(gr.hier_block2):
    """
    Narrow Band FM Transmitter.

    Takes a single float input stream of audio samples in the range [-1,+1]
    and produces a single FM modulated complex baseband output.

    Args:
        audio_rate: sample rate of audio stream, >= 16k (integer)
        quad_rate: sample rate of output stream (integer)
        tau: preemphasis time constant (default 75e-6) (float)
        max_dev: maximum deviation in Hz (default 5e3) (float)
        fh: high frequency at which to flatten preemphasis; < 0 means default of 0.925*quad_rate/2.0 (float)

    quad_rate must be an integer multiple of audio_rate.
    """

    def __init__(self, audio_rate, quad_rate, tau=7.5e-05, max_dev=5000.0, fh=-1.0):
        if False:
            while True:
                i = 10
        gr.hier_block2.__init__(self, 'nbfm_tx', gr.io_signature(1, 1, gr.sizeof_float), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self._audio_rate = audio_rate = int(audio_rate)
        self._quad_rate = quad_rate = int(quad_rate)
        if quad_rate % audio_rate != 0:
            raise ValueError('quad_rate is not an integer multiple of audio_rate')
        do_interp = audio_rate != quad_rate
        if do_interp:
            interp_factor = int(quad_rate / audio_rate)
            interp_taps = filter.optfir.low_pass(interp_factor, quad_rate, 4500, 7000, 0.1, 40)
            self.interpolator = filter.interp_fir_filter_fff(interp_factor, interp_taps)
        self.preemph = fm_preemph(quad_rate, tau=tau, fh=fh)
        k = 2 * math.pi * max_dev / quad_rate
        self.modulator = analog.frequency_modulator_fc(k)
        if do_interp:
            self.connect(self, self.interpolator, self.preemph, self.modulator, self)
        else:
            self.connect(self, self.preemph, self.modulator, self)

    def set_max_deviation(self, max_dev):
        if False:
            while True:
                i = 10
        k = 2 * math.pi * max_dev / self._quad_rate
        self.modulator.set_sensitivity(k)

class ctcss_gen_f(gr.hier_block2):

    def __init__(self, sample_rate, tone_freq):
        if False:
            i = 10
            return i + 15
        gr.hier_block2.__init__(self, 'ctcss_gen_f', gr.io_signature(0, 0, 0), gr.io_signature(1, 1, gr.sizeof_float))
        self.plgen = analog.sig_source_f(sample_rate, analog.GR_SIN_WAVE, tone_freq, 0.1, 0.0)
        self.connect(self.plgen, self)