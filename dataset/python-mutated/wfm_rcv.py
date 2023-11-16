import math
from gnuradio import gr, filter, fft
from . import analog_python as analog
from .fm_emph import fm_deemph

class wfm_rcv(gr.hier_block2):

    def __init__(self, quad_rate, audio_decimation, deemph_tau=7.5e-05):
        if False:
            print('Hello World!')
        '\n        Hierarchical block for demodulating a broadcast FM signal.\n\n        The input is the downconverted complex baseband signal (gr_complex).\n        The output is the demodulated audio (float).\n\n        Args:\n            quad_rate: input sample rate of complex baseband input. (float)\n            audio_decimation: how much to decimate quad_rate to get to audio. (integer)\n            deemph_tau: deemphasis ime constant in seconds (75us in US and South Korea, 50us everywhere else). (float)\n        '
        gr.hier_block2.__init__(self, 'wfm_rcv', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(1, 1, gr.sizeof_float))
        if audio_decimation != int(audio_decimation):
            raise ValueError('audio_decimation needs to be an integer')
        audio_decimation = int(audio_decimation)
        volume = 20.0
        max_dev = 75000.0
        fm_demod_gain = quad_rate / (2 * math.pi * max_dev)
        audio_rate = quad_rate / audio_decimation
        self.fm_demod = analog.quadrature_demod_cf(fm_demod_gain)
        self.deemph_tau = deemph_tau
        self.deemph = fm_deemph(audio_rate, tau=deemph_tau)
        width_of_transition_band = audio_rate / 32
        audio_coeffs = filter.firdes.low_pass(1.0, quad_rate, audio_rate / 2 - width_of_transition_band, width_of_transition_band, fft.window.WIN_HAMMING)
        self.audio_filter = filter.fir_filter_fff(audio_decimation, audio_coeffs)
        self.connect(self, self.fm_demod, self.audio_filter, self.deemph, self)