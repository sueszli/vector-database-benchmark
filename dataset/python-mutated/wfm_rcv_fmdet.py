import math
from gnuradio import gr
from gnuradio import blocks
from gnuradio import filter, fft
from . import analog_python as analog
from .fm_emph import fm_deemph

class wfm_rcv_fmdet(gr.hier_block2):

    def __init__(self, demod_rate, audio_decimation):
        if False:
            i = 10
            return i + 15
        '\n        Hierarchical block for demodulating a broadcast FM signal.\n\n        The input is the downconverted complex baseband signal\n        (gr_complex).  The output is two streams of the demodulated\n        audio (float) 0=Left, 1=Right.\n\n        Args:\n            demod_rate: input sample rate of complex baseband input. (float)\n            audio_decimation: how much to decimate demod_rate to get to audio. (integer)\n        '
        gr.hier_block2.__init__(self, 'wfm_rcv_fmdet', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(2, 2, gr.sizeof_float))
        if audio_decimation != int(audio_decimation):
            raise ValueError('audio_decimation needs to be an integer')
        audio_decimation = int(audio_decimation)
        lowfreq = -125000.0 / demod_rate
        highfreq = 125000.0 / demod_rate
        audio_rate = demod_rate / audio_decimation
        self.fm_demod = analog.fmdet_cf(demod_rate, lowfreq, highfreq, 0.05)
        self.deemph_Left = fm_deemph(audio_rate)
        self.deemph_Right = fm_deemph(audio_rate)
        width_of_transition_band = audio_rate / 32
        audio_coeffs = filter.firdes.low_pass(1.0, demod_rate, 15000, width_of_transition_band, fft.window.WIN_HAMMING)
        self.audio_filter = filter.fir_filter_fff(audio_decimation, audio_coeffs)
        if 1:
            stereo_carrier_filter_coeffs = filter.firdes.complex_band_pass(10.0, demod_rate, -19020, -18980, width_of_transition_band, fft.window.WIN_HAMMING)
            stereo_dsbsc_filter_coeffs = filter.firdes.complex_band_pass(20.0, demod_rate, 38000 - 15000 / 2, 38000 + 15000 / 2, width_of_transition_band, fft.window.WIN_HAMMING)
            self.stereo_carrier_filter = filter.fir_filter_fcc(audio_decimation, stereo_carrier_filter_coeffs)
            self.stereo_carrier_generator = blocks.multiply_cc()
            stereo_rds_filter_coeffs = filter.firdes.complex_band_pass(30.0, demod_rate, 57000 - 1500, 57000 + 1500, width_of_transition_band, fft.window.WIN_HAMMING)
            self.rds_signal_filter = filter.fir_filter_fcc(audio_decimation, stereo_rds_filter_coeffs)
            self.rds_carrier_generator = blocks.multiply_cc()
            self.rds_signal_generator = blocks.multiply_cc()
            self_rds_signal_processor = blocks.null_sink(gr.sizeof_gr_complex)
            loop_bw = 2 * math.pi / 100.0
            max_freq = -2.0 * math.pi * 18990 / audio_rate
            min_freq = -2.0 * math.pi * 19010 / audio_rate
            self.stereo_carrier_pll_recovery = analog.pll_refout_cc(loop_bw, max_freq, min_freq)
            self.stereo_basebander = blocks.multiply_cc()
            self.LmR_real = blocks.complex_to_real()
            self.Make_Left = blocks.add_ff()
            self.Make_Right = blocks.sub_ff()
            self.stereo_dsbsc_filter = filter.fir_filter_fcc(audio_decimation, stereo_dsbsc_filter_coeffs)
        if 1:
            self.connect(self, self.fm_demod, self.stereo_carrier_filter, self.stereo_carrier_pll_recovery, (self.stereo_carrier_generator, 0))
            self.connect(self.stereo_carrier_pll_recovery, (self.stereo_carrier_generator, 1))
            self.connect(self.stereo_carrier_generator, (self.stereo_basebander, 0))
            self.connect(self.fm_demod, self.stereo_dsbsc_filter, (self.stereo_basebander, 1))
            self.connect(self.stereo_basebander, self.LmR_real, (self.Make_Left, 0))
            self.connect(self.LmR_real, (self.Make_Right, 1))
            self.connect(self.stereo_basebander, (self.rds_carrier_generator, 0))
            self.connect(self.stereo_carrier_pll_recovery, (self.rds_carrier_generator, 1))
            self.connect(self.fm_demod, self.rds_signal_filter, (self.rds_signal_generator, 0))
            self.connect(self.rds_carrier_generator, (self.rds_signal_generator, 1))
            self.connect(self.rds_signal_generator, self_rds_signal_processor)
        if 1:
            self.connect(self.fm_demod, self.audio_filter, (self.Make_Left, 1))
            self.connect(self.audio_filter, (self.Make_Right, 0))
            self.connect(self.Make_Left, self.deemph_Left, (self, 0))
            self.connect(self.Make_Right, self.deemph_Right, (self, 1))