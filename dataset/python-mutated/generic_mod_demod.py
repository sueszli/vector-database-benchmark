"""
Generic modulation and demodulation.
"""
from gnuradio import gr, blocks, filter, analog
from .modulation_utils import extract_kwargs_from_options_for_class
from .utils import mod_codes
from . import digital_python as digital
import math
_def_samples_per_symbol = 2
_def_excess_bw = 0.35
_def_verbose = False
_def_log = False
_def_truncate = False
_def_freq_bw = 2 * math.pi / 100.0
_def_timing_bw = 2 * math.pi / 100.0
_def_timing_max_dev = 1.5
_def_phase_bw = 2 * math.pi / 100.0
_def_constellation_points = 16
_def_differential = False

def add_common_options(parser):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets options common to both modulator and demodulator.\n    '
    parser.add_option('-p', '--constellation-points', type='int', default=_def_constellation_points, help='set the number of constellation points (must be a power of 2 for psk, power of 4 for QAM) [default=%default]')
    parser.add_option('', '--non-differential', action='store_false', dest='differential', help='do not use differential encoding [default=False]')
    parser.add_option('', '--differential', action='store_true', dest='differential', default=True, help='use differential encoding [default=%default]')
    parser.add_option('', '--mod-code', type='choice', choices=mod_codes.codes, default=mod_codes.NO_CODE, help='Select modulation code from: %s [default=%%default]' % (', '.join(mod_codes.codes),))
    parser.add_option('', '--excess-bw', type='float', default=_def_excess_bw, help='set RRC excess bandwidth factor [default=%default]')

class generic_mod(gr.hier_block2):
    """
    Hierarchical block for RRC-filtered differential generic modulation.

    The input is a byte stream (unsigned char) and the
    output is the complex modulated signal at baseband.

    Args:
        constellation: determines the modulation type (gnuradio.digital.digital_constellation)
        samples_per_symbol: samples per baud >= 2 (float)
        differential: whether to use differential encoding (boolean)
        pre_diff_code: whether to use apply a pre-differential mapping (boolean)
        excess_bw: Root-raised cosine filter excess bandwidth (float)
        verbose: Print information about modulator? (boolean)
        log: Log modulation data to files? (boolean)
        truncate: Truncate the modulated output to account for the RRC filter response (boolean)
    """

    def __init__(self, constellation, differential=_def_differential, samples_per_symbol=_def_samples_per_symbol, pre_diff_code=True, excess_bw=_def_excess_bw, verbose=_def_verbose, log=_def_log, truncate=_def_truncate):
        if False:
            i = 10
            return i + 15
        gr.hier_block2.__init__(self, 'generic_mod', gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self._constellation = constellation
        self._samples_per_symbol = samples_per_symbol
        self._excess_bw = excess_bw
        self._differential = differential
        self.pre_diff_code = pre_diff_code and self._constellation.apply_pre_diff_code()
        if self._samples_per_symbol < 2:
            raise TypeError('sps must be >= 2, is %f' % self._samples_per_symbol)
        arity = pow(2, self.bits_per_symbol())
        self.bytes2chunks = blocks.packed_to_unpacked_bb(self.bits_per_symbol(), gr.GR_MSB_FIRST)
        if self.pre_diff_code:
            self.symbol_mapper = digital.map_bb(self._constellation.pre_diff_code())
        if differential:
            self.diffenc = digital.diff_encoder_bb(arity)
        self.chunks2symbols = digital.chunks_to_symbols_bc(self._constellation.points())
        nfilts = 32
        ntaps_per_filt = 11
        ntaps = nfilts * ntaps_per_filt * int(self._samples_per_symbol)
        self.rrc_taps = filter.firdes.root_raised_cosine(nfilts, nfilts, 1.0, self._excess_bw, ntaps)
        self.rrc_filter = filter.pfb_arb_resampler_ccf(self._samples_per_symbol, self.rrc_taps)
        if truncate:
            fsps = float(self._samples_per_symbol)
            len_filt_delay = int((ntaps_per_filt * fsps * fsps - fsps) / 2.0)
            self.skiphead = blocks.skiphead(gr.sizeof_gr_complex * 1, len_filt_delay)
        self._blocks = [self, self.bytes2chunks]
        if self.pre_diff_code:
            self._blocks.append(self.symbol_mapper)
        if differential:
            self._blocks.append(self.diffenc)
        self._blocks += [self.chunks2symbols, self.rrc_filter]
        if truncate:
            self._blocks.append(self.skiphead)
        self._blocks.append(self)
        self.connect(*self._blocks)
        if verbose:
            self._print_verbage()
        if log:
            self._setup_logging()

    def samples_per_symbol(self):
        if False:
            print('Hello World!')
        return self._samples_per_symbol

    def bits_per_symbol(self):
        if False:
            return 10
        return self._constellation.bits_per_symbol()

    @staticmethod
    def add_options(parser):
        if False:
            return 10
        '\n        Adds generic modulation options to the standard parser\n        '
        add_common_options(parser)

    def extract_kwargs_from_options(cls, options):
        if False:
            while True:
                i = 10
        '\n        Given command line options, create dictionary suitable for passing to __init__\n        '
        return extract_kwargs_from_options_for_class(cls, options)
    extract_kwargs_from_options = classmethod(extract_kwargs_from_options)

    def _print_verbage(self):
        if False:
            for i in range(10):
                print('nop')
        print('\nModulator:')
        print('bits per symbol:     %d' % self.bits_per_symbol())
        print('RRC roll-off factor: %.2f' % self._excess_bw)

    def _setup_logging(self):
        if False:
            for i in range(10):
                print('nop')
        print('Modulation logging turned on.')
        self.connect(self.bytes2chunks, blocks.file_sink(gr.sizeof_char, 'tx_bytes2chunks.8b'))
        if self.pre_diff_code:
            self.connect(self.symbol_mapper, blocks.file_sink(gr.sizeof_char, 'tx_symbol_mapper.8b'))
        if self._differential:
            self.connect(self.diffenc, blocks.file_sink(gr.sizeof_char, 'tx_diffenc.8b'))
        self.connect(self.chunks2symbols, blocks.file_sink(gr.sizeof_gr_complex, 'tx_chunks2symbols.32fc'))
        self.connect(self.rrc_filter, blocks.file_sink(gr.sizeof_gr_complex, 'tx_rrc_filter.32fc'))

class generic_demod(gr.hier_block2):
    """
    Hierarchical block for RRC-filtered differential generic demodulation.

    The input is the complex modulated signal at baseband.
    The output is a stream of bits packed 1 bit per byte (LSB)

    Args:
        constellation: determines the modulation type (gnuradio.digital.digital_constellation)
        samples_per_symbol: samples per baud >= 2 (float)
        differential: whether to use differential encoding (boolean)
        pre_diff_code: whether to use apply a pre-differential mapping (boolean)
        excess_bw: Root-raised cosine filter excess bandwidth (float)
        freq_bw: loop filter lock-in bandwidth (float)
        timing_bw: timing recovery loop lock-in bandwidth (float)
        phase_bw: phase recovery loop bandwidth (float)
        verbose: Print information about modulator? (boolean)
        log: Log modulation data to files? (boolean)
    """

    def __init__(self, constellation, differential=_def_differential, samples_per_symbol=_def_samples_per_symbol, pre_diff_code=True, excess_bw=_def_excess_bw, freq_bw=_def_freq_bw, timing_bw=_def_timing_bw, phase_bw=_def_phase_bw, verbose=_def_verbose, log=_def_log):
        if False:
            while True:
                i = 10
        gr.hier_block2.__init__(self, 'generic_demod', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(1, 1, gr.sizeof_char))
        self._constellation = constellation
        self._samples_per_symbol = samples_per_symbol
        self._excess_bw = excess_bw
        self._phase_bw = phase_bw
        self._freq_bw = freq_bw
        self._timing_bw = timing_bw
        self._timing_max_dev = _def_timing_max_dev
        self._differential = differential
        if self._samples_per_symbol < 2:
            raise TypeError('sps must be >= 2, is %d' % self._samples_per_symbol)
        self.pre_diff_code = pre_diff_code and self._constellation.apply_pre_diff_code()
        arity = pow(2, self.bits_per_symbol())
        nfilts = 32
        ntaps = 11 * int(self._samples_per_symbol * nfilts)
        self.agc = analog.agc2_cc(0.06, 0.001, 1, 1)
        fll_ntaps = 55
        self.freq_recov = digital.fll_band_edge_cc(self._samples_per_symbol, self._excess_bw, fll_ntaps, self._freq_bw)
        taps = filter.firdes.root_raised_cosine(nfilts, nfilts * self._samples_per_symbol, 1.0, self._excess_bw, ntaps)
        self.time_recov = digital.pfb_clock_sync_ccf(self._samples_per_symbol, self._timing_bw, taps, nfilts, nfilts // 2, self._timing_max_dev)
        fmin = -0.25
        fmax = 0.25
        self.receiver = digital.constellation_receiver_cb(self._constellation.base(), self._phase_bw, fmin, fmax)
        if differential:
            self.diffdec = digital.diff_decoder_bb(arity)
        if self.pre_diff_code:
            self.symbol_mapper = digital.map_bb(mod_codes.invert_code(self._constellation.pre_diff_code()))
        self.unpack = blocks.unpack_k_bits_bb(self.bits_per_symbol())
        if verbose:
            self._print_verbage()
        if log:
            self._setup_logging()
        self._blocks = [self, self.agc, self.freq_recov, self.time_recov, self.receiver]
        if differential:
            self._blocks.append(self.diffdec)
        if self.pre_diff_code:
            self._blocks.append(self.symbol_mapper)
        self._blocks += [self.unpack, self]
        self.connect(*self._blocks)

    def samples_per_symbol(self):
        if False:
            return 10
        return self._samples_per_symbol

    def bits_per_symbol(self):
        if False:
            for i in range(10):
                print('nop')
        return self._constellation.bits_per_symbol()

    def _print_verbage(self):
        if False:
            while True:
                i = 10
        print('\nDemodulator:')
        print('bits per symbol:     %d' % self.bits_per_symbol())
        print('RRC roll-off factor: %.2f' % self._excess_bw)
        print('FLL bandwidth:       %.2e' % self._freq_bw)
        print('Timing bandwidth:    %.2e' % self._timing_bw)
        print('Phase bandwidth:     %.2e' % self._phase_bw)

    def _setup_logging(self):
        if False:
            for i in range(10):
                print('nop')
        print('Modulation logging turned on.')
        self.connect(self.agc, blocks.file_sink(gr.sizeof_gr_complex, 'rx_agc.32fc'))
        self.connect((self.freq_recov, 0), blocks.file_sink(gr.sizeof_gr_complex, 'rx_freq_recov.32fc'))
        self.connect((self.freq_recov, 1), blocks.file_sink(gr.sizeof_float, 'rx_freq_recov_freq.32f'))
        self.connect((self.freq_recov, 2), blocks.file_sink(gr.sizeof_float, 'rx_freq_recov_phase.32f'))
        self.connect((self.freq_recov, 3), blocks.file_sink(gr.sizeof_float, 'rx_freq_recov_error.32f'))
        self.connect((self.time_recov, 0), blocks.file_sink(gr.sizeof_gr_complex, 'rx_time_recov.32fc'))
        self.connect((self.time_recov, 1), blocks.file_sink(gr.sizeof_float, 'rx_time_recov_error.32f'))
        self.connect((self.time_recov, 2), blocks.file_sink(gr.sizeof_float, 'rx_time_recov_rate.32f'))
        self.connect((self.time_recov, 3), blocks.file_sink(gr.sizeof_float, 'rx_time_recov_phase.32f'))
        self.connect((self.receiver, 0), blocks.file_sink(gr.sizeof_char, 'rx_receiver.8b'))
        self.connect((self.receiver, 1), blocks.file_sink(gr.sizeof_float, 'rx_receiver_error.32f'))
        self.connect((self.receiver, 2), blocks.file_sink(gr.sizeof_float, 'rx_receiver_phase.32f'))
        self.connect((self.receiver, 3), blocks.file_sink(gr.sizeof_float, 'rx_receiver_freq.32f'))
        if self._differential:
            self.connect(self.diffdec, blocks.file_sink(gr.sizeof_char, 'rx_diffdec.8b'))
        if self.pre_diff_code:
            self.connect(self.symbol_mapper, blocks.file_sink(gr.sizeof_char, 'rx_symbol_mapper.8b'))
        self.connect(self.unpack, blocks.file_sink(gr.sizeof_char, 'rx_unpack.8b'))

    @staticmethod
    def add_options(parser):
        if False:
            return 10
        '\n        Adds generic demodulation options to the standard parser\n        '
        add_common_options(parser)
        parser.add_option('', '--freq-bw', type='float', default=_def_freq_bw, help='set frequency lock loop lock-in bandwidth [default=%default]')
        parser.add_option('', '--phase-bw', type='float', default=_def_phase_bw, help='set phase tracking loop lock-in bandwidth [default=%default]')
        parser.add_option('', '--timing-bw', type='float', default=_def_timing_bw, help='set timing symbol sync loop gain lock-in bandwidth [default=%default]')

    def extract_kwargs_from_options(cls, options):
        if False:
            return 10
        '\n        Given command line options, create dictionary suitable for passing to __init__\n        '
        return extract_kwargs_from_options_for_class(cls, options)
    extract_kwargs_from_options = classmethod(extract_kwargs_from_options)
shared_demod_args = '    samples_per_symbol: samples per baud >= 2 (float)\n        excess_bw: Root-raised cosine filter excess bandwidth (float)\n        freq_bw: loop filter lock-in bandwidth (float)\n        timing_bw: timing recovery loop lock-in bandwidth (float)\n        phase_bw: phase recovery loop bandwidth (float)\n        verbose: Print information about modulator? (boolean)\n        log: Log modulation data to files? (boolean)\n'
shared_mod_args = '    samples_per_symbol: samples per baud >= 2 (float)\n        excess_bw: Root-raised cosine filter excess bandwidth (float)\n        verbose: Print information about modulator? (boolean)\n        log: Log modulation data to files? (boolean)\n'