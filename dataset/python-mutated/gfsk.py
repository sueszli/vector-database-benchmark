from math import log as ln
import numpy
from gnuradio import gr
from gnuradio import analog
from gnuradio import blocks, filter
from . import modulation_utils
from . import digital_python as digital
_def_samples_per_symbol = 2
_def_sensitivity = 1
_def_bt = 0.35
_def_verbose = False
_def_log = False
_def_do_unpack = True
_def_gain_mu = None
_def_mu = 0.5
_def_freq_error = 0.0
_def_omega_relative_limit = 0.005

class gfsk_mod(gr.hier_block2):
    """
    Hierarchical block for Gaussian Frequency Shift Key (GFSK)
    modulation.

    The input is a byte stream (unsigned char) and the
    output is the complex modulated signal at baseband.

    Args:
        samples_per_symbol: samples per baud >= 2 (integer)
        bt: Gaussian filter bandwidth * symbol time (float)
        verbose: Print information about modulator? (bool)
        debug: Print modualtion data to files? (bool)
        unpack: Unpack input byte stream? (bool)
    """

    def __init__(self, samples_per_symbol=_def_samples_per_symbol, sensitivity=_def_sensitivity, bt=_def_bt, verbose=_def_verbose, log=_def_log, do_unpack=_def_do_unpack):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'gfsk_mod', gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        samples_per_symbol = int(samples_per_symbol)
        self._samples_per_symbol = samples_per_symbol
        self._bt = bt
        self._differential = False
        if not isinstance(samples_per_symbol, int) or samples_per_symbol < 2:
            raise TypeError('samples_per_symbol must be an integer >= 2, is %r' % (samples_per_symbol,))
        ntaps = 4 * samples_per_symbol
        self.nrz = digital.chunks_to_symbols_bf([-1, 1])
        self.gaussian_taps = filter.firdes.gaussian(1.0, samples_per_symbol, bt, ntaps)
        self.sqwave = (1,) * samples_per_symbol
        self.taps = numpy.convolve(numpy.array(self.gaussian_taps), numpy.array(self.sqwave))
        self.gaussian_filter = filter.interp_fir_filter_fff(samples_per_symbol, self.taps)
        self.fmmod = analog.frequency_modulator_fc(sensitivity)
        self.amp = blocks.multiply_const_cc(0.999)
        if verbose:
            self._print_verbage()
        if log:
            self._setup_logging()
        if do_unpack:
            self.unpack = blocks.packed_to_unpacked_bb(1, gr.GR_MSB_FIRST)
            self.connect(self, self.unpack, self.nrz, self.gaussian_filter, self.fmmod, self.amp, self)
        else:
            self.connect(self, self.nrz, self.gaussian_filter, self.fmmod, self.amp, self)

    def samples_per_symbol(self):
        if False:
            for i in range(10):
                print('nop')
        return self._samples_per_symbol

    @staticmethod
    def bits_per_symbol(self=None):
        if False:
            i = 10
            return i + 15
        return 1

    def _print_verbage(self):
        if False:
            while True:
                i = 10
        print('bits per symbol = %d' % self.bits_per_symbol())
        print('Gaussian filter bt = %.2f' % self._bt)

    def _setup_logging(self):
        if False:
            i = 10
            return i + 15
        print('Modulation logging turned on.')
        self.connect(self.nrz, blocks.file_sink(gr.sizeof_float, 'nrz.dat'))
        self.connect(self.gaussian_filter, blocks.file_sink(gr.sizeof_float, 'gaussian_filter.dat'))
        self.connect(self.fmmod, blocks.file_sink(gr.sizeof_gr_complex, 'fmmod.dat'))

    @staticmethod
    def add_options(parser):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds GFSK modulation-specific options to the standard parser\n        '
        parser.add_option('', '--bt', type='float', default=_def_bt, help='set bandwidth-time product [default=%default] (GFSK)')

    @staticmethod
    def extract_kwargs_from_options(options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given command line options, create dictionary suitable for passing to __init__\n        '
        return modulation_utils.extract_kwargs_from_options(gfsk_mod.__init__, ('self',), options)

class gfsk_demod(gr.hier_block2):
    """
    Hierarchical block for Gaussian Minimum Shift Key (GFSK)
    demodulation.

    The input is the complex modulated signal at baseband.
    The output is a stream of bits packed 1 bit per byte (the LSB)

    Args:
        samples_per_symbol: samples per baud (integer)
        verbose: Print information about modulator? (bool)
        log: Print modualtion data to files? (bool)

    Clock recovery parameters.  These all have reasonable defaults.

    Args:
        gain_mu: controls rate of mu adjustment (float)
        mu: unused but unremoved for backward compatibility (unused)
        omega_relative_limit: sets max variation in omega (float, typically 0.000200 (200 ppm))
        freq_error: bit rate error as a fraction
    """

    def __init__(self, samples_per_symbol=_def_samples_per_symbol, sensitivity=_def_sensitivity, gain_mu=_def_gain_mu, mu=_def_mu, omega_relative_limit=_def_omega_relative_limit, freq_error=_def_freq_error, verbose=_def_verbose, log=_def_log):
        if False:
            for i in range(10):
                print('nop')
        gr.hier_block2.__init__(self, 'gfsk_demod', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(1, 1, gr.sizeof_char))
        self._samples_per_symbol = samples_per_symbol
        self._gain_mu = gain_mu
        self._omega_relative_limit = omega_relative_limit
        self._freq_error = freq_error
        self._differential = False
        if samples_per_symbol < 2:
            raise TypeError('samples_per_symbol >= 2, is %f' % samples_per_symbol)
        self._omega = samples_per_symbol * (1 + self._freq_error)
        if not self._gain_mu:
            self._gain_mu = 0.175
        self._gain_omega = 0.25 * self._gain_mu * self._gain_mu
        self._damping = 1.0
        self._loop_bw = -ln((self._gain_mu + self._gain_omega) / -2.0 + 1)
        self._max_dev = self._omega_relative_limit * self._samples_per_symbol
        self.fmdemod = analog.quadrature_demod_cf(1.0 / sensitivity)
        self.clock_recovery = self.digital_symbol_sync_xx_0 = digital.symbol_sync_ff(digital.TED_MUELLER_AND_MULLER, self._omega, self._loop_bw, self._damping, 1.0, self._max_dev, 1, digital.constellation_bpsk().base(), digital.IR_MMSE_8TAP, 128, [])
        self.slicer = digital.binary_slicer_fb()
        if verbose:
            self._print_verbage()
        if log:
            self._setup_logging()
        self.connect(self, self.fmdemod, self.clock_recovery, self.slicer, self)

    def samples_per_symbol(self):
        if False:
            print('Hello World!')
        return self._samples_per_symbol

    @staticmethod
    def bits_per_symbol(self=None):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def _print_verbage(self):
        if False:
            i = 10
            return i + 15
        print('bits per symbol = %d' % self.bits_per_symbol())
        print('Symbol Sync M&M omega = %f' % self._omega)
        print('Symbol Sync M&M gain mu = %f' % self._gain_mu)
        print('Symbol Sync M&M omega rel. limit = %f' % self._omega_relative_limit)
        print('frequency error = %f' % self._freq_error)

    def _setup_logging(self):
        if False:
            return 10
        print('Demodulation logging turned on.')
        self.connect(self.fmdemod, blocks.file_sink(gr.sizeof_float, 'fmdemod.dat'))
        self.connect(self.clock_recovery, blocks.file_sink(gr.sizeof_float, 'clock_recovery.dat'))
        self.connect(self.slicer, blocks.file_sink(gr.sizeof_char, 'slicer.dat'))

    @staticmethod
    def add_options(parser):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds GFSK demodulation-specific options to the standard parser\n        '
        parser.add_option('', '--gain-mu', type='float', default=_def_gain_mu, help='Symbol Sync M&M gain mu [default=%default] (GFSK/PSK)')
        parser.add_option('', '--mu', type='float', default=_def_mu, help='M&M clock recovery mu [default=%default] (Unused)')
        parser.add_option('', '--omega-relative-limit', type='float', default=_def_omega_relative_limit, help='Symbol Sync M&M omega relative limit [default=%default] (GFSK/PSK)')
        parser.add_option('', '--freq-error', type='float', default=_def_freq_error, help='Symbol Sync M&M frequency error [default=%default] (GFSK)')

    @staticmethod
    def extract_kwargs_from_options(options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given command line options, create dictionary suitable for passing to __init__\n        '
        return modulation_utils.extract_kwargs_from_options(gfsk_demod.__init__, ('self',), options)
modulation_utils.add_type_1_mod('gfsk', gfsk_mod)
modulation_utils.add_type_1_demod('gfsk', gfsk_demod)