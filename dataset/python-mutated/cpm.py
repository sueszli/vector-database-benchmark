from math import pi
import numpy
from gnuradio import gr, filter
from gnuradio import analog
from gnuradio import blocks
from . import digital_python
from . import modulation_utils
_def_samples_per_symbol = 2
_def_bits_per_symbol = 1
_def_h_numerator = 1
_def_h_denominator = 2
_def_cpm_type = 0
_def_bt = 0.35
_def_symbols_per_pulse = 1
_def_generic_taps = numpy.empty(1)
_def_verbose = False
_def_log = False

class cpm_mod(gr.hier_block2):
    """
    Hierarchical block for Continuous Phase modulation.

    The input is a byte stream (unsigned char) representing packed
    bits and the output is the complex modulated signal at baseband.

    See Proakis for definition of generic CPM signals:
    s(t)=exp(j phi(t))
    phi(t)= 2 pi h int_0^t f(t') dt'
    f(t)=sum_k a_k g(t-kT)
    (normalizing assumption: int_0^infty g(t) dt = 1/2)

    Args:
        samples_per_symbol: samples per baud >= 2 (integer)
        bits_per_symbol: bits per symbol (integer)
        h_numerator: numerator of modulation index (integer)
        h_denominator: denominator of modulation index (numerator and denominator must be relative primes) (integer)
        cpm_type: supported types are: 0=CPFSK, 1=GMSK, 2=RC, 3=GENERAL (integer)
        bt: bandwidth symbol time product for GMSK (float)
        symbols_per_pulse: shaping pulse duration in symbols (integer)
        generic_taps: define a generic CPM pulse shape (sum = samples_per_symbol/2) (list/array of floats)
        verbose: Print information about modulator? (boolean)
        debug: Print modulation data to files? (boolean)
    """

    def __init__(self, samples_per_symbol=_def_samples_per_symbol, bits_per_symbol=_def_bits_per_symbol, h_numerator=_def_h_numerator, h_denominator=_def_h_denominator, cpm_type=_def_cpm_type, bt=_def_bt, symbols_per_pulse=_def_symbols_per_pulse, generic_taps=_def_generic_taps, verbose=_def_verbose, log=_def_log):
        if False:
            for i in range(10):
                print('nop')
        gr.hier_block2.__init__(self, 'cpm_mod', gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self._samples_per_symbol = samples_per_symbol
        self._bits_per_symbol = bits_per_symbol
        self._h_numerator = h_numerator
        self._h_denominator = h_denominator
        self._cpm_type = cpm_type
        self._bt = bt
        if cpm_type == 0 or cpm_type == 2 or cpm_type == 3:
            self._symbols_per_pulse = symbols_per_pulse
        elif cpm_type == 1:
            self._symbols_per_pulse = 4
        else:
            raise TypeError('cpm_type must be an integer in {0,1,2,3}, is %r' % (cpm_type,))
        self._generic_taps = numpy.array(generic_taps)
        if samples_per_symbol < 2:
            raise TypeError('samples_per_symbol must be >= 2, is %r' % (samples_per_symbol,))
        self.nsymbols = 2 ** bits_per_symbol
        self.sym_alphabet = numpy.arange(-(self.nsymbols - 1), self.nsymbols, 2).tolist()
        self.ntaps = int(self._symbols_per_pulse * samples_per_symbol)
        sensitivity = 2 * pi * h_numerator / h_denominator / samples_per_symbol
        self.B2s = blocks.packed_to_unpacked_bb(bits_per_symbol, gr.GR_MSB_FIRST)
        self.pam = digital_python.chunks_to_symbols_bf(self.sym_alphabet, 1)
        if cpm_type == 0:
            self.taps = (1.0 / self._symbols_per_pulse / 2,) * self.ntaps
        elif cpm_type == 1:
            gaussian_taps = filter.firdes.gaussian(1.0 / 2, samples_per_symbol, bt, self.ntaps)
            sqwave = (1,) * samples_per_symbol
            self.taps = numpy.convolve(numpy.array(gaussian_taps), numpy.array(sqwave))
        elif cpm_type == 2:
            self.taps = (1 - numpy.cos(2 * pi * numpy.arange(0 / self.ntaps / samples_per_symbol / self._symbols_per_pulse)), 2 * self._symbols_per_pulse)
        elif cpm_type == 3:
            self.taps = generic_taps
        else:
            raise TypeError('cpm_type must be an integer in {0,1,2,3}, is %r' % (cpm_type,))
        self.filter = filter.pfb.arb_resampler_fff(samples_per_symbol, self.taps)
        self.fmmod = analog.frequency_modulator_fc(sensitivity)
        if verbose:
            self._print_verbage()
        if log:
            self._setup_logging()
        self.connect(self, self.B2s, self.pam, self.filter, self.fmmod, self)

    def samples_per_symbol(self):
        if False:
            print('Hello World!')
        return self._samples_per_symbol

    def bits_per_symbol(self):
        if False:
            i = 10
            return i + 15
        return self._bits_per_symbol

    def h_numerator(self):
        if False:
            while True:
                i = 10
        return self._h_numerator

    def h_denominator(self):
        if False:
            return 10
        return self._h_denominator

    def cpm_type(self):
        if False:
            i = 10
            return i + 15
        return self._cpm_type

    def bt(self):
        if False:
            print('Hello World!')
        return self._bt

    def symbols_per_pulse(self):
        if False:
            return 10
        return self._symbols_per_pulse

    def _print_verbage(self):
        if False:
            for i in range(10):
                print('nop')
        print('Samples per symbol = %d' % self._samples_per_symbol)
        print('Bits per symbol = %d' % self._bits_per_symbol)
        print('h = ', self._h_numerator, ' / ', self._h_denominator)
        print('Symbol alphabet = ', self.sym_alphabet)
        print('Symbols per pulse = %d' % self._symbols_per_pulse)
        print('taps = ', self.taps)
        print('CPM type = %d' % self._cpm_type)
        if self._cpm_type == 1:
            print('Gaussian filter BT = %.2f' % self._bt)

    def _setup_logging(self):
        if False:
            print('Hello World!')
        print('Modulation logging turned on.')
        self.connect(self.B2s, blocks.file_sink(gr.sizeof_float, 'symbols.dat'))
        self.connect(self.pam, blocks.file_sink(gr.sizeof_float, 'pam.dat'))
        self.connect(self.filter, blocks.file_sink(gr.sizeof_float, 'filter.dat'))
        self.connect(self.fmmod, blocks.file_sink(gr.sizeof_gr_complex, 'fmmod.dat'))

    @staticmethod
    def add_options(parser):
        if False:
            return 10
        '\n        Adds CPM modulation-specific options to the standard parser\n        '
        parser.add_option('', '--bt', type='float', default=_def_bt, help='set bandwidth-time product [default=%default] (GMSK)')

    @staticmethod
    def extract_kwargs_from_options(options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given command line options, create dictionary suitable for passing to __init__\n        '
        return modulation_utils.extract_kwargs_from_options(cpm_mod.__init__, ('self',), options)
modulation_utils.add_type_1_mod('cpm', cpm_mod)