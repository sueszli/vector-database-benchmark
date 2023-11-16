from gnuradio import gr
from gnuradio import eng_notation
from gnuradio import blocks
from gnuradio import digital
import copy
import sys

class transmit_path(gr.hier_block2):

    def __init__(self, options):
        if False:
            i = 10
            return i + 15
        '\n        See below for what options should hold\n        '
        gr.hier_block2.__init__(self, 'transmit_path', gr.io_signature(0, 0, 0), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        options = copy.copy(options)
        self._verbose = options.verbose
        self._tx_amplitude = options.tx_amplitude
        self.ofdm_tx = digital.ofdm_mod(options, msgq_limit=4, pad_for_usrp=False)
        self.amp = blocks.multiply_const_cc(1)
        self.set_tx_amplitude(self._tx_amplitude)
        if self._verbose:
            self._print_verbage()
        self.connect(self.ofdm_tx, self.amp, self)

    def set_tx_amplitude(self, ampl):
        if False:
            return 10
        '\n        Sets the transmit amplitude sent to the USRP\n\n        Args:\n            : ampl 0 <= ampl < 1.0.  Try 0.10\n        '
        self._tx_amplitude = max(0.0, min(ampl, 1))
        self.amp.set_k(self._tx_amplitude)

    def send_pkt(self, payload='', eof=False):
        if False:
            i = 10
            return i + 15
        '\n        Calls the transmitter method to send a packet\n        '
        return self.ofdm_tx.send_pkt(payload, eof)

    @staticmethod
    def add_options(normal, expert):
        if False:
            i = 10
            return i + 15
        '\n        Adds transmitter-specific options to the Options Parser\n        '
        normal.add_option('', '--tx-amplitude', type='eng_float', default=0.1, metavar='AMPL', help='set transmitter digital amplitude: 0 <= AMPL < 1.0 [default=%default]')
        normal.add_option('-W', '--bandwidth', type='eng_float', default=500000.0, help='set symbol bandwidth [default=%default]')
        normal.add_option('-v', '--verbose', action='store_true', default=False)
        expert.add_option('', '--log', action='store_true', default=False, help='Log all parts of flow graph to file (CAUTION: lots of data)')

    def _print_verbage(self):
        if False:
            while True:
                i = 10
        '\n        Prints information about the transmit path\n        '
        print('Tx amplitude     %s' % self._tx_amplitude)