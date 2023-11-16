from gnuradio import gr
from gnuradio import eng_notation
from gnuradio import digital
from gnuradio import analog
import copy
import sys

class receive_path(gr.hier_block2):

    def __init__(self, rx_callback, options):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'receive_path', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(0, 0, 0))
        options = copy.copy(options)
        self._verbose = options.verbose
        self._log = options.log
        self._rx_callback = rx_callback
        self.ofdm_rx = digital.ofdm_demod(options, callback=self._rx_callback)
        alpha = 0.001
        thresh = 30
        self.probe = analog.probe_avg_mag_sqrd_c(thresh, alpha)
        self.connect(self, self.ofdm_rx)
        self.connect(self.ofdm_rx, self.probe)
        if self._verbose:
            self._print_verbage()

    def carrier_sensed(self):
        if False:
            print('Hello World!')
        '\n        Return True if we think carrier is present.\n        '
        return self.probe.unmuted()

    def carrier_threshold(self):
        if False:
            i = 10
            return i + 15
        '\n        Return current setting in dB.\n        '
        return self.probe.threshold()

    def set_carrier_threshold(self, threshold_in_db):
        if False:
            while True:
                i = 10
        '\n        Set carrier threshold.\n\n        Args:\n            threshold_in_db: set detection threshold (float (dB))\n        '
        self.probe.set_threshold(threshold_in_db)

    @staticmethod
    def add_options(normal, expert):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds receiver-specific options to the Options Parser\n        '
        normal.add_option('-W', '--bandwidth', type='eng_float', default=500000.0, help='set symbol bandwidth [default=%default]')
        normal.add_option('-v', '--verbose', action='store_true', default=False)
        expert.add_option('', '--log', action='store_true', default=False, help='Log all parts of flow graph to files (CAUTION: lots of data)')

    def _print_verbage(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prints information about the receive path\n        '
        pass