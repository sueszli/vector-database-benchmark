from gnuradio import blocks
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes

class conj_fs_iqcorr(gr.hier_block2):

    def __init__(self, delay=0, taps=[]):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'Conj FS IQBal', gr.io_signature(1, 1, gr.sizeof_gr_complex * 1), gr.io_signature(1, 1, gr.sizeof_gr_complex * 1))
        self.delay = delay
        self.taps = taps
        self.filter_fir_filter_xxx_0 = filter.fir_filter_ccc(1, taps)
        self.delay_0 = blocks.delay(gr.sizeof_gr_complex * 1, delay)
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.connect((self.blocks_add_xx_0, 0), (self, 0))
        self.connect((self, 0), (self.blocks_conjugate_cc_0, 0))
        self.connect((self.filter_fir_filter_xxx_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.filter_fir_filter_xxx_0, 0))
        self.connect((self, 0), (self.delay_0, 0))
        self.connect((self.delay_0, 0), (self.blocks_add_xx_0, 0))

    def get_delay(self):
        if False:
            for i in range(10):
                print('nop')
        return self.delay

    def set_delay(self, delay):
        if False:
            print('Hello World!')
        self.delay = delay
        self.delay_0.set_dly(self.delay)

    def get_taps(self):
        if False:
            while True:
                i = 10
        return self.taps

    def set_taps(self, taps):
        if False:
            print('Hello World!')
        self.taps = taps
        self.filter_fir_filter_xxx_0.set_taps(self.taps)