from gnuradio import gr, filter, analog
from .atsc_rx_filter import *

class atsc_rx(gr.hier_block2):

    def __init__(self, input_rate, sps):
        if False:
            while True:
                i = 10
        gr.hier_block2.__init__(self, 'atsc_rx', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(1, 1, gr.sizeof_char))
        rx_filt = atsc_rx_filter(input_rate, sps)
        output_rate = ATSC_SYMBOL_RATE * sps
        pll = dtv.atsc_fpll(output_rate)
        dcr = filter.dc_blocker_ff(4096)
        agc = analog.agc_ff(1e-05, 4.0)
        btl = dtv.atsc_sync(output_rate)
        fsc = dtv.atsc_fs_checker()
        equ = dtv.atsc_equalizer()
        vit = dtv.atsc_viterbi_decoder()
        dei = dtv.atsc_deinterleaver()
        rsd = dtv.atsc_rs_decoder()
        der = dtv.atsc_derandomizer()
        dep = dtv.atsc_depad()
        self.connect(self, rx_filt, pll, dcr, agc, btl, fsc)
        self.connect((fsc, 0), (equ, 0))
        self.connect((fsc, 1), (equ, 1))
        self.connect((equ, 0), (vit, 0))
        self.connect((equ, 1), (vit, 1))
        self.connect((vit, 0), (dei, 0))
        self.connect((vit, 1), (dei, 1))
        self.connect((dei, 0), (rsd, 0))
        self.connect((dei, 1), (rsd, 1))
        self.connect((rsd, 0), (der, 0))
        self.connect((rsd, 1), (der, 1))
        self.connect((der, 0), (dep, 0))
        self.connect((dep, 0), (self, 0))