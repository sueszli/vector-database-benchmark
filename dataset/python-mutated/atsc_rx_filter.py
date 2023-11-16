from gnuradio import gr, filter
from . import dtv_python as dtv
ATSC_CHANNEL_BW = 6000000.0
ATSC_SYMBOL_RATE = 4500000.0 / 286 * 684
ATSC_RRC_SYMS = 8

class atsc_rx_filter(gr.hier_block2):

    def __init__(self, input_rate, sps):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'atsc_rx_filter', gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(1, 1, gr.sizeof_gr_complex))
        nfilts = 16
        output_rate = ATSC_SYMBOL_RATE * sps
        filter_rate = input_rate * nfilts
        symbol_rate = ATSC_SYMBOL_RATE / 2.0
        excess_bw = 0.1152
        ntaps = int((2 * ATSC_RRC_SYMS + 1) * sps * nfilts)
        interp = output_rate / input_rate
        gain = nfilts * symbol_rate / filter_rate
        rrc_taps = filter.firdes.root_raised_cosine(gain, filter_rate, symbol_rate, excess_bw, ntaps)
        pfb = filter.pfb_arb_resampler_ccf(interp, rrc_taps, nfilts)
        self.connect(self, pfb, self)