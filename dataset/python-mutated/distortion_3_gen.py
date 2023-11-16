from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import math

class distortion_3_gen(gr.hier_block2):

    def __init__(self, beta=0):
        if False:
            i = 10
            return i + 15
        gr.hier_block2.__init__(self, 'Third Order Distortion', gr.io_signature(1, 1, gr.sizeof_gr_complex * 1), gr.io_signature(1, 1, gr.sizeof_gr_complex * 1))
        self.beta = beta
        self.blocks_null_source_0 = blocks.null_source(gr.sizeof_float * 1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vcc((beta,))
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_null_source_0, 0), (self.blocks_float_to_complex_0, 1))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self, 0))

    def get_beta(self):
        if False:
            i = 10
            return i + 15
        return self.beta

    def set_beta(self, beta):
        if False:
            return 10
        self.beta = beta
        self.blocks_multiply_const_vxx_0.set_k((self.beta,))