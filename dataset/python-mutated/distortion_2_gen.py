from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import math

class distortion_2_gen(gr.hier_block2):

    def __init__(self, beta=0):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'Second Order Distortion', gr.io_signature(1, 1, gr.sizeof_gr_complex * 1), gr.io_signature(1, 1, gr.sizeof_gr_complex * 1))
        self.beta = beta
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vcc((beta,))
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_add_xx_0_0 = blocks.add_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self, 0))
        self.connect((self, 0), (self.blocks_conjugate_cc_0, 0))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self, 0), (self.blocks_multiply_xx_0_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.blocks_multiply_const_vxx_0, 0))

    def get_beta(self):
        if False:
            while True:
                i = 10
        return self.beta

    def set_beta(self, beta):
        if False:
            for i in range(10):
                print('nop')
        self.beta = beta
        self.blocks_multiply_const_vxx_0.set_k((self.beta,))