from gnuradio import gr, blocks
from . import fec_python as fec

class threaded_encoder(gr.hier_block2):

    def __init__(self, encoder_list_0, input_size, output_size):
        if False:
            i = 10
            return i + 15
        gr.hier_block2.__init__(self, 'Threaded Encoder', gr.io_signature(1, 1, input_size * 1), gr.io_signature(1, 1, output_size * 1))
        self.encoder_list_0 = encoder_list_0
        self.fec_deinterleave_0 = blocks.deinterleave(input_size, fec.get_encoder_input_size(encoder_list_0[0]))
        self.generic_encoders_0 = []
        for i in range(len(encoder_list_0)):
            self.generic_encoders_0.append(fec.encoder(encoder_list_0[i], input_size, output_size))
        self.fec_interleave_0 = blocks.interleave(output_size, fec.get_encoder_output_size(encoder_list_0[0]))
        for i in range(len(encoder_list_0)):
            self.connect((self.fec_deinterleave_0, i), (self.generic_encoders_0[i], 0))
        for i in range(len(encoder_list_0)):
            self.connect((self.generic_encoders_0[i], 0), (self.fec_interleave_0, i))
        self.connect((self, 0), (self.fec_deinterleave_0, 0))
        self.connect((self.fec_interleave_0, 0), (self, 0))

    def get_encoder_list_0(self):
        if False:
            return 10
        return self.encoder_list_0

    def set_encoder_list_0(self, encoder_list_0):
        if False:
            for i in range(10):
                print('nop')
        self.encoder_list_0 = encoder_list_0