from gnuradio.fec.bitflip import read_bitlist
from gnuradio import gr, blocks, analog
import math
import sys
if 'gnuradio.digital' in sys.modules:
    digital = sys.modules['gnuradio.digital']
else:
    from gnuradio import digital
from .extended_encoder import extended_encoder
from .extended_decoder import extended_decoder

class fec_test(gr.hier_block2):

    def __init__(self, generic_encoder=0, generic_decoder=0, esno=0, samp_rate=3200000, threading='capillary', puncpat='11', seed=0):
        if False:
            while True:
                i = 10
        gr.hier_block2.__init__(self, 'fec_test', gr.io_signature(1, 1, gr.sizeof_char * 1), gr.io_signature(2, 2, gr.sizeof_char * 1))
        self.generic_encoder = generic_encoder
        self.generic_decoder = generic_decoder
        self.esno = esno
        self.samp_rate = samp_rate
        self.threading = threading
        self.puncpat = puncpat
        self.map_bb = digital.map_bb([-1, 1])
        self.b2f = blocks.char_to_float(1, 1)
        self.unpack8 = blocks.unpack_k_bits_bb(8)
        self.pack8 = blocks.pack_k_bits_bb(8)
        self.encoder = extended_encoder(encoder_obj_list=generic_encoder, threading=threading, puncpat=puncpat)
        self.decoder = extended_decoder(decoder_obj_list=generic_decoder, threading=threading, ann=None, puncpat=puncpat, integration_period=10000, rotator=None)
        noise = math.sqrt(10.0 ** (-esno / 10.0) / 2.0)
        self.fastnoise = analog.noise_source_f(analog.GR_GAUSSIAN, noise, seed)
        self.addnoise = blocks.add_ff(1)
        self.copy_packed = blocks.copy(gr.sizeof_char)
        self.connect(self, self.copy_packed)
        self.connect(self.copy_packed, (self, 1))
        self.connect(self, self.unpack8)
        self.connect(self.unpack8, self.encoder)
        self.connect(self.encoder, self.map_bb)
        self.connect(self.map_bb, self.b2f)
        self.connect(self.b2f, (self.addnoise, 0))
        self.connect(self.fastnoise, (self.addnoise, 1))
        self.connect(self.addnoise, self.decoder)
        self.connect(self.decoder, self.pack8)
        self.connect(self.pack8, (self, 0))

    def get_generic_encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.generic_encoder

    def set_generic_encoder(self, generic_encoder):
        if False:
            i = 10
            return i + 15
        self.generic_encoder = generic_encoder

    def get_generic_decoder(self):
        if False:
            while True:
                i = 10
        return self.generic_decoder

    def set_generic_decoder(self, generic_decoder):
        if False:
            while True:
                i = 10
        self.generic_decoder = generic_decoder

    def get_esno(self):
        if False:
            i = 10
            return i + 15
        return self.esno

    def set_esno(self, esno):
        if False:
            i = 10
            return i + 15
        self.esno = esno

    def get_samp_rate(self):
        if False:
            for i in range(10):
                print('nop')
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        if False:
            print('Hello World!')
        self.samp_rate = samp_rate

    def get_threading(self):
        if False:
            for i in range(10):
                print('nop')
        return self.threading

    def set_threading(self, threading):
        if False:
            return 10
        self.threading = threading

    def get_puncpat(self):
        if False:
            while True:
                i = 10
        return self.puncpat

    def set_puncpat(self, puncpat):
        if False:
            while True:
                i = 10
        self.puncpat = puncpat