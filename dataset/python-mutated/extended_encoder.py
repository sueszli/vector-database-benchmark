from gnuradio import gr, blocks
from . import fec_python as fec
from .threaded_encoder import threaded_encoder
from .capillary_threaded_encoder import capillary_threaded_encoder
from .bitflip import read_bitlist

class extended_encoder(gr.hier_block2):

    def __init__(self, encoder_obj_list, threading, puncpat=None):
        if False:
            for i in range(10):
                print('nop')
        gr.hier_block2.__init__(self, 'extended_encoder', gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_char))
        self.blocks = []
        self.puncpat = puncpat
        if type(encoder_obj_list) == list:
            if type(encoder_obj_list[0]) == list:
                gr.log.info('fec.extended_encoder: Parallelism must be 1.')
                raise AttributeError
        else:
            encoder_obj_list = [encoder_obj_list]
        if fec.get_encoder_input_conversion(encoder_obj_list[0]) == 'pack':
            self.blocks.append(blocks.pack_k_bits_bb(8))
        if threading == 'capillary':
            self.blocks.append(capillary_threaded_encoder(encoder_obj_list, gr.sizeof_char, gr.sizeof_char))
        elif threading == 'ordinary':
            self.blocks.append(threaded_encoder(encoder_obj_list, gr.sizeof_char, gr.sizeof_char))
        else:
            self.blocks.append(fec.encoder(encoder_obj_list[0], gr.sizeof_char, gr.sizeof_char))
        if fec.get_encoder_output_conversion(encoder_obj_list[0]) == 'packed_bits':
            self.blocks.append(blocks.packed_to_unpacked_bb(1, gr.GR_MSB_FIRST))
        if self.puncpat != '11':
            self.blocks.append(fec.puncture_bb(len(puncpat), read_bitlist(puncpat), 0))
        self.connect((self, 0), (self.blocks[0], 0))
        self.connect((self.blocks[-1], 0), (self, 0))
        for i in range(len(self.blocks) - 1):
            self.connect((self.blocks[i], 0), (self.blocks[i + 1], 0))