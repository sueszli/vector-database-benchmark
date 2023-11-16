from gnuradio import gr, blocks
from . import fec_python as fec
from .bitflip import read_bitlist

class extended_tagged_encoder(gr.hier_block2):

    def __init__(self, encoder_obj_list, puncpat=None, lentagname=None, mtu=1500):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'extended_tagged_encoder', gr.io_signature(1, 1, gr.sizeof_char), gr.io_signature(1, 1, gr.sizeof_char))
        self.blocks = []
        self.puncpat = puncpat
        if type(encoder_obj_list) == list:
            if type(encoder_obj_list[0]) == list:
                gr.log.info('fec.extended_tagged_encoder: Parallelism must be 0 or 1.')
                raise AttributeError
            encoder_obj = encoder_obj_list[0]
        else:
            encoder_obj = encoder_obj_list
        if type(lentagname) == str:
            if lentagname.lower() == 'none':
                lentagname = None
        if fec.get_encoder_input_conversion(encoder_obj) == 'pack':
            self.blocks.append(blocks.pack_k_bits_bb(8))
        if not lentagname:
            self.blocks.append(fec.encoder(encoder_obj, gr.sizeof_char, gr.sizeof_char))
        else:
            self.blocks.append(fec.tagged_encoder(encoder_obj, gr.sizeof_char, gr.sizeof_char, lentagname, mtu))
        if self.puncpat != '11':
            self.blocks.append(fec.puncture_bb(len(puncpat), read_bitlist(puncpat), 0))
        self.connect((self, 0), (self.blocks[0], 0))
        self.connect((self.blocks[-1], 0), (self, 0))
        for i in range(len(self.blocks) - 1):
            self.connect((self.blocks[i], 0), (self.blocks[i + 1], 0))