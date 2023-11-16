import weakref
from gnuradio import gr
from . import fec_python as fec
from .bitflip import read_bitlist

class extended_async_encoder(gr.hier_block2):

    def __init__(self, encoder_obj_list, puncpat=None):
        if False:
            print('Hello World!')
        gr.hier_block2.__init__(self, 'extended_async_encoder', gr.io_signature(0, 0, 0), gr.io_signature(0, 0, 0))
        self.message_port_register_hier_in('in')
        self.message_port_register_hier_out('out')
        self.puncpat = puncpat
        if type(encoder_obj_list) == list:
            if type(encoder_obj_list[0]) == list:
                gr.log.info('fec.extended_encoder: Parallelism must be 0 or 1.')
                raise AttributeError
            encoder_obj = encoder_obj_list[0]
        else:
            encoder_obj = encoder_obj_list
        self.encoder = fec.async_encoder(encoder_obj)
        self.msg_connect(weakref.proxy(self), 'in', self.encoder, 'in')
        self.msg_connect(self.encoder, 'out', weakref.proxy(self), 'out')