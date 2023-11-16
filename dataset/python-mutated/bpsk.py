"""
BPSK modulation and demodulation.
"""
from math import pi, log
from cmath import exp
from gnuradio import gr
from gnuradio.digital.generic_mod_demod import generic_mod, generic_demod
from gnuradio.digital.generic_mod_demod import shared_mod_args, shared_demod_args
from . import digital_python
from . import modulation_utils

def bpsk_constellation():
    if False:
        print('Hello World!')
    return digital_python.constellation_bpsk()

def dbpsk_constellation():
    if False:
        return 10
    return digital_python.constellation_dbpsk()
modulation_utils.add_type_1_constellation('bpsk', bpsk_constellation)
modulation_utils.add_type_1_constellation('dbpsk', dbpsk_constellation)