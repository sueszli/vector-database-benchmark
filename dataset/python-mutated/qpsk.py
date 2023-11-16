"""
QPSK modulation.

Demodulation is not included since the generic_mod_demod
"""
from gnuradio import gr
from gnuradio.digital.generic_mod_demod import generic_mod, generic_demod
from gnuradio.digital.generic_mod_demod import shared_mod_args, shared_demod_args
from .utils import mod_codes
from . import digital_python as digital
from . import modulation_utils
_def_mod_code = mod_codes.GRAY_CODE

def qpsk_constellation(mod_code=_def_mod_code):
    if False:
        while True:
            i = 10
    '\n    Creates a QPSK constellation.\n    '
    if mod_code != mod_codes.GRAY_CODE:
        raise ValueError('This QPSK mod/demod works only for gray-coded constellations.')
    return digital.constellation_qpsk()

def dqpsk_constellation(mod_code=_def_mod_code):
    if False:
        for i in range(10):
            print('nop')
    if mod_code != mod_codes.GRAY_CODE:
        raise ValueError("The DQPSK constellation is only generated for gray_coding.  But it can be used for non-grayed coded modulation if one doesn't use the pre-differential code.")
    return digital.constellation_dqpsk()
modulation_utils.add_type_1_constellation('qpsk', qpsk_constellation)
modulation_utils.add_type_1_constellation('dqpsk', dqpsk_constellation)