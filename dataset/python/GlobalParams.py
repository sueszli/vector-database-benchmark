"""Load arrays or pickled objects from .npy, .npz or pickled files."""

import os
import numpy as np

params = {}
init_token = ''


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in pp.iteritems():
        if 0:
            if kk not in params:
                pass
                raise Warning('%s is not in the archive' % kk)
            # if options['use_global']==0 and kk=='C_lstm_end_once':
            #     pass
        params[kk] = pp[kk]

    return params


load_params(os.path.abspath('../resource/initialvalue_resource/randomvalue.npz'), params)

'''
keys in params: 
W_O
lstm_de_U
lstm_de_W
W_A
V_A
lstm_U_l
input_bias
V_O
lstm_U
lstm_W
YUN_O
W_S
out_bias
W_yun_pre
input_de_bias
W_lv_pre
C_O
LV_O
lstm_W_l
U_A
lstm_W_i
C_lstm_end_once
U_O
input_bias_l
C_lstm
'''
