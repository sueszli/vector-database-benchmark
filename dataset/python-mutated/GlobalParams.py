"""Load arrays or pickled objects from .npy, .npz or pickled files."""
import os
import numpy as np
params = {}
init_token = ''

def load_params(path, params):
    if False:
        print('Hello World!')
    pp = np.load(path)
    for (kk, vv) in pp.iteritems():
        if 0:
            if kk not in params:
                pass
                raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params
load_params(os.path.abspath('../resource/initialvalue_resource/randomvalue.npz'), params)
'\nkeys in params: \nW_O\nlstm_de_U\nlstm_de_W\nW_A\nV_A\nlstm_U_l\ninput_bias\nV_O\nlstm_U\nlstm_W\nYUN_O\nW_S\nout_bias\nW_yun_pre\ninput_de_bias\nW_lv_pre\nC_O\nLV_O\nlstm_W_l\nU_A\nlstm_W_i\nC_lstm_end_once\nU_O\ninput_bias_l\nC_lstm\n'