import unittest
import numpy as np
from op_test import OpTest
from test_lstm_op import ACTIVATION, lstm

def fc(x, w, b):
    if False:
        return 10
    return np.dot(x, w) + b

def fusion_lstm(x, lod, wx=None, bx=None, h0=None, c0=None, w_h=None, w_b=None, w_c=None, is_reverse=False, act_gate=None, act_cell=None, act_cand=None):
    if False:
        print('Hello World!')
    return lstm(fc(x, wx, bx), lod, h0, c0, w_h, w_b, w_c, is_reverse, act_gate, act_cell, act_cand)

class TestFusionLSTMOp(OpTest):

    def set_conf(self):
        if False:
            while True:
                i = 10
        pass

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'fusion_lstm'
        self.lod = [[2, 3, 5, 4]]
        self.M = 8
        self.D = 16
        self.has_initial_state = False
        self.use_peepholes = False
        self.is_reverse = False
        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'
        self.use_mkldnn = False
        self.set_conf()
        T = sum(self.lod[0])
        bs = len(self.lod[0])
        x = np.random.normal(size=(T, self.M)).astype('float32')
        if self.has_initial_state:
            h0 = np.random.normal(size=(bs, self.D)).astype('float32')
            c0 = np.random.normal(size=(bs, self.D)).astype('float32')
        else:
            h0 = np.zeros((bs, self.D)).astype('float32')
            c0 = np.zeros((bs, self.D)).astype('float32')
        wh = np.random.normal(size=(self.D, 4 * self.D)).astype('float32')
        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float32')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float32')
        w_b = np.copy(b[:, 0:4 * self.D])
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None
        wx = np.random.normal(size=(self.M, 4 * self.D)).astype('float32')
        bx = np.random.normal(size=(1, 4 * self.D)).astype('float32')
        b[0, 0:4 * self.D] += bx[0, :]
        (h, c) = fusion_lstm(x, self.lod, wx, bx, h0, c0, wh, w_b, w_c, self.is_reverse, ACTIVATION[self.act_gate], ACTIVATION[self.act_cell], ACTIVATION[self.act_cand])
        self.inputs = {'X': (x, self.lod), 'WeightX': wx, 'WeightH': wh, 'Bias': b}
        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0
        self.outputs = {'Hidden': (h, self.lod), 'Cell': (c, self.lod)}
        self.attrs = {'use_peepholes': self.use_peepholes, 'is_reverse': self.is_reverse, 'gate_activation': self.act_gate, 'cell_activation': self.act_cell, 'candidate_activation': self.act_cand, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(check_dygraph=False)

class TestFusionLSTMOpInit(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.has_initial_state = True

class TestFusionLSTMOpReverse(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            return 10
        self.is_reverse = True

class TestFusionLSTMOpInitReverse(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            return 10
        self.has_initial_state = True
        self.is_reverse = True

class TestFusionLSTMOpMD1(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            return 10
        self.M = 36
        self.D = 8

class TestFusionLSTMOpMD2(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.M = 8
        self.D = 8

class TestFusionLSTMOpMD3(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.M = 15
        self.D = 3

class TestFusionLSTMOpBS1(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.lod = [[3]]
        self.D = 16

class TestFusionLSTMOpPeepholes(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.use_peepholes = True

class TestFusionLSTMOpPeepholesInit(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.use_peepholes = True
        self.has_initial_state = True

class TestFusionLSTMOpPeepholesReverse(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.use_peepholes = True
        self.is_reverse = True

class TestFusionLSTMOpPeepholesInitReverse(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.use_peepholes = True
        self.has_initial_state = True
        self.is_reverse = True

class TestFusionLSTMOpPeepholesBS1(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_peepholes = True
        self.lod = [[2]]
        self.D = 8
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()