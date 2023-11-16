import unittest
import numpy as np
from op_test import OpTest
SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0

def identity(x):
    if False:
        i = 10
        return i + 15
    return x

def sigmoid(x):
    if False:
        i = 10
        return i + 15
    y = np.copy(x)
    y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
    y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
    return 1.0 / (1.0 + np.exp(-y))

def tanh(x):
    if False:
        for i in range(10):
            print('nop')
    y = -2.0 * x
    y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
    return 2.0 / (1.0 + np.exp(y)) - 1.0

def relu(x):
    if False:
        print('Hello World!')
    return np.maximum(x, 0)
ACTIVATION = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

def lstm(input, lod, h0=None, c0=None, w_h=None, w_b=None, w_c=None, is_reverse=False, act_gate=None, act_cell=None, act_cand=None):
    if False:
        return 10

    def _step(x, w_h, w_c, h_pre, c_pre, act_gate, act_cell, act_cand):
        if False:
            i = 10
            return i + 15
        g = np.dot(h_pre, w_h)
        g = g + x
        g = np.reshape(g, (1, g.size))
        (c, g_i, g_f, g_o) = np.split(g, 4, axis=1)
        if w_c is None:
            g_i = act_gate(g_i)
            g_f = act_gate(g_f)
        else:
            (w_ic, w_fc, w_oc) = np.split(w_c, 3, axis=1)
            g_i = act_gate(g_i + w_ic * c_pre)
            g_f = act_gate(g_f + w_fc * c_pre)
        c = g_f * c_pre + g_i * act_cand(c)
        if w_c is None:
            g_o = act_gate(g_o)
        else:
            (_, _, w_oc) = np.split(w_c, 3, axis=1)
            g_o = act_gate(g_o + w_oc * c)
        h = g_o * act_cell(c)
        return (h, c)

    def _reverse(x, offset):
        if False:
            return 10
        y = np.zeros_like(x)
        for i in range(len(offset) - 1):
            (b, e) = (offset[i], offset[i + 1])
            y[b:e, :] = np.flip(x[b:e, :], 0)
        return y
    offset = [0]
    for l in lod[0]:
        offset.append(offset[-1] + l)
    batch_size = len(lod[0])
    hidden = []
    cell = []
    input = _reverse(input, offset) if is_reverse else input
    if w_b is not None:
        input = input + np.tile(w_b, (offset[-1], 1))
    for i in range(batch_size):
        seq_len = lod[0][i]
        x = input[offset[i]:offset[i + 1], :]
        h_pre = h0[i]
        c_pre = c0[i]
        for j in range(seq_len):
            (h_pre, c_pre) = _step(x[j], w_h, w_c, h_pre, c_pre, act_gate, act_cell, act_cand)
            hidden.append(h_pre.flatten())
            cell.append(c_pre.flatten())
    hidden = np.array(hidden).astype('float64')
    cell = np.array(cell).astype('float64')
    hidden = _reverse(hidden, offset) if is_reverse else hidden
    cell = _reverse(cell, offset) if is_reverse else cell
    assert hidden.shape == (input.shape[0], input.shape[1] / 4)
    assert cell.shape == (input.shape[0], input.shape[1] / 4)
    return (hidden, cell)

class TestLstmOp(OpTest):

    def set_is_test(self):
        if False:
            i = 10
            return i + 15
        self.is_test = False

    def set_lod(self):
        if False:
            print('Hello World!')
        self.lod = [[2, 3, 2]]

    def set_argument(self):
        if False:
            print('Hello World!')
        self.set_is_test()
        self.set_lod()
        self.D = 16
        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'
        self.has_initial_state = False
        self.is_reverse = False
        self.use_peepholes = True

    def setUp(self):
        if False:
            return 10
        self.set_argument()
        self.op_type = 'lstm'
        T = sum(self.lod[0])
        N = len(self.lod[0])
        x = np.random.normal(size=(T, 4 * self.D)).astype('float64')
        if self.has_initial_state:
            h0 = np.random.normal(size=(N, self.D)).astype('float64')
            c0 = np.random.normal(size=(N, self.D)).astype('float64')
        else:
            h0 = np.zeros((N, self.D)).astype('float64')
            c0 = np.zeros((N, self.D)).astype('float64')
        w = np.random.normal(size=(self.D, 4 * self.D)).astype('float64')
        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float64')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float64')
        w_b = b[:, 0:4 * self.D]
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None
        (h, c) = lstm(x, self.lod, h0, c0, w, w_b, w_c, self.is_reverse, ACTIVATION[self.act_gate], ACTIVATION[self.act_cell], ACTIVATION[self.act_cand])
        self.inputs = {'Input': (x, self.lod), 'Weight': w}
        self.inputs['Bias'] = b
        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0
        self.outputs = {'Hidden': (h, self.lod), 'Cell': (c, self.lod)}
        self.attrs = {'use_peepholes': self.use_peepholes, 'is_reverse': self.is_reverse, 'gate_activation': self.act_gate, 'cell_activation': self.act_cell, 'candidate_activation': self.act_cand, 'is_test': self.is_test}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(atol=1e-08, check_dygraph=False)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros((N, self.D)).astype('float64')
        self.check_grad(['Input', 'Weight', 'Bias'], ['Hidden'], max_relative_error=0.0005, check_dygraph=False)

class TestLstmOpCase1(TestLstmOp):

    def set_lod(self):
        if False:
            for i in range(10):
                print('nop')
        self.lod = [[0, 3, 2]]

class TestLstmOpCase2(TestLstmOp):

    def set_lod(self):
        if False:
            return 10
        self.lod = [[0, 3, 0]]

class TestLstmOpCase3(TestLstmOp):

    def set_lod(self):
        if False:
            i = 10
            return i + 15
        self.lod = [[2, 0, 4]]

class TestLstmOpInference(TestLstmOp):

    def set_is_test(self):
        if False:
            while True:
                i = 10
        self.is_test = True

    def test_check_grad(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    unittest.main()