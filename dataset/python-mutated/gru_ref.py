"""
This is a Minimal single layer GRU implementation adapted from DoctorTeeth's code:
https://github.com/DoctorTeeth/gru/blob/master/gru.py

The adaptation includes
  - remove the GRU to output affine transformation
  - provide inputs for forward pass and errors for backward pass
  - being able to provide init_state
  - initialize weights and biases into zeros, as the main test script will externally
    initialize the weights and biases
  - being able to read out hashable values to compare with another GRU
    implementation
"""
import numpy as np

class GRU(object):

    def __init__(self, in_size, hidden_size):
        if False:
            print('Hello World!')
        '\n        This class implements a GRU.\n        '
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.Wxc = np.zeros((hidden_size, in_size))
        self.Wxr = np.zeros((hidden_size, in_size))
        self.Wxz = np.zeros((hidden_size, in_size))
        self.Rhc = np.zeros((hidden_size, hidden_size))
        self.Rhr = np.zeros((hidden_size, hidden_size))
        self.Rhz = np.zeros((hidden_size, hidden_size))
        self.bc = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bz = np.zeros((hidden_size, 1))
        self.weights = [self.Wxc, self.Wxr, self.Wxz, self.Rhc, self.Rhr, self.Rhz, self.bc, self.br, self.bz]
        self.names = ['Wxc', 'Wxr', 'Wxz', 'Rhc', 'Rhr', 'Rhz', 'bc', 'br', 'bz']
        self.weights_ind_Wxc = 0
        self.weights_ind_Wxr = 1
        self.weights_ind_Wxz = 2
        self.weights_ind_Rhc = 3
        self.weights_ind_Rhr = 4
        self.weights_ind_Rhz = 5
        self.weights_ind_bc = 6
        self.weights_ind_br = 7
        self.weights_ind_bz = 8

    def lossFun(self, inputs, errors, init_state=None):
        if False:
            print('Hello World!')
        '\n        Does a forward and backward pass on the network using (inputs, errors)\n        inputs is a bit-vector of seq-length\n        errors is a bit-vector of seq-length\n        '
        (xs, rbars, rs, zbars, zs, cbars, cs, hs) = ({}, {}, {}, {}, {}, {}, {}, {})
        if init_state is None:
            hs[-1] = np.zeros((self.hidden_size, 1))
        else:
            hs[-1] = init_state
        seq_len = len(inputs)
        hs_list = np.zeros((self.hidden_size, seq_len))
        for t in range(seq_len):
            xs[t] = np.matrix(inputs[t])
            rbars[t] = np.dot(self.Wxr, xs[t]) + np.dot(self.Rhr, hs[t - 1]) + self.br
            rs[t] = 1.0 / (1 + np.exp(-rbars[t]))
            zbars[t] = np.dot(self.Wxz, xs[t]) + np.dot(self.Rhz, hs[t - 1]) + self.bz
            zs[t] = 1.0 / (1 + np.exp(-zbars[t]))
            cbars[t] = np.dot(self.Wxc, xs[t]) + np.dot(self.Rhc, np.multiply(rs[t], hs[t - 1])) + self.bc
            cs[t] = np.tanh(cbars[t])
            ones = np.ones_like(zs[t])
            hs[t] = np.multiply(cs[t], zs[t]) + np.multiply(hs[t - 1], ones - zs[t])
            hs_list[:, t] = hs[t].flatten()
        dWxc = np.zeros_like(self.Wxc)
        dWxr = np.zeros_like(self.Wxr)
        dWxz = np.zeros_like(self.Wxz)
        dRhc = np.zeros_like(self.Rhc)
        dRhr = np.zeros_like(self.Rhr)
        dRhz = np.zeros_like(self.Rhz)
        dbc = np.zeros_like(self.bc)
        dbr = np.zeros_like(self.br)
        dbz = np.zeros_like(self.bz)
        dhnext = np.zeros_like(hs[0])
        drbarnext = np.zeros_like(rbars[0])
        dzbarnext = np.zeros_like(zbars[0])
        dcbarnext = np.zeros_like(cbars[0])
        zs[len(inputs)] = np.zeros_like(zs[0])
        rs[len(inputs)] = np.zeros_like(rs[0])
        dh_list = errors
        dh_list_out = np.zeros_like(dh_list)
        dr_list = np.zeros((self.hidden_size, seq_len))
        dz_list = np.zeros((self.hidden_size, seq_len))
        dc_list = np.zeros((self.hidden_size, seq_len))
        for t in reversed(range(seq_len)):
            dha = np.multiply(dhnext, ones - zs[t + 1])
            dhb = np.dot(self.Rhr.T, drbarnext)
            dhc = np.dot(self.Rhz.T, dzbarnext)
            dhd = np.multiply(rs[t + 1], np.dot(self.Rhc.T, dcbarnext))
            dhe = dh_list[t]
            dh = dha + dhb + dhc + dhd + dhe
            dh_list_out[t] = dh
            dc = np.multiply(dh, zs[t])
            dcbar = np.multiply(dc, ones - np.square(cs[t]))
            dr = np.multiply(hs[t - 1], np.dot(self.Rhc.T, dcbar))
            dz = np.multiply(dh, cs[t] - hs[t - 1])
            drbar = np.multiply(dr, np.multiply(rs[t], ones - rs[t]))
            dzbar = np.multiply(dz, np.multiply(zs[t], ones - zs[t]))
            dWxr += np.dot(drbar, xs[t].T)
            dWxz += np.dot(dzbar, xs[t].T)
            dWxc += np.dot(dcbar, xs[t].T)
            dRhr += np.dot(drbar, hs[t - 1].T)
            dRhz += np.dot(dzbar, hs[t - 1].T)
            dRhc += np.dot(dcbar, np.multiply(rs[t], hs[t - 1]).T)
            dbr += drbar
            dbc += dcbar
            dbz += dzbar
            dhnext = dh
            drbarnext = drbar
            dzbarnext = dzbar
            dcbarnext = dcbar
            dr_list[:, t] = drbar.flatten()
            dz_list[:, t] = dzbar.flatten()
            dc_list[:, t] = dcbar.flatten()
        dw = [dWxc, dWxr, dWxz, dRhc, dRhr, dRhz, dbc, dbr, dbz]
        self.dW_ind_Wxc = 0
        self.dW_ind_Wxr = 1
        self.dW_ind_Wxz = 2
        self.dW_ind_Rhc = 3
        self.dW_ind_Rhr = 4
        self.dW_ind_Rhz = 5
        self.dW_ind_bc = 6
        self.dW_ind_br = 7
        self.dW_ind_bz = 8
        return (dw, hs_list, dh_list_out, dr_list, dz_list, dc_list)