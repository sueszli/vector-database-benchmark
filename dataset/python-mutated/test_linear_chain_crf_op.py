import random
import unittest
import numpy as np
from op_test import OpTest

class LinearChainCrfForward:

    def __init__(self, seq_start_positions, emission_weights, emission_row_max, emission_exps, transition_weights, transition_exps, labels):
        if False:
            return 10
        self.tag_num = emission_weights.shape[1]
        self.seq_num = len(seq_start_positions) - 1
        self.seq_start_positions = seq_start_positions
        self.labels = labels
        self.x = emission_weights
        self.x_row_max = emission_row_max
        self.x_exps = emission_exps
        self.a = transition_weights[0, :]
        self.a_exps = transition_exps[0, :]
        self.b = transition_weights[1, :]
        self.b_exps = transition_exps[1, :]
        self.w = transition_weights[2:, :]
        self.w_exps = transition_exps[2:, :]
        self.alpha = np.zeros((seq_start_positions[-1], self.tag_num), dtype='float64')
        self.log_likelihood = np.zeros((self.seq_num, 1))

    def _l1_norm(self, x):
        if False:
            for i in range(10):
                print('nop')
        s = np.sum(x)
        x /= s
        return s

    def _forward_a_sequence(self, x, x_row_max, x_exps, label, alpha):
        if False:
            while True:
                i = 10
        seq_len = x_row_max.shape[0]
        log_likelihood = 0.0
        for i in range(self.tag_num):
            alpha[0, i] = self.a_exps[i] * x_exps[0, i]
        log_likelihood = -x_row_max[0] - np.log(self._l1_norm(alpha[0, :]))
        for k in range(1, seq_len):
            for i in range(self.tag_num):
                s = 0.0
                for j in range(self.tag_num):
                    s += alpha[k - 1, j] * self.w_exps[j, i]
                alpha[k, i] = x_exps[k, i] * s
            log_likelihood -= x_row_max[k] + np.log(self._l1_norm(alpha[k, :]))
        s = 0.0
        for i in range(self.tag_num):
            s += alpha[-1, i] * self.b_exps[i]
        log_likelihood -= np.log(s)
        log_likelihood += self.a[label[0]] + x[0, label[0]] + self.b[label[-1]]
        for k in range(1, seq_len):
            log_likelihood += x[k, label[k]] + self.w[label[k - 1], label[k]]
        return -log_likelihood

    def crf_forward_compute(self):
        if False:
            while True:
                i = 10
        for i in range(self.seq_num):
            start = self.seq_start_positions[i]
            end = self.seq_start_positions[i + 1]
            if start >= end:
                continue
            self.log_likelihood[i] = self._forward_a_sequence(self.x[start:end, :], self.x_row_max[start:end, :], self.x_exps[start:end, :], self.labels[start:end, :], self.alpha[start:end, :])
        return (self.alpha, self.log_likelihood)

class TestLinearChainCrfOp(OpTest):

    def set_test_data(self):
        if False:
            for i in range(10):
                print('nop')
        SEQ_NUM = 3
        TAG_NUM = 17
        MAX_SEQ_LEN = 5
        lod = [[]]
        seq_start_pos = [0]
        for i in range(SEQ_NUM):
            lod[-1].append(random.randint(1, MAX_SEQ_LEN))
            seq_start_pos.append(seq_start_pos[-1] + lod[-1][-1])
        emission = np.random.uniform(-1, 1, [seq_start_pos[-1], TAG_NUM]).astype('float64')
        emission_row_max = np.amax(emission, axis=1, keepdims=True)
        emission_exps = np.exp(emission - emission_row_max)
        transition = np.random.uniform(-0.5, 0.5, [TAG_NUM + 2, TAG_NUM]).astype('float64')
        transition_exps = np.exp(transition)
        labels = np.random.randint(low=0, high=TAG_NUM, size=(seq_start_pos[-1], 1), dtype='int64')
        self.inputs = {'Emission': (emission, lod), 'Transition': transition, 'Label': (labels, lod)}
        crf = LinearChainCrfForward(seq_start_pos, emission, emission_row_max, emission_exps, transition, transition_exps, labels)
        (alpha, log_likelihood) = crf.crf_forward_compute()
        self.outputs = {'Alpha': alpha, 'EmissionExps': emission_exps, 'TransitionExps': transition_exps, 'LogLikelihood': log_likelihood}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'linear_chain_crf'
        self.set_test_data()

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['Emission', 'Transition'], 'LogLikelihood')

    def test_check_grad_ignore_transition(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['Emission'], 'LogLikelihood', no_grad_set=set('Transition'))

class TestLinearChainCrfPaddingTensor(OpTest):

    def seq_pad(self, data, length):
        if False:
            return 10
        max_len = np.max(length)
        shape = [len(length), max_len] + list(data.shape[1:])
        padded = np.zeros(shape).astype(data.dtype)
        offset = 0
        for (i, l) in enumerate(length):
            padded[i, 0:l] = data[offset:offset + l]
            offset += l
        return padded

    def seq_pad_exps(self, data, length):
        if False:
            return 10
        max_len = np.max(length)
        shape = [len(length), max_len] + list(data.shape[1:])
        padded = np.ones(shape).astype(data.dtype)
        offset = 0
        for (i, l) in enumerate(length):
            padded[i, 0:l] = data[offset:offset + l]
            offset += l
        return padded

    def set_test_data_1(self):
        if False:
            for i in range(10):
                print('nop')
        SEQ_NUM = 3
        TAG_NUM = 17
        MAX_SEQ_LEN = 5
        lod = [[]]
        seq_start_pos = [0]
        for i in range(SEQ_NUM):
            lod[-1].append(random.randint(1, MAX_SEQ_LEN))
            seq_start_pos.append(seq_start_pos[-1] + lod[-1][-1])
        emission = np.random.uniform(-1, 1, [seq_start_pos[-1], TAG_NUM]).astype('float64')
        emission_row_max = np.amax(emission, axis=1, keepdims=True)
        emission_exps = np.exp(emission - emission_row_max)
        transition = np.random.uniform(-0.5, 0.5, [TAG_NUM + 2, TAG_NUM]).astype('float64')
        transition_exps = np.exp(transition)
        labels = np.random.randint(low=0, high=TAG_NUM, size=(seq_start_pos[-1], 1), dtype='int64')
        self.inputs = {'Emission': self.seq_pad(emission, lod[0]), 'Transition': transition, 'Label': self.seq_pad(labels, lod[0]), 'Length': np.array(lod).astype('int64')}
        crf = LinearChainCrfForward(seq_start_pos, emission, emission_row_max, emission_exps, transition, transition_exps, labels)
        (alpha, log_likelihood) = crf.crf_forward_compute()
        self.outputs = {'Alpha': self.seq_pad(alpha, lod[0]), 'EmissionExps': self.seq_pad_exps(emission_exps, lod[0]), 'TransitionExps': transition_exps, 'LogLikelihood': log_likelihood}

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'linear_chain_crf'
        self.set_test_data_1()

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['Emission', 'Transition'], 'LogLikelihood')

    def test_check_grad_ignore_transition(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['Emission'], 'LogLikelihood', no_grad_set=set('Transition'))
if __name__ == '__main__':
    unittest.main()