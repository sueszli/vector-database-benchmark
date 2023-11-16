import unittest
from test_fusion_lstm_op import TestFusionLSTMOp

class TestFusionLSTMONEDNNOp(TestFusionLSTMOp):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.use_mkldnn = True

    def test_check_output(self):
        if False:
            while True:
                i = 10
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(check_dygraph=False, no_check_set=['Cell'])

class TestFusionLSTMONEDNNOpReverse(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_reverse = True
        self.use_mkldnn = True

class TestFusionLSTMONEDNNOpInitReverse(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.has_initial_state = True
        self.is_reverse = True
        self.use_mkldnn = True

class TestFusionLSTMONEDNNOpMD1(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            i = 10
            return i + 15
        self.M = 36
        self.D = 8
        self.use_mkldnn = True

class TestFusionLSTMONEDNNOpMD2(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.M = 8
        self.D = 8
        self.use_mkldnn = True

class TestFusionLSTMONEDNNOpMD3(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            while True:
                i = 10
        self.M = 15
        self.D = 3
        self.use_mkldnn = True

class TestFusionLSTMONEDNNOpBS1(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            return 10
        self.lod = [[3]]
        self.D = 16
        self.use_mkldnn = True

class TestFusionLSTMONEDNNOpPeepholesInit(TestFusionLSTMONEDNNOp):

    def set_conf(self):
        if False:
            print('Hello World!')
        self.use_peepholes = True
        self.has_initial_state = True
        self.use_mkldnn = True
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()