import unittest
from test_fusion_gru_op import TestFusionGRUOp

class TestFusionGRUMKLDNNOp(TestFusionGRUOp):

    def set_confs(self):
        if False:
            while True:
                i = 10
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpNoInitial(TestFusionGRUOp):

    def set_confs(self):
        if False:
            i = 10
            return i + 15
        self.with_h0 = False
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpNoBias(TestFusionGRUOp):

    def set_confs(self):
        if False:
            for i in range(10):
                print('nop')
        self.with_bias = False
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpReverse(TestFusionGRUOp):

    def set_confs(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_reverse = True
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpOriginMode(TestFusionGRUOp):

    def set_confs(self):
        if False:
            return 10
        self.origin_mode = True
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpMD1(TestFusionGRUOp):

    def set_confs(self):
        if False:
            for i in range(10):
                print('nop')
        self.M = 36
        self.D = 8
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpMD2(TestFusionGRUOp):

    def set_confs(self):
        if False:
            while True:
                i = 10
        self.M = 8
        self.D = 8
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpMD3(TestFusionGRUOp):

    def set_confs(self):
        if False:
            while True:
                i = 10
        self.M = 17
        self.D = 15
        self.use_mkldnn = True

class TestFusionGRUMKLDNNOpBS1(TestFusionGRUOp):

    def set_confs(self):
        if False:
            print('Hello World!')
        self.lod = [[3]]
        self.D = 16
        self.use_mkldnn = True
if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()