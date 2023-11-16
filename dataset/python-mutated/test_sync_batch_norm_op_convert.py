import unittest
import numpy as np
import paddle

class SyBNNet(paddle.nn.Layer):

    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        if False:
            print('Hello World!')
        super().__init__()
        self.bn_s1 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(paddle.nn.BatchNorm3D(out_ch, weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.0))))
        self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC'))

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.bn_s1(x)
        out = paddle.sum(paddle.abs(self.bn_s2(x)))
        return out

class BNNet(paddle.nn.Layer):

    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.bn_s1 = paddle.nn.BatchNorm3D(out_ch, weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.0)))
        self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC'))

    def forward(self, x):
        if False:
            return 10
        x = self.bn_s1(x)
        out = paddle.sum(paddle.abs(self.bn_s2(x)))
        return out

class TestConvertSyncBatchNormCase(unittest.TestCase):

    def test_convert(self):
        if False:
            i = 10
            return i + 15
        if not paddle.is_compiled_with_cuda():
            return
        bn_model = BNNet()
        sybn_model = SyBNNet()
        np.random.seed(10)
        data = np.random.random([3, 3, 3, 3, 3]).astype('float32')
        x = paddle.to_tensor(data)
        bn_out = bn_model(x)
        sybn_out = sybn_model(x)
        np.testing.assert_allclose(bn_out.numpy(), sybn_out.numpy(), rtol=1e-05, err_msg='Output has diff. \n' + '\nBN     ' + str(bn_out.numpy()) + '\n' + 'Sync BN ' + str(sybn_out.numpy()))
if __name__ == '__main__':
    unittest.main()