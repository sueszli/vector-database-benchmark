import unittest
import numpy as np
from op_test import OpTest, OpTestTool, skip_check_grad_ci
import paddle

class TestReduceSumDefaultOneDNNOp(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype('float32')}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_dygraph=False, check_pir=False)

class TestReduceDefaultWithGradOneDNNOp(TestReduceSumDefaultOneDNNOp):

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', check_dygraph=False, check_pir=False)

class TestReduceSum4DOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 10, 5, 5)).astype('float32')}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [2]}
        self.outputs = {'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))}

class TestReduceSum4DReduceAllDimAttributeBF16OneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 10, 5, 3)).astype('float32')}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': [0, 1, 2, 3]}
        self.outputs = {'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))}

class TestReduceSum5DKeepDimsOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((2, 5, 3, 2, 2)).astype('float32')}
        self.attrs = {'dim': (2, 3, 4), 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']), keepdims=self.attrs['keep_dim'])}

class TestReduceSum0DOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random(()).astype('float32')}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': []}
        self.outputs = {'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))}

class TestReduceSum5DReduceAllKeepDimsOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((2, 5, 3, 2, 2)).astype('float32')}
        self.attrs = {'reduce_all': True, 'keep_dim': True, 'use_mkldnn': True}
        self.outputs = {'Out': self.inputs['X'].sum(keepdims=self.attrs['keep_dim'])}

class TestReduceSum4DReduceAllOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype('float32')}
        self.attrs = {'reduce_all': True, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].sum()}

@OpTestTool.skip_if(True, reason='According to Paddle API, None dim means reduce all instead of copy, so just skip this test to avoid potential failure')
class TestReduceSum4DNoReduceSimpleCopyOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'reduce_sum'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype('float32')}
        self.attrs = {'dim': (), 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': np.copy(self.inputs['X'])}

@skip_check_grad_ci(reason='reduce_max is discontinuous non-derivable function, its gradient check is not supported by unittest framework.')
class TestReduceMax3DOneDNNOp(TestReduceSumDefaultOneDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'reduce_max'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype('float32')}
        self.attrs = {'dim': [-1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))}

@skip_check_grad_ci(reason='reduce_max is discontinuous non-derivable function, its gradient check is not supported by unittest framework.')
class TestReduceMax0DOneDNNOp(TestReduceSumDefaultOneDNNOp):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'reduce_max'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random(()).astype('float32')}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': []}
        self.outputs = {'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))}

@skip_check_grad_ci(reason='reduce_max is discontinuous non-derivable function, its gradient check is not supported by unittest framework.')
class TestReduceMax4DNegativeAndPositiveDimsOneDNNOp(TestReduceSumDefaultOneDNNOp):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'reduce_max'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10, 9)).astype('float32')}
        self.attrs = {'dim': [-1, 0, 1], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))}

@skip_check_grad_ci(reason='reduce_min is discontinuous non-derivable function, its gradient check is not supported by unittest framework.')
class TestReduceMin3DOneDNNOp(TestReduceSumDefaultOneDNNOp):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'reduce_min'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype('float32')}
        self.attrs = {'dim': [2], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))}

@skip_check_grad_ci(reason='reduce_min is discontinuous non-derivable function, its gradient check is not supported by unittest framework.')
class TestReduceMin0DOneDNNOp(TestReduceSumDefaultOneDNNOp):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'reduce_min'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random(()).astype('float32')}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': []}
        self.outputs = {'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))}

class TestReduceMean3DOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'reduce_mean'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 10)).astype('float32')}
        self.attrs = {'dim': [0], 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0) / self.inputs['X'].shape[0]}

class TestReduceMean0DOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'reduce_mean'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random(()).astype('float32')}
        self.attrs = {'use_mkldnn': self.use_mkldnn, 'dim': []}
        self.outputs = {'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))}

class TestReduceMean4DReduceAllOneDNNOp(TestReduceDefaultWithGradOneDNNOp):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'reduce_mean'
        self.use_mkldnn = True
        self.inputs = {'X': np.random.random((5, 6, 8, 10)).astype('float32')}
        self.attrs = {'reduce_all': True, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.inputs['X'].sum() / np.asarray(self.inputs['X'].shape).prod()}
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()