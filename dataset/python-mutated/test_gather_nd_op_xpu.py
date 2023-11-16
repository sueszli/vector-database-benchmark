import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestGatherNd(XPUOpTestWrapper):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.op_name = 'gather_nd'

    class XPUTestGatherNdBase(XPUOpTest):

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.op_type = 'gather_nd'
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.init_data()
            self.inputs = {'X': self.xnp, 'Index': self.inp}
            self.outputs = {'Out': self.output}

        def test_check_output(self):
            if False:
                return 10
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_grad(['X'], 'Out', check_dygraph=False)

        def init_data(self):
            if False:
                print('Hello World!')
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([[], []]).astype('int32')
            self.output = np.vstack((self.xnp[np.newaxis, :], self.xnp[np.newaxis, :]))

        def infer_dtype_from_inputs_outputs(self, inputs, outputs):
            if False:
                print('Hello World!')
            self.__class__.dtype = self.dtype
            self.output_dtype = self.dtype

    class XPUTestGatherNdOpWithEmptyIndex1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([[], []]).astype('int32')
            self.output = np.vstack((self.xnp[np.newaxis, :], self.xnp[np.newaxis, :]))

    class XPUTestGatherNdOpWithEmptyIndex2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                return 10
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([[], []]).astype('int64')
            self.output = np.vstack((self.xnp[np.newaxis, :], self.xnp[np.newaxis, :]))

    class XPUTestGatherNdOpWithIndex1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([1]).astype('int32')
            self.output = self.xnp[tuple(self.inp)]

    class XPUTestGatherNdOpWithIndex2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([1]).astype('int64')
            self.output = self.xnp[tuple(self.inp)]

    class XPUTestGatherNdOpWithLowIndex1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([[1], [2]]).astype('int32')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithLowIndex2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([1, 2]).astype('int64')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankSame1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                while True:
                    i = 10
            shape = (5, 2, 3, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack([np.random.randint(0, s, size=2) for s in shape]).T.astype('int32')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankSame2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                while True:
                    i = 10
            shape = (5, 2, 3, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack([np.random.randint(0, s, size=2) for s in shape]).T.astype('int64')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankDiff1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                while True:
                    i = 10
            shape = (2, 3, 4, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack([np.random.randint(0, s, size=200) for s in shape]).T.astype('int32')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankDiff2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            shape = (2, 3, 4, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack([np.random.randint(0, s, size=200) for s in shape]).T.astype('int64')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithSameIndexAsX1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                print('Hello World!')
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([[1, 1], [2, 1]]).astype('int32')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithSameIndexAsX2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([[1, 1], [2, 1]]).astype('int64')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpIndex1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([1, 2]).astype('int32')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpIndex2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([1, 2]).astype('int64')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpMultiDimIndex1(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                i = 10
                return i + 15
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([2, 2]).astype('int32')
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpMultiDimIndex2(XPUTestGatherNdBase):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([2, 2]).astype('int64')
            self.output = self.xnp[tuple(self.inp.T)]
support_types = get_xpu_op_support_types('gather_nd')
for stype in support_types:
    create_test_class(globals(), XPUTestGatherNd, stype)
if __name__ == '__main__':
    unittest.main()