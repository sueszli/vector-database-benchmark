import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def numpy_scatter_nd(ref, index, updates, fun):
    if False:
        while True:
            i = 10
    ref_shape = ref.shape
    index_shape = index.shape
    end_size = index_shape[-1]
    remain_numl = np.prod(index_shape[:-1]).astype('int32')
    slice_size = np.prod(ref_shape[end_size:len(ref_shape)]).astype('int32')
    flat_index = index.reshape([remain_numl] + list(index_shape[-1:]))
    flat_updates = updates.reshape((remain_numl, slice_size))
    flat_output = ref.reshape(list(ref_shape[:end_size]) + [slice_size])
    for (i_up, i_out) in enumerate(flat_index):
        i_out = tuple(i_out)
        flat_output[i_out] = fun(flat_output[i_out], flat_updates[i_up])
    return flat_output.reshape(ref.shape)

def numpy_scatter_nd_add(ref, index, updates):
    if False:
        print('Hello World!')
    return numpy_scatter_nd(ref, index, updates, lambda x, y: x + y)

def judge_update_shape(ref, index):
    if False:
        return 10
    ref_shape = ref.shape
    index_shape = index.shape
    update_shape = []
    for i in range(len(index_shape) - 1):
        update_shape.append(index_shape[i])
    for i in range(index_shape[-1], len(ref_shape), 1):
        update_shape.append(ref_shape[i])
    return update_shape

class XPUTestScatterNdAdd(XPUOpTestWrapper):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_name = 'scatter_nd_add'

    class TestScatterNdAdd(XPUOpTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'scatter_nd_add'
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.init_data()
            self.inputs = {'X': self.x_np, 'Index': self.index_np, 'Updates': self.updates_np}
            output = numpy_scatter_nd_add(self.x_np.copy(), self.index_np, self.updates_np)
            self.outputs = {'Out': output}

        def test_check_output(self):
            if False:
                for i in range(10):
                    print('nop')
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if False:
                while True:
                    i = 10
            self.check_grad_with_place(self.place, ['X', 'Updates'], 'Out')

        def init_data(self):
            if False:
                while True:
                    i = 10
            self.x_np = np.random.random([100]).astype(self.dtype)
            self.index_np = np.random.randint(0, 100, [100, 1]).astype('int32')
            self.updates_np = np.random.random([100]).astype(self.dtype)

        def infer_dtype_from_inputs_outputs(self, inputs, outputs):
            if False:
                print('Hello World!')
            self.__class__.dtype = self.dtype
            self.output_dtype = self.dtype

    class TestScatterNdAddWithEmptyIndex(TestScatterNdAdd):

        def init_data(self):
            if False:
                print('Hello World!')
            self.x_np = np.random.random((10, 10)).astype(self.dtype)
            self.index_np = np.array([[[], []], [[], []]]).astype('int32')
            self.updates_np = np.random.random((2, 2, 10, 10)).astype(self.dtype)

    class TestScatterNdAddOpWithHighRankSame(TestScatterNdAdd):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            shape = (3, 2, 2, 1, 10)
            self.x_np = np.random.rand(*shape).astype(self.dtype)
            self.index_np = np.vstack([np.random.randint(0, s, size=100) for s in shape]).T.astype('int32')
            update_shape = judge_update_shape(self.x_np, self.index_np)
            self.updates_np = np.random.rand(*update_shape).astype(self.dtype)

    class TestScatterNdAddWithHighRankDiff(TestScatterNdAdd):

        def init_data(self):
            if False:
                for i in range(10):
                    print('nop')
            shape = (8, 2, 2, 1, 10)
            self.x_np = np.random.rand(*shape).astype(self.dtype)
            index_tmp = np.vstack([np.random.randint(0, s, size=500) for s in shape]).T
            self.index_np = index_tmp.reshape([10, 5, 10, 5]).astype('int64')
            update_shape = judge_update_shape(self.x_np, self.index_np)
            self.updates_np = np.random.rand(*update_shape).astype(self.dtype)

    class TestScatterNdAddWithMultiDimIndex(TestScatterNdAdd):

        def init_data(self):
            if False:
                return 10
            shape = (16, 3, 20, 20)
            self.x_np = np.random.rand(*shape).astype(self.dtype)
            self.index_np = np.random.rand(796, 4).astype('int32')
            update_shape = judge_update_shape(self.x_np, self.index_np)
            self.updates_np = np.random.rand(*update_shape).astype(self.dtype)

    class TestScatterNdAddWithZeroDimUpdates(TestScatterNdAdd):

        def init_data(self):
            if False:
                while True:
                    i = 10
            shape = (10,)
            self.x_np = np.random.rand(*shape).astype(self.dtype)
            self.index_np = np.random.randint(0, 10, [1]).astype('int32')
            self.updates_np = np.array(np.random.rand()).astype(self.dtype)
support_types = get_xpu_op_support_types('scatter_nd_add')
for stype in support_types:
    create_test_class(globals(), XPUTestScatterNdAdd, stype)
if __name__ == '__main__':
    unittest.main()