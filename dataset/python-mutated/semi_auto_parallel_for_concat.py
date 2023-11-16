from semi_auto_parallel_util import SemiAutoParallelTestBase
import paddle
import paddle.distributed as dist
'\ntest for concat、slice 、split\n'

class TestSplitAndConcatSemiAutoParallel(SemiAutoParallelTestBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def check_dim_mapping(self, output, expected_dim_mapping):
        if False:
            print('Hello World!')
        assert output.dist_attr.dims_mapping == expected_dim_mapping, f'{output.dist_attr.dims_mapping}  vs {expected_dim_mapping}'

    def test_concat_forward(self):
        if False:
            while True:
                i = 10
        shapes = [[16, 4, 4], [64, 4, 4]]
        specs = [[None, None, 'x'], [None, None, 'x']]
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.concat, with_backward=False, axis=0)
        self.check_dim_mapping(outputs, [-1, -1, 0])

    def test_concat_forward_reshard(self):
        if False:
            i = 10
            return i + 15
        shapes = [[16, 4, 4], [64, 4, 4]]
        specs = [['x', None, None], [None, None, 'x']]
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.concat, with_backward=False, axis=0)
        self.check_dim_mapping(outputs, [-1, -1, 0])

    def test_stack_forward(self):
        if False:
            return 10
        shapes = [[16, 4, 4], [16, 4, 4]]
        specs = [[None, None, 'x'], [None, None, 'x']]
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.stack, with_backward=False, axis=0)
        self.check_dim_mapping(outputs, [-1, -1, -1, 0])

    def test_stack_forward_reshard(self):
        if False:
            print('Hello World!')
        shapes = [[16, 4, 4], [16, 4, 4]]
        specs = [['x', None, None], [None, None, 'x']]
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.stack, with_backward=False, axis=0)
        self.check_dim_mapping(outputs, [-1, 0, -1, -1])

    def test_slice(self):
        if False:
            i = 10
            return i + 15
        shapes = [64, 4, 4]
        specs = [None, None, 'x']
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.slice, with_backward=True, axes=[0, 1], starts=[1, 1], ends=[3, 3])

    def test_slice_reshard(self):
        if False:
            return 10
        shapes = [64, 4, 4]
        specs = [None, 'x', None]
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.slice, with_backward=True, axes=[0, 1], starts=[1, 1], ends=[3, 3])

    def test_stride_slice(self):
        if False:
            print('Hello World!')
        shapes = [64, 4, 4]
        specs = [None, None, 'x']
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.strided_slice, with_backward=True, axes=[0, 1], starts=[1, 3], ends=[3, 1], strides=[1, -1])

    def test_stride_slice_reshard(self):
        if False:
            while True:
                i = 10
        shapes = [64, 4, 4]
        specs = [None, 'x', None]
        (inputs, outputs) = self.runfunc_and_check(inputs_shape=shapes, inputs_specs=specs, op_func=paddle.strided_slice, with_backward=True, axes=[0, 1], starts=[1, 3], ends=[3, 1], strides=[1, -1])

    def run_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        if self._backend == 'cpu':
            paddle.set_device('cpu')
        elif self._backend == 'gpu':
            paddle.set_device('gpu:' + str(dist.get_rank()))
        else:
            raise ValueError('Only support cpu or gpu backend.')
        self.test_concat_forward()
        self.test_stack_forward()
        self.test_slice()
        self.test_stride_slice()
        if self._backend == 'gpu':
            self.test_concat_forward_reshard()
            self.test_slice_reshard()
            self.test_stride_slice_reshard()
            self.test_stack_forward_reshard()
if __name__ == '__main__':
    TestSplitAndConcatSemiAutoParallel().run_test_case()