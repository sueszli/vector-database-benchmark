from collections import namedtuple
import benchmark_utils
from benchmark_test_generator import _register_test
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
'Caffe2 performance microbenchmarks.\n\nThis module contains Caffe2-specific functionalities for performance\nmicrobenchmarks.\n'

class Caffe2BenchmarkBase:
    """This is a base class used to create Caffe2 operator benchmark"""
    tensor_index = 0
    test_index = 0

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.args = {}
        self.user_provided_name = None
        self._num_inputs_require_grads = 0
        self._pass_count = 0

    def _set_backward_test(self, is_backward):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _device_option(self, device):
        if False:
            print('Hello World!')
        'This method is used to set device option.'
        if device not in ['cuda', 'cpu']:
            raise ValueError('Missing attrs in configs')
        if 'cuda' in device:
            self.dev = core.DeviceOption(caffe2_pb2.CUDA, 0)
        else:
            self.dev = core.DeviceOption(caffe2_pb2.CPU)
        return self.dev

    def tensor(self, shapes, dtype='float32', device='cpu'):
        if False:
            return 10
        'A wapper function to create C2 tensor filled with random data.\n        The name/label of the tensor is returned and it is available\n        throughout the benchmark execution phase.\n        Args:\n            shapes: int or a sequence of ints to defining the shapes of the tensor\n            dtype: use the dtypes from numpy\n                (https://docs.scipy.org/doc/numpy/user/basics.types.html)\n        Return:\n            C2 tensor of dtype\n        '
        return self.feed_tensor(benchmark_utils.numpy_random(dtype, *shapes), device)

    def feed_tensor(self, tensor, device='cpu'):
        if False:
            return 10
        'Similar to tensor, but can supply any data compatible with FeedBlob'
        blob_name = 'blob_' + str(Caffe2BenchmarkBase.tensor_index)
        dev = self._device_option(device)
        with core.DeviceScope(dev):
            workspace.FeedBlob(blob_name, tensor)
        Caffe2BenchmarkBase.tensor_index += 1
        return blob_name

    def module_name(self):
        if False:
            for i in range(10):
                print('nop')
        'this is used to label the operator being benchmarked'
        if self.user_provided_name:
            return self.user_provided_name
        return self.__class__.__name__

    def set_module_name(self, name):
        if False:
            return 10
        self.user_provided_name = name

    def _value_to_str(self, value):
        if False:
            print('Hello World!')
        'if value is bool, we will convert it to 0 and 1'
        ret = value
        if type(value) == bool:
            ret = int(value)
        return str(ret)

    def test_name(self, name_type='long', **kargs):
        if False:
            for i in range(10):
                print('nop')
        'this is a globally unique name which can be used to\n        label a specific test\n        '
        if name_type == 'long':
            test_name_str = []
            for key in kargs:
                value = kargs[key]
                test_name_str.append(key + self._value_to_str(value))
            name = (self.module_name() + '_' + '_'.join(test_name_str)).replace(' ', '')
        elif name_type == 'short':
            name = '_'.join([self.module_name(), 'test', str(Caffe2BenchmarkBase.test_index)])
            Caffe2BenchmarkBase.test_index += 1
        return name

    def extract_inputs_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Caffe2OperatorTestCase:
    """This class includes all the information needed to benchmark an operator.
    op_bench: it's a user-defined class (child of Caffe2BenchmarkBase)
    which includes input and operator, .etc
    test_config: a namedtuple includes test_name, input_shape, tag, run_backward.
    When run_backward is false, the run_forward method will be executed, otherwise
    run_backward method will be executed.
    """

    def __init__(self, op_bench, test_config):
        if False:
            for i in range(10):
                print('nop')
        self.op_bench = op_bench
        self.test_config = test_config
        self.framework = 'Caffe2'

    def run_forward(self, num_runs, print_per_iter=False, cuda_sync=False):
        if False:
            return 10
        'Run the forward path of an operator in a loop'
        with core.DeviceScope(self.op_bench.dev):
            op = self.op_bench.forward()
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise ValueError(f'Unable to run operator test case: {self.test_name}')

    def run_backward(self, num_runs, print_per_iter=False):
        if False:
            for i in range(10):
                print('nop')
        'Run the backward path of an operator in a loop'
        with core.DeviceScope(self.op_bench.dev):
            op = self.op_bench.backward()
        if not workspace.RunOperatorMultiple(op, num_runs):
            raise ValueError(f'Unable to run operator gradient test case: {self.test_name}')

    def _print_per_iter(self):
        if False:
            return 10
        pass

def create_caffe2_op_test_case(op_bench, test_config):
    if False:
        i = 10
        return i + 15
    test_case = Caffe2OperatorTestCase(op_bench, test_config)
    test_config = test_case.test_config
    op = test_case.op_bench
    func_name = f'{op.module_name()}{test_case.framework}{str(test_config)}'
    return (func_name, test_case)
OpMeta = namedtuple('OpMeta', 'op_type num_inputs input_dims input_types                     output_dims num_outputs args device')

def generate_c2_test_from_ops(ops_metadata, bench_op, tags):
    if False:
        i = 10
        return i + 15
    "\n    This function is used to generate Caffe2 tests based on the metadata\n    of operators. The metadata includes seven fields which are 1) op_type:\n    the name of the operator. 2) num_inputs: the number of input blobs.\n    3) input_dims: a dictionary which includes the shapes of the input blobs.\n    4) input_types: a list which includes the types of input blobs. 5)\n    output_dims: a dictionary which includes the shapes of output blobs.\n    6) num_oupts: the number of output blobs. 7) args: a dictionary which\n    includes the args for th operator.\n    Here is an example to show the metadata for the WeighedSum operator\n    op_type : WeightedSum\n    num_inputs: 4\n    input_dims: {'0': [256], '1': [1], '2': [256], '3': [1]}\n    input_types: ['float', 'float', 'float', 'float']\n    output_dims:  {'0': [256]}\n    num_outputs: 4\n    args: {}\n    TODO(mingzhe0908): introduce device and add it to the benchmark name\n    "
    for op_metadata in ops_metadata:
        tmp_attrs = OpMeta(op_metadata.op_type, op_metadata.num_inputs, op_metadata.input_dims, op_metadata.input_types, op_metadata.output_dims, op_metadata.num_outputs, op_metadata.args, op_metadata.device)
        test_attrs = tmp_attrs._asdict()
        op = bench_op()
        op.init(**test_attrs)
        test_name = op.test_name('short')
        input_config = 'Shapes: {}, Type: {}, Args: {}'.format(op_metadata.input_dims, op_metadata.input_types, str(op_metadata.args))
        test_config = TestConfig(test_name, input_config, tags, run_backward=False)
        if op is not None:
            create_caffe2_op_test_case(op, test_config)

def generate_c2_test(configs, c2_bench_op):
    if False:
        for i in range(10):
            print('nop')
    'This function creates Caffe2 op test based on the given operator'
    return _register_test(configs, c2_bench_op, create_caffe2_op_test_case, False)

def generate_c2_gradient_test(configs, c2_bench_op):
    if False:
        i = 10
        return i + 15
    'This function creates Caffe2 op test based on the given operator'
    return _register_test(configs, c2_bench_op, create_caffe2_op_test_case, True)