"""Benchmark for split and grad of split."""
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

def build_graph(device, input_shape, output_sizes, axis):
    if False:
        return 10
    'Build a graph containing a sequence of split operations.\n\n  Args:\n    device: string, the device to run on.\n    input_shape: shape of the input tensor.\n    output_sizes: size of each output along axis.\n    axis: axis to be split along.\n\n  Returns:\n    An array of tensors to run()\n  '
    with ops.device('/%s:0' % device):
        inp = array_ops.zeros(input_shape)
        outputs = []
        for _ in range(100):
            outputs.extend(array_ops.split(inp, output_sizes, axis))
        return control_flow_ops.group(*outputs)

class SplitBenchmark(test.Benchmark):
    """Benchmark split!"""

    def _run_graph(self, device, output_shape, variable, num_outputs, axis):
        if False:
            print('Hello World!')
        'Run the graph and print its execution time.\n\n    Args:\n      device: string, the device to run on.\n      output_shape: shape of each output tensors.\n      variable: whether or not the output shape should be fixed\n      num_outputs: the number of outputs to split the input into\n      axis: axis to be split\n\n    Returns:\n      The duration of the run in seconds.\n    '
        graph = ops.Graph()
        with graph.as_default():
            if not variable:
                if axis == 0:
                    input_shape = [output_shape[0] * num_outputs, output_shape[1]]
                    sizes = [output_shape[0] for _ in range(num_outputs)]
                else:
                    input_shape = [output_shape[0], output_shape[1] * num_outputs]
                    sizes = [output_shape[1] for _ in range(num_outputs)]
            else:
                sizes = np.random.randint(low=max(1, output_shape[axis] - 2), high=output_shape[axis] + 2, size=num_outputs)
                total_size = np.sum(sizes)
                if axis == 0:
                    input_shape = [total_size, output_shape[1]]
                else:
                    input_shape = [output_shape[0], total_size]
            outputs = build_graph(device, input_shape, sizes, axis)
        config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(optimizer_options=config_pb2.OptimizerOptions(opt_level=config_pb2.OptimizerOptions.L0)))
        with session_lib.Session(graph=graph, config=config) as session:
            logging.set_verbosity('info')
            variables.global_variables_initializer().run()
            bench = benchmark.TensorFlowBenchmark()
            bench.run_op_benchmark(session, outputs, mbs=input_shape[0] * input_shape[1] * 4 * 2 * 100 / 1000000.0, extras={'input_shape': input_shape, 'variable': variable, 'axis': axis})

    def benchmark_split(self):
        if False:
            print('Hello World!')
        print('Forward vs backward concat')
        shapes = [[2000, 8], [8, 2000], [100, 18], [1000, 18], [10000, 18], [100, 97], [1000, 97], [10000, 1], [1, 10000]]
        axis_ = [1]
        num_outputs = 100
        variable = [False, True]
        for shape in shapes:
            for axis in axis_:
                for v in variable:
                    self._run_graph('gpu', shape, v, num_outputs, axis)
if __name__ == '__main__':
    test.main()