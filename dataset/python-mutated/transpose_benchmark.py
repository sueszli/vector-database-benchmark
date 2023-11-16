"""Benchmark for Transpose op."""
import time
import numpy as np
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def build_graph(device, input_shape, perm, datatype, num_iters):
    if False:
        return 10
    "builds a graph containing a sequence of conv2d operations.\n\n  Args:\n    device: String, the device to run on.\n    input_shape: Shape of the input tensor.\n    perm: A list of ints with the same length as input tensor's dimension.\n    datatype: numpy data type of the input tensor.\n    num_iters: number of iterations to run transpose.\n\n  Returns:\n    An array of tensors to run()\n  "
    with ops.device('/%s:0' % device):
        total_size = np.prod(input_shape)
        inp = np.arange(1, total_size + 1, dtype=datatype).reshape(input_shape)
        t = constant_op.constant(inp, shape=input_shape)
        outputs = []
        transpose_op = array_ops.transpose(t, perm)
        outputs.append(transpose_op)
        for _ in range(1, num_iters):
            with ops.control_dependencies([transpose_op]):
                transpose_op = array_ops.transpose(t, perm)
                outputs.append(transpose_op)
        return control_flow_ops.group(*outputs)

class TransposeBenchmark(test.Benchmark):
    """Benchmark transpose!"""

    def _run_graph(self, device, input_shape, perm, num_iters, datatype):
        if False:
            for i in range(10):
                print('nop')
        "runs the graph and print its execution time.\n\n    Args:\n      device: String, the device to run on.\n      input_shape: Shape of the input tensor.\n      perm: A list of ints with the same length as input tensor's dimension.\n      num_iters: Number of iterations to run the benchmark.\n      datatype: numpy data type of the input tensor.\n\n    Returns:\n      The duration of the run in seconds.\n    "
        graph = ops.Graph()
        with graph.as_default():
            outputs = build_graph(device, input_shape, perm, datatype, num_iters)
            with session_lib.Session(graph=graph) as session:
                variables.global_variables_initializer().run()
                session.run(outputs)
                start_time = time.time()
                session.run(outputs)
                duration = (time.time() - start_time) / num_iters
                throughput = np.prod(np.array(input_shape)) * datatype().itemsize * 2 / duration / 1000000000.0
                print('%s %s inputshape:%s perm:%s %d %.6fsec, %.4fGB/s.' % (device, str(datatype), str(input_shape).replace(' ', ''), str(perm).replace(' ', ''), num_iters, duration, throughput))
        name_template = 'transpose_{device}_{dtype}_input_shape_{inputshape}_perm_{perm}'
        self.report_benchmark(name=name_template.format(device=device, dtype=str(datatype).replace(' ', ''), inputshape=str(input_shape).replace(' ', ''), perm=str(perm).replace(' ', '')).replace(' ', ''), iters=num_iters, wall_time=duration)
        return duration

    def benchmark_transpose(self):
        if False:
            for i in range(10):
                print('nop')
        print('transpose benchmark:')
        datatypes = [np.complex128, np.float64, np.float32, np.float16, np.int8]
        small_shapes = [[2, 20, 20, 20, 16], [2, 16, 20, 20, 20]] * 2
        small_shapes += [[2, 100, 100, 16], [2, 16, 100, 100]] * 2
        small_shapes += [[2, 5000, 16], [2, 16, 5000]] * 2
        small_perms = [[0, 4, 1, 2, 3], [0, 2, 3, 4, 1]] + [[4, 1, 2, 3, 0]] * 2
        small_perms += [[0, 3, 1, 2], [0, 2, 3, 1]] + [[3, 1, 2, 0]] * 2
        small_perms += [[0, 2, 1]] * 2 + [[2, 1, 0]] * 2
        large_shapes = [[2, 40, 40, 40, 32], [2, 40, 40, 40, 64]] * 2 + [[2, 300, 300, 32], [2, 300, 300, 64]] * 2 + [[2, 100000, 32], [2, 100000, 64]] * 2
        large_perms = [[0, 4, 1, 2, 3], [0, 2, 3, 4, 1]] + [[4, 1, 2, 3, 0]] * 2 + [[0, 3, 1, 2], [0, 2, 3, 1]] + [[3, 1, 2, 0]] * 2 + [[0, 2, 1]] * 2 + [[2, 1, 0]] * 2
        num_iters = 40
        for datatype in datatypes:
            for (ishape, perm) in zip(small_shapes, small_perms):
                self._run_graph('gpu', ishape, perm, num_iters, datatype)
            if datatype is not np.complex128:
                if datatype is not np.float16:
                    for (ishape, perm) in zip(large_shapes, large_perms):
                        self._run_graph('gpu', ishape, perm, num_iters, datatype)
        small_dim_large_shapes = [[2, 10000, 3], [2, 3, 10000], [2, 10000, 8], [2, 8, 10000]]
        small_dim_small_shapes = [[2, 5000, 3], [2, 3, 5000], [2, 5000, 8], [2, 8, 5000]]
        small_dim_perms = [[0, 2, 1]] * 4
        num_iters = 320
        small_dim_large_shape_datatypes = [np.float64, np.float32, np.int8]
        for datatype in small_dim_large_shape_datatypes:
            for (ishape, perm) in zip(small_dim_large_shapes, small_dim_perms):
                self._run_graph('gpu', ishape, perm, num_iters, datatype)
        small_dim_small_shape_datatypes = [np.complex128, np.float16]
        for datatype in small_dim_small_shape_datatypes:
            for (ishape, perm) in zip(small_dim_small_shapes, small_dim_perms):
                self._run_graph('gpu', ishape, perm, num_iters, datatype)
if __name__ == '__main__':
    test.main()