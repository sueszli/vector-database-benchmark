import cupy
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import test_utils
import random
import os

def random_seed():
    if False:
        print('Hello World!')
    return int(random.random() * (1 << 32))
test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')
DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 32
SEED = random_seed()
NUM_WORKERS = 6

class PythonFunctionPipeline(Pipeline):

    def __init__(self, function, device, num_outputs=1):
        if False:
            while True:
                i = 10
        super(PythonFunctionPipeline, self).__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, seed=SEED)
        self.device = device
        self.reader = ops.readers.File(file_root=images_dir)
        self.decode = ops.decoders.Image(device='cpu', output_type=types.RGB)
        self.norm = ops.CropMirrorNormalize(std=255.0, mean=0.0, device=device, output_layout='HWC')
        self.func = ops.PythonFunction(device=device, function=function, num_outputs=num_outputs)

    def define_graph(self):
        if False:
            return 10
        (jpegs, labels) = self.reader()
        decoded = self.decode(jpegs)
        images = decoded if self.device == 'cpu' else decoded.gpu()
        normalized = self.norm(images)
        return self.func(normalized, normalized)

def validate_cpu_vs_gpu(gpu_fun, cpu_fun, num_outputs=1):
    if False:
        print('Hello World!')
    gpu_pipe = PythonFunctionPipeline(gpu_fun, 'gpu', num_outputs)
    cpu_pipe = PythonFunctionPipeline(cpu_fun, 'cpu', num_outputs)
    test_utils.compare_pipelines(gpu_pipe, cpu_pipe, BATCH_SIZE, ITERS)

def arrays_arithmetic(in1, in2):
    if False:
        i = 10
        return i + 15
    return (in1 + in2, in1 - in2 / 2.0)

def test_simple_arithm():
    if False:
        i = 10
        return i + 15
    validate_cpu_vs_gpu(arrays_arithmetic, arrays_arithmetic, num_outputs=2)
square_diff_kernel = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x*x - y*y', 'square_diff')

def square_diff(in1, in2):
    if False:
        print('Hello World!')
    return in1 * in1 - in2 * in2

def test_cupy_kernel():
    if False:
        while True:
            i = 10
    validate_cpu_vs_gpu(square_diff_kernel, square_diff)

def test_builtin_func():
    if False:
        for i in range(10):
            print('nop')
    validate_cpu_vs_gpu(cupy.logaddexp, np.logaddexp)