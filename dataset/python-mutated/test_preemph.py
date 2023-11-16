from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import numpy as np
from functools import partial
from test_utils import compare_pipelines
from test_utils import RandomlyShapedDataIterator
SEED = 12345

def preemph_func(border, coeff, signal):
    if False:
        while True:
            i = 10
    in_shape = signal.shape
    assert len(in_shape) == 1
    out = np.copy(signal)
    if border == 'clamp':
        out[0] -= coeff * signal[0]
    elif border == 'reflect':
        out[0] -= coeff * signal[1]
    out[1:] -= coeff * signal[0:in_shape[0] - 1]
    return out

class PreemphasisPipeline(Pipeline):

    def __init__(self, device, batch_size, iterator, border='clamp', preemph_coeff=0.97, per_sample_coeff=False, num_threads=4, device_id=0):
        if False:
            for i in range(10):
                print('nop')
        super(PreemphasisPipeline, self).__init__(batch_size, num_threads, device_id, seed=SEED)
        self.device = device
        self.iterator = iterator
        self.per_sample_coeff = per_sample_coeff
        self.uniform = ops.random.Uniform(range=(0.5, 0.97), seed=1234)
        if self.per_sample_coeff:
            self.preemph = ops.PreemphasisFilter(device=device, border=border)
        else:
            self.preemph = ops.PreemphasisFilter(device=device, border=border, preemph_coeff=preemph_coeff)

    def define_graph(self):
        if False:
            return 10
        data = fn.external_source(lambda : next(self.iterator))
        out = data.gpu() if self.device == 'gpu' else data
        if self.per_sample_coeff:
            preemph_coeff_arg = self.uniform()
            return self.preemph(out, preemph_coeff=preemph_coeff_arg)
        else:
            return self.preemph(out)

class PreemphasisPythonPipeline(Pipeline):

    def __init__(self, device, batch_size, iterator, border='clamp', preemph_coeff=0.97, per_sample_coeff=False, num_threads=4, device_id=0):
        if False:
            print('Hello World!')
        super(PreemphasisPythonPipeline, self).__init__(batch_size, num_threads, device_id, seed=SEED, exec_async=False, exec_pipelined=False)
        self.device = 'cpu'
        self.iterator = iterator
        self.per_sample_coeff = per_sample_coeff
        self.uniform = ops.random.Uniform(range=(0.5, 0.97), seed=1234)
        if self.per_sample_coeff:
            function = partial(preemph_func, border)
        else:
            function = partial(preemph_func, border, preemph_coeff)
        self.preemph = ops.PythonFunction(function=function)

    def define_graph(self):
        if False:
            i = 10
            return i + 15
        data = fn.external_source(lambda : next(self.iterator))
        if self.per_sample_coeff:
            coef = self.uniform()
            return self.preemph(coef, data)
        else:
            return self.preemph(data)

def check_preemphasis_operator(device, batch_size, border, preemph_coeff, per_sample_coeff):
    if False:
        return 10
    eii1 = RandomlyShapedDataIterator(batch_size, min_shape=(100,), max_shape=(10000,), dtype=np.float32)
    eii2 = RandomlyShapedDataIterator(batch_size, min_shape=(100,), max_shape=(10000,), dtype=np.float32)
    compare_pipelines(PreemphasisPipeline(device, batch_size, iter(eii1), border=border, preemph_coeff=preemph_coeff, per_sample_coeff=per_sample_coeff), PreemphasisPythonPipeline(device, batch_size, iter(eii2), border=border, preemph_coeff=preemph_coeff, per_sample_coeff=per_sample_coeff), batch_size=batch_size, N_iterations=3)

def test_preemphasis_operator():
    if False:
        for i in range(10):
            print('nop')
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 3, 128]:
            for border in ['zero', 'clamp', 'reflect']:
                for (coef, per_sample_coeff) in [(0.97, False), (0.0, False), (None, True)]:
                    yield (check_preemphasis_operator, device, batch_size, border, coef, per_sample_coeff)