from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np
from functools import partial
from test_utils import compare_pipelines
from test_utils import RandomDataIterator
import librosa as librosa
from nose_utils import assert_raises

class MFCCPipeline(Pipeline):

    def __init__(self, device, batch_size, iterator, axis=0, dct_type=2, lifter=1.0, n_mfcc=20, norm=None, num_threads=1, device_id=0):
        if False:
            i = 10
            return i + 15
        super(MFCCPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.mfcc = ops.MFCC(device=self.device, axis=axis, dct_type=dct_type, lifter=lifter, n_mfcc=n_mfcc, normalize=norm)

    def define_graph(self):
        if False:
            return 10
        self.data = self.inputs()
        out = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.mfcc(out)
        return out

    def iter_setup(self):
        if False:
            i = 10
            return i + 15
        data = self.iterator.next()
        self.feed_input(self.data, data)

def mfcc_func(axis, dct_type, lifter, n_mfcc, norm, input_data):
    if False:
        return 10
    if axis == 1:
        input_data = np.transpose(input_data)
    in_shape = input_data.shape
    assert len(in_shape) == 2
    norm_str = 'ortho' if norm else None
    out = librosa.feature.mfcc(S=input_data, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm_str, lifter=lifter)
    if not norm:
        out = out / 2
    if axis == 1:
        out = np.transpose(out)
    return out

class MFCCPythonPipeline(Pipeline):

    def __init__(self, device, batch_size, iterator, axis=0, dct_type=2, lifter=1.0, n_mfcc=20, norm=None, num_threads=1, device_id=0, func=mfcc_func):
        if False:
            print('Hello World!')
        super(MFCCPythonPipeline, self).__init__(batch_size, num_threads, device_id, seed=12345, exec_async=False, exec_pipelined=False)
        self.device = 'cpu'
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        function = partial(func, axis, dct_type, lifter, n_mfcc, norm)
        self.mfcc = ops.PythonFunction(function=function)

    def define_graph(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = self.inputs()
        out = self.mfcc(self.data)
        return out

    def iter_setup(self):
        if False:
            return 10
        data = self.iterator.next()
        self.feed_input(self.data, data)

def check_operator_mfcc_vs_python(device, batch_size, input_shape, axis, dct_type, lifter, n_mfcc, norm):
    if False:
        print('Hello World!')
    eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    eii2 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
    compare_pipelines(MFCCPipeline(device, batch_size, iter(eii1), axis=axis, dct_type=dct_type, lifter=lifter, n_mfcc=n_mfcc, norm=norm), MFCCPythonPipeline(device, batch_size, iter(eii2), axis=axis, dct_type=dct_type, lifter=lifter, n_mfcc=n_mfcc, norm=norm), batch_size=batch_size, N_iterations=3, eps=0.001)

def test_operator_mfcc_vs_python():
    if False:
        print('Hello World!')
    for device in ['cpu', 'gpu']:
        for batch_size in [1, 3]:
            for dct_type in [1, 2, 3]:
                for norm in [False] if dct_type == 1 else [True, False]:
                    for (axis, n_mfcc, lifter, shape) in [(0, 17, 0.0, (17, 1)), (1, 80, 2.0, (513, 100)), (1, 90, 0.0, (513, 100)), (1, 20, 202.0, (513, 100))]:
                        yield (check_operator_mfcc_vs_python, device, batch_size, shape, axis, dct_type, lifter, n_mfcc, norm)

def check_operator_mfcc_wrong_args(device, batch_size, input_shape, axis, dct_type, lifter, n_mfcc, norm, msg):
    if False:
        i = 10
        return i + 15
    with assert_raises(RuntimeError, regex=msg):
        eii1 = RandomDataIterator(batch_size, shape=input_shape, dtype=np.float32)
        pipe = MFCCPipeline(device, batch_size, iter(eii1), axis=axis, dct_type=dct_type, lifter=lifter, n_mfcc=n_mfcc, norm=norm)
        pipe.build()
        pipe.run()

def test_operator_mfcc_wrong_args():
    if False:
        i = 10
        return i + 15
    batch_size = 3
    for device in ['cpu', 'gpu']:
        for (dct_type, norm, axis, n_mfcc, lifter, shape, msg) in [(1, True, 0, 20, 0.0, (100, 100), 'Ortho-normalization is not supported for DCT type I'), (2, False, -1, 20, 0.0, (100, 100), 'Provided axis cannot be negative'), (2, False, 2, 20, 0.0, (100, 100), 'Axis [\\d]+ is out of bounds \\[[\\d]+,[\\d]+\\)'), (10, False, 0, 20, 0.0, (100, 100), 'Unsupported DCT type: 10. Supported types are: 1, 2, 3, 4')]:
            yield (check_operator_mfcc_wrong_args, device, batch_size, shape, axis, dct_type, lifter, n_mfcc, norm, msg)