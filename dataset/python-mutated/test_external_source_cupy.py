from nose.plugins.attrib import attr
from test_external_source_impl import *
from test_external_source_impl import use_cupy
from test_utils import check_output, check_output_pattern
import nvidia.dali
from nvidia.dali import Pipeline, pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.tensors import TensorGPU
import numpy as np
use_cupy()
import cupy as cp
assert nvidia.dali.types._is_cupy_array(cp.array([1, 2, 3])), 'CuPy array not recognized'

def test_external_source_with_iter_cupy_stream():
    if False:
        i = 10
        return i + 15
    with cp.cuda.Stream(non_blocking=True):
        for attempt in range(10):
            pipe = Pipeline(1, 3, 0)

            def get_data(i):
                if False:
                    return 10
                return [cp.array([attempt * 100 + i * 10 + 1.5], dtype=cp.float32)]
            pipe.set_outputs(fn.external_source(get_data))
            pipe.build()
            for i in range(10):
                check_output(pipe.run(), [np.array([attempt * 100 + i * 10 + 1.5], dtype=np.float32)])

def test_external_source_mixed_contiguous():
    if False:
        while True:
            i = 10
    batch_size = 2
    iterations = 4

    def generator(i):
        if False:
            print('Hello World!')
        if i % 2:
            return cp.array([[100 + i * 10 + 1.5]] * batch_size, dtype=cp.float32)
        else:
            return batch_size * [cp.array([100 + i * 10 + 1.5], dtype=cp.float32)]
    pipe = Pipeline(batch_size, 3, 0)
    pipe.set_outputs(fn.external_source(device='gpu', source=generator, no_copy=True))
    pipe.build()
    pattern = 'ExternalSource operator should not mix contiguous and noncontiguous inputs. In such a case the internal memory used to gather data in a contiguous chunk of memory would be trashed.'
    with check_output_pattern(pattern):
        for _ in range(iterations):
            pipe.run()

def _test_cross_device(src, dst, use_dali_tensor=False):
    if False:
        for i in range(10):
            print('nop')
    import nvidia.dali.fn as fn
    import numpy as np
    pipe = Pipeline(1, 3, dst)
    iter = 0

    def get_data():
        if False:
            for i in range(10):
                print('nop')
        nonlocal iter
        with cp.cuda.Device(src):
            data = cp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=cp.float32) + iter
            iter += 1
        if use_dali_tensor:
            return TensorGPU(data.toDlpack())
        return data
    with pipe:
        pipe.set_outputs(fn.external_source(get_data, batch=False, device='gpu'))
    pipe.build()
    for i in range(10):
        (out,) = pipe.run()
        assert np.array_equal(np.array(out[0].as_cpu()), np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) + i)

@attr('multigpu')
def test_cross_device():
    if False:
        return 10
    if cp.cuda.runtime.getDeviceCount() > 1:
        for src in [0, 1]:
            for dst in [0, 1]:
                for use_dali_tensor in [True, False]:
                    yield (_test_cross_device, src, dst, use_dali_tensor)

def _test_memory_consumption(device, test_case):
    if False:
        print('Hello World!')
    batch_size = 32
    num_iters = 128
    if device == 'cpu':
        import numpy as np
        fw = np
    else:
        fw = cp

    def no_copy_sample():
        if False:
            while True:
                i = 10
        batch = [fw.full((1024, 1024, 4), i, dtype=fw.int32) for i in range(batch_size)]

        def cb(sample_info):
            if False:
                while True:
                    i = 10
            return batch[sample_info.idx_in_batch]
        return cb

    def copy_sample():
        if False:
            i = 10
            return i + 15

        def cb(sample_info):
            if False:
                print('Hello World!')
            return fw.full((1024, 1024, 4), sample_info.idx_in_batch, dtype=fw.int32)
        return cb

    def copy_batch():
        if False:
            print('Hello World!')

        def cb():
            if False:
                while True:
                    i = 10
            return fw.full((batch_size, 1024, 1024, 4), 42, dtype=fw.int32)
        return cb
    cases = {'no_copy_sample': (no_copy_sample, True, False), 'copy_sample': (copy_sample, False, False), 'copy_batch': (copy_batch, False, True)}
    (cb, no_copy, batch_mode) = cases[test_case]

    @pipeline_def
    def pipeline():
        if False:
            while True:
                i = 10
        return fn.external_source(source=cb(), device=device, batch=batch_mode, no_copy=no_copy)
    pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    pipe.build()
    for _ in range(num_iters):
        pipe.run()

def test_memory_consumption():
    if False:
        i = 10
        return i + 15
    for device in ['cpu', 'gpu']:
        for test_case in ['no_copy_sample', 'copy_sample', 'copy_batch']:
            yield (_test_memory_consumption, device, test_case)