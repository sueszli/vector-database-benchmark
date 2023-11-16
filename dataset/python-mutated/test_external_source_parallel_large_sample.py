import numpy as np
from nose.tools import with_setup
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from test_external_source_parallel_utils import setup_function, teardown_function, capture_processes

def large_sample_cb(sample_info):
    if False:
        i = 10
        return i + 15
    return np.full((512, 1024, 1024), sample_info.idx_in_epoch, dtype=np.int32)

@with_setup(setup_function, teardown_function)
def _test_large_sample(start_method):
    if False:
        return 10
    batch_size = 2

    @pipeline_def
    def create_pipeline():
        if False:
            return 10
        large = fn.external_source(large_sample_cb, batch=False, parallel=True, prefetch_queue_depth=1)
        reduced = fn.reductions.sum(large, axes=(1, 2))
        return reduced
    pipe = create_pipeline(batch_size=batch_size, py_num_workers=2, py_start_method=start_method, prefetch_queue_depth=1, num_threads=2, device_id=0)
    pipe.build()
    capture_processes(pipe._py_pool)
    for batch_idx in range(8):
        (out,) = pipe.run()
        for idx_in_batch in range(batch_size):
            idx_in_epoch = batch_size * batch_idx + idx_in_batch
            expected_val = idx_in_epoch * 1024 * 1024
            a = np.array(out[idx_in_batch])
            assert a.shape == (512,), 'Expected shape (512,) but got {}'.format(a.shape)
            for val in a.flat:
                assert val == expected_val, f'Unexpected value in batch: got {val}, expected {expected_val}, for batch {batch_idx}, sample {idx_in_batch}'

def test_large_sample():
    if False:
        for i in range(10):
            print('nop')
    for start_method in ('fork', 'spawn'):
        yield (_test_large_sample, start_method)