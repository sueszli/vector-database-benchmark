import numpy as np
import nvidia.dali as dali
from nose.tools import with_setup
from nvidia.dali.types import SampleInfo, BatchInfo
import test_external_source_parallel_utils as utils
from nose_utils import raises

def no_arg_fun():
    if False:
        i = 10
        return i + 15
    pass

def multi_arg_fun(a, b, c):
    if False:
        return 10
    pass

class Iterable:

    def __init__(self, batch_size=4, shape=(10, 10), epoch_size=None, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        self.count = 0
        self.batch_size = batch_size
        self.shape = shape
        self.epoch_size = epoch_size
        self.dtype = dtype or np.int16

    def __iter__(self):
        if False:
            print('Hello World!')
        self.count = 0
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.epoch_size is not None and self.epoch_size <= self.count:
            raise StopIteration
        batch = [np.full(self.shape, self.count + i, dtype=self.dtype) for i in range(self.batch_size)]
        self.count += 1
        return batch

class FaultyResetIterable(Iterable):

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

class SampleCallbackBatched:

    def __init__(self, sample_cb, batch_size, batch_info):
        if False:
            for i in range(10):
                print('nop')
        self.sample_cb = sample_cb
        self.batch_size = batch_size
        self.batch_info = batch_info

    def __call__(self, batch_info):
        if False:
            print('Hello World!')
        if not self.batch_info:
            batch_i = batch_info
            epoch_idx = 0
        else:
            batch_i = batch_info.iteration
            epoch_idx = batch_info.epoch_idx
        epoch_offset = batch_i * self.batch_size
        return [self.sample_cb(SampleInfo(epoch_offset + i, i, batch_i, epoch_idx)) for i in range(self.batch_size)]

class SampleCallbackIterator:

    def __init__(self, sample_cb, batch_size, batch_info):
        if False:
            return 10
        self.iters = 0
        self.batch_info = batch_info
        self.epoch_idx = 0
        self.batched = SampleCallbackBatched(sample_cb, batch_size, batch_info)

    def __iter__(self):
        if False:
            while True:
                i = 10
        if self.iters > 0:
            self.epoch_idx += 1
        self.iters = 0
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        batch_info = BatchInfo(self.iters, self.epoch_idx) if self.batch_info else self.iters
        batch = self.batched(batch_info)
        self.iters += 1
        return batch

def generator_fun():
    if False:
        for i in range(10):
            print('nop')
    while True:
        yield [np.full((2, 2), 42)]

def check_source_build(source):
    if False:
        i = 10
        return i + 15
    pipe = utils.create_pipe(source, 'cpu', 10, py_num_workers=4, py_start_method='spawn', parallel=True)
    pipe.build()

def test_wrong_source():
    if False:
        i = 10
        return i + 15
    callable_msg = 'Callable passed to External Source in parallel mode (when `parallel=True`) must accept exactly one argument*. Got {} instead.'
    batch_required_msg = 'Parallel external source with {} must be run in a batch mode'
    disallowed_sources = [(no_arg_fun, (TypeError, callable_msg.format('a callable that does not accept arguments'))), (multi_arg_fun, (TypeError, 'External source callback must be a callable with 0 or 1 argument')), (Iterable(), (TypeError, batch_required_msg.format('an iterable'))), (generator_fun, (TypeError, batch_required_msg.format('a generator function'))), (generator_fun(), (TypeError, batch_required_msg.format('an iterable')))]
    for (source, (error_type, error_msg)) in disallowed_sources:
        yield (raises(error_type, error_msg)(check_source_build), source)

@with_setup(utils.setup_function, utils.teardown_function)
def test_parallel_fork_cpu_only():
    if False:
        print('Hello World!')
    pipeline_pairs = 4
    batch_size = 10
    iters = 40
    callback = utils.ExtCallback((4, 5), iters * batch_size, np.int32)
    parallel_pipes = [(utils.create_pipe(callback, 'cpu', batch_size, py_num_workers=4, py_start_method='fork', parallel=True, device_id=None), utils.create_pipe(callback, 'cpu', batch_size, py_num_workers=4, py_start_method='fork', parallel=True, device_id=None)) for i in range(pipeline_pairs)]
    for (pipe0, pipe1) in parallel_pipes:
        pipe0.build()
        pipe1.build()
        utils.capture_processes(pipe0._py_pool)
        utils.capture_processes(pipe1._py_pool)
        utils.compare_pipelines(pipe0, pipe1, batch_size, iters)

def test_parallel_no_workers():
    if False:
        i = 10
        return i + 15
    batch_size = 10
    iters = 4
    callback = utils.ExtCallback((4, 5), iters * batch_size, np.int32)
    parallel_pipe = utils.create_pipe(callback, 'cpu', batch_size, py_num_workers=0, py_start_method='spawn', parallel=True, device_id=None)
    parallel_pipe.build()
    assert parallel_pipe._py_pool is None
    assert not parallel_pipe._py_pool_started

@with_setup(utils.setup_function, utils.teardown_function)
def test_parallel_fork():
    if False:
        return 10
    epoch_size = 250
    callback = utils.ExtCallback((4, 5), epoch_size, np.int32)
    pipes = [(utils.create_pipe(callback, 'cpu', batch_size, py_num_workers=num_workers, py_start_method='fork', parallel=True), utils.create_pipe(callback, 'cpu', batch_size, parallel=False), dtype, batch_size) for dtype in [np.float32, np.int16] for num_workers in [1, 3, 4] for batch_size in [1, 16, 150, 250]]
    pipes.append((utils.create_pipe(Iterable(32, (4, 5), dtype=np.int16), 'cpu', 32, py_num_workers=1, py_start_method='fork', parallel=True, batch=True), utils.create_pipe(Iterable(32, (4, 5), dtype=np.int16), 'cpu', 32, parallel=False, batch=True), np.int16, 32))
    for (parallel_pipe, _, _, _) in pipes:
        parallel_pipe.start_py_workers()
    for (parallel_pipe, pipe, dtype, batch_size) in pipes:
        yield (utils.check_callback, parallel_pipe, pipe, epoch_size, batch_size, dtype)
        parallel_pipe._py_pool.close()
    parallel_pipe = utils.create_pipe(callback, 'cpu', 16, py_num_workers=4, py_start_method='fork', parallel=True)
    yield (raises(RuntimeError, 'Cannot fork a process when the CUDA has been initialized in the process.')(utils.build_and_run_pipeline), parallel_pipe, 1)

def test_dtypes():
    if False:
        return 10
    yield from utils.check_spawn_with_callback(utils.ExtCallback)

def test_random_data():
    if False:
        while True:
            i = 10
    yield from utils.check_spawn_with_callback(utils.ExtCallback, shapes=[(100, 40, 3), (8, 64, 64, 3)], random_data=True)

def test_randomly_shaped_data():
    if False:
        return 10
    yield from utils.check_spawn_with_callback(utils.ExtCallback, shapes=[(100, 40, 3), (8, 64, 64, 3)], random_data=True, random_shape=True)

def test_num_outputs():
    if False:
        i = 10
        return i + 15
    yield from utils.check_spawn_with_callback(utils.ExtCallbackMultipleOutputs, utils.ExtCallbackMultipleOutputs, num_outputs=2, dtypes=[np.uint8, np.float])

def test_tensor_cpu():
    if False:
        while True:
            i = 10
    yield from utils.check_spawn_with_callback(utils.ExtCallbackTensorCPU)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_exception_propagation(callback, batch_size, num_workers, expected):
    if False:
        while True:
            i = 10
    pipe = utils.create_pipe(callback, 'cpu', batch_size, py_num_workers=num_workers, py_start_method='spawn', parallel=True)
    raises(expected)(utils.build_and_run_pipeline)(pipe, None)

def test_exception_propagation():
    if False:
        i = 10
        return i + 15
    for (raised, expected) in [(StopIteration, StopIteration), (utils.CustomException, Exception)]:
        callback = utils.ExtCallback((4, 4), 250, np.int32, exception_class=raised)
        for num_workers in [1, 4]:
            for batch_size in [1, 15, 150]:
                yield (_test_exception_propagation, callback, batch_size, num_workers, expected)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_stop_iteration_resume(callback, batch_size, layout, num_workers):
    if False:
        for i in range(10):
            print('nop')
    pipe = utils.create_pipe(callback, 'cpu', batch_size, layout=layout, py_num_workers=num_workers, py_start_method='spawn', parallel=True)
    utils.check_stop_iteration_resume(pipe, batch_size, layout)

def test_stop_iteration_resume():
    if False:
        i = 10
        return i + 15
    callback = utils.ExtCallback((4, 4), 250, 'int32')
    layout = 'XY'
    for num_workers in [1, 4]:
        for batch_size in [1, 15, 150]:
            yield (_test_stop_iteration_resume, callback, batch_size, layout, num_workers)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_layout(callback, batch_size, layout, num_workers):
    if False:
        i = 10
        return i + 15
    pipe = utils.create_pipe(callback, 'cpu', batch_size, layout=layout, py_num_workers=num_workers, py_start_method='spawn', parallel=True)
    utils.check_layout(pipe, layout)

def test_layout():
    if False:
        i = 10
        return i + 15
    for (layout, dims) in zip(['X', 'XY', 'XYZ'], ((4,), (4, 4), (4, 4, 4))):
        callback = utils.ExtCallback(dims, 1024, 'int32')
        for num_workers in [1, 4]:
            for batch_size in [1, 256, 600]:
                yield (_test_layout, callback, batch_size, layout, num_workers)

class ext_cb:

    def __init__(self, name, shape):
        if False:
            while True:
                i = 10
        self.name = name
        self.shape = shape

    def __call__(self, sinfo):
        if False:
            while True:
                i = 10
        return np.full(self.shape, sinfo.idx_in_epoch, dtype=np.int32)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_vs_non_parallel(batch_size, cb_parallel, cb_seq, batch, py_num_workers):
    if False:
        while True:
            i = 10
    pipe = dali.Pipeline(batch_size=batch_size, device_id=None, num_threads=5, py_num_workers=py_num_workers, py_start_method='spawn')
    with pipe:
        ext_seq = dali.fn.external_source(cb_parallel, batch=batch, parallel=False)
        ext_par = dali.fn.external_source(cb_seq, batch=batch, parallel=True)
        pipe.set_outputs(ext_seq, ext_par)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    for i in range(10):
        (seq, par) = pipe.run()
        for j in range(batch_size):
            s = seq.at(j)
            p = par.at(j)
            assert np.array_equal(s, p)

def test_vs_non_parallel():
    if False:
        print('Hello World!')
    for shape in [[], [10], [100, 100, 100]]:
        for (batch_size, cb_parallel, cb_seq, batch, py_num_workers) in [(50, ext_cb('cb 1', shape), ext_cb('cb 2', shape), False, 14), (50, Iterable(50, shape), Iterable(50, shape), True, 1)]:
            yield (_test_vs_non_parallel, batch_size, cb_parallel, cb_seq, batch, py_num_workers)

def generator_shape_empty():
    if False:
        i = 10
        return i + 15
    count = 0
    while True:
        yield [np.full([], count + i) for i in range(50)]

def generator_shape_10():
    if False:
        return 10
    count = 0
    while True:
        yield [np.full([10], count + i) for i in range(50)]

def generator_shape_100x3():
    if False:
        for i in range(10):
            print('nop')
    count = 0
    while True:
        yield [np.full([10, 10, 10], count + i) for i in range(50)]

def test_generator_vs_non_parallel():
    if False:
        while True:
            i = 10
    for cb in [generator_shape_empty, generator_shape_10, generator_shape_100x3]:
        yield (_test_vs_non_parallel, 50, cb, cb, True, 1)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_cycle_raise(cb, is_gen_fun, batch_size, epoch_size, reader_queue_size):
    if False:
        return 10
    pipe = utils.create_pipe(cb, 'cpu', batch_size=batch_size, py_num_workers=1, py_start_method='spawn', parallel=True, device_id=None, batch=True, num_threads=5, cycle='raise', reader_queue_depth=reader_queue_size)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    if is_gen_fun:
        refer_iter = cb()
    else:
        refer_iter = cb
    for _ in range(3):
        i = 0
        while True:
            try:
                (batch,) = pipe.run()
                expected_batch = next(refer_iter)
                assert len(batch) == len(expected_batch), f'Batch length mismatch: expected {len(expected_batch)}, got {len(batch)}'
                for (sample, expected_sample) in zip(batch, expected_batch):
                    np.testing.assert_equal(sample, expected_sample)
                i += 1
            except StopIteration:
                pipe.reset()
                if is_gen_fun:
                    refer_iter = cb()
                else:
                    refer_iter = iter(cb)
                assert i == epoch_size, f'Number of iterations mismatch: expected {epoch_size}, got {i}'
                break

def generator_epoch_size_1():
    if False:
        while True:
            i = 10
    yield [np.full((4, 5), i) for i in range(20)]

def generator_epoch_size_4():
    if False:
        return 10
    for j in range(4):
        yield [np.full((4, 5), j + i) for i in range(20)]

def test_cycle_raise():
    if False:
        while True:
            i = 10
    batch_size = 20
    for (epoch_size, cb, is_gen_fun) in [(1, Iterable(batch_size, (4, 5), epoch_size=1), False), (4, Iterable(batch_size, (4, 5), epoch_size=4), False), (1, generator_epoch_size_1, True), (4, generator_epoch_size_4, True)]:
        for reader_queue_size in (1, 2, 6):
            yield (_test_cycle_raise, cb, is_gen_fun, batch_size, epoch_size, reader_queue_size)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_cycle_quiet(cb, is_gen_fun, batch_size, epoch_size, reader_queue_size):
    if False:
        print('Hello World!')
    pipe = utils.create_pipe(cb, 'cpu', batch_size=batch_size, py_num_workers=1, py_start_method='spawn', parallel=True, device_id=None, batch=True, num_threads=5, cycle='quiet', reader_queue_depth=reader_queue_size)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    refer_iter = cb
    for i in range(3 * epoch_size + 1):
        if i % epoch_size == 0:
            if is_gen_fun:
                refer_iter = cb()
            else:
                refer_iter = iter(cb)
        (batch,) = pipe.run()
        expected_batch = next(refer_iter)
        assert len(batch) == len(expected_batch), f'Batch length mismatch: expected {len(expected_batch)}, got {len(batch)}'
        for (sample, expected_sample) in zip(batch, expected_batch):
            np.testing.assert_equal(sample, expected_sample)

def test_cycle_quiet():
    if False:
        print('Hello World!')
    batch_size = 20
    for (epoch_size, cb, is_gen_fun) in [(1, Iterable(batch_size, (4, 5), epoch_size=1), False), (4, Iterable(batch_size, (4, 5), epoch_size=4), False), (1, generator_epoch_size_1, True), (4, generator_epoch_size_4, True)]:
        for reader_queue_size in (1, 2, 6):
            yield (_test_cycle_quiet, cb, is_gen_fun, batch_size, epoch_size, reader_queue_size)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_cycle_quiet_non_resetable(iterable, reader_queue_size, batch_size, epoch_size):
    if False:
        while True:
            i = 10
    pipe = utils.create_pipe(iterable, 'cpu', batch_size=batch_size, py_num_workers=1, py_start_method='spawn', parallel=True, device_id=None, batch=True, num_threads=5, cycle='quiet', reader_queue_depth=reader_queue_size)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    for _ in range(epoch_size):
        pipe.run()
    try:
        pipe.run()
    except StopIteration:
        pipe.reset()
        try:
            pipe.run()
        except StopIteration:
            pass
        else:
            assert False, 'Expected stop iteration'
    else:
        assert False, 'Expected stop iteration at the end of the epoch'

def test_cycle_quiet_non_resetable():
    if False:
        while True:
            i = 10
    epoch_size = 3
    batch_size = 20
    iterable = FaultyResetIterable(batch_size, (5, 4), epoch_size=epoch_size)
    for reader_queue_size in (1, 3, 6):
        yield (_test_cycle_quiet_non_resetable, iterable, reader_queue_size, batch_size, epoch_size)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_cycle_no_resetting(cb, batch_size, epoch_size, reader_queue_size):
    if False:
        for i in range(10):
            print('nop')
    pipe = utils.create_pipe(cb, 'cpu', batch_size=batch_size, py_num_workers=1, py_start_method='spawn', parallel=True, device_id=None, batch=True, num_threads=5, cycle=None, reader_queue_depth=reader_queue_size)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    for _ in range(epoch_size):
        pipe.run()
    try:
        pipe.run()
    except StopIteration:
        pipe.reset()
    else:
        assert False, 'Expected stop iteration'
    pipe.run()

def test_cycle_no_resetting():
    if False:
        i = 10
        return i + 15
    batch_size = 20
    for (epoch_size, cb) in [(1, Iterable(batch_size, (4, 5), epoch_size=1)), (4, Iterable(batch_size, (4, 5), epoch_size=4)), (1, generator_epoch_size_1), (4, generator_epoch_size_4)]:
        for reader_queue_size in (1, 2, 6):
            yield (raises(StopIteration)(_test_cycle_no_resetting), cb, batch_size, epoch_size, reader_queue_size)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_all_kinds_parallel(sample_cb, batch_cb, iter_cb, batch_size, py_num_workers, reader_queue_sizes, num_iters):
    if False:
        return 10

    @dali.pipeline_def(batch_size=batch_size, num_threads=4, device_id=None, py_num_workers=py_num_workers, py_start_method='spawn')
    def pipeline():
        if False:
            print('Hello World!')
        (queue_size_1, queue_size_2, queue_size_3) = reader_queue_sizes
        sample_out = dali.fn.external_source(source=sample_cb, parallel=True, batch=False, prefetch_queue_depth=queue_size_1)
        batch_out = dali.fn.external_source(source=batch_cb, parallel=True, batch=True, prefetch_queue_depth=queue_size_2, batch_info=True)
        iter_out = dali.fn.external_source(source=iter_cb, parallel=True, batch=True, prefetch_queue_depth=queue_size_3, cycle='raise')
        return (sample_out, batch_out, iter_out)
    pipe = pipeline()
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    for _ in range(3):
        i = 0
        while True:
            try:
                (sample_outs, batch_outs, iter_outs) = pipe.run()
                assert len(sample_outs) == len(batch_outs), f'Batch length mismatch: sample: {len(sample_outs)}, batch: {len(batch_outs)}'
                assert len(batch_outs) == len(iter_outs), f'Batch length mismatch: batch: {len(batch_outs)}, iter: {len(iter_outs)}'
                for (sample_out, batch_out, iter_out) in zip(sample_outs, batch_outs, iter_outs):
                    np.testing.assert_equal(np.array(sample_out), np.array(batch_out))
                    np.testing.assert_equal(np.array(batch_out), np.array(iter_out))
                i += 1
            except StopIteration:
                pipe.reset()
                assert i == num_iters, f'Number of iterations mismatch: expected {num_iters}, got {i}'
                break

def test_all_kinds_parallel():
    if False:
        i = 10
        return i + 15
    for batch_size in (1, 17):
        for num_iters in (1, 3, 31):
            for trailing in (0, 30):
                if trailing >= batch_size:
                    continue
                epoch_size = num_iters * batch_size + trailing
                sample_cb = utils.ExtCallback((4, 5), epoch_size, np.int32)
                batch_cb = SampleCallbackBatched(sample_cb, batch_size, batch_info=True)
                iterator_cb = SampleCallbackIterator(sample_cb, batch_size, batch_info=True)
                for reader_queue_sizes in ((1, 1, 1), (2, 2, 2), (5, 5, 5), (3, 1, 1), (1, 3, 1), (1, 1, 3)):
                    for num_workers in (1, 7):
                        yield (_test_all_kinds_parallel, sample_cb, batch_cb, iterator_cb, batch_size, num_workers, reader_queue_sizes, num_iters)

def collect_iterations(pipe, num_iters):
    if False:
        i = 10
        return i + 15
    outs = []
    for _ in range(num_iters):
        try:
            out = pipe.run()
            outs.append([[np.copy(sample) for sample in batch] for batch in out])
        except StopIteration:
            outs.append(StopIteration)
            pipe.reset()
    return outs

@with_setup(utils.setup_function, utils.teardown_function)
def _test_cycle_multiple_iterators(batch_size, iters_num, py_num_workers, reader_queue_sizes, cycle_policies, epoch_sizes):
    if False:
        i = 10
        return i + 15

    @dali.pipeline_def(batch_size=batch_size, num_threads=4, device_id=None, py_num_workers=py_num_workers, py_start_method='spawn')
    def pipeline(sample_cb, iter_1, iter_2, parallel):
        if False:
            while True:
                i = 10
        if parallel:
            (queue_size_0, queue_size_1, queue_size_2) = reader_queue_sizes
        else:
            (queue_size_0, queue_size_1, queue_size_2) = (None, None, None)
        (cycle_1, cycle_2) = cycle_policies
        sample_out = dali.fn.external_source(source=sample_cb, parallel=parallel, batch=False, prefetch_queue_depth=queue_size_0)
        iter1_out = dali.fn.external_source(source=iter_1, parallel=parallel, batch=True, prefetch_queue_depth=queue_size_1, cycle=cycle_1)
        iter2_out = dali.fn.external_source(source=iter_2, parallel=parallel, batch=True, prefetch_queue_depth=queue_size_2, cycle=cycle_2)
        return (sample_out, iter1_out, iter2_out)
    shape = (2, 3)
    (sample_epoch_size, iter_1_epoch_size, iter_2_epoch_size) = epoch_sizes
    sample_cb = utils.ExtCallback((4, 5), sample_epoch_size * batch_size, np.int32)
    iter_1 = Iterable(batch_size, shape, epoch_size=iter_1_epoch_size, dtype=np.int32)
    iter_2 = Iterable(batch_size, shape, epoch_size=iter_2_epoch_size, dtype=np.int32)
    pipe_parallel = pipeline(sample_cb, iter_1, iter_2, parallel=True)
    pipe_seq = pipeline(sample_cb, iter_1, iter_2, parallel=False)
    pipe_parallel.build()
    utils.capture_processes(pipe_parallel._py_pool)
    pipe_seq.build()
    parallel_outs = collect_iterations(pipe_parallel, iters_num)
    seq_outs = collect_iterations(pipe_seq, iters_num)
    assert len(parallel_outs) == len(seq_outs)
    for (parallel_out, seq_out) in zip(parallel_outs, seq_outs):
        if parallel_out == StopIteration or seq_out == StopIteration:
            assert parallel_out == seq_out
            continue
        assert len(parallel_out) == len(seq_out) == 3
        for (batch_parallel, batch_seq) in zip(parallel_out, seq_out):
            assert len(batch_parallel) == len(batch_seq) == batch_size
            for (sample_parallel, sample_seq) in zip(batch_parallel, batch_seq):
                np.testing.assert_equal(np.array(sample_parallel), np.array(sample_seq))

def test_cycle_multiple_iterators():
    if False:
        for i in range(10):
            print('nop')
    batch_size = 50
    iters_num = 17
    num_workers = 4
    for prefetch_queue_depths in ((3, 1, 1), (1, 3, 1), (1, 1, 3), (1, 1, 1), (3, 3, 3)):
        for cycle_policies in (('raise', 'raise'), ('quiet', 'raise'), ('raise', 'quiet'), ('quiet', 'quiet'), (True, True)):
            for epoch_sizes in ((8, 4, 6), (8, 6, 4), (4, 6, 8), (1, 1, 1)):
                yield (_test_cycle_multiple_iterators, batch_size, iters_num, num_workers, prefetch_queue_depths, cycle_policies, epoch_sizes)

def ext_cb2(sinfo):
    if False:
        return 10
    return np.array([sinfo.idx_in_epoch, sinfo.idx_in_batch, sinfo.iteration], dtype=np.int32)

@with_setup(utils.setup_function, utils.teardown_function)
def test_discard():
    if False:
        return 10
    bs = 5
    pipe = dali.Pipeline(batch_size=bs, device_id=None, num_threads=5, py_num_workers=4, py_start_method='spawn')
    with pipe:
        ext1 = dali.fn.external_source([[np.float32(i) for i in range(bs)]] * 3, cycle='raise')
        ext2 = dali.fn.external_source(ext_cb2, batch=False, parallel=True)
        ext3 = dali.fn.external_source(ext_cb2, batch=False, parallel=False)
        pipe.set_outputs(ext1, ext2, ext3)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    sample_in_epoch = 0
    iteration = 0
    for i in range(10):
        try:
            (e1, e2, e3) = pipe.run()
            for i in range(bs):
                assert e1.at(i) == i
                assert np.array_equal(e2.at(i), np.array([sample_in_epoch, i, iteration]))
                assert np.array_equal(e3.at(i), np.array([sample_in_epoch, i, iteration]))
                sample_in_epoch += 1
            iteration += 1
        except StopIteration:
            sample_in_epoch = 0
            iteration = 0
            pipe.reset()

class SampleCb:

    def __init__(self, batch_size, epoch_size):
        if False:
            print('Hello World!')
        self.batch_size = batch_size
        self.epoch_size = epoch_size

    def __call__(self, sample_info):
        if False:
            for i in range(10):
                print('nop')
        if sample_info.iteration >= self.epoch_size:
            raise StopIteration
        return np.array([sample_info.idx_in_epoch, sample_info.idx_in_batch, sample_info.iteration, sample_info.epoch_idx], dtype=np.int32)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_epoch_idx(batch_size, epoch_size, cb, py_num_workers, prefetch_queue_depth, reader_queue_depth, batch_mode, batch_info):
    if False:
        for i in range(10):
            print('nop')
    num_epochs = 3
    pipe = utils.create_pipe(cb, 'cpu', batch_size=batch_size, py_num_workers=py_num_workers, py_start_method='spawn', parallel=True, device_id=0, batch=batch_mode, num_threads=1, cycle=None, batch_info=batch_info, prefetch_queue_depth=prefetch_queue_depth, reader_queue_depth=reader_queue_depth)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    for epoch_idx in range(num_epochs):
        for iteration in range(epoch_size):
            (batch,) = pipe.run()
            assert len(batch) == batch_size
            for (sample_i, sample) in enumerate(batch):
                expected = np.array([iteration * batch_size + sample_i, sample_i, iteration, epoch_idx if not batch_mode or batch_info else 0])
                np.testing.assert_array_equal(sample, expected)
        try:
            pipe.run()
        except StopIteration:
            pipe.reset()
        else:
            assert False, 'expected StopIteration'

def test_epoch_idx():
    if False:
        return 10
    num_workers = 4
    prefetch_queue_depth = 2
    for batch_size in (1, 50):
        for epoch_size in (1, 3, 7):
            for reader_queue_depth in (1, 5):
                sample_cb = SampleCb(batch_size, epoch_size)
                yield (_test_epoch_idx, batch_size, epoch_size, sample_cb, num_workers, prefetch_queue_depth, reader_queue_depth, False, None)
                batch_cb = SampleCallbackBatched(sample_cb, batch_size, True)
                yield (_test_epoch_idx, batch_size, epoch_size, batch_cb, num_workers, prefetch_queue_depth, reader_queue_depth, True, True)
                batch_cb = SampleCallbackBatched(sample_cb, batch_size, False)
                yield (_test_epoch_idx, batch_size, epoch_size, batch_cb, num_workers, prefetch_queue_depth, reader_queue_depth, True, False)

class PermutableSampleCb:

    def __init__(self, batch_size, epoch_size, trailing_samples):
        if False:
            return 10
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.trailing_samples = trailing_samples
        self.last_seen_epoch = None
        self.perm = None

    def __call__(self, sample_info):
        if False:
            while True:
                i = 10
        if sample_info.iteration > self.epoch_size or (sample_info.iteration == self.epoch_size and sample_info.idx_in_batch >= self.trailing_samples):
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            rng = np.random.default_rng(seed=42 + self.last_seen_epoch)
            self.perm = rng.permutation(self.batch_size * self.epoch_size + self.trailing_samples)
        return np.array([self.perm[sample_info.idx_in_epoch]], dtype=np.int32)

@with_setup(utils.setup_function, utils.teardown_function)
def _test_permute_dataset(batch_size, epoch_size, trailing_samples, cb, py_num_workers, prefetch_queue_depth, reader_queue_depth):
    if False:
        for i in range(10):
            print('nop')
    num_epochs = 3
    pipe = utils.create_pipe(cb, 'cpu', batch_size=batch_size, py_num_workers=py_num_workers, py_start_method='spawn', parallel=True, device_id=0, batch=False, num_threads=1, cycle=None, prefetch_queue_depth=prefetch_queue_depth, reader_queue_depth=reader_queue_depth)
    pipe.build()
    utils.capture_processes(pipe._py_pool)
    for epoch_idx in range(num_epochs):
        epoch_data = [False for _ in range(epoch_size * batch_size + trailing_samples)]
        for _ in range(epoch_size):
            (batch,) = pipe.run()
            assert len(batch) == batch_size
            for sample in batch:
                epoch_data[np.array(sample)[0]] = True
        assert sum(epoch_data) == epoch_size * batch_size, 'Epoch number {} did not contain some samples from data set'.format(epoch_idx)
        try:
            pipe.run()
        except StopIteration:
            pipe.reset()
        else:
            assert False, 'expected StopIteration'

def test_permute_dataset():
    if False:
        i = 10
        return i + 15
    for (batch_size, trailing_samples) in ((4, 0), (100, 0), (100, 99)):
        for epoch_size in (3, 7):
            cb = PermutableSampleCb(batch_size, epoch_size, trailing_samples=trailing_samples)
            for reader_queue_depth in (1, 5):
                yield (_test_permute_dataset, batch_size, epoch_size, trailing_samples, cb, 4, 1, reader_queue_depth)

class PerIterShapeSource:

    def __init__(self, shapes):
        if False:
            return 10
        self.shapes = shapes

    def __call__(self, sample_info):
        if False:
            print('Hello World!')
        batch_idx = sample_info.iteration
        shape = self.shapes[batch_idx % len(self.shapes)]
        return np.full(shape, sample_info.idx_in_epoch, dtype=np.uint8)

def per_iter_shape_pipeline(shapes, py_num_workers=4, batch_size=4, parallel=True, bytes_per_sample_hint=None):
    if False:
        i = 10
        return i + 15

    @dali.pipeline_def
    def pipeline():
        if False:
            i = 10
            return i + 15
        return dali.fn.external_source(PerIterShapeSource(shapes), batch=False, parallel=parallel, bytes_per_sample_hint=bytes_per_sample_hint)
    pipe = pipeline(batch_size=batch_size, py_num_workers=py_num_workers, device_id=0, num_threads=4, py_start_method='spawn')
    pipe.build()
    return pipe

def test_no_parallel_no_shm():
    if False:
        i = 10
        return i + 15
    shapes = [(4, 1024, 1024)]
    pipe = per_iter_shape_pipeline(shapes, parallel=False)
    for _ in range(5):
        pipe.run()
    assert pipe.external_source_shm_statistics()['capacities'] == []

def test_default_shm_size():
    if False:
        for i in range(10):
            print('nop')
    default_shm_size = 1024 * 1024
    shapes = [(16, 1024, 1024)]
    pipe_default = per_iter_shape_pipeline(shapes)
    default_sizes = pipe_default.external_source_shm_statistics()['capacities']
    assert len(default_sizes) > 0
    for size in default_sizes:
        assert size == default_shm_size, f'Expected initial size to be {default_shm_size}, got {size}.'
    pipe_too_small_hint = per_iter_shape_pipeline(shapes, bytes_per_sample_hint=1024)
    sizes = pipe_too_small_hint.external_source_shm_statistics()['capacities']
    assert len(sizes) > 0
    for size in sizes:
        assert size == default_shm_size, f'Expected initial size to be {default_shm_size}, got {size}.'

def test_initial_hint():
    if False:
        for i in range(10):
            print('nop')
    sample_size = 32 * 1024 * 1024
    shapes = [(32, 1024, 1024)]
    bytes_per_sample_hint = 4 * 1024 * 1024
    batch_size = 7
    num_workers = 3
    min_samples_in_mini_batch = batch_size // num_workers
    max_samples_in_mini_batch = (batch_size + num_workers - 1) // num_workers
    initial_shm_size = max_samples_in_mini_batch * bytes_per_sample_hint
    expected_min_chunk_size = min_samples_in_mini_batch * sample_size
    pipe = per_iter_shape_pipeline(shapes, bytes_per_sample_hint=bytes_per_sample_hint, batch_size=batch_size, py_num_workers=num_workers)
    sizes = pipe.external_source_shm_statistics()['capacities']
    assert len(sizes) > 0
    for size in sizes:
        assert size == initial_shm_size, f'Expected initial size to be {initial_shm_size}, got {size}.'
    for _ in range(5):
        pipe.run()
    sizes = pipe.external_source_shm_statistics()['capacities']
    assert len(sizes) > 0
    for size in sizes:
        assert size >= expected_min_chunk_size, f'Expected the size to be at least {expected_min_chunk_size}, got {size}.'

def test_variable_sample_size():
    if False:
        i = 10
        return i + 15
    shapes = [(31, 1024, 1024), (32, 1024, 1024)]
    bytes_per_sample_hint = 32 * 1024 * 1024
    bytes_per_sample_hint += 4096
    batch_size = 8
    num_workers = 4
    max_samples_in_mini_batch = batch_size // num_workers
    initial_shm_size = max_samples_in_mini_batch * bytes_per_sample_hint
    pipe = per_iter_shape_pipeline(shapes, bytes_per_sample_hint=bytes_per_sample_hint, batch_size=batch_size, py_num_workers=num_workers)
    no_hint_pipe = per_iter_shape_pipeline(shapes, batch_size=batch_size, py_num_workers=num_workers)
    sizes = pipe.external_source_shm_statistics()['capacities']
    assert len(sizes) > 0
    for size in sizes:
        assert size == initial_shm_size, f'Expected initial size to be {initial_shm_size}, got {size}.'
    for _ in range(5):
        pipe.run()
        no_hint_pipe.run()
    sizes = pipe.external_source_shm_statistics()['capacities']
    assert len(sizes) > 0
    for size in sizes:
        assert size >= initial_shm_size, f'Expected the size to be unchanged and equal {initial_shm_size}, got {size}.'
    per_sample_sizes = pipe.external_source_shm_statistics()['per_sample_capacities']
    assert len(sizes) == len(per_sample_sizes)
    for size in per_sample_sizes:
        assert size == bytes_per_sample_hint, f'Expected initial per sample size to be {bytes_per_sample_hint}, got {size}.'
    no_hint_pipe_shm_size = min(no_hint_pipe.external_source_shm_statistics()['capacities'])
    sizes = pipe.external_source_shm_statistics()['capacities']
    for size in sizes:
        assert size < no_hint_pipe_shm_size, f'Expected the size to be less than {no_hint_pipe_shm_size}, got {size}.'