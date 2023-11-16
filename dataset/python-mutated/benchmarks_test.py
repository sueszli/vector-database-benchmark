"""Benchmarks for checkpoint-related APIs."""
import os
import time
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.platform import test
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader

class _TrivialRestore(base.Trackable):

    def _serialize_to_tensors(self):
        if False:
            print('Hello World!')
        return {base.VARIABLE_VALUE_KEY: array_ops.ones([])}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            return 10
        return control_flow_ops.no_op()

class _LazyTrivialObjects(module.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.existing = [_TrivialRestore() for _ in range(5)]
        self.lazy = []

    def __call__(self):
        if False:
            while True:
                i = 10
        if not self.lazy:
            self.lazy.extend((_TrivialRestore() for _ in range(5)))
        return

def _save_checkpoint():
    if False:
        i = 10
        return i + 15
    original_checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
    original_checkpoint.m()
    return original_checkpoint.write(os.path.join(test.get_temp_dir(), 'ckpt'))

class SavingBenchmarks(test.Benchmark):

    def _run(self, func, num_iters, execution_mode=None):
        if False:
            print('Hello World!')
        func()
        start = time.time()
        for _ in range(num_iters):
            func()
        end = time.time()
        mean_us = (end - start) * 1000000.0 / num_iters
        self.report_benchmark(iters=num_iters, wall_time=mean_us, extras={'examples_per_sec': num_iters / (end - start)})

    def benchmark_baseline_no_restore(self):
        if False:
            for i in range(10):
                print('nop')

        def _create_and_call():
            if False:
                print('Hello World!')
            checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
            checkpoint.m()
        self._run(_create_and_call, 3)

    def benchmark_batch_restore(self):
        if False:
            for i in range(10):
                print('nop')
        checkpoint_path = _save_checkpoint()

        def _create_and_call():
            if False:
                print('Hello World!')
            checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
            checkpoint.m()
            checkpoint.restore(checkpoint_path)
        self._run(_create_and_call, 3)

    def benchmark_restore_on_create(self):
        if False:
            return 10
        checkpoint_path = _save_checkpoint()

        def _create_and_call():
            if False:
                i = 10
                return i + 15
            checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
            checkpoint.restore(checkpoint_path)
            checkpoint.m()
        self._run(_create_and_call, 3)

    def benchmark_raw_restore(self):
        if False:
            for i in range(10):
                print('nop')
        checkpoint_path = _save_checkpoint()
        (all_names, all_dtypes) = zip(*py_checkpoint_reader.NewCheckpointReader(checkpoint_path).get_variable_to_dtype_map().items())

        def _call_restore_v2():
            if False:
                while True:
                    i = 10
            gen_io_ops.restore_v2(checkpoint_path, all_names, [''] * len(all_names), all_dtypes)
        self._run(_call_restore_v2, 3)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()