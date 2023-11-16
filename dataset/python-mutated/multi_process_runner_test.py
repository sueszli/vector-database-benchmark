"""Tests for `multi_process_runner`."""
import ctypes
import json
import os
import sys
import threading
import time
import unittest
from absl import logging
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
try:
    import dill
    _REGISTER_DECORATOR = dill.register
except ImportError:
    _REGISTER_DECORATOR = lambda fn, *_: fn

def fn_that_adds_task_type_in_return_data():
    if False:
        print('Hello World!')
    return multi_worker_test_base.get_task_type()

def fn_that_errors():
    if False:
        return 10
    raise ValueError('This is an error.')

def fn_that_does_nothing():
    if False:
        while True:
            i = 10
    pass

def fn_that_adds_simple_return_data():
    if False:
        while True:
            i = 10
    return 'dummy_data'

def fn_that_returns_args_and_kwargs(*args, **kwargs):
    if False:
        while True:
            i = 10
    return list(args) + list(kwargs.items())

def fn_with_barrier():
    if False:
        while True:
            i = 10
    return multi_process_runner.get_barrier()

def fn_that_returns_pid():
    if False:
        return 10
    return os.getpid()
V = None

def fn_that_sets_global(val):
    if False:
        i = 10
        return i + 15
    global V
    old_val = V
    V = val
    return old_val

@combinations.generate(combinations.combine(required_gpus=0))
class MultiProcessRunnerTest(test.TestCase, parameterized.TestCase):

    def _worker_idx(self):
        if False:
            print('Hello World!')
        config_task = json.loads(os.environ['TF_CONFIG'])['task']
        return config_task['index']

    def test_multi_process_runner(self):
        if False:
            i = 10
            return i + 15
        mpr_result = multi_process_runner.run(fn_that_adds_task_type_in_return_data, multi_worker_test_base.create_cluster_spec(num_workers=2, num_ps=3, has_chief=True))
        job_count_dict = {'worker': 2, 'ps': 3, 'chief': 1}
        for data in mpr_result.return_value:
            job_count_dict[data] -= 1
        self.assertEqual(job_count_dict['worker'], 0)
        self.assertEqual(job_count_dict['ps'], 0)
        self.assertEqual(job_count_dict['chief'], 0)

    def test_multi_process_runner_error_propagates_from_subprocesses(self):
        if False:
            print('Hello World!')
        runner = multi_process_runner.MultiProcessRunner(fn_that_errors, multi_worker_test_base.create_cluster_spec(num_workers=1, num_ps=1), max_run_time=20)
        runner.start()
        with self.assertRaisesRegex(ValueError, 'This is an error.'):
            runner.join()

    def test_multi_process_runner_queue_emptied_between_runs(self):
        if False:
            i = 10
            return i + 15
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        return_value = multi_process_runner.run(fn_that_adds_simple_return_data, cluster_spec).return_value
        self.assertTrue(return_value)
        self.assertEqual(return_value[0], 'dummy_data')
        self.assertEqual(return_value[1], 'dummy_data')
        return_value = multi_process_runner.run(fn_that_does_nothing, cluster_spec).return_value
        self.assertFalse(return_value)

    def test_multi_process_runner_args_passed_correctly(self):
        if False:
            while True:
                i = 10
        return_value = multi_process_runner.run(fn_that_returns_args_and_kwargs, multi_worker_test_base.create_cluster_spec(num_workers=1), args=('a', 'b'), kwargs={'c_k': 'c_v'}).return_value
        self.assertEqual(return_value[0][0], 'a')
        self.assertEqual(return_value[0][1], 'b')
        self.assertEqual(return_value[0][2], ('c_k', 'c_v'))

    def test_stdout_captured(self):
        if False:
            i = 10
            return i + 15

        def simple_print_func():
            if False:
                for i in range(10):
                    print('nop')
            print('This is something printed.', flush=True)
            return 'This is returned data.'
        mpr_result = multi_process_runner.run(simple_print_func, multi_worker_test_base.create_cluster_spec(num_workers=2), return_output=True)
        std_stream_results = mpr_result.stdout
        return_value = mpr_result.return_value
        self.assertIn('[worker-0]:    This is something printed.\n', std_stream_results)
        self.assertIn('[worker-1]:    This is something printed.\n', std_stream_results)
        self.assertIn('This is returned data.', return_value)

    def test_termination(self):
        if False:
            return 10

        def fn():
            if False:
                i = 10
                return i + 15
            for i in range(0, 10):
                print('index {}, iteration {}'.format(self._worker_idx(), i), flush=True)
                time.sleep(5)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=2), return_output=True)
        mpr.start()
        time.sleep(5)
        mpr.terminate('worker', 0)
        std_stream_results = mpr.join().stdout
        self.assertIn('[worker-0]:    index 0, iteration 0\n', std_stream_results)
        self.assertNotIn('[worker-0]:    index 0, iteration 9\n', std_stream_results)
        self.assertIn('[worker-1]:    index 1, iteration 0\n', std_stream_results)
        self.assertIn('[worker-1]:    index 1, iteration 9\n', std_stream_results)

    def test_termination_and_start_single_process(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                return 10
            for i in range(0, 10):
                print('index {}, iteration {}'.format(self._worker_idx(), i), flush=True)
                time.sleep(1)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=2), return_output=True)
        mpr.start()
        time.sleep(3)
        mpr.terminate('worker', 0)
        mpr.start_single_process('worker', 0)
        std_stream_results = mpr.join().stdout
        self.assertLen([s for s in std_stream_results if 'index 0, iteration 0' in s], 2)
        self.assertIn('[worker-0]:    index 0, iteration 9\n', std_stream_results)
        self.assertIn('[worker-1]:    index 1, iteration 0\n', std_stream_results)
        self.assertIn('[worker-1]:    index 1, iteration 9\n', std_stream_results)

    def test_streaming(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                return 10
            for i in range(5):
                logging.info('(logging) %s-%d, i: %d', multi_worker_test_base.get_task_type(), self._worker_idx(), i)
                print('(print) {}-{}, i: {}'.format(multi_worker_test_base.get_task_type(), self._worker_idx(), i), flush=True)
                time.sleep(1)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=2, num_ps=2), return_output=True)
        mpr._dependence_on_chief = False
        mpr.start()
        mpr.start_single_process('worker', 2)
        mpr.start_single_process('ps', 2)
        mpr_result = mpr.join()
        list_to_assert = mpr_result.stdout
        for job in ['chief']:
            for iteration in range(5):
                self.assertTrue(any(('(logging) {}-0, i: {}'.format(job, iteration) in line for line in list_to_assert)))
                self.assertTrue(any(('(print) {}-0, i: {}'.format(job, iteration) in line for line in list_to_assert)))
        for job in ['worker', 'ps']:
            for iteration in range(5):
                for task in range(3):
                    self.assertTrue(any(('(logging) {}-{}, i: {}'.format(job, task, iteration) in line for line in list_to_assert)))
                    self.assertTrue(any(('(print) {}-{}, i: {}'.format(job, task, iteration) in line for line in list_to_assert)))
                task = 3
                self.assertFalse(any(('(logging) {}-{}, i: {}'.format(job, task, iteration) in line for line in list_to_assert)))
                self.assertFalse(any(('(print) {}-{}, i: {}'.format(job, task, iteration) in line for line in list_to_assert)))

    def test_start_in_process_as(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                return 10
            for i in range(5):
                logging.info('%s-%d, i: %d', multi_worker_test_base.get_task_type(), self._worker_idx(), i)
                time.sleep(1)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1), return_output=True)

        def eval_func():
            if False:
                return 10
            time.sleep(1)
            mpr.start_single_process(task_type='evaluator', task_id=0)
        eval_thread = threading.Thread(target=eval_func)
        eval_thread.start()
        mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
        eval_thread.join()
        list_to_assert = mpr.join().stdout
        for job in ['worker', 'evaluator']:
            for iteration in range(5):
                self.assertTrue(any(('{}-0, i: {}'.format(job, iteration) in line for line in list_to_assert)))

    def test_terminate_all_does_not_ignore_error(self):
        if False:
            return 10
        mpr = multi_process_runner.MultiProcessRunner(fn_that_errors, multi_worker_test_base.create_cluster_spec(num_workers=2), return_output=True)
        mpr.start()
        time.sleep(60)
        mpr.terminate_all()
        with self.assertRaisesRegex(ValueError, 'This is an error.'):
            mpr.join()

    def test_barrier(self):
        if False:
            print('Hello World!')
        multi_process_runner.run(fn_with_barrier, cluster_spec=multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1))

    def test_barrier_called_in_main_process(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            multi_process_runner.get_barrier()

    def test_stdout_available_when_timeout(self):
        if False:
            return 10

        def fn():
            if False:
                return 10
            logging.info('something printed')
            time.sleep(10000)
        with self.assertRaises(multi_process_runner.SubprocessTimeoutError) as cm:
            mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=1), return_output=True)
            mpr.start()
            mpr.join(timeout=60)
        mpr.terminate_all()
        list_to_assert = cm.exception.mpr_result.stdout
        self.assertTrue(any(('something printed' in line for line in list_to_assert)))

    def test_seg_fault_raises_error(self):
        if False:
            for i in range(10):
                print('nop')
        if multi_process_runner.is_oss() or sys.version_info >= (3, 7):
            self.skipTest('TODO(b/171004637): Failing in OSS and Python 3.7+')

        def fn_expected_to_seg_fault():
            if False:
                for i in range(10):
                    print('nop')
            ctypes.string_at(0)
        with self.assertRaises(multi_process_runner.UnexpectedSubprocessExitError) as cm:
            multi_process_runner.run(fn_expected_to_seg_fault, multi_worker_test_base.create_cluster_spec(num_workers=1), return_output=True)
        self.assertIn('Subprocess worker-0 exited with exit code', str(cm.exception))
        list_to_assert = cm.exception.mpr_result.stdout
        self.assertTrue(any(('Segmentation fault' in line for line in list_to_assert)))

    def test_seg_fault_in_chief_raises_error(self):
        if False:
            print('Hello World!')
        if multi_process_runner.is_oss() or sys.version_info >= (3, 7):
            self.skipTest('TODO(b/171004637): Failing in OSS and Python 3.7+')

        def fn_expected_to_seg_fault():
            if False:
                for i in range(10):
                    print('nop')
            if multi_worker_test_base.get_task_type() == 'worker':
                time.sleep(10000)
            ctypes.string_at(0)
        with self.assertRaises(multi_process_runner.UnexpectedSubprocessExitError) as cm:
            multi_process_runner.run(fn_expected_to_seg_fault, multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1), return_output=True)
        self.assertIn('Subprocess chief-0 exited with exit code', str(cm.exception))
        list_to_assert = cm.exception.mpr_result.stdout
        self.assertTrue(any(('Segmentation fault' in line for line in list_to_assert)))

    def test_exit_code_is_reported_by_chief_subprocess(self):
        if False:
            for i in range(10):
                print('nop')

        def fn_expected_to_exit_with_20():
            if False:
                print('Hello World!')
            if multi_worker_test_base.get_task_type() == 'worker':
                time.sleep(10000)
            sys.exit(20)
        mpr = multi_process_runner.MultiProcessRunner(fn_expected_to_exit_with_20, multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1))
        mpr.start()
        with self.assertRaisesRegex(multi_process_runner.UnexpectedSubprocessExitError, 'Subprocess chief-0 exited with exit code 20'):
            mpr.join()

    def test_exit_code_is_reported_by_subprocess(self):
        if False:
            print('Hello World!')

        def fn_expected_to_exit_with_10():
            if False:
                while True:
                    i = 10
            sys.exit(10)
        mpr = multi_process_runner.MultiProcessRunner(fn_expected_to_exit_with_10, multi_worker_test_base.create_cluster_spec(num_workers=1))
        mpr.start()
        with self.assertRaisesRegex(multi_process_runner.UnexpectedSubprocessExitError, 'Subprocess worker-0 exited with exit code 10'):
            mpr.join()

    def test_auto_restart(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(counter):
            if False:
                for i in range(10):
                    print('nop')
            counter.value += 1
            if counter.value == 1:
                raise ValueError
        manager = multi_process_runner.manager()
        counter = manager.Value(int, 0)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=1), args=(counter,), auto_restart=True)
        mpr.start()
        mpr.join()
        self.assertEqual(counter.value, 2)

    def test_auto_restart_and_timeout(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                print('Hello World!')
            logging.info('Running')
            time.sleep(1)
            raise ValueError
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=1), auto_restart=True, return_output=True)
        mpr.start()
        with self.assertRaises(ValueError) as cm:
            mpr.join(timeout=10)
        self.assertGreater(sum(['Running' in msg for msg in cm.exception.mpr_result.stdout]), 1)

    def test_auto_restart_and_chief(self):
        if False:
            return 10

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(1)
            if multi_worker_test_base.get_task_type() != 'chief':
                raise ValueError
        manager = multi_process_runner.manager()
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1), auto_restart=True)
        mpr.start()
        with self.assertRaises(ValueError):
            mpr.join(timeout=10)

    def test_auto_restart_failure_immediate_after_restart(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(5)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(has_chief=False, num_workers=2), auto_restart=True)
        mpr.start()
        pid = mpr.get_process_id('worker', 1)
        mpr.terminate('worker', 1)
        while mpr.get_process_id('worker', 1) == pid:
            time.sleep(0.1)
        mpr.terminate('worker', 0)
        mpr.join(timeout=20)

    def test_auto_restart_terminate(self):
        if False:
            return 10

        def fn(counter):
            if False:
                print('Hello World!')
            counter.value += 1
            if counter.value == 1:
                time.sleep(100)
        manager = multi_process_runner.manager()
        counter = manager.Value(int, 0)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(has_chief=False, num_workers=1), args=(counter,), auto_restart=True)
        mpr.start()
        time.sleep(3)
        mpr.terminate('worker', 0)
        mpr.join(timeout=20)
        self.assertEqual(counter.value, 2)

    def test_error_reporting_overrides_timeout_reporting(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            if self._worker_idx() == 1:
                time.sleep(10000)
            raise ValueError('Worker 0 errored')
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=2))
        mpr.start()
        with self.assertRaisesRegex(ValueError, 'Worker 0 errored'):
            mpr.join(timeout=20)

    def test_process_exists(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                i = 10
                return i + 15
            time.sleep(100000)
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=1))
        mpr.start()
        self.assertTrue(mpr.process_exists('worker', 0))
        mpr.terminate('worker', 0)
        while mpr.process_exists('worker', 0):
            time.sleep(1)

    def test_timeout_none(self):
        if False:
            while True:
                i = 10
        if multi_process_runner.is_oss():
            self.skipTest('Intentionally skipping longer test in OSS.')

        def fn():
            if False:
                while True:
                    i = 10
            time.sleep(250)
            raise ValueError('Worker 0 errored')
        mpr = multi_process_runner.MultiProcessRunner(fn, multi_worker_test_base.create_cluster_spec(num_workers=1))
        mpr.start()
        with self.assertRaisesRegex(ValueError, 'Worker 0 errored'):
            mpr.join(timeout=None)
_global_pool = multi_process_runner.MultiProcessPoolRunner(multi_worker_test_base.create_cluster_spec(num_workers=2))

class MultiProcessPoolRunnerTest(test.TestCase):

    def test_same_process_across_runs(self):
        if False:
            print('Hello World!')
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
        pid = runner.run(fn_that_returns_pid)
        for _ in range(3):
            self.assertAllEqual(runner.run(fn_that_returns_pid), pid)

    def test_exceptions_in_sub_process(self):
        if False:
            while True:
                i = 10
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
        pid = runner.run(fn_that_returns_pid)
        with self.assertRaisesRegex(ValueError, 'This is an error.'):
            runner.run(fn_that_errors)
        self.assertAllEqual(runner.run(fn_that_returns_pid), pid)

    def test_tf_config(self):
        if False:
            print('Hello World!')
        cluster_spec = multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=2)
        runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
        result = runner.run(fn_that_adds_task_type_in_return_data)
        job_count_dict = {'worker': 2, 'chief': 1}
        for data in result:
            job_count_dict[data] -= 1
        self.assertEqual(job_count_dict['worker'], 0)
        self.assertEqual(job_count_dict['chief'], 0)

    @unittest.expectedFailure
    def test_exception_in_main_process(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=2)
        runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
        runner.run(fn_that_returns_pid)
        raise ValueError('failure')

    def test_initializer(self):
        if False:
            print('Hello World!')
        cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
        runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec, initializer=lambda : fn_that_sets_global(1))
        result = runner.run(fn_that_sets_global, args=(2,))
        self.assertAllEqual(result, [1, 1])

    def test_global_pool(self):
        if False:
            return 10
        _global_pool.run(fn_that_does_nothing)

    def test_nested_pool(self):
        if False:
            while True:
                i = 10

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            _global_pool.run(fn_that_does_nothing)
        _global_pool.run(fn)

@combinations.generate(combinations.combine(required_physical_gpus=2))
class MultiProcessRunnerMultiGPUTest(test.TestCase, parameterized.TestCase):

    def test_not_share_gpu(self):
        if False:
            print('Hello World!')
        num_gpus = len(context.context().list_physical_devices('GPU'))
        if num_gpus != 2 and num_gpus != 4:
            self.skipTest('requires 2 or 4 GPUs')
        cluster_spec = multi_worker_test_base.create_cluster_spec(has_chief=True, num_workers=1)

        def cuda_visible_devices_fn():
            if False:
                for i in range(10):
                    print('nop')
            return os.getenv('CUDA_VISIBLE_DEVICES')
        runner = multi_process_runner.MultiProcessRunner(cuda_visible_devices_fn, cluster_spec, share_gpu=False)
        runner.start()
        result = runner.join()
        if num_gpus == 2:
            self.assertAllEqual(sorted(result.return_value), ['0', '1'])
        else:
            self.assertAllEqual(sorted(result.return_value), ['0,2', '1,3'])

        def num_gpus_fn():
            if False:
                return 10
            return len(context.context().list_physical_devices('GPU'))
        runner = multi_process_runner.MultiProcessRunner(num_gpus_fn, cluster_spec, share_gpu=False)
        runner.start()
        result = runner.join()
        if num_gpus == 2:
            self.assertAllEqual(result.return_value, [1, 1])
        else:
            self.assertAllEqual(result.return_value, [2, 2])

@_REGISTER_DECORATOR(MultiProcessRunnerTest)
def _save_multi_process_runner_test(pickler, obj):
    if False:
        return 10

    def reconstruct(*args, **kwargs):
        if False:
            print('Hello World!')
        del args, kwargs
        return MultiProcessRunnerTest()
    return pickler.save_reduce(reconstruct, (), obj=obj)

@_REGISTER_DECORATOR(MultiProcessPoolRunnerTest)
def _save_multi_process_pool_runner_test(pickler, obj):
    if False:
        for i in range(10):
            print('nop')

    def reconstruct(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        del args, kwargs
        return MultiProcessPoolRunnerTest()
    return pickler.save_reduce(reconstruct, (), obj=obj)

@_REGISTER_DECORATOR(MultiProcessRunnerMultiGPUTest)
def _save_multi_process_runner_multi_gpu_test(pickler, obj):
    if False:
        return 10

    def reconstruct(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        del args, kwargs
        return MultiProcessRunnerMultiGPUTest()
    return pickler.save_reduce(reconstruct, (), obj=obj)
if __name__ == '__main__':
    multi_process_runner.test_main()