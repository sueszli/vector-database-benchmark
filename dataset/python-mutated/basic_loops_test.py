"""Tests for basic_loops.py."""
import os
import shutil
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import basic_loops
from tensorflow.python.training import supervisor

def _test_dir(test_name):
    if False:
        return 10
    test_dir = os.path.join(test.get_temp_dir(), test_name)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    return test_dir

class BasicTrainLoopTest(test.TestCase):

    def testBasicTrainLoop(self):
        if False:
            return 10
        logdir = _test_dir('basic_train_loop')
        num_calls = [0]

        def train_fn(unused_sess, sv, y, a):
            if False:
                return 10
            num_calls[0] += 1
            self.assertEqual('y', y)
            self.assertEqual('A', a)
            if num_calls[0] == 3:
                sv.request_stop()
        with ops.Graph().as_default():
            sv = supervisor.Supervisor(logdir=logdir)
            basic_loops.basic_train_loop(sv, train_fn, args=(sv, 'y'), kwargs={'a': 'A'})
            self.assertEqual(3, num_calls[0])

    def testBasicTrainLoopExceptionAborts(self):
        if False:
            for i in range(10):
                print('nop')
        logdir = _test_dir('basic_train_loop_exception_aborts')

        def train_fn(unused_sess):
            if False:
                for i in range(10):
                    print('nop')
            train_fn.counter += 1
            if train_fn.counter == 3:
                raise RuntimeError('Failed')
        train_fn.counter = 0
        with ops.Graph().as_default():
            sv = supervisor.Supervisor(logdir=logdir)
            with self.assertRaisesRegex(RuntimeError, 'Failed'):
                basic_loops.basic_train_loop(sv, train_fn)

    def testBasicTrainLoopRetryOnAborted(self):
        if False:
            i = 10
            return i + 15
        logdir = _test_dir('basic_train_loop_exception_aborts')

        class AbortAndRetry:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.num_calls = 0
                self.retries_left = 2

            def train_fn(self, unused_sess):
                if False:
                    return 10
                self.num_calls += 1
                if self.num_calls % 3 == 2:
                    self.retries_left -= 1
                if self.retries_left > 0:
                    raise errors_impl.AbortedError(None, None, 'Aborted here')
                else:
                    raise RuntimeError('Failed Again')
        with ops.Graph().as_default():
            sv = supervisor.Supervisor(logdir=logdir)
            aar = AbortAndRetry()
            with self.assertRaisesRegex(RuntimeError, 'Failed Again'):
                basic_loops.basic_train_loop(sv, aar.train_fn)
            self.assertEqual(0, aar.retries_left)
if __name__ == '__main__':
    test.main()