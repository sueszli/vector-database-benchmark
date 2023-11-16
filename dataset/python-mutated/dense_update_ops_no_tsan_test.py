"""Tests for state updating ops that may have benign race conditions."""
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class AssignOpTest(test.TestCase):

    def testParallelUpdateWithoutLocking(self):
        if False:
            while True:
                i = 10
        ops.get_default_graph().switch_to_thread_local()
        with self.cached_session() as sess:
            ones_t = array_ops.fill([1024, 1024], 1.0)
            p = variables.Variable(array_ops.zeros([1024, 1024]))
            adds = [state_ops.assign_add(p, ones_t, use_locking=False) for _ in range(20)]
            self.evaluate(variables.global_variables_initializer())

            def run_add(add_op):
                if False:
                    return 10
                self.evaluate(add_op)
            threads = [self.checkedThread(target=run_add, args=(add_op,)) for add_op in adds]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            vals = self.evaluate(p)
            ones = np.ones((1024, 1024)).astype(np.float32)
            self.assertTrue((vals >= ones).all())
            self.assertTrue((vals <= ones * 20).all())

    def testParallelAssignWithoutLocking(self):
        if False:
            for i in range(10):
                print('nop')
        ops.get_default_graph().switch_to_thread_local()
        with self.cached_session() as sess:
            ones_t = array_ops.fill([1024, 1024], float(1))
            p = variables.Variable(array_ops.zeros([1024, 1024]))
            assigns = [state_ops.assign(p, math_ops.multiply(ones_t, float(i)), False) for i in range(1, 21)]
            self.evaluate(variables.global_variables_initializer())

            def run_assign(assign_op):
                if False:
                    return 10
                self.evaluate(assign_op)
            threads = [self.checkedThread(target=run_assign, args=(assign_op,)) for assign_op in assigns]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            vals = self.evaluate(p)
            self.assertTrue((vals > 0).all())
            self.assertTrue((vals <= 20).all())

    def testParallelUpdateWithLocking(self):
        if False:
            i = 10
            return i + 15
        ops.get_default_graph().switch_to_thread_local()
        with self.cached_session() as sess:
            zeros_t = array_ops.fill([1024, 1024], 0.0)
            ones_t = array_ops.fill([1024, 1024], 1.0)
            p = variables.Variable(zeros_t)
            adds = [state_ops.assign_add(p, ones_t, use_locking=True) for _ in range(20)]
            self.evaluate(p.initializer)

            def run_add(add_op):
                if False:
                    while True:
                        i = 10
                self.evaluate(add_op)
            threads = [self.checkedThread(target=run_add, args=(add_op,)) for add_op in adds]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            vals = self.evaluate(p)
            ones = np.ones((1024, 1024)).astype(np.float32)
            self.assertAllEqual(vals, ones * 20)

    def testParallelAssignWithLocking(self):
        if False:
            print('Hello World!')
        ops.get_default_graph().switch_to_thread_local()
        with self.cached_session() as sess:
            zeros_t = array_ops.fill([1024, 1024], 0.0)
            ones_t = array_ops.fill([1024, 1024], 1.0)
            p = variables.Variable(zeros_t)
            assigns = [state_ops.assign(p, math_ops.multiply(ones_t, float(i)), use_locking=True) for i in range(1, 21)]
            self.evaluate(p.initializer)

            def run_assign(assign_op):
                if False:
                    for i in range(10):
                        print('nop')
                self.evaluate(assign_op)
            threads = [self.checkedThread(target=run_assign, args=(assign_op,)) for assign_op in assigns]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            vals = self.evaluate(p)
            self.assertTrue(vals[0, 0] > 0)
            self.assertTrue(vals[0, 0] <= 20)
            self.assertAllEqual(vals, np.ones([1024, 1024]) * vals[0, 0])
if __name__ == '__main__':
    test.main()