from caffe2.python import workspace
from caffe2.python.core import Plan, to_execution_step, Net
from caffe2.python.task import Task, TaskGroup, final_output
from caffe2.python.net_builder import ops, NetBuilder
from caffe2.python.session import LocalSession
import unittest
import threading

class PythonOpStats:
    lock = threading.Lock()
    num_instances = 0
    num_calls = 0

def python_op_builder():
    if False:
        for i in range(10):
            print('nop')
    PythonOpStats.lock.acquire()
    PythonOpStats.num_instances += 1
    PythonOpStats.lock.release()

    def my_op(inputs, outputs):
        if False:
            for i in range(10):
                print('nop')
        PythonOpStats.lock.acquire()
        PythonOpStats.num_calls += 1
        PythonOpStats.lock.release()
    return my_op

def _test_loop():
    if False:
        print('Hello World!')
    x = ops.Const(5)
    y = ops.Const(0)
    with ops.loop():
        ops.stop_if(ops.EQ([x, ops.Const(0)]))
        ops.Add([x, ops.Const(-1)], [x])
        ops.Add([y, ops.Const(1)], [y])
    return y

def _test_inner_stop(x):
    if False:
        print('Hello World!')
    ops.stop_if(ops.LT([x, ops.Const(5)]))

def _test_outer():
    if False:
        i = 10
        return i + 15
    x = ops.Const(10)
    with ops.stop_guard() as g1:
        _test_inner_stop(x)
    y = ops.Const(3)
    with ops.stop_guard() as g2:
        _test_inner_stop(y)
    with ops.stop_guard() as g4:
        ops.Const(0)
    with ops.stop_guard() as g3:
        pass
    return (g1.has_stopped(), g2.has_stopped(), g3.has_stopped(), g4.has_stopped())

def _test_if(x):
    if False:
        i = 10
        return i + 15
    y = ops.Const(1)
    with ops.If(ops.GT([x, ops.Const(50)])):
        ops.Const(2, blob_out=y)
    with ops.If(ops.LT([x, ops.Const(50)])):
        ops.Const(3, blob_out=y)
        ops.stop()
        ops.Const(4, blob_out=y)
    return y

class TestNetBuilder(unittest.TestCase):

    def test_ops(self):
        if False:
            return 10
        with NetBuilder() as nb:
            y = _test_loop()
            (z, w, a, b) = _test_outer()
            p = _test_if(ops.Const(75))
            q = _test_if(ops.Const(25))
        plan = Plan('name')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        expected_results = [(y, 5), (z, False), (w, True), (a, False), (b, False), (p, 2), (q, 3)]
        for (b, expected) in expected_results:
            actual = ws.blobs[str(b)].fetch()
            self.assertEqual(actual, expected)

    def _expected_loop(self):
        if False:
            for i in range(10):
                print('nop')
        total = 0
        total_large = 0
        total_small = 0
        total_tiny = 0
        for loop_iter in range(10):
            outer = loop_iter * 10
            for inner_iter in range(loop_iter):
                val = outer + inner_iter
                if val >= 80:
                    total_large += val
                elif val >= 50:
                    total_small += val
                else:
                    total_tiny += val
                total += val
        return (total, total_large, total_small, total_tiny)

    def _actual_loop(self):
        if False:
            i = 10
            return i + 15
        total = ops.Const(0)
        total_large = ops.Const(0)
        total_small = ops.Const(0)
        total_tiny = ops.Const(0)
        with ops.loop(10) as loop:
            outer = ops.Mul([loop.iter(), ops.Const(10)])
            with ops.loop(loop.iter()) as inner:
                val = ops.Add([outer, inner.iter()])
                with ops.If(ops.GE([val, ops.Const(80)])) as c:
                    ops.Add([total_large, val], [total_large])
                with c.Elif(ops.GE([val, ops.Const(50)])) as c:
                    ops.Add([total_small, val], [total_small])
                with c.Else():
                    ops.Add([total_tiny, val], [total_tiny])
                ops.Add([total, val], total)
        return [final_output(x) for x in [total, total_large, total_small, total_tiny]]

    def test_net_multi_use(self):
        if False:
            i = 10
            return i + 15
        with Task() as task:
            total = ops.Const(0)
            net = Net('my_net')
            net.Add([total, net.Const(1)], [total])
            ops.net(net)
            ops.net(net)
            result = final_output(total)
        with LocalSession() as session:
            session.run(task)
            self.assertEqual(2, result.fetch())

    def test_loops(self):
        if False:
            i = 10
            return i + 15
        with Task() as task:
            out_actual = self._actual_loop()
        with LocalSession() as session:
            session.run(task)
            expected = self._expected_loop()
            actual = [o.fetch() for o in out_actual]
            for (e, a) in zip(expected, actual):
                self.assertEqual(e, a)

    def test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        with Task() as task:
            with ops.task_init():
                one = ops.Const(1)
            two = ops.Add([one, one])
            with ops.task_init():
                three = ops.Const(3)
            accum = ops.Add([two, three])
            with ops.task_exit():
                seven_1 = ops.Add([accum, one])
            six = ops.Add([accum, one])
            ops.Add([accum, one], [accum])
            seven_2 = ops.Add([accum, one])
            o6 = final_output(six)
            o7_1 = final_output(seven_1)
            o7_2 = final_output(seven_2)
        with LocalSession() as session:
            session.run(task)
            self.assertEqual(o6.fetch(), 6)
            self.assertEqual(o7_1.fetch(), 7)
            self.assertEqual(o7_2.fetch(), 7)

    def test_multi_instance_python_op(self):
        if False:
            while True:
                i = 10
        '\n        When task instances are created at runtime, C++ concurrently creates\n        multiple instances of operators in C++, and concurrently destroys them\n        once the task is finished. This means that the destructor of PythonOp\n        will be called concurrently, so the GIL must be acquired. This\n        test exercises this condition.\n        '
        with Task(num_instances=64) as task:
            with ops.loop(4):
                ops.Python((python_op_builder, [], {}))([], [])
        with LocalSession() as session:
            PythonOpStats.num_instances = 0
            PythonOpStats.num_calls = 0
            session.run(task)
            self.assertEqual(PythonOpStats.num_instances, 64)
            self.assertEqual(PythonOpStats.num_calls, 256)

    def test_multi_instance(self):
        if False:
            print('Hello World!')
        NUM_INSTANCES = 10
        NUM_ITERS = 15
        with TaskGroup() as tg:
            with Task(num_instances=NUM_INSTANCES):
                with ops.task_init():
                    counter1 = ops.CreateCounter([], ['global_counter'])
                    counter2 = ops.CreateCounter([], ['global_counter2'])
                    counter3 = ops.CreateCounter([], ['global_counter3'])
                with ops.task_instance_init():
                    task_counter = ops.CreateCounter([], ['task_counter'])
                local_counter = ops.CreateCounter([], ['local_counter'])
                with ops.loop(NUM_ITERS):
                    ops.CountUp(counter1)
                    ops.CountUp(task_counter)
                    ops.CountUp(local_counter)
                with ops.task_instance_exit():
                    count2 = ops.RetrieveCount(task_counter)
                    with ops.loop(ops.Mul([count2, count2])):
                        ops.CountUp(counter2)
                count3 = ops.RetrieveCount(local_counter)
                with ops.loop(ops.Mul([count3, count3])):
                    ops.CountUp(counter3)
                with ops.task_exit():
                    total1 = final_output(ops.RetrieveCount(counter1))
                    total2 = final_output(ops.RetrieveCount(counter2))
                    total3 = final_output(ops.RetrieveCount(counter3))
        with LocalSession() as session:
            session.run(tg)
            self.assertEqual(total1.fetch(), NUM_INSTANCES * NUM_ITERS)
            self.assertEqual(total2.fetch(), NUM_INSTANCES * NUM_ITERS ** 2)
            self.assertEqual(total3.fetch(), NUM_INSTANCES * NUM_ITERS ** 2)

    def test_if_net(self):
        if False:
            for i in range(10):
                print('nop')
        with NetBuilder() as nb:
            x0 = ops.Const(0)
            x1 = ops.Const(1)
            x2 = ops.Const(2)
            y0 = ops.Const(0)
            y1 = ops.Const(1)
            y2 = ops.Const(2)
            first_res = ops.Const(0)
            with ops.IfNet(ops.Const(True)):
                ops.Const(1, blob_out=first_res)
            with ops.Else():
                ops.Const(2, blob_out=first_res)
            second_res = ops.Const(0)
            with ops.IfNet(ops.Const(False)):
                ops.Const(1, blob_out=second_res)
            with ops.Else():
                ops.Const(2, blob_out=second_res)
            with ops.IfNet(ops.LT([x0, x1])):
                local_blob = ops.Const(900)
                ops.Add([ops.Const(100), local_blob], [y0])
                gt = ops.GT([x1, x2])
                with ops.IfNet(gt):
                    pass
                with ops.Else():
                    ops.Add([y1, local_blob], [local_blob])
                    ops.Add([ops.Const(100), y1], [y1])
                with ops.IfNet(ops.EQ([local_blob, ops.Const(901)])):
                    ops.Const(7, blob_out=y2)
                    ops.Add([y1, y2], [y2])
            with ops.Else():
                pass
        plan = Plan('if_net_test')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        first_res_value = ws.blobs[str(first_res)].fetch()
        second_res_value = ws.blobs[str(second_res)].fetch()
        y0_value = ws.blobs[str(y0)].fetch()
        y1_value = ws.blobs[str(y1)].fetch()
        y2_value = ws.blobs[str(y2)].fetch()
        self.assertEqual(first_res_value, 1)
        self.assertEqual(second_res_value, 2)
        self.assertEqual(y0_value, 1000)
        self.assertEqual(y1_value, 101)
        self.assertEqual(y2_value, 108)
        self.assertTrue(str(local_blob) not in ws.blobs)

    def test_while_net(self):
        if False:
            print('Hello World!')
        with NetBuilder() as nb:
            x = ops.Const(0)
            y = ops.Const(0)
            with ops.WhileNet():
                with ops.Condition():
                    ops.Add([x, ops.Const(1)], [x])
                    ops.LT([x, ops.Const(7)])
                ops.Add([x, y], [y])
        plan = Plan('while_net_test')
        plan.AddStep(to_execution_step(nb))
        ws = workspace.C.Workspace()
        ws.run(plan)
        x_value = ws.blobs[str(x)].fetch()
        y_value = ws.blobs[str(y)].fetch()
        self.assertEqual(x_value, 7)
        self.assertEqual(y_value, 21)