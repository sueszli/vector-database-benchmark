import os
import random
import stat
import sys
import tempfile
import time
import unittest
from pyspark import SparkConf, SparkContext, TaskContext, BarrierTaskContext
from pyspark.testing.utils import PySparkTestCase, SPARK_HOME, eventually

class TaskContextTests(PySparkTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._old_sys_path = list(sys.path)
        class_name = self.__class__.__name__
        self.sc = SparkContext('local[4, 2]', class_name)

    def test_stage_id(self):
        if False:
            while True:
                i = 10
        'Test the stage ids are available and incrementing as expected.'
        rdd = self.sc.parallelize(range(10))
        stage1 = rdd.map(lambda x: TaskContext.get().stageId()).take(1)[0]
        stage2 = rdd.map(lambda x: TaskContext.get().stageId()).take(1)[0]
        stage3 = rdd.map(lambda x: TaskContext().stageId()).take(1)[0]
        self.assertEqual(stage1 + 1, stage2)
        self.assertEqual(stage1 + 2, stage3)
        self.assertEqual(stage2 + 1, stage3)

    def test_resources(self):
        if False:
            i = 10
            return i + 15
        'Test the resources are empty by default.'
        rdd = self.sc.parallelize(range(10))
        resources1 = rdd.map(lambda x: TaskContext.get().resources()).take(1)[0]
        resources2 = rdd.map(lambda x: TaskContext().resources()).take(1)[0]
        self.assertEqual(len(resources1), 0)
        self.assertEqual(len(resources2), 0)

    def test_partition_id(self):
        if False:
            while True:
                i = 10
        'Test the partition id.'
        rdd1 = self.sc.parallelize(range(10), 1)
        rdd2 = self.sc.parallelize(range(10), 2)
        pids1 = rdd1.map(lambda x: TaskContext.get().partitionId()).collect()
        pids2 = rdd2.map(lambda x: TaskContext.get().partitionId()).collect()
        self.assertEqual(0, pids1[0])
        self.assertEqual(0, pids1[9])
        self.assertEqual(0, pids2[0])
        self.assertEqual(1, pids2[9])

    def test_attempt_number(self):
        if False:
            i = 10
            return i + 15
        'Verify the attempt numbers are correctly reported.'
        rdd = self.sc.parallelize(range(10))
        attempt_numbers = rdd.map(lambda x: TaskContext.get().attemptNumber()).collect()
        map(lambda attempt: self.assertEqual(0, attempt), attempt_numbers)

        def fail_on_first(x):
            if False:
                return 10
            'Fail on the first attempt so we get a positive attempt number'
            tc = TaskContext.get()
            attempt_number = tc.attemptNumber()
            partition_id = tc.partitionId()
            attempt_id = tc.taskAttemptId()
            if attempt_number == 0 and partition_id == 0:
                raise RuntimeError('Failing on first attempt')
            else:
                return [x, partition_id, attempt_number, attempt_id]
        result = rdd.map(fail_on_first).collect()
        self.assertEqual([0, 0, 1], result[0][0:3])
        self.assertEqual([9, 3, 0], result[9][0:3])
        first_partition = filter(lambda x: x[1] == 0, result)
        map(lambda x: self.assertEqual(1, x[2]), first_partition)
        other_partitions = filter(lambda x: x[1] != 0, result)
        map(lambda x: self.assertEqual(0, x[2]), other_partitions)
        self.assertTrue(result[0][3] != result[9][3])

    def test_tc_on_driver(self):
        if False:
            return 10
        'Verify that getting the TaskContext on the driver returns None.'
        tc = TaskContext.get()
        self.assertTrue(tc is None)

    def test_get_local_property(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that local properties set on the driver are available in TaskContext.'
        key = 'testkey'
        value = 'testvalue'
        self.sc.setLocalProperty(key, value)
        try:
            rdd = self.sc.parallelize(range(1), 1)
            prop1 = rdd.map(lambda _: TaskContext.get().getLocalProperty(key)).collect()[0]
            self.assertEqual(prop1, value)
            prop2 = rdd.map(lambda _: TaskContext.get().getLocalProperty('otherkey')).collect()[0]
            self.assertTrue(prop2 is None)
        finally:
            self.sc.setLocalProperty(key, None)

    def test_barrier(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that BarrierTaskContext.barrier() performs global sync among all barrier tasks\n        within a stage.\n        '
        rdd = self.sc.parallelize(range(10), 4)

        def f(iterator):
            if False:
                i = 10
                return i + 15
            yield sum(iterator)

        def context_barrier(x):
            if False:
                for i in range(10):
                    print('nop')
            tc = BarrierTaskContext.get()
            time.sleep(random.randint(1, 5) * 2)
            tc.barrier()
            return time.time()
        times = rdd.barrier().mapPartitions(f).map(context_barrier).collect()
        self.assertTrue(max(times) - min(times) < 2)

    def test_all_gather(self):
        if False:
            while True:
                i = 10
        '\n        Verify that BarrierTaskContext.allGather() performs global sync among all barrier tasks\n        within a stage and passes messages properly.\n        '
        rdd = self.sc.parallelize(range(10), 4)

        def f(iterator):
            if False:
                print('Hello World!')
            yield sum(iterator)

        def context_barrier(x):
            if False:
                return 10
            tc = BarrierTaskContext.get()
            time.sleep(random.randint(1, 10))
            out = tc.allGather(str(tc.partitionId()))
            pids = [int(e) for e in out]
            return pids
        pids = rdd.barrier().mapPartitions(f).map(context_barrier).collect()[0]
        self.assertEqual(pids, [0, 1, 2, 3])

    def test_barrier_infos(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that BarrierTaskContext.getTaskInfos() returns a list of all task infos in the\n        barrier stage.\n        '
        rdd = self.sc.parallelize(range(10), 4)

        def f(iterator):
            if False:
                while True:
                    i = 10
            yield sum(iterator)
        taskInfos = rdd.barrier().mapPartitions(f).map(lambda x: BarrierTaskContext.get().getTaskInfos()).collect()
        self.assertTrue(len(taskInfos) == 4)
        self.assertTrue(len(taskInfos[0]) == 4)

    def test_context_get(self):
        if False:
            return 10
        '\n        Verify that TaskContext.get() works both in or not in a barrier stage.\n        '
        rdd = self.sc.parallelize(range(10), 4)

        def f(iterator):
            if False:
                while True:
                    i = 10
            taskContext = TaskContext.get()
            if isinstance(taskContext, BarrierTaskContext):
                yield (taskContext.partitionId() + 1)
            elif isinstance(taskContext, TaskContext):
                yield (taskContext.partitionId() + 2)
            else:
                yield (-1)
        result1 = rdd.mapPartitions(f).collect()
        self.assertTrue(result1 == [2, 3, 4, 5])
        result2 = rdd.barrier().mapPartitions(f).collect()
        self.assertTrue(result2 == [1, 2, 3, 4])

    def test_barrier_context_get(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that BarrierTaskContext.get() should only works in a barrier stage.\n        '
        rdd = self.sc.parallelize(range(10), 4)

        def f(iterator):
            if False:
                print('Hello World!')
            try:
                taskContext = BarrierTaskContext.get()
            except Exception:
                yield (-1)
            else:
                yield taskContext.partitionId()
        result1 = rdd.mapPartitions(f).collect()
        self.assertTrue(result1 == [-1, -1, -1, -1])
        result2 = rdd.barrier().mapPartitions(f).collect()
        self.assertTrue(result2 == [0, 1, 2, 3])

class TaskContextTestsWithWorkerReuse(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        class_name = self.__class__.__name__
        conf = SparkConf().set('spark.python.worker.reuse', 'true')
        self.sc = SparkContext('local[2]', class_name, conf=conf)

    def test_barrier_with_python_worker_reuse(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Regression test for SPARK-25921: verify that BarrierTaskContext.barrier() with\n        reused python worker.\n        '
        worker_pids = self.sc.parallelize(range(2), 2).map(lambda x: os.getpid()).collect()
        rdd = self.sc.parallelize(range(10), 2)

        def f(iterator):
            if False:
                while True:
                    i = 10
            yield sum(iterator)

        def context_barrier(x):
            if False:
                while True:
                    i = 10
            tc = BarrierTaskContext.get()
            time.sleep(random.randint(1, 5) * 2)
            tc.barrier()
            return (time.time(), os.getpid())
        result = rdd.barrier().mapPartitions(f).map(context_barrier).collect()
        times = list(map(lambda x: x[0], result))
        pids = list(map(lambda x: x[1], result))
        self.assertTrue(max(times) - min(times) < 2)
        for pid in pids:
            self.assertTrue(pid in worker_pids)

    def check_task_context_correct_with_python_worker_reuse(self):
        if False:
            return 10
        'Verify the task context correct when reused python worker'
        worker_pids = self.sc.parallelize(range(2), 2).map(lambda x: os.getpid()).collect()
        rdd = self.sc.parallelize(range(10), 2)

        def context(iterator):
            if False:
                print('Hello World!')
            tp = TaskContext.get().partitionId()
            try:
                bp = BarrierTaskContext.get().partitionId()
            except Exception:
                bp = -1
            yield (tp, bp, os.getpid())
        normal_result = rdd.mapPartitions(context).collect()
        (tps, bps, pids) = zip(*normal_result)
        self.assertTrue(tps == (0, 1))
        self.assertTrue(bps == (-1, -1))
        for pid in pids:
            self.assertTrue(pid in worker_pids)
        barrier_result = rdd.barrier().mapPartitions(context).collect()
        (tps, bps, pids) = zip(*barrier_result)
        self.assertTrue(tps == (0, 1))
        self.assertTrue(bps == (0, 1))
        for pid in pids:
            self.assertTrue(pid in worker_pids)
        normal_result2 = rdd.mapPartitions(context).collect()
        (tps, bps, pids) = zip(*normal_result2)
        self.assertTrue(tps == (0, 1))
        self.assertTrue(bps == (-1, -1))
        for pid in pids:
            self.assertTrue(pid in worker_pids)
        return True

    @eventually(catch_assertions=True)
    def test_task_context_correct_with_python_worker_reuse(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_task_context_correct_with_python_worker_reuse()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.sc.stop()

class TaskContextTestsWithResources(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        class_name = self.__class__.__name__
        self.tempFile = tempfile.NamedTemporaryFile(delete=False)
        self.tempFile.write(b'echo {\\"name\\": \\"gpu\\", \\"addresses\\": [\\"0\\"]}')
        self.tempFile.close()
        self.tempdir = tempfile.NamedTemporaryFile(delete=False)
        os.unlink(self.tempdir.name)
        os.chmod(self.tempFile.name, stat.S_IRWXU | stat.S_IXGRP | stat.S_IRGRP | stat.S_IROTH | stat.S_IXOTH)
        conf = SparkConf().set('spark.test.home', SPARK_HOME)
        conf = conf.set('spark.worker.resource.gpu.discoveryScript', self.tempFile.name)
        conf = conf.set('spark.worker.resource.gpu.amount', 1)
        conf = conf.set('spark.task.cpus', 2)
        conf = conf.set('spark.task.resource.gpu.amount', '1')
        conf = conf.set('spark.executor.resource.gpu.amount', '1')
        self.sc = SparkContext('local-cluster[2,2,1024]', class_name, conf=conf)

    def test_cpus(self):
        if False:
            return 10
        'Test the cpus are available.'
        rdd = self.sc.parallelize(range(10))
        cpus = rdd.map(lambda x: TaskContext.get().cpus()).take(1)[0]
        self.assertEqual(cpus, 2)

    def test_resources(self):
        if False:
            while True:
                i = 10
        'Test the resources are available.'
        rdd = self.sc.parallelize(range(10))
        resources = rdd.map(lambda x: TaskContext.get().resources()).take(1)[0]
        self.assertEqual(len(resources), 1)
        self.assertTrue('gpu' in resources)
        self.assertEqual(resources['gpu'].name, 'gpu')
        self.assertEqual(resources['gpu'].addresses, ['0'])

    def tearDown(self):
        if False:
            return 10
        os.unlink(self.tempFile.name)
        self.sc.stop()
if __name__ == '__main__':
    import unittest
    from pyspark.tests.test_taskcontext import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)