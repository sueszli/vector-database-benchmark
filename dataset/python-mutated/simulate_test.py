from helpers import unittest
import luigi
from luigi.contrib.simulate import RunAnywayTarget
from multiprocessing import Process
import os
import tempfile

def temp_dir():
    if False:
        print('Hello World!')
    return os.path.join(tempfile.gettempdir(), 'luigi-simulate')

def is_writable():
    if False:
        print('Hello World!')
    d = temp_dir()
    fn = os.path.join(d, 'luigi-simulate-write-test')
    exists = True
    try:
        try:
            os.makedirs(d)
        except OSError:
            pass
        open(fn, 'w').close()
        os.remove(fn)
    except BaseException:
        exists = False
    return unittest.skipIf(not exists, "Can't write to temporary directory")

class TaskA(luigi.Task):
    i = luigi.IntParameter(default=0)

    def output(self):
        if False:
            print('Hello World!')
        return RunAnywayTarget(self)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        fn = os.path.join(temp_dir(), 'luigi-simulate-test.tmp')
        try:
            os.makedirs(os.path.dirname(fn))
        except OSError:
            pass
        with open(fn, 'a') as f:
            f.write('{0}={1}\n'.format(self.__class__.__name__, self.i))
        self.output().done()

class TaskB(TaskA):

    def requires(self):
        if False:
            i = 10
            return i + 15
        return TaskA(i=10)

class TaskC(TaskA):

    def requires(self):
        if False:
            i = 10
            return i + 15
        return TaskA(i=5)

class TaskD(TaskA):

    def requires(self):
        if False:
            print('Hello World!')
        return [TaskB(), TaskC(), TaskA(i=20)]

class TaskWrap(luigi.WrapperTask):

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        return [TaskA(), TaskD()]

def reset():
    if False:
        for i in range(10):
            print('nop')
    t = TaskA().output()
    with t.unique.get_lock():
        t.unique.value = 0

class RunAnywayTargetTest(unittest.TestCase):

    @is_writable()
    def test_output(self):
        if False:
            print('Hello World!')
        reset()
        fn = os.path.join(temp_dir(), 'luigi-simulate-test.tmp')
        luigi.build([TaskWrap()], local_scheduler=True)
        with open(fn, 'r') as f:
            data = f.read().strip().split('\n')
        data.sort()
        reference = ['TaskA=0', 'TaskA=10', 'TaskA=20', 'TaskA=5', 'TaskB=0', 'TaskC=0', 'TaskD=0']
        reference.sort()
        os.remove(fn)
        self.assertEqual(data, reference)

    @is_writable()
    def test_output_again(self):
        if False:
            while True:
                i = 10
        p = Process(target=self.test_output)
        p.start()
        p.join()