from helpers import unittest
import luigi
import luigi.notifications
luigi.notifications.DEBUG = True

class LinearSum(luigi.Task):
    lo = luigi.IntParameter()
    hi = luigi.IntParameter()

    def requires(self):
        if False:
            i = 10
            return i + 15
        if self.hi > self.lo:
            return self.clone(hi=self.hi - 1)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.hi > self.lo:
            self.s = self.requires().s + self.f(self.hi - 1)
        else:
            self.s = 0
        self.complete = lambda : True

    def complete(self):
        if False:
            return 10
        return False

    def f(self, x):
        if False:
            print('Hello World!')
        return x

class PowerSum(LinearSum):
    p = luigi.IntParameter()

    def f(self, x):
        if False:
            return 10
        return x ** self.p

class CloneTest(unittest.TestCase):

    def test_args(self):
        if False:
            return 10
        t = LinearSum(lo=42, hi=45)
        self.assertEqual(t.param_args, (42, 45))
        self.assertEqual(t.param_kwargs, {'lo': 42, 'hi': 45})

    def test_recursion(self):
        if False:
            while True:
                i = 10
        t = LinearSum(lo=42, hi=45)
        luigi.build([t], local_scheduler=True)
        self.assertEqual(t.s, 42 + 43 + 44)

    def test_inheritance(self):
        if False:
            while True:
                i = 10
        t = PowerSum(lo=42, hi=45, p=2)
        luigi.build([t], local_scheduler=True)
        self.assertEqual(t.s, 42 ** 2 + 43 ** 2 + 44 ** 2)

    def test_inheritance_from_non_parameter(self):
        if False:
            while True:
                i = 10
        '\n        Cloning can pull non-source-parameters from source to target parameter.\n        '

        class SubTask(luigi.Task):
            lo = 1

            @property
            def hi(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 2
        t1 = SubTask()
        t2 = t1.clone(cls=LinearSum)
        self.assertEqual(t2.lo, 1)
        self.assertEqual(t2.hi, 2)