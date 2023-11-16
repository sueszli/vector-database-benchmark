from helpers import unittest
import luigi
import luigi.interface
from luigi.mock import MockTarget

class Fib(luigi.Task):
    n = luigi.IntParameter(default=100)

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        if self.n >= 2:
            return [Fib(self.n - 1), Fib(self.n - 2)]
        else:
            return []

    def output(self):
        if False:
            print('Hello World!')
        return MockTarget('/tmp/fib_%d' % self.n)

    def run(self):
        if False:
            return 10
        if self.n == 0:
            s = 0
        elif self.n == 1:
            s = 1
        else:
            s = 0
            for input in self.input():
                for line in input.open('r'):
                    s += int(line.strip())
        f = self.output().open('w')
        f.write('%d\n' % s)
        f.close()

class FibTestBase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        MockTarget.fs.clear()

class FibTest(FibTestBase):

    def test_invoke(self):
        if False:
            while True:
                i = 10
        luigi.build([Fib(100)], local_scheduler=True)
        self.assertEqual(MockTarget.fs.get_data('/tmp/fib_10'), b'55\n')
        self.assertEqual(MockTarget.fs.get_data('/tmp/fib_100'), b'354224848179261915075\n')

    def test_cmdline(self):
        if False:
            i = 10
            return i + 15
        luigi.run(['--local-scheduler', '--no-lock', 'Fib', '--n', '100'])
        self.assertEqual(MockTarget.fs.get_data('/tmp/fib_10'), b'55\n')
        self.assertEqual(MockTarget.fs.get_data('/tmp/fib_100'), b'354224848179261915075\n')

    def test_build_internal(self):
        if False:
            while True:
                i = 10
        luigi.build([Fib(100)], local_scheduler=True)
        self.assertEqual(MockTarget.fs.get_data('/tmp/fib_10'), b'55\n')
        self.assertEqual(MockTarget.fs.get_data('/tmp/fib_100'), b'354224848179261915075\n')