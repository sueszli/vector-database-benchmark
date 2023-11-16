import abc
from helpers import unittest
import luigi

class AbstractTask(luigi.Task):
    k = luigi.IntParameter()

    @property
    @abc.abstractmethod
    def foo(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abc.abstractmethod
    def helper_function(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def run(self):
        if False:
            while True:
                i = 10
        return ','.join([self.foo, self.helper_function()])

class Implementation(AbstractTask):

    @property
    def foo(self):
        if False:
            print('Hello World!')
        return 'bar'

    def helper_function(self):
        if False:
            return 10
        return 'hello' * self.k

class AbstractSubclassTest(unittest.TestCase):

    def test_instantiate_abstract(self):
        if False:
            while True:
                i = 10

        def try_instantiate():
            if False:
                while True:
                    i = 10
            AbstractTask(k=1)
        self.assertRaises(TypeError, try_instantiate)

    def test_instantiate(self):
        if False:
            while True:
                i = 10
        self.assertEqual('bar,hellohello', Implementation(k=2).run())