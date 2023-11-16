from helpers import unittest
import luigi

class Factorial(luigi.Task):
    """ This calculates factorials *online* and does not write its results anywhere

    Demonstrates the ability for dependencies between Tasks and not just between their output.
    """
    n = luigi.IntParameter(default=100)

    def requires(self):
        if False:
            i = 10
            return i + 15
        if self.n > 1:
            return Factorial(self.n - 1)

    def run(self):
        if False:
            return 10
        if self.n > 1:
            self.value = self.n * self.requires().value
        else:
            self.value = 1
        self.complete = lambda : True

    def complete(self):
        if False:
            print('Hello World!')
        return False

class FactorialTest(unittest.TestCase):

    def test_invoke(self):
        if False:
            i = 10
            return i + 15
        luigi.build([Factorial(100)], local_scheduler=True)
        self.assertEqual(Factorial(42).value, 1405006117752879898543142606244511569936384000000000)