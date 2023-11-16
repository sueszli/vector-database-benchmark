from random import Random
from robot.model import SuiteVisitor

class Randomizer(SuiteVisitor):

    def __init__(self, randomize_suites=True, randomize_tests=True, seed=None):
        if False:
            while True:
                i = 10
        self.randomize_suites = randomize_suites
        self.randomize_tests = randomize_tests
        self.seed = seed
        self._shuffle = Random(seed).shuffle

    def start_suite(self, suite):
        if False:
            i = 10
            return i + 15
        if not self.randomize_suites and (not self.randomize_tests):
            return False
        if self.randomize_suites:
            self._shuffle(suite.suites)
        if self.randomize_tests:
            self._shuffle(suite.tests)
        if not suite.parent:
            suite.metadata['Randomized'] = self._get_message()

    def _get_message(self):
        if False:
            for i in range(10):
                print('nop')
        possibilities = {(True, True): 'Suites and tests', (True, False): 'Suites', (False, True): 'Tests'}
        randomized = (self.randomize_suites, self.randomize_tests)
        return '%s (seed %s)' % (possibilities[randomized], self.seed)

    def visit_test(self, test):
        if False:
            return 10
        pass

    def visit_keyword(self, kw):
        if False:
            i = 10
            return i + 15
        pass