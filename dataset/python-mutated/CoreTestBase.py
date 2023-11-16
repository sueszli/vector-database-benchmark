import unittest
from coalib.core.Core import run

class CoreTestBase(unittest.TestCase):

    def execute_run(self, bears, cache=None, executor=None):
        if False:
            return 10
        '\n        Executes a coala run and returns the results.\n\n        This function has multiple ways to provide a different executor than\n        the default one (topmost item has highest precedence):\n\n        - Pass it via the ``executor`` parameter.\n        - Assign an executor class and the according instantiation arguments to\n          ``self.executor`` during ``setUp()``.\n\n        :param bears:\n            The bears to run.\n        :param cache:\n            A cache the bears can use to speed up runs. If ``None``, no cache\n            will be used.\n        :param executor:\n            The executor to run bears on.\n        :return:\n            A list of results.\n        '
        if executor is None:
            executor = getattr(self, 'executor', None)
            if executor is not None:
                (cls, args, kwargs) = self.executor
                executor = cls(*args, **kwargs)
        results = []

        def capture_results(result):
            if False:
                while True:
                    i = 10
            results.append(result)
        run(bears, capture_results, cache, executor)
        return results