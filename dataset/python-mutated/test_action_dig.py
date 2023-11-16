from __future__ import absolute_import
from st2tests.base import BaseActionTestCase
from dig import DigAction

class DigActionTestCase(BaseActionTestCase):
    action_cls = DigAction

    def test_run_with_empty_hostname(self):
        if False:
            while True:
                i = 10
        action = self.get_action_instance()
        result = action.run(rand=False, count=0, nameserver=None, hostname='', queryopts='short')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_run_with_empty_queryopts(self):
        if False:
            print('Hello World!')
        action = self.get_action_instance()
        results = action.run(rand=False, count=0, nameserver=None, hostname='google.com', queryopts='')
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_run_with_empty_querytype(self):
        if False:
            print('Hello World!')
        action = self.get_action_instance()
        results = action.run(rand=False, count=0, nameserver=None, hostname='google.com', queryopts='short', querytype='')
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_run(self):
        if False:
            while True:
                i = 10
        action = self.get_action_instance()
        results = action.run(rand=False, count=0, nameserver=None, hostname='google.com', queryopts='short', querytype='A')
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)