from __future__ import absolute_import
from unittest2 import TestCase
from st2common.expressions.functions import time

class TestTimeJinjaFilters(TestCase):

    def test_to_human_time_from_seconds(self):
        if False:
            while True:
                i = 10
        self.assertEqual('0s', time.to_human_time_from_seconds(seconds=0))
        self.assertEqual('0.1μs', time.to_human_time_from_seconds(seconds=0.1))
        self.assertEqual('56s', time.to_human_time_from_seconds(seconds=56))
        self.assertEqual('56s', time.to_human_time_from_seconds(seconds=56.2))
        self.assertEqual('7m36s', time.to_human_time_from_seconds(seconds=456))
        self.assertEqual('1h16m0s', time.to_human_time_from_seconds(seconds=4560))
        self.assertEqual('1y12d16h36m37s', time.to_human_time_from_seconds(seconds=45678997))
        self.assertRaises(AssertionError, time.to_human_time_from_seconds, seconds='stuff')