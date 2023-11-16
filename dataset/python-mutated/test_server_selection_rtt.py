"""Test the topology module."""
from __future__ import annotations
import json
import os
import sys
sys.path[0:0] = ['']
from test import unittest
from pymongo.read_preferences import MovingAverage
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'server_selection/rtt')

class TestAllScenarios(unittest.TestCase):
    pass

def create_test(scenario_def):
    if False:
        for i in range(10):
            print('nop')

    def run_scenario(self):
        if False:
            return 10
        moving_average = MovingAverage()
        if scenario_def['avg_rtt_ms'] != 'NULL':
            moving_average.add_sample(scenario_def['avg_rtt_ms'])
        if scenario_def['new_rtt_ms'] != 'NULL':
            moving_average.add_sample(scenario_def['new_rtt_ms'])
        self.assertAlmostEqual(moving_average.get(), scenario_def['new_avg_rtt'])
    return run_scenario

def create_tests():
    if False:
        i = 10
        return i + 15
    for (dirpath, _, filenames) in os.walk(_TEST_PATH):
        dirname = os.path.split(dirpath)[-1]
        for filename in filenames:
            with open(os.path.join(dirpath, filename)) as scenario_stream:
                scenario_def = json.load(scenario_stream)
            new_test = create_test(scenario_def)
            test_name = f'test_{dirname}_{os.path.splitext(filename)[0]}'
            new_test.__name__ = test_name
            setattr(TestAllScenarios, new_test.__name__, new_test)
create_tests()
if __name__ == '__main__':
    unittest.main()