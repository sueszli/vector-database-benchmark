from __future__ import annotations
import os
import sys
from typing import Any
sys.path[0:0] = ['']
from test import unittest
from test.unified_format import MatchEvaluatorUtil, generate_test_classes
from bson import ObjectId
_TEST_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unified-test-format')
globals().update(generate_test_classes(os.path.join(_TEST_PATH, 'valid-pass'), module=__name__, class_name_prefix='UnifiedTestFormat', expected_failures=['Client side error in command starting transaction'], RUN_ON_SERVERLESS=False))
globals().update(generate_test_classes(os.path.join(_TEST_PATH, 'valid-fail'), module=__name__, class_name_prefix='UnifiedTestFormat', bypass_test_generation_errors=True, expected_failures=['.*'], RUN_ON_SERVERLESS=False))

class TestMatchEvaluatorUtil(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.match_evaluator = MatchEvaluatorUtil(self)

    def test_unsetOrMatches(self):
        if False:
            i = 10
            return i + 15
        spec: dict[str, Any] = {'$$unsetOrMatches': {'y': {'$$unsetOrMatches': 2}}}
        for actual in [{}, {'y': 2}, None]:
            self.match_evaluator.match_result(spec, actual)
        spec = {'x': {'$$unsetOrMatches': {'y': {'$$unsetOrMatches': 2}}}}
        for actual in [{}, {'x': {}}, {'x': {'y': 2}}]:
            self.match_evaluator.match_result(spec, actual)
        spec = {'y': {'$$unsetOrMatches': {'$$exists': True}}}
        self.match_evaluator.match_result(spec, {})
        self.match_evaluator.match_result(spec, {'y': 2})
        self.match_evaluator.match_result(spec, {'x': 1})
        self.match_evaluator.match_result(spec, {'y': {}})

    def test_type(self):
        if False:
            print('Hello World!')
        self.match_evaluator.match_result({'operationType': 'insert', 'ns': {'db': 'change-stream-tests', 'coll': 'test'}, 'fullDocument': {'_id': {'$$type': 'objectId'}, 'x': 1}}, {'operationType': 'insert', 'fullDocument': {'_id': ObjectId('5fc93511ac93941052098f0c'), 'x': 1}, 'ns': {'db': 'change-stream-tests', 'coll': 'test'}})
if __name__ == '__main__':
    unittest.main()