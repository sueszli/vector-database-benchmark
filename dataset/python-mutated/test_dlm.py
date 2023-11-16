import unittest
from troposphere.dlm import CreateRule

class TestDlmCreateRule(unittest.TestCase):

    def test_createrule_interval_bad_value(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Interval must be one of'):
            CreateRule('CreateRule', Interval=25)

    def test_createrule_intervalunit_bad_value(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'Interval unit must be one of'):
            CreateRule('CreateRule', Interval=24, IntervalUnit='HOUR')

    def test_createrule(self):
        if False:
            return 10
        CreateRule('CreateRule', Interval=24, IntervalUnit='HOURS')