import unittest
import troposphere.cloudwatch as cloudwatch
from troposphere.cloudwatch import Dashboard

class TestModel(unittest.TestCase):

    def test_dashboard(self):
        if False:
            return 10
        dashboard = Dashboard('dashboard', DashboardBody='{"a": "b"}')
        dashboard.validate()
        dashboard = Dashboard('dashboard', DashboardBody='{"a: "b"}')
        with self.assertRaises(ValueError):
            dashboard.validate()
        d = {'c': 'd'}
        dashboard = Dashboard('dashboard', DashboardBody=d)
        dashboard.validate()
        self.assertEqual(dashboard.properties['DashboardBody'], '{"c": "d"}')
        with self.assertRaises(TypeError):
            dashboard = Dashboard('dashboard', DashboardBody=1)

class TestCloudWatchValidators(unittest.TestCase):

    def test_validate_units(self):
        if False:
            for i in range(10):
                print('nop')
        cloudwatch.validate_unit('Bytes/Second')
        for bad_unit in ['Minutes', 'Bytes/Minute', 'Bits/Hour', '']:
            with self.assertRaisesRegex(ValueError, 'must be one of'):
                cloudwatch.validate_unit(bad_unit)

    def test_validate_treat_missing_data(self):
        if False:
            return 10
        cloudwatch.validate_treat_missing_data('missing')
        for bad_value in ['exists', 'notMissing', '']:
            with self.assertRaisesRegex(ValueError, 'must be one of'):
                cloudwatch.validate_treat_missing_data(bad_value)
if __name__ == '__main__':
    unittest.main()