import unittest
import hamcrest as hc
import apache_beam.runners.dataflow.internal.clients.dataflow as dataflow
from apache_beam.internal.gcp.json_value import to_json_value
from apache_beam.runners.dataflow.internal.clients.dataflow import message_matchers
try:
    from apitools.base.py import base_api
except ImportError:
    base_api = None

@unittest.skipIf(base_api is None, 'GCP dependencies are not installed')
class TestMatchers(unittest.TestCase):

    def test_structured_name_matcher_basic(self):
        if False:
            return 10
        metric_name = dataflow.MetricStructuredName()
        metric_name.name = 'metric1'
        metric_name.origin = 'origin2'
        matcher = message_matchers.MetricStructuredNameMatcher(name='metric1', origin='origin2')
        hc.assert_that(metric_name, hc.is_(matcher))
        with self.assertRaises(AssertionError):
            matcher = message_matchers.MetricStructuredNameMatcher(name='metric1', origin='origin1')
            hc.assert_that(metric_name, hc.is_(matcher))

    def test_metric_update_basic(self):
        if False:
            while True:
                i = 10
        metric_update = dataflow.MetricUpdate()
        metric_update.name = dataflow.MetricStructuredName()
        metric_update.name.name = 'metric1'
        metric_update.name.origin = 'origin1'
        metric_update.cumulative = False
        metric_update.kind = 'sum'
        metric_update.scalar = to_json_value(1, with_type=True)
        name_matcher = message_matchers.MetricStructuredNameMatcher(name='metric1', origin='origin1')
        matcher = message_matchers.MetricUpdateMatcher(name=name_matcher, kind='sum', scalar=1)
        hc.assert_that(metric_update, hc.is_(matcher))
        with self.assertRaises(AssertionError):
            matcher.kind = 'suma'
            hc.assert_that(metric_update, hc.is_(matcher))
if __name__ == '__main__':
    unittest.main()