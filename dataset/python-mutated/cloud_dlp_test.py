"""Unit tests for Google Cloud Video Intelligence API transforms."""
import logging
import unittest
import mock
import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.testing.test_pipeline import TestPipeline
try:
    from google.cloud import dlp_v2
except ImportError:
    dlp_v2 = None
else:
    from apache_beam.ml.gcp.cloud_dlp import InspectForDetails
    from apache_beam.ml.gcp.cloud_dlp import MaskDetectedDetails
    from apache_beam.ml.gcp.cloud_dlp import _DeidentifyFn
    from apache_beam.ml.gcp.cloud_dlp import _InspectFn
    from google.cloud.dlp_v2.types import dlp
_LOGGER = logging.getLogger(__name__)

@unittest.skipIf(dlp_v2 is None, 'GCP dependencies are not installed')
class TestDeidentifyText(unittest.TestCase):

    def test_exception_raised_when_no_config_is_provided(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            with TestPipeline() as p:
                p | MaskDetectedDetails()

@unittest.skipIf(dlp_v2 is None, 'GCP dependencies are not installed')
class TestDeidentifyFn(unittest.TestCase):

    def test_deidentify_called(self):
        if False:
            while True:
                i = 10

        class ClientMock(object):

            def deidentify_content(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                dlp.DeidentifyContentRequest(kwargs['request'])
                called = Metrics.counter('test_deidentify_text', 'called')
                called.inc()
                operation = mock.Mock()
                item = mock.Mock()
                item.value = [None]
                operation.item = item
                return operation

            def common_project_path(self, *args):
                if False:
                    while True:
                        i = 10
                return 'test'
        with mock.patch('google.cloud.dlp_v2.DlpServiceClient', ClientMock):
            p = TestPipeline()
            config = {'deidentify_config': {'info_type_transformations': {'transformations': [{'primitive_transformation': {'character_mask_config': {'masking_character': '#'}}}]}}}
            p | beam.Create(['mary.sue@example.com', 'john.doe@example.com']) | beam.ParDo(_DeidentifyFn(config=config))
            result = p.run()
            result.wait_until_finish()
        called = result.metrics().query()['counters'][0]
        self.assertEqual(called.result, 2)

@unittest.skipIf(dlp_v2 is None, 'GCP dependencies are not installed')
class TestInspectText(unittest.TestCase):

    def test_exception_raised_then_no_config_provided(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            with TestPipeline() as p:
                p | InspectForDetails()

@unittest.skipIf(dlp_v2 is None, 'GCP dependencies are not installed')
class TestInspectFn(unittest.TestCase):

    def test_inspect_called(self):
        if False:
            print('Hello World!')

        class ClientMock(object):

            def inspect_content(self, *args, **kwargs):
                if False:
                    return 10
                dlp.InspectContentRequest(kwargs['request'])
                called = Metrics.counter('test_inspect_text', 'called')
                called.inc()
                operation = mock.Mock()
                operation.result = mock.Mock()
                operation.result.findings = [None]
                return operation

            def common_project_path(self, *args):
                if False:
                    i = 10
                    return i + 15
                return 'test'
        with mock.patch('google.cloud.dlp_v2.DlpServiceClient', ClientMock):
            p = TestPipeline()
            config = {'inspect_config': {'info_types': [{'name': 'EMAIL_ADDRESS'}]}}
            p | beam.Create(['mary.sue@example.com', 'john.doe@example.com']) | beam.ParDo(_InspectFn(config=config))
            result = p.run()
            result.wait_until_finish()
            called = result.metrics().query()['counters'][0]
            self.assertEqual(called.result, 2)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()