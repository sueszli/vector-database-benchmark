"""Tests for bigquery_io_metadata."""
import logging
import unittest
from apache_beam.io.gcp import bigquery_io_metadata

class BigqueryIoMetadataTest(unittest.TestCase):

    def test_is_valid_cloud_label_value(self):
        if False:
            print('Hello World!')
        test_str = '2020-06-29_15_26_09-12838749047888422749'
        self.assertTrue(bigquery_io_metadata._is_valid_cloud_label_value(test_str))
        test_str = '0'
        self.assertTrue(bigquery_io_metadata._is_valid_cloud_label_value(test_str))
        test_str = '0123456789abcdefghij0123456789abcdefghij0123456789abcdefghij012'
        self.assertTrue(bigquery_io_metadata._is_valid_cloud_label_value(test_str))
        test_str = 'abcdefghijklmnopqrstuvwxyz'
        for test_char in test_str:
            self.assertTrue(bigquery_io_metadata._is_valid_cloud_label_value(test_char))
        test_str = ''
        self.assertFalse(bigquery_io_metadata._is_valid_cloud_label_value(test_str))
        test_str = '0123456789abcdefghij0123456789abcdefghij0123456789abcdefghij0123'
        self.assertFalse(bigquery_io_metadata._is_valid_cloud_label_value(test_str))
        test_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for test_char in test_str:
            self.assertFalse(bigquery_io_metadata._is_valid_cloud_label_value(test_char))
        test_str = '!@#$%^&*()+=[{]};:\'"\\|,<.>?/`~'
        for test_char in test_str:
            self.assertFalse(bigquery_io_metadata._is_valid_cloud_label_value(test_char))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()