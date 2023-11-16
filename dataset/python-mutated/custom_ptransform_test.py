"""Tests for the various custom Count implementation examples."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.cookbook import custom_ptransform
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class CustomCountTest(unittest.TestCase):
    WORDS = ['CAT', 'DOG', 'CAT', 'CAT', 'DOG']

    def create_content_input_file(self, path, contents):
        if False:
            while True:
                i = 10
        logging.info('Creating temp file: %s', path)
        with open(path, 'w') as f:
            f.write(contents)

    def test_count1(self):
        if False:
            return 10
        self.run_pipeline(custom_ptransform.Count1())

    def test_count2(self):
        if False:
            return 10
        self.run_pipeline(custom_ptransform.Count2())

    def test_count3(self):
        if False:
            print('Hello World!')
        factor = 2
        self.run_pipeline(custom_ptransform.Count3(factor), factor=factor)

    def run_pipeline(self, count_implementation, factor=1):
        if False:
            for i in range(10):
                print('nop')
        with TestPipeline() as p:
            words = p | beam.Create(self.WORDS)
            result = words | count_implementation
            assert_that(result, equal_to([('CAT', 3 * factor), ('DOG', 2 * factor)]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()