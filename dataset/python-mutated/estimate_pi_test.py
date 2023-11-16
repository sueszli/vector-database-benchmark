"""Test for the estimate_pi example."""
import logging
import unittest
from apache_beam.examples.complete import estimate_pi
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import BeamAssertException
from apache_beam.testing.util import assert_that

def in_between(lower, upper):
    if False:
        return 10

    def _in_between(actual):
        if False:
            print('Hello World!')
        (_, _, estimate) = actual[0]
        if estimate < lower or estimate > upper:
            raise BeamAssertException('Failed assert: %f not in [%f, %f]' % (estimate, lower, upper))
    return _in_between

class EstimatePiTest(unittest.TestCase):

    def test_basics(self):
        if False:
            while True:
                i = 10
        with TestPipeline() as p:
            result = p | 'Estimate' >> estimate_pi.EstimatePiTransform(5000)
            assert_that(result, in_between(3.125, 3.155))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()