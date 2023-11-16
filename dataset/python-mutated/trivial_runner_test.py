import logging
import unittest
import apache_beam as beam
from apache_beam.runners.trivial_runner import TrivialRunner
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class TrivialRunnerTest(unittest.TestCase):

    def test_trivial(self):
        if False:
            while True:
                i = 10
        with beam.Pipeline(runner=TrivialRunner()) as p:
            _ = p | beam.Impulse()

    def test_assert_that(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'Failed assert'):
            with beam.Pipeline(runner=TrivialRunner()) as p:
                assert_that(p | beam.Impulse(), equal_to(['a']))

    def test_impulse(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(runner=TrivialRunner()) as p:
            assert_that(p | beam.Impulse(), equal_to([b'']))

    def test_create(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(runner=TrivialRunner()) as p:
            assert_that(p | beam.Create(['a', 'b']), equal_to(['a', 'b']))

    def test_flatten(self):
        if False:
            i = 10
            return i + 15
        with beam.Pipeline(runner=TrivialRunner()) as p:
            ab = p | 'AB' >> beam.Create(['a', 'b'], reshuffle=False)
            c = p | 'C' >> beam.Create(['c'], reshuffle=False)
            assert_that((ab, c, c) | beam.Flatten(), equal_to(['a', 'b', 'c', 'c']))

    def test_map(self):
        if False:
            print('Hello World!')
        with beam.Pipeline(runner=TrivialRunner()) as p:
            assert_that(p | beam.Create(['a', 'b'], reshuffle=False) | beam.Map(str.upper), equal_to(['A', 'B']))

    def test_gbk(self):
        if False:
            return 10
        with beam.Pipeline(runner=TrivialRunner()) as p:
            result = p | beam.Create([('a', 1), ('b', 2), ('b', 3)], reshuffle=False) | beam.GroupByKey() | beam.MapTuple(lambda k, vs: (k, sorted(vs)))
            assert_that(result, equal_to([('a', [1]), ('b', [2, 3])]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()