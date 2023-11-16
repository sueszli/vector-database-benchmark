"""Unit tests for DataflowDistributionCounter
When Cython is available, unit tests will test on cythonized module,
otherwise, test on pure python module
"""
import unittest
from mock import Mock
from apache_beam.transforms import DataflowDistributionCounter
INT64_MAX = 2 ** 63 - 1

class DataflowDistributionAccumulatorTest(unittest.TestCase):

    def test_calculate_bucket_index_with_input_0(self):
        if False:
            i = 10
            return i + 15
        counter = DataflowDistributionCounter()
        index = counter.calculate_bucket_index(0)
        self.assertEqual(index, 0)

    def test_calculate_bucket_index_within_max_long(self):
        if False:
            print('Hello World!')
        counter = DataflowDistributionCounter()
        bucket = 1
        power_of_ten = 1
        while power_of_ten <= INT64_MAX:
            for multiplier in [1, 2, 5]:
                value = multiplier * power_of_ten
                actual_bucket = counter.calculate_bucket_index(value - 1)
                self.assertEqual(actual_bucket, bucket - 1)
                bucket += 1
            power_of_ten *= 10

    def test_add_input(self):
        if False:
            return 10
        counter = DataflowDistributionCounter()
        expected_buckets = [1, 3, 0, 0, 0, 0, 0, 0, 1, 1]
        expected_sum = 1510
        expected_first_bucket_index = 1
        expected_count = 6
        expected_min = 1
        expected_max = 1000
        for element in [1, 500, 2, 3, 1000, 4]:
            counter.add_input(element)
        histogram = Mock(firstBucketOffset=None, bucketCounts=None)
        counter.translate_to_histogram(histogram)
        self.assertEqual(counter.sum, expected_sum)
        self.assertEqual(counter.count, expected_count)
        self.assertEqual(counter.min, expected_min)
        self.assertEqual(counter.max, expected_max)
        self.assertEqual(histogram.firstBucketOffset, expected_first_bucket_index)
        self.assertEqual(histogram.bucketCounts, expected_buckets)

    def test_translate_to_histogram_with_input_0(self):
        if False:
            i = 10
            return i + 15
        counter = DataflowDistributionCounter()
        counter.add_input(0)
        histogram = Mock(firstBucketOffset=None, bucketCounts=None)
        counter.translate_to_histogram(histogram)
        self.assertEqual(histogram.firstBucketOffset, 0)
        self.assertEqual(histogram.bucketCounts, [1])

    def test_translate_to_histogram_with_max_input(self):
        if False:
            for i in range(10):
                print('nop')
        counter = DataflowDistributionCounter()
        counter.add_input(INT64_MAX)
        histogram = Mock(firstBucketOffset=None, bucketCounts=None)
        counter.translate_to_histogram(histogram)
        self.assertEqual(histogram.firstBucketOffset, 57)
        self.assertEqual(histogram.bucketCounts, [1])
if __name__ == '__main__':
    unittest.main()