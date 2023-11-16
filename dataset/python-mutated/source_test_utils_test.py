import logging
import tempfile
import unittest
from apache_beam.io import source_test_utils
from apache_beam.io.filebasedsource_test import LineSource

class SourceTestUtilsTest(unittest.TestCase):

    def _create_file_with_data(self, lines):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(lines, list)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            for line in lines:
                f.write(line + b'\n')
            return f.name

    def _create_data(self, num_lines):
        if False:
            print('Hello World!')
        return [b'line ' + str(i).encode('latin1') for i in range(num_lines)]

    def _create_source(self, data):
        if False:
            for i in range(10):
                print('nop')
        source = LineSource(self._create_file_with_data(data))
        for bundle in source.split(float('inf')):
            return bundle.source

    def test_read_from_source(self):
        if False:
            return 10
        data = self._create_data(100)
        source = self._create_source(data)
        self.assertCountEqual(data, source_test_utils.read_from_source(source, None, None))

    def test_source_equals_reference_source(self):
        if False:
            i = 10
            return i + 15
        data = self._create_data(100)
        reference_source = self._create_source(data)
        sources_info = [(split.source, split.start_position, split.stop_position) for split in reference_source.split(desired_bundle_size=50)]
        if len(sources_info) < 2:
            raise ValueError('Test is too trivial since splitting only generated %dbundles. Please adjust the test so that at least two splits get generated.' % len(sources_info))
        source_test_utils.assert_sources_equal_reference_source((reference_source, None, None), sources_info)

    def test_split_at_fraction_successful(self):
        if False:
            print('Hello World!')
        data = self._create_data(100)
        source = self._create_source(data)
        result1 = source_test_utils.assert_split_at_fraction_behavior(source, 10, 0.5, source_test_utils.ExpectedSplitOutcome.MUST_SUCCEED_AND_BE_CONSISTENT)
        result2 = source_test_utils.assert_split_at_fraction_behavior(source, 20, 0.5, source_test_utils.ExpectedSplitOutcome.MUST_SUCCEED_AND_BE_CONSISTENT)
        self.assertEqual(result1, result2)
        self.assertEqual(100, result1[0] + result1[1])
        result3 = source_test_utils.assert_split_at_fraction_behavior(source, 30, 0.8, source_test_utils.ExpectedSplitOutcome.MUST_SUCCEED_AND_BE_CONSISTENT)
        result4 = source_test_utils.assert_split_at_fraction_behavior(source, 50, 0.8, source_test_utils.ExpectedSplitOutcome.MUST_SUCCEED_AND_BE_CONSISTENT)
        self.assertEqual(result3, result4)
        self.assertEqual(100, result3[0] + result4[1])
        self.assertTrue(result1[0] < result3[0])
        self.assertTrue(result1[1] > result3[1])

    def test_split_at_fraction_fails(self):
        if False:
            i = 10
            return i + 15
        data = self._create_data(100)
        source = self._create_source(data)
        result = source_test_utils.assert_split_at_fraction_behavior(source, 90, 0.1, source_test_utils.ExpectedSplitOutcome.MUST_FAIL)
        self.assertEqual(result[0], 100)
        self.assertEqual(result[1], -1)
        with self.assertRaises(ValueError):
            source_test_utils.assert_split_at_fraction_behavior(source, 10, 0.5, source_test_utils.ExpectedSplitOutcome.MUST_FAIL)

    def test_split_at_fraction_binary(self):
        if False:
            return 10
        data = self._create_data(100)
        source = self._create_source(data)
        stats = source_test_utils.SplitFractionStatistics([], [])
        source_test_utils.assert_split_at_fraction_binary(source, data, 10, 0.5, None, 0.8, None, stats)
        self.assertTrue(stats.successful_fractions)
        self.assertTrue(stats.non_trivial_fractions)

    def test_split_at_fraction_exhaustive(self):
        if False:
            while True:
                i = 10
        data = self._create_data(10)
        source = self._create_source(data)
        source_test_utils.assert_split_at_fraction_exhaustive(source)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()