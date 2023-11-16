"""Unit tests for the PValue and PCollection classes."""
import unittest
from apache_beam.pvalue import AsSingleton
from apache_beam.pvalue import PValue
from apache_beam.pvalue import Row
from apache_beam.pvalue import TaggedOutput
from apache_beam.testing.test_pipeline import TestPipeline

class PValueTest(unittest.TestCase):

    def test_pvalue_expected_arguments(self):
        if False:
            i = 10
            return i + 15
        pipeline = TestPipeline()
        value = PValue(pipeline)
        self.assertEqual(pipeline, value.pipeline)

    def test_assingleton_multi_element(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'PCollection of size 2 with more than one element accessed as a singleton view. First two elements encountered are "1", "2".'):
            AsSingleton._from_runtime_iterable([1, 2], {})

class TaggedValueTest(unittest.TestCase):

    def test_passed_tuple_as_tag(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(TypeError, 'Attempting to create a TaggedOutput with non-string tag \\(1, 2, 3\\)'):
            TaggedOutput((1, 2, 3), 'value')

class RowTest(unittest.TestCase):

    def test_row_eq(self):
        if False:
            i = 10
            return i + 15
        row = Row(a=1, b=2)
        same = Row(a=1, b=2)
        self.assertEqual(row, same)

    def test_trailing_column_row_neq(self):
        if False:
            while True:
                i = 10
        row = Row(a=1, b=2)
        trail = Row(a=1, b=2, c=3)
        self.assertNotEqual(row, trail)

    def test_row_comparison_respects_element_order(self):
        if False:
            while True:
                i = 10
        row = Row(a=1, b=2)
        different = Row(b=2, a=1)
        self.assertNotEqual(row, different)
if __name__ == '__main__':
    unittest.main()