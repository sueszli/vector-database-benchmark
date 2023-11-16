"""Unit tests for the Create and _CreateSource classes."""
import logging
import unittest
from apache_beam import Create
from apache_beam import coders
from apache_beam.coders import FastPrimitivesCoder
from apache_beam.internal import pickler
from apache_beam.io import source_test_utils
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class CreateTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.coder = FastPrimitivesCoder()

    def test_create_transform(self):
        if False:
            while True:
                i = 10
        with TestPipeline() as p:
            assert_that(p | 'Empty' >> Create([]), equal_to([]), label='empty')
            assert_that(p | 'One' >> Create([None]), equal_to([None]), label='one')
            assert_that(p | Create(list(range(10))), equal_to(list(range(10))))

    def test_create_source_read(self):
        if False:
            i = 10
            return i + 15
        self.check_read([], self.coder)
        self.check_read([1], self.coder)
        self.check_read(list(range(10)), self.coder)

    def check_read(self, values, coder):
        if False:
            return 10
        source = Create._create_source_from_iterable(values, coder)
        read_values = source_test_utils.read_from_source(source)
        self.assertEqual(sorted(values), sorted(read_values))

    def test_create_source_read_with_initial_splits(self):
        if False:
            return 10
        self.check_read_with_initial_splits([], self.coder, num_splits=2)
        self.check_read_with_initial_splits([1], self.coder, num_splits=2)
        values = list(range(8))
        self.check_read_with_initial_splits(values, self.coder, num_splits=1)
        self.check_read_with_initial_splits(values, self.coder, num_splits=0.5)
        self.check_read_with_initial_splits(values, self.coder, num_splits=3)
        self.check_read_with_initial_splits(values, self.coder, num_splits=4)
        self.check_read_with_initial_splits(values, self.coder, num_splits=len(values))
        self.check_read_with_initial_splits(values, self.coder, num_splits=30)

    def check_read_with_initial_splits(self, values, coder, num_splits):
        if False:
            i = 10
            return i + 15
        'A test that splits the given source into `num_splits` and verifies that\n    the data read from original source is equal to the union of the data read\n    from the split sources.\n    '
        source = Create._create_source_from_iterable(values, coder)
        desired_bundle_size = source._total_size // num_splits
        splits = source.split(desired_bundle_size)
        splits_info = [(split.source, split.start_position, split.stop_position) for split in splits]
        source_test_utils.assert_sources_equal_reference_source((source, None, None), splits_info)

    def test_create_source_read_reentrant(self):
        if False:
            for i in range(10):
                print('nop')
        source = Create._create_source_from_iterable(range(9), self.coder)
        source_test_utils.assert_reentrant_reads_succeed((source, None, None))

    def test_create_source_read_reentrant_with_initial_splits(self):
        if False:
            for i in range(10):
                print('nop')
        source = Create._create_source_from_iterable(range(24), self.coder)
        for split in source.split(desired_bundle_size=5):
            source_test_utils.assert_reentrant_reads_succeed((split.source, split.start_position, split.stop_position))

    def test_create_source_dynamic_splitting(self):
        if False:
            i = 10
            return i + 15
        source = Create._create_source_from_iterable(range(2), self.coder)
        source_test_utils.assert_split_at_fraction_exhaustive(source)
        source = Create._create_source_from_iterable(range(11), self.coder)
        source_test_utils.assert_split_at_fraction_exhaustive(source, perform_multi_threaded_test=True)

    def test_create_source_progress(self):
        if False:
            print('Hello World!')
        num_values = 10
        source = Create._create_source_from_iterable(range(num_values), self.coder)
        splits = [split for split in source.split(desired_bundle_size=100)]
        assert len(splits) == 1
        fraction_consumed_report = []
        split_points_report = []
        range_tracker = splits[0].source.get_range_tracker(splits[0].start_position, splits[0].stop_position)
        for _ in splits[0].source.read(range_tracker):
            fraction_consumed_report.append(range_tracker.fraction_consumed())
            split_points_report.append(range_tracker.split_points())
        self.assertEqual([float(i) / num_values for i in range(num_values)], fraction_consumed_report)
        expected_split_points_report = [(i - 1, num_values - (i - 1)) for i in range(1, num_values + 1)]
        self.assertEqual(expected_split_points_report, split_points_report)

    def test_create_uses_coder_for_pickling(self):
        if False:
            while True:
                i = 10
        coders.registry.register_coder(_Unpicklable, _UnpicklableCoder)
        create = Create([_Unpicklable(1), _Unpicklable(2), _Unpicklable(3)])
        unpickled_create = pickler.loads(pickler.dumps(create))
        self.assertEqual(sorted(create.values, key=lambda v: v.value), sorted(unpickled_create.values, key=lambda v: v.value))
        with self.assertRaises(NotImplementedError):
            create_mixed_types = Create([_Unpicklable(1), 2])
            pickler.dumps(create_mixed_types)

class _Unpicklable(object):

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.value == other.value

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class _UnpicklableCoder(coders.Coder):

    def encode(self, value):
        if False:
            while True:
                i = 10
        return str(value.value).encode()

    def decode(self, encoded):
        if False:
            print('Hello World!')
        return _Unpicklable(int(encoded.decode()))

    def to_type_hint(self):
        if False:
            print('Hello World!')
        return _Unpicklable
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()