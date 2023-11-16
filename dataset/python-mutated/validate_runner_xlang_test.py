"""
###########################################################
Runner Validation Test Suite for Cross-language Transforms
###########################################################
 As per Beams's Portability Framework design, Cross-language transforms
 should work out of the box. In spite of this, there always exists a
 possibility of rough edges existing. It could be caused due to unpolished
 implementation of any part of the execution code path, for example:
 - Transform expansion [SDK]
 - Pipeline construction [SDK]
 - Cross-language artifact staging [Runner]
 - Language specific serialization/deserialization of PCollection (and
 other data types) [Runner/SDK]

 In an effort to improve developer visibility into potential problems,
 this test suite validates correct execution of 5 Core Beam transforms when
 used as cross-language transforms within the Python SDK from any foreign SDK:
  - ParDo
  (https://beam.apache.org/documentation/programming-guide/#pardo)
  - GroupByKey
  (https://beam.apache.org/documentation/programming-guide/#groupbykey)
  - CoGroupByKey
  (https://beam.apache.org/documentation/programming-guide/#cogroupbykey)
  - Combine
  (https://beam.apache.org/documentation/programming-guide/#combine)
  - Flatten
  (https://beam.apache.org/documentation/programming-guide/#flatten)
  - Partition
  (https://beam.apache.org/documentation/programming-guide/#partition)

  See Runner Validation Test Plan for Cross-language transforms at
https://docs.google.com/document/d/1xQp0ElIV84b8OCVz8CD2hvbiWdR8w4BvWxPTZJZA6NA
  for further details.
"""
import logging
import os
import typing
import unittest
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
TEST_PREFIX_URN = 'beam:transforms:xlang:test:prefix'
TEST_MULTI_URN = 'beam:transforms:xlang:test:multi'
TEST_GBK_URN = 'beam:transforms:xlang:test:gbk'
TEST_CGBK_URN = 'beam:transforms:xlang:test:cgbk'
TEST_COMGL_URN = 'beam:transforms:xlang:test:comgl'
TEST_COMPK_URN = 'beam:transforms:xlang:test:compk'
TEST_FLATTEN_URN = 'beam:transforms:xlang:test:flatten'
TEST_PARTITION_URN = 'beam:transforms:xlang:test:partition'

class CrossLanguageTestPipelines(object):

    def __init__(self, expansion_service=None):
        if False:
            while True:
                i = 10
        self.expansion_service = expansion_service or 'localhost:%s' % os.environ.get('EXPANSION_PORT')

    def run_prefix(self, pipeline):
        if False:
            while True:
                i = 10
        '\n    Target transform - ParDo\n    (https://beam.apache.org/documentation/programming-guide/#pardo)\n    Test scenario - Mapping elements from a single input collection to a\n    single output collection\n    Boundary conditions checked -\n     - PCollection<?> to external transforms\n     - PCollection<?> from external transforms\n    '
        with pipeline as p:
            res = p | beam.Create(['a', 'b']).with_output_types(str) | beam.ExternalTransform(TEST_PREFIX_URN, ImplicitSchemaPayloadBuilder({'data': '0'}), self.expansion_service)
            assert_that(res, equal_to(['0a', '0b']))

    def run_multi_input_output_with_sideinput(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        '\n    Target transform - ParDo\n    (https://beam.apache.org/documentation/programming-guide/#pardo)\n    Test scenario - Mapping elements from multiple input collections (main\n    and side) to multiple output collections (main and side)\n    Boundary conditions checked -\n     - PCollectionTuple to external transforms\n     - PCollectionTuple from external transforms\n    '
        with pipeline as p:
            main1 = p | 'Main1' >> beam.Create(['a', 'bb'], reshuffle=False).with_output_types(str)
            main2 = p | 'Main2' >> beam.Create(['x', 'yy', 'zzz'], reshuffle=False).with_output_types(str)
            side = p | 'Side' >> beam.Create(['s']).with_output_types(str)
            res = dict(main1=main1, main2=main2, side=side) | beam.ExternalTransform(TEST_MULTI_URN, None, self.expansion_service)
            assert_that(res['main'], equal_to(['as', 'bbs', 'xs', 'yys', 'zzzs']))
            assert_that(res['side'], equal_to(['ss']), label='CheckSide')

    def run_group_by_key(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        '\n    Target transform - GroupByKey\n    (https://beam.apache.org/documentation/programming-guide/#groupbykey)\n    Test scenario - Grouping a collection of KV<K,V> to a collection of\n    KV<K, Iterable<V>> by key\n    Boundary conditions checked -\n     - PCollection<KV<?, ?>> to external transforms\n     - PCollection<KV<?, Iterable<?>>> from external transforms\n    '
        with pipeline as p:
            res = p | beam.Create([(0, '1'), (0, '2'), (1, '3')], reshuffle=False).with_output_types(typing.Tuple[int, str]) | beam.ExternalTransform(TEST_GBK_URN, None, self.expansion_service) | beam.Map(lambda x: '{}:{}'.format(x[0], ','.join(sorted(x[1]))))
            assert_that(res, equal_to(['0:1,2', '1:3']))

    def run_cogroup_by_key(self, pipeline):
        if False:
            i = 10
            return i + 15
        '\n    Target transform - CoGroupByKey\n    (https://beam.apache.org/documentation/programming-guide/#cogroupbykey)\n    Test scenario - Grouping multiple input collections with keys to a\n    collection of KV<K, CoGbkResult> by key\n    Boundary conditions checked -\n     - KeyedPCollectionTuple<?> to external transforms\n     - PCollection<KV<?, Iterable<?>>> from external transforms\n    '
        with pipeline as p:
            col1 = p | 'create_col1' >> beam.Create([(0, '1'), (0, '2'), (1, '3')], reshuffle=False).with_output_types(typing.Tuple[int, str])
            col2 = p | 'create_col2' >> beam.Create([(0, '4'), (1, '5'), (1, '6')], reshuffle=False).with_output_types(typing.Tuple[int, str])
            res = dict(col1=col1, col2=col2) | beam.ExternalTransform(TEST_CGBK_URN, None, self.expansion_service) | beam.Map(lambda x: '{}:{}'.format(x[0], ','.join(sorted(x[1]))))
            assert_that(res, equal_to(['0:1,2,4', '1:3,5,6']))

    def run_combine_globally(self, pipeline):
        if False:
            return 10
        '\n    Target transform - Combine\n    (https://beam.apache.org/documentation/programming-guide/#combine)\n    Test scenario - Combining elements globally with a predefined simple\n    CombineFn\n    Boundary conditions checked -\n     - PCollection<?> to external transforms\n     - PCollection<?> from external transforms\n    '
        with pipeline as p:
            res = p | beam.Create([1, 2, 3]).with_output_types(int) | beam.ExternalTransform(TEST_COMGL_URN, None, self.expansion_service)
            assert_that(res, equal_to([6]))

    def run_combine_per_key(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        '\n    Target transform - Combine\n    (https://beam.apache.org/documentation/programming-guide/#combine)\n    Test scenario - Combining elements per key with a predefined simple\n    merging function\n    Boundary conditions checked -\n     - PCollection<?> to external transforms\n     - PCollection<?> from external transforms\n    '
        with pipeline as p:
            res = p | beam.Create([('a', 1), ('a', 2), ('b', 3)]).with_output_types(typing.Tuple[str, int]) | beam.ExternalTransform(TEST_COMPK_URN, None, self.expansion_service)
            assert_that(res, equal_to([('a', 3), ('b', 3)]))

    def run_flatten(self, pipeline):
        if False:
            i = 10
            return i + 15
        '\n    Target transform - Flatten\n    (https://beam.apache.org/documentation/programming-guide/#flatten)\n    Test scenario - Merging multiple collections into a single collection\n    Boundary conditions checked -\n     - PCollectionList<?> to external transforms\n     - PCollection<?> from external transforms\n    '
        with pipeline as p:
            col1 = p | 'col1' >> beam.Create([1, 2, 3]).with_output_types(int)
            col2 = p | 'col2' >> beam.Create([4, 5, 6]).with_output_types(int)
            res = (col1, col2) | beam.ExternalTransform(TEST_FLATTEN_URN, None, self.expansion_service)
            assert_that(res, equal_to([1, 2, 3, 4, 5, 6]))

    def run_partition(self, pipeline):
        if False:
            while True:
                i = 10
        '\n    Target transform - Partition\n    (https://beam.apache.org/documentation/programming-guide/#partition)\n    Test scenario - Splitting a single collection into multiple collections\n    with a predefined simple PartitionFn\n    Boundary conditions checked -\n     - PCollection<?> to external transforms\n     - PCollectionList<?> from external transforms\n    '
        with pipeline as p:
            res = p | beam.Create([1, 2, 3, 4, 5, 6]).with_output_types(int) | beam.ExternalTransform(TEST_PARTITION_URN, None, self.expansion_service)
            assert_that(res['0'], equal_to([2, 4, 6]), label='check_even')
            assert_that(res['1'], equal_to([1, 3, 5]), label='check_odd')

@unittest.skipUnless(os.environ.get('EXPANSION_PORT'), 'EXPANSION_PORT environment var is not provided.')
class ValidateRunnerXlangTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def create_pipeline(self):
        if False:
            while True:
                i = 10
        test_pipeline = TestPipeline()
        test_pipeline.not_use_test_runner_api = True
        return test_pipeline

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_prefix(self, test_pipeline=None):
        if False:
            i = 10
            return i + 15
        CrossLanguageTestPipelines().run_prefix(test_pipeline or self.create_pipeline())

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_multi_input_output_with_sideinput(self, test_pipeline=None):
        if False:
            return 10
        CrossLanguageTestPipelines().run_multi_input_output_with_sideinput(test_pipeline or self.create_pipeline())

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_group_by_key(self, test_pipeline=None):
        if False:
            while True:
                i = 10
        CrossLanguageTestPipelines().run_group_by_key(test_pipeline or self.create_pipeline())

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_cogroup_by_key(self, test_pipeline=None):
        if False:
            while True:
                i = 10
        CrossLanguageTestPipelines().run_cogroup_by_key(test_pipeline or self.create_pipeline())

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_combine_globally(self, test_pipeline=None):
        if False:
            for i in range(10):
                print('nop')
        CrossLanguageTestPipelines().run_combine_globally(test_pipeline or self.create_pipeline())

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_combine_per_key(self, test_pipeline=None):
        if False:
            return 10
        CrossLanguageTestPipelines().run_combine_per_key(test_pipeline or self.create_pipeline())

    def test_flatten(self, test_pipeline=None):
        if False:
            i = 10
            return i + 15
        CrossLanguageTestPipelines().run_flatten(test_pipeline or self.create_pipeline())

    @pytest.mark.uses_java_expansion_service
    @pytest.mark.uses_python_expansion_service
    def test_partition(self, test_pipeline=None):
        if False:
            return 10
        CrossLanguageTestPipelines().run_partition(test_pipeline or self.create_pipeline())
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()