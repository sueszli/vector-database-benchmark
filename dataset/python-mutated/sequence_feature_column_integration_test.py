"""Integration test for sequence feature columns with SequenceExamples."""
import string
import tempfile
from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class SequenceExampleParsingTest(test.TestCase):

    def test_seq_ex_in_sequence_categorical_column_with_identity(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_parsed_sequence_example('int_list', sfc.sequence_categorical_column_with_identity, 10, [3, 6], [2, 4, 6])

    def test_seq_ex_in_sequence_categorical_column_with_hash_bucket(self):
        if False:
            print('Hello World!')
        self._test_parsed_sequence_example('bytes_list', sfc.sequence_categorical_column_with_hash_bucket, 10, [3, 4], [compat.as_bytes(x) for x in 'acg'])

    def test_seq_ex_in_sequence_categorical_column_with_vocabulary_list(self):
        if False:
            return 10
        self._test_parsed_sequence_example('bytes_list', sfc.sequence_categorical_column_with_vocabulary_list, list(string.ascii_lowercase), [3, 4], [compat.as_bytes(x) for x in 'acg'])

    def test_seq_ex_in_sequence_categorical_column_with_vocabulary_file(self):
        if False:
            for i in range(10):
                print('nop')
        (_, fname) = tempfile.mkstemp()
        with open(fname, 'w') as f:
            f.write(string.ascii_lowercase)
        self._test_parsed_sequence_example('bytes_list', sfc.sequence_categorical_column_with_vocabulary_file, fname, [3, 4], [compat.as_bytes(x) for x in 'acg'])

    def _test_parsed_sequence_example(self, col_name, col_fn, col_arg, shape, values):
        if False:
            return 10
        'Helper function to check that each FeatureColumn parses correctly.\n\n    Args:\n      col_name: string, name to give to the feature column. Should match\n        the name that the column will parse out of the features dict.\n      col_fn: function used to create the feature column. For example,\n        sequence_numeric_column.\n      col_arg: second arg that the target feature column is expecting.\n      shape: the expected dense_shape of the feature after parsing into\n        a SparseTensor.\n      values: the expected values at index [0, 2, 6] of the feature\n        after parsing into a SparseTensor.\n    '
        example = _make_sequence_example()
        columns = [fc.categorical_column_with_identity('int_ctx', num_buckets=100), fc.numeric_column('float_ctx'), col_fn(col_name, col_arg)]
        (context, seq_features) = parsing_ops.parse_single_sequence_example(example.SerializeToString(), context_features=fc.make_parse_example_spec_v2(columns[:2]), sequence_features=fc.make_parse_example_spec_v2(columns[2:]))
        with self.cached_session() as sess:
            (ctx_result, seq_result) = sess.run([context, seq_features])
            self.assertEqual(list(seq_result[col_name].dense_shape), shape)
            self.assertEqual(list(seq_result[col_name].values[[0, 2, 6]]), values)
            self.assertEqual(list(ctx_result['int_ctx'].dense_shape), [1])
            self.assertEqual(ctx_result['int_ctx'].values[0], 5)
            self.assertEqual(list(ctx_result['float_ctx'].shape), [1])
            self.assertAlmostEqual(ctx_result['float_ctx'][0], 123.6, places=1)
_SEQ_EX_PROTO = '\ncontext {\n  feature {\n    key: "float_ctx"\n    value {\n      float_list {\n        value: 123.6\n      }\n    }\n  }\n  feature {\n    key: "int_ctx"\n    value {\n      int64_list {\n        value: 5\n      }\n    }\n  }\n}\nfeature_lists {\n  feature_list {\n    key: "bytes_list"\n    value {\n      feature {\n        bytes_list {\n          value: "a"\n        }\n      }\n      feature {\n        bytes_list {\n          value: "b"\n          value: "c"\n        }\n      }\n      feature {\n        bytes_list {\n          value: "d"\n          value: "e"\n          value: "f"\n          value: "g"\n        }\n      }\n    }\n  }\n  feature_list {\n    key: "float_list"\n    value {\n      feature {\n        float_list {\n          value: 1.0\n        }\n      }\n      feature {\n        float_list {\n          value: 3.0\n          value: 3.0\n          value: 3.0\n        }\n      }\n      feature {\n        float_list {\n          value: 5.0\n          value: 5.0\n          value: 5.0\n          value: 5.0\n          value: 5.0\n        }\n      }\n    }\n  }\n  feature_list {\n    key: "int_list"\n    value {\n      feature {\n        int64_list {\n          value: 2\n          value: 2\n        }\n      }\n      feature {\n        int64_list {\n          value: 4\n          value: 4\n          value: 4\n          value: 4\n        }\n      }\n      feature {\n        int64_list {\n          value: 6\n          value: 6\n          value: 6\n          value: 6\n          value: 6\n          value: 6\n        }\n      }\n    }\n  }\n}\n'

def _make_sequence_example():
    if False:
        print('Hello World!')
    example = example_pb2.SequenceExample()
    return text_format.Parse(_SEQ_EX_PROTO, example)
if __name__ == '__main__':
    test.main()