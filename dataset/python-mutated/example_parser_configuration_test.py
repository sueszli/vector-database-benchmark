"""Tests for ExampleParserConfiguration."""
from google.protobuf import text_format
from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util.example_parser_configuration import extract_example_parser_configuration
EXPECTED_CONFIG_V1 = '\nfeature_map {\n  key: "x"\n  value {\n    fixed_len_feature {\n      dtype: DT_FLOAT\n      shape {\n        dim {\n          size: 1\n        }\n      }\n      default_value {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        float_val: 33.0\n      }\n      values_output_tensor_name: "ParseExample/ParseExample:3"\n    }\n  }\n}\nfeature_map {\n  key: "y"\n  value {\n    var_len_feature {\n      dtype: DT_STRING\n      values_output_tensor_name: "ParseExample/ParseExample:1"\n      indices_output_tensor_name: "ParseExample/ParseExample:0"\n      shapes_output_tensor_name: "ParseExample/ParseExample:2"\n    }\n  }\n}\n'
EXPECTED_CONFIG_V2 = EXPECTED_CONFIG_V1.replace('ParseExample/ParseExample:', 'ParseExample/ParseExampleV2:')

class ExampleParserConfigurationTest(test.TestCase):

    def getExpectedConfig(self, op_type):
        if False:
            print('Hello World!')
        expected = example_parser_configuration_pb2.ExampleParserConfiguration()
        if op_type == 'ParseExampleV2':
            text_format.Parse(EXPECTED_CONFIG_V2, expected)
        else:
            text_format.Parse(EXPECTED_CONFIG_V1, expected)
        return expected

    def testBasic(self):
        if False:
            return 10
        with session.Session() as sess:
            examples = array_ops.placeholder(dtypes.string, shape=[1])
            feature_to_type = {'x': parsing_ops.FixedLenFeature([1], dtypes.float32, 33.0), 'y': parsing_ops.VarLenFeature(dtypes.string)}
            result = parsing_ops.parse_example(examples, feature_to_type)
            parse_example_op = result['x'].op
            config = extract_example_parser_configuration(parse_example_op, sess)
            expected = self.getExpectedConfig(parse_example_op.type)
            self.assertProtoEquals(expected, config)
if __name__ == '__main__':
    test.main()