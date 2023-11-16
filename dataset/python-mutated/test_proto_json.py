import assertpy
import pytest
from google.protobuf.json_format import MessageToDict, Parse
from feast import proto_json
from feast.protos.feast.serving.ServingService_pb2 import FeatureList, GetOnlineFeaturesResponse
from feast.protos.feast.types.Value_pb2 import RepeatedValue
FeatureVector = GetOnlineFeaturesResponse.FeatureVector

def test_feature_vector_values(proto_json_patch):
    if False:
        i = 10
        return i + 15
    feature_vector_str = '{\n        "values": [\n            1,\n            2.0,\n            true,\n            "foo",\n            [1, 2, 3],\n            [2.0, 3.0, 4.0, null],\n            [true, false, true],\n            ["foo", "bar", "foobar"]\n        ]\n    }'
    feature_vector_proto = FeatureVector()
    Parse(feature_vector_str, feature_vector_proto)
    assertpy.assert_that(len(feature_vector_proto.values)).is_equal_to(8)
    assertpy.assert_that(feature_vector_proto.values[0].int64_val).is_equal_to(1)
    assertpy.assert_that(feature_vector_proto.values[1].double_val).is_equal_to(2.0)
    assertpy.assert_that(feature_vector_proto.values[2].bool_val).is_equal_to(True)
    assertpy.assert_that(feature_vector_proto.values[3].string_val).is_equal_to('foo')
    assertpy.assert_that(feature_vector_proto.values[4].int64_list_val.val).is_equal_to([1, 2, 3])
    assertpy.assert_that(feature_vector_proto.values[5].double_list_val.val[:3]).is_equal_to([2.0, 3.0, 4.0])
    assertpy.assert_that(feature_vector_proto.values[5].double_list_val.val[3]).is_nan()
    assertpy.assert_that(feature_vector_proto.values[6].bool_list_val.val).is_equal_to([True, False, True])
    assertpy.assert_that(feature_vector_proto.values[7].string_list_val.val).is_equal_to(['foo', 'bar', 'foobar'])
    feature_vector_json = MessageToDict(feature_vector_proto)
    assertpy.assert_that(len(feature_vector_json['values'])).is_equal_to(8)
    assertpy.assert_that(feature_vector_json['values'][0]).is_equal_to(1)
    assertpy.assert_that(feature_vector_json['values'][1]).is_equal_to(2.0)
    assertpy.assert_that(feature_vector_json['values'][2]).is_equal_to(True)
    assertpy.assert_that(feature_vector_json['values'][3]).is_equal_to('foo')
    assertpy.assert_that(feature_vector_json['values'][4]).is_equal_to([1, 2, 3])
    assertpy.assert_that(feature_vector_json['values'][5][:3]).is_equal_to([2.0, 3.0, 4.0])
    assertpy.assert_that(feature_vector_json['values'][5][3]).is_nan()
    assertpy.assert_that(feature_vector_json['values'][6]).is_equal_to([True, False, True])
    assertpy.assert_that(feature_vector_json['values'][7]).is_equal_to(['foo', 'bar', 'foobar'])

def test_feast_repeated_value(proto_json_patch):
    if False:
        while True:
            i = 10
    repeated_value_str = '[1,2,3]'
    repeated_value_proto = RepeatedValue()
    Parse(repeated_value_str, repeated_value_proto, '')
    assertpy.assert_that(len(repeated_value_proto.val)).is_equal_to(3)
    assertpy.assert_that(repeated_value_proto.val[0].int64_val).is_equal_to(1)
    assertpy.assert_that(repeated_value_proto.val[1].int64_val).is_equal_to(2)
    assertpy.assert_that(repeated_value_proto.val[2].int64_val).is_equal_to(3)
    repeated_value_json = MessageToDict(repeated_value_proto)
    assertpy.assert_that(repeated_value_json).is_equal_to([1, 2, 3])

def test_feature_list(proto_json_patch):
    if False:
        i = 10
        return i + 15
    feature_list_str = '["feature-a", "feature-b", "feature-c"]'
    feature_list_proto = FeatureList()
    Parse(feature_list_str, feature_list_proto)
    assertpy.assert_that(len(feature_list_proto.val)).is_equal_to(3)
    assertpy.assert_that(feature_list_proto.val[0]).is_equal_to('feature-a')
    assertpy.assert_that(feature_list_proto.val[1]).is_equal_to('feature-b')
    assertpy.assert_that(feature_list_proto.val[2]).is_equal_to('feature-c')
    feature_list_json = MessageToDict(feature_list_proto)
    assertpy.assert_that(feature_list_json).is_equal_to(['feature-a', 'feature-b', 'feature-c'])

@pytest.fixture(scope='module')
def proto_json_patch():
    if False:
        return 10
    proto_json.patch()