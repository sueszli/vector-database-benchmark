import uuid
from typing import Any, Callable, Type
from google.protobuf.json_format import _WKTJSONMETHODS, ParseError, _Parser, _Printer
from importlib_metadata import version as importlib_version
from packaging import version
from feast.protos.feast.serving.ServingService_pb2 import FeatureList
from feast.protos.feast.types.Value_pb2 import RepeatedValue, Value
ProtoMessage = Any
JsonObject = Any

def _patch_proto_json_encoding(proto_type: Type[ProtoMessage], to_json_object: Callable[[_Printer, ProtoMessage], JsonObject], from_json_object: Callable[[_Parser, JsonObject, ProtoMessage], None]) -> None:
    if False:
        print('Hello World!')
    'Patch Protobuf JSON Encoder / Decoder for a desired Protobuf type with to_json & from_json methods.'
    to_json_fn_name = '_' + uuid.uuid4().hex
    from_json_fn_name = '_' + uuid.uuid4().hex
    setattr(_Printer, to_json_fn_name, to_json_object)
    setattr(_Parser, from_json_fn_name, from_json_object)
    _WKTJSONMETHODS[proto_type.DESCRIPTOR.full_name] = [to_json_fn_name, from_json_fn_name]

def _patch_feast_value_json_encoding():
    if False:
        print('Hello World!')
    'Patch Protobuf JSON Encoder / Decoder with a Feast Value type.\n\n    This allows encoding the proto object as a native type, without the dummy structural wrapper.\n\n    Here\'s a before example:\n\n    {\n        "value_1": {\n            "int64_val": 1\n        },\n        "value_2": {\n            "double_list_val": [1.0, 2.0, 3.0]\n        },\n    }\n\n    And here\'s an after example:\n\n    {\n        "value_1": 1,\n        "value_2": [1.0, 2.0, 3.0]\n    }\n    '

    def to_json_object(printer: _Printer, message: ProtoMessage) -> JsonObject:
        if False:
            i = 10
            return i + 15
        which = message.WhichOneof('val')
        if which is None or which == 'null_val':
            return None
        elif '_list_' in which:
            value = list(getattr(message, which).val)
        else:
            value = getattr(message, which)
        return value

    def from_json_object(parser: _Parser, value: JsonObject, message: ProtoMessage) -> None:
        if False:
            return 10
        if value is None:
            message.null_val = 0
        elif isinstance(value, bool):
            message.bool_val = value
        elif isinstance(value, str):
            message.string_val = value
        elif isinstance(value, int):
            message.int64_val = value
        elif isinstance(value, float):
            message.double_val = value
        elif isinstance(value, list):
            if len(value) == 0:
                message.int64_list_val.Clear()
            elif isinstance(value[0], bool):
                message.bool_list_val.val.extend(value)
            elif isinstance(value[0], str):
                message.string_list_val.val.extend(value)
            elif isinstance(value[0], (float, int, type(None))):
                if all((isinstance(item, int) for item in value)):
                    message.int64_list_val.val.extend(value)
                else:
                    message.double_list_val.val.extend([item if item is not None else float('nan') for item in value])
            else:
                raise ParseError('Value {0} has unexpected type {1}.'.format(value[0], type(value[0])))
        else:
            raise ParseError('Value {0} has unexpected type {1}.'.format(value, type(value)))

    def from_json_object_updated(parser: _Parser, value: JsonObject, message: ProtoMessage, path: str):
        if False:
            return 10
        from_json_object(parser, value, message)
    current_version = importlib_version('protobuf')
    if version.parse(current_version) < version.parse('3.20'):
        _patch_proto_json_encoding(Value, to_json_object, from_json_object)
    else:
        _patch_proto_json_encoding(Value, to_json_object, from_json_object_updated)

def _patch_feast_repeated_value_json_encoding():
    if False:
        return 10
    'Patch Protobuf JSON Encoder / Decoder with a Feast RepeatedValue type.\n\n    This allows list of lists without dummy field name "val".\n\n    Here\'s a before example:\n\n    {\n        "repeated_value": [\n            {"val": [1,2,3]},\n            {"val": [4,5,6]}\n        ]\n    }\n\n    And here\'s an after example:\n\n    {\n        "repeated_value": [\n            [1,2,3],\n            [4,5,6]\n        ]\n    }\n    '

    def to_json_object(printer: _Printer, message: ProtoMessage) -> JsonObject:
        if False:
            return 10
        return [printer._MessageToJsonObject(item) for item in message.val]

    def from_json_object_updated(parser: _Parser, value: JsonObject, message: ProtoMessage, path: str) -> None:
        if False:
            while True:
                i = 10
        array = value if isinstance(value, list) else value['val']
        for item in array:
            parser.ConvertMessage(item, message.val.add(), path)

    def from_json_object(parser: _Parser, value: JsonObject, message: ProtoMessage) -> None:
        if False:
            while True:
                i = 10
        array = value if isinstance(value, list) else value['val']
        for item in array:
            parser.ConvertMessage(item, message.val.add())
    current_version = importlib_version('protobuf')
    if version.parse(current_version) < version.parse('3.20'):
        _patch_proto_json_encoding(RepeatedValue, to_json_object, from_json_object)
    else:
        _patch_proto_json_encoding(RepeatedValue, to_json_object, from_json_object_updated)

def _patch_feast_feature_list_json_encoding():
    if False:
        print('Hello World!')
    'Patch Protobuf JSON Encoder / Decoder with a Feast FeatureList type.\n\n    This allows list of lists without dummy field name "features".\n\n    Here\'s a before example:\n\n    {\n        "feature_list": {\n            "features": [\n                "feature-1",\n                "feature-2",\n                "feature-3"\n            ]\n        }\n    }\n\n    And here\'s an after example:\n\n    {\n        "feature_list": [\n            "feature-1",\n            "feature-2",\n            "feature-3"\n        ]\n    }\n    '

    def to_json_object(printer: _Printer, message: ProtoMessage) -> JsonObject:
        if False:
            i = 10
            return i + 15
        return list(message.val)

    def from_json_object(parser: _Parser, value: JsonObject, message: ProtoMessage) -> None:
        if False:
            for i in range(10):
                print('nop')
        array = value if isinstance(value, list) else value['val']
        message.val.extend(array)

    def from_json_object_updated(parser: _Parser, value: JsonObject, message: ProtoMessage, path: str) -> None:
        if False:
            i = 10
            return i + 15
        from_json_object(parser, value, message)
    current_version = importlib_version('protobuf')
    if version.parse(current_version) < version.parse('3.20'):
        _patch_proto_json_encoding(FeatureList, to_json_object, from_json_object)
    else:
        _patch_proto_json_encoding(FeatureList, to_json_object, from_json_object_updated)

def patch():
    if False:
        return 10
    'Patch Protobuf JSON Encoder / Decoder with all desired Feast types.'
    _patch_feast_value_json_encoding()
    _patch_feast_repeated_value_json_encoding()
    _patch_feast_feature_list_json_encoding()