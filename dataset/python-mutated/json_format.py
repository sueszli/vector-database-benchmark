"""Contains routines for printing protocol messages in JSON format.

Simple usage example:

  # Create a proto object and serialize it to a json format string.
  message = my_proto_pb2.MyMessage(foo='bar')
  json_string = json_format.MessageToJson(message)

  # Parse a json format string to proto object.
  message = json_format.Parse(json_string, my_proto_pb2.MyMessage())
"""
__author__ = 'jieluo@google.com (Jie Luo)'
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import base64
import json
import math
import six
import sys
from google.protobuf import descriptor
from google.protobuf import symbol_database
_TIMESTAMPFOMAT = '%Y-%m-%dT%H:%M:%S'
_INT_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_INT32, descriptor.FieldDescriptor.CPPTYPE_UINT32, descriptor.FieldDescriptor.CPPTYPE_INT64, descriptor.FieldDescriptor.CPPTYPE_UINT64])
_INT64_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_INT64, descriptor.FieldDescriptor.CPPTYPE_UINT64])
_FLOAT_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_FLOAT, descriptor.FieldDescriptor.CPPTYPE_DOUBLE])
_INFINITY = 'Infinity'
_NEG_INFINITY = '-Infinity'
_NAN = 'NaN'

class Error(Exception):
    """Top-level module error for json_format."""

class SerializeToJsonError(Error):
    """Thrown if serialization to JSON fails."""

class ParseError(Error):
    """Thrown in case of parsing error."""

def MessageToJson(message, including_default_value_fields=False):
    if False:
        while True:
            i = 10
    'Converts protobuf message to JSON format.\n\n  Args:\n    message: The protocol buffers message instance to serialize.\n    including_default_value_fields: If True, singular primitive fields,\n        repeated fields, and map fields will always be serialized.  If\n        False, only serialize non-empty fields.  Singular message fields\n        and oneof fields are not affected by this option.\n\n  Returns:\n    A string containing the JSON formatted protocol buffer message.\n  '
    js = _MessageToJsonObject(message, including_default_value_fields)
    return json.dumps(js, indent=2)

def _MessageToJsonObject(message, including_default_value_fields):
    if False:
        print('Hello World!')
    'Converts message to an object according to Proto3 JSON Specification.'
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
        return _WrapperMessageToJsonObject(message)
    if full_name in _WKTJSONMETHODS:
        return _WKTJSONMETHODS[full_name][0](message, including_default_value_fields)
    js = {}
    return _RegularMessageToJsonObject(message, js, including_default_value_fields)

def _IsMapEntry(field):
    if False:
        for i in range(10):
            print('nop')
    return field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options and field.message_type.GetOptions().map_entry

def _RegularMessageToJsonObject(message, js, including_default_value_fields):
    if False:
        i = 10
        return i + 15
    'Converts normal message according to Proto3 JSON Specification.'
    fields = message.ListFields()
    include_default = including_default_value_fields
    try:
        for (field, value) in fields:
            name = field.camelcase_name
            if _IsMapEntry(field):
                v_field = field.message_type.fields_by_name['value']
                js_map = {}
                for key in value:
                    if isinstance(key, bool):
                        if key:
                            recorded_key = 'true'
                        else:
                            recorded_key = 'false'
                    else:
                        recorded_key = key
                    js_map[recorded_key] = _FieldToJsonObject(v_field, value[key], including_default_value_fields)
                js[name] = js_map
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                js[name] = [_FieldToJsonObject(field, k, include_default) for k in value]
            else:
                js[name] = _FieldToJsonObject(field, value, include_default)
        if including_default_value_fields:
            message_descriptor = message.DESCRIPTOR
            for field in message_descriptor.fields:
                if field.label != descriptor.FieldDescriptor.LABEL_REPEATED and field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE or field.containing_oneof:
                    continue
                name = field.camelcase_name
                if name in js:
                    continue
                if _IsMapEntry(field):
                    js[name] = {}
                elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                    js[name] = []
                else:
                    js[name] = _FieldToJsonObject(field, field.default_value)
    except ValueError as e:
        raise SerializeToJsonError('Failed to serialize {0} field: {1}.'.format(field.name, e))
    return js

def _FieldToJsonObject(field, value, including_default_value_fields=False):
    if False:
        while True:
            i = 10
    'Converts field value according to Proto3 JSON Specification.'
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        return _MessageToJsonObject(value, including_default_value_fields)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
        enum_value = field.enum_type.values_by_number.get(value, None)
        if enum_value is not None:
            return enum_value.name
        else:
            raise SerializeToJsonError('Enum field contains an integer value which can not mapped to an enum value.')
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
        if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            return base64.b64encode(value).decode('utf-8')
        else:
            return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
        return bool(value)
    elif field.cpp_type in _INT64_TYPES:
        return str(value)
    elif field.cpp_type in _FLOAT_TYPES:
        if math.isinf(value):
            if value < 0.0:
                return _NEG_INFINITY
            else:
                return _INFINITY
        if math.isnan(value):
            return _NAN
    return value

def _AnyMessageToJsonObject(message, including_default):
    if False:
        while True:
            i = 10
    'Converts Any message according to Proto3 JSON Specification.'
    if not message.ListFields():
        return {}
    js = OrderedDict()
    type_url = message.type_url
    js['@type'] = type_url
    sub_message = _CreateMessageFromTypeUrl(type_url)
    sub_message.ParseFromString(message.value)
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
        js['value'] = _WrapperMessageToJsonObject(sub_message)
        return js
    if full_name in _WKTJSONMETHODS:
        js['value'] = _WKTJSONMETHODS[full_name][0](sub_message, including_default)
        return js
    return _RegularMessageToJsonObject(sub_message, js, including_default)

def _CreateMessageFromTypeUrl(type_url):
    if False:
        return 10
    db = symbol_database.Default()
    type_name = type_url.split('/')[-1]
    try:
        message_descriptor = db.pool.FindMessageTypeByName(type_name)
    except KeyError:
        raise TypeError('Can not find message descriptor by type_url: {0}.'.format(type_url))
    message_class = db.GetPrototype(message_descriptor)
    return message_class()

def _GenericMessageToJsonObject(message, unused_including_default):
    if False:
        while True:
            i = 10
    'Converts message by ToJsonString according to Proto3 JSON Specification.'
    return message.ToJsonString()

def _ValueMessageToJsonObject(message, unused_including_default=False):
    if False:
        print('Hello World!')
    'Converts Value message according to Proto3 JSON Specification.'
    which = message.WhichOneof('kind')
    if which is None or which == 'null_value':
        return None
    if which == 'list_value':
        return _ListValueMessageToJsonObject(message.list_value)
    if which == 'struct_value':
        value = message.struct_value
    else:
        value = getattr(message, which)
    oneof_descriptor = message.DESCRIPTOR.fields_by_name[which]
    return _FieldToJsonObject(oneof_descriptor, value)

def _ListValueMessageToJsonObject(message, unused_including_default=False):
    if False:
        print('Hello World!')
    'Converts ListValue message according to Proto3 JSON Specification.'
    return [_ValueMessageToJsonObject(value) for value in message.values]

def _StructMessageToJsonObject(message, unused_including_default=False):
    if False:
        print('Hello World!')
    'Converts Struct message according to Proto3 JSON Specification.'
    fields = message.fields
    ret = {}
    for key in fields:
        ret[key] = _ValueMessageToJsonObject(fields[key])
    return ret

def _IsWrapperMessage(message_descriptor):
    if False:
        while True:
            i = 10
    return message_descriptor.file.name == 'google/protobuf/wrappers.proto'

def _WrapperMessageToJsonObject(message):
    if False:
        i = 10
        return i + 15
    return _FieldToJsonObject(message.DESCRIPTOR.fields_by_name['value'], message.value)

def _DuplicateChecker(js):
    if False:
        print('Hello World!')
    result = {}
    for (name, value) in js:
        if name in result:
            raise ParseError('Failed to load JSON: duplicate key {0}.'.format(name))
        result[name] = value
    return result

def Parse(text, message):
    if False:
        while True:
            i = 10
    'Parses a JSON representation of a protocol message into a message.\n\n  Args:\n    text: Message JSON representation.\n    message: A protocol beffer message to merge into.\n\n  Returns:\n    The same message passed as argument.\n\n  Raises::\n    ParseError: On JSON parsing problems.\n  '
    if not isinstance(text, six.text_type):
        text = text.decode('utf-8')
    try:
        if sys.version_info < (2, 7):
            js = json.loads(text)
        else:
            js = json.loads(text, object_pairs_hook=_DuplicateChecker)
    except ValueError as e:
        raise ParseError('Failed to load JSON: {0}.'.format(str(e)))
    _ConvertMessage(js, message)
    return message

def _ConvertFieldValuePair(js, message):
    if False:
        for i in range(10):
            print('nop')
    'Convert field value pairs into regular message.\n\n  Args:\n    js: A JSON object to convert the field value pairs.\n    message: A regular protocol message to record the data.\n\n  Raises:\n    ParseError: In case of problems converting.\n  '
    names = []
    message_descriptor = message.DESCRIPTOR
    for name in js:
        try:
            field = message_descriptor.fields_by_camelcase_name.get(name, None)
            if not field:
                raise ParseError('Message type "{0}" has no field named "{1}".'.format(message_descriptor.full_name, name))
            if name in names:
                raise ParseError('Message type "{0}" should not have multiple "{1}" fields.'.format(message.DESCRIPTOR.full_name, name))
            names.append(name)
            if field.containing_oneof is not None:
                oneof_name = field.containing_oneof.name
                if oneof_name in names:
                    raise ParseError('Message type "{0}" should not have multiple "{1}" oneof fields.'.format(message.DESCRIPTOR.full_name, oneof_name))
                names.append(oneof_name)
            value = js[name]
            if value is None:
                message.ClearField(field.name)
                continue
            if _IsMapEntry(field):
                message.ClearField(field.name)
                _ConvertMapFieldValue(value, message, field)
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                message.ClearField(field.name)
                if not isinstance(value, list):
                    raise ParseError('repeated field {0} must be in [] which is {1}.'.format(name, value))
                if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                    for item in value:
                        sub_message = getattr(message, field.name).add()
                        if item is None and sub_message.DESCRIPTOR.full_name != 'google.protobuf.Value':
                            raise ParseError('null is not allowed to be used as an element in a repeated field.')
                        _ConvertMessage(item, sub_message)
                else:
                    for item in value:
                        if item is None:
                            raise ParseError('null is not allowed to be used as an element in a repeated field.')
                        getattr(message, field.name).append(_ConvertScalarFieldValue(item, field))
            elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                sub_message = getattr(message, field.name)
                _ConvertMessage(value, sub_message)
            else:
                setattr(message, field.name, _ConvertScalarFieldValue(value, field))
        except ParseError as e:
            if field and field.containing_oneof is None:
                raise ParseError('Failed to parse {0} field: {1}'.format(name, e))
            else:
                raise ParseError(str(e))
        except ValueError as e:
            raise ParseError('Failed to parse {0} field: {1}.'.format(name, e))
        except TypeError as e:
            raise ParseError('Failed to parse {0} field: {1}.'.format(name, e))

def _ConvertMessage(value, message):
    if False:
        for i in range(10):
            print('nop')
    'Convert a JSON object into a message.\n\n  Args:\n    value: A JSON object.\n    message: A WKT or regular protocol message to record the data.\n\n  Raises:\n    ParseError: In case of convert problems.\n  '
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
        _ConvertWrapperMessage(value, message)
    elif full_name in _WKTJSONMETHODS:
        _WKTJSONMETHODS[full_name][1](value, message)
    else:
        _ConvertFieldValuePair(value, message)

def _ConvertAnyMessage(value, message):
    if False:
        while True:
            i = 10
    'Convert a JSON representation into Any message.'
    if isinstance(value, dict) and (not value):
        return
    try:
        type_url = value['@type']
    except KeyError:
        raise ParseError('@type is missing when parsing any message.')
    sub_message = _CreateMessageFromTypeUrl(type_url)
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
        _ConvertWrapperMessage(value['value'], sub_message)
    elif full_name in _WKTJSONMETHODS:
        _WKTJSONMETHODS[full_name][1](value['value'], sub_message)
    else:
        del value['@type']
        _ConvertFieldValuePair(value, sub_message)
    message.value = sub_message.SerializeToString()
    message.type_url = type_url

def _ConvertGenericMessage(value, message):
    if False:
        i = 10
        return i + 15
    'Convert a JSON representation into message with FromJsonString.'
    message.FromJsonString(value)
_INT_OR_FLOAT = six.integer_types + (float,)

def _ConvertValueMessage(value, message):
    if False:
        for i in range(10):
            print('nop')
    'Convert a JSON representation into Value message.'
    if isinstance(value, dict):
        _ConvertStructMessage(value, message.struct_value)
    elif isinstance(value, list):
        _ConvertListValueMessage(value, message.list_value)
    elif value is None:
        message.null_value = 0
    elif isinstance(value, bool):
        message.bool_value = value
    elif isinstance(value, six.string_types):
        message.string_value = value
    elif isinstance(value, _INT_OR_FLOAT):
        message.number_value = value
    else:
        raise ParseError('Unexpected type for Value message.')

def _ConvertListValueMessage(value, message):
    if False:
        for i in range(10):
            print('nop')
    'Convert a JSON representation into ListValue message.'
    if not isinstance(value, list):
        raise ParseError('ListValue must be in [] which is {0}.'.format(value))
    message.ClearField('values')
    for item in value:
        _ConvertValueMessage(item, message.values.add())

def _ConvertStructMessage(value, message):
    if False:
        i = 10
        return i + 15
    'Convert a JSON representation into Struct message.'
    if not isinstance(value, dict):
        raise ParseError('Struct must be in a dict which is {0}.'.format(value))
    for key in value:
        _ConvertValueMessage(value[key], message.fields[key])
    return

def _ConvertWrapperMessage(value, message):
    if False:
        return 10
    'Convert a JSON representation into Wrapper message.'
    field = message.DESCRIPTOR.fields_by_name['value']
    setattr(message, 'value', _ConvertScalarFieldValue(value, field))

def _ConvertMapFieldValue(value, message, field):
    if False:
        for i in range(10):
            print('nop')
    'Convert map field value for a message map field.\n\n  Args:\n    value: A JSON object to convert the map field value.\n    message: A protocol message to record the converted data.\n    field: The descriptor of the map field to be converted.\n\n  Raises:\n    ParseError: In case of convert problems.\n  '
    if not isinstance(value, dict):
        raise ParseError('Map field {0} must be in a dict which is {1}.'.format(field.name, value))
    key_field = field.message_type.fields_by_name['key']
    value_field = field.message_type.fields_by_name['value']
    for key in value:
        key_value = _ConvertScalarFieldValue(key, key_field, True)
        if value_field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            _ConvertMessage(value[key], getattr(message, field.name)[key_value])
        else:
            getattr(message, field.name)[key_value] = _ConvertScalarFieldValue(value[key], value_field)

def _ConvertScalarFieldValue(value, field, require_str=False):
    if False:
        return 10
    'Convert a single scalar field value.\n\n  Args:\n    value: A scalar value to convert the scalar field value.\n    field: The descriptor of the field to convert.\n    require_str: If True, the field value must be a str.\n\n  Returns:\n    The converted scalar field value\n\n  Raises:\n    ParseError: In case of convert problems.\n  '
    if field.cpp_type in _INT_TYPES:
        return _ConvertInteger(value)
    elif field.cpp_type in _FLOAT_TYPES:
        return _ConvertFloat(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
        return _ConvertBool(value, require_str)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
        if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            return base64.b64decode(value)
        else:
            return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
        enum_value = field.enum_type.values_by_name.get(value, None)
        if enum_value is None:
            raise ParseError('Enum value must be a string literal with double quotes. Type "{0}" has no value named {1}.'.format(field.enum_type.full_name, value))
        return enum_value.number

def _ConvertInteger(value):
    if False:
        for i in range(10):
            print('nop')
    "Convert an integer.\n\n  Args:\n    value: A scalar value to convert.\n\n  Returns:\n    The integer value.\n\n  Raises:\n    ParseError: If an integer couldn't be consumed.\n  "
    if isinstance(value, float):
        raise ParseError("Couldn't parse integer: {0}.".format(value))
    if isinstance(value, six.text_type) and value.find(' ') != -1:
        raise ParseError('Couldn\'t parse integer: "{0}".'.format(value))
    return int(value)

def _ConvertFloat(value):
    if False:
        for i in range(10):
            print('nop')
    'Convert an floating point number.'
    if value == 'nan':
        raise ParseError('Couldn\'t parse float "nan", use "NaN" instead.')
    try:
        return float(value)
    except ValueError:
        if value == _NEG_INFINITY:
            return float('-inf')
        elif value == _INFINITY:
            return float('inf')
        elif value == _NAN:
            return float('nan')
        else:
            raise ParseError("Couldn't parse float: {0}.".format(value))

def _ConvertBool(value, require_str):
    if False:
        for i in range(10):
            print('nop')
    "Convert a boolean value.\n\n  Args:\n    value: A scalar value to convert.\n    require_str: If True, value must be a str.\n\n  Returns:\n    The bool parsed.\n\n  Raises:\n    ParseError: If a boolean value couldn't be consumed.\n  "
    if require_str:
        if value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise ParseError('Expected "true" or "false", not {0}.'.format(value))
    if not isinstance(value, bool):
        raise ParseError('Expected true or false without quotes.')
    return value
_WKTJSONMETHODS = {'google.protobuf.Any': [_AnyMessageToJsonObject, _ConvertAnyMessage], 'google.protobuf.Duration': [_GenericMessageToJsonObject, _ConvertGenericMessage], 'google.protobuf.FieldMask': [_GenericMessageToJsonObject, _ConvertGenericMessage], 'google.protobuf.ListValue': [_ListValueMessageToJsonObject, _ConvertListValueMessage], 'google.protobuf.Struct': [_StructMessageToJsonObject, _ConvertStructMessage], 'google.protobuf.Timestamp': [_GenericMessageToJsonObject, _ConvertGenericMessage], 'google.protobuf.Value': [_ValueMessageToJsonObject, _ConvertValueMessage]}