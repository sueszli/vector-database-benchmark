from apache_beam.coders import typecoders
from apache_beam.coders.coder_impl import LogicalTypeCoderImpl
from apache_beam.coders.coder_impl import RowCoderImpl
from apache_beam.coders.coders import BigEndianShortCoder
from apache_beam.coders.coders import BooleanCoder
from apache_beam.coders.coders import BytesCoder
from apache_beam.coders.coders import Coder
from apache_beam.coders.coders import DecimalCoder
from apache_beam.coders.coders import FastCoder
from apache_beam.coders.coders import FloatCoder
from apache_beam.coders.coders import IterableCoder
from apache_beam.coders.coders import MapCoder
from apache_beam.coders.coders import NullableCoder
from apache_beam.coders.coders import SinglePrecisionFloatCoder
from apache_beam.coders.coders import StrUtf8Coder
from apache_beam.coders.coders import TimestampCoder
from apache_beam.coders.coders import VarIntCoder
from apache_beam.portability import common_urns
from apache_beam.portability.api import schema_pb2
from apache_beam.typehints import row_type
from apache_beam.typehints.schemas import PYTHON_ANY_URN
from apache_beam.typehints.schemas import LogicalType
from apache_beam.typehints.schemas import named_tuple_from_schema
from apache_beam.typehints.schemas import schema_from_element_type
from apache_beam.utils import proto_utils
__all__ = ['RowCoder']

class RowCoder(FastCoder):
    """ Coder for `typing.NamedTuple` instances.

  Implements the beam:coder:row:v1 standard coder spec.
  """

    def __init__(self, schema, force_deterministic=False):
        if False:
            print('Hello World!')
        'Initializes a :class:`RowCoder`.\n\n    Args:\n      schema (apache_beam.portability.api.schema_pb2.Schema): The protobuf\n        representation of the schema of the data that the RowCoder will be used\n        to encode/decode.\n    '
        self.schema = schema
        self._type_hint = named_tuple_from_schema(self.schema)
        self.components = [_nonnull_coder_from_type(field.type) for field in self.schema.fields]
        if force_deterministic:
            self.components = [c.as_deterministic_coder(force_deterministic) for c in self.components]
        self.forced_deterministic = bool(force_deterministic)

    def _create_impl(self):
        if False:
            print('Hello World!')
        return RowCoderImpl(self.schema, self.components)

    def is_deterministic(self):
        if False:
            print('Hello World!')
        return all((c.is_deterministic() for c in self.components))

    def as_deterministic_coder(self, step_label, error_message=None):
        if False:
            print('Hello World!')
        if self.is_deterministic():
            return self
        else:
            return RowCoder(self.schema, error_message or step_label)

    def to_type_hint(self):
        if False:
            while True:
                i = 10
        return self._type_hint

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.schema.SerializeToString())

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return type(self) == type(other) and self.schema == other.schema and (self.forced_deterministic == other.forced_deterministic)

    def to_runner_api_parameter(self, unused_context):
        if False:
            i = 10
            return i + 15
        return (common_urns.coders.ROW.urn, self.schema, [])

    @staticmethod
    @Coder.register_urn(common_urns.coders.ROW.urn, schema_pb2.Schema)
    def from_runner_api_parameter(schema, components, unused_context):
        if False:
            print('Hello World!')
        return RowCoder(schema)

    @classmethod
    def from_type_hint(cls, type_hint, registry):
        if False:
            print('Hello World!')
        if isinstance(type_hint, str):
            import importlib
            main_module = importlib.import_module('__main__')
            type_hint = getattr(main_module, type_hint, type_hint)
        schema = schema_from_element_type(type_hint)
        return cls(schema)

    @staticmethod
    def from_payload(payload):
        if False:
            for i in range(10):
                print('nop')
        return RowCoder(proto_utils.parse_Bytes(payload, schema_pb2.Schema))

    def __reduce__(self):
        if False:
            return 10
        return (RowCoder.from_payload, (self.schema.SerializeToString(),))
typecoders.registry.register_coder(row_type.RowTypeConstraint, RowCoder)
typecoders.registry.register_coder(row_type.GeneratedClassRowTypeConstraint, RowCoder)

def _coder_from_type(field_type):
    if False:
        i = 10
        return i + 15
    coder = _nonnull_coder_from_type(field_type)
    if field_type.nullable:
        return NullableCoder(coder)
    else:
        return coder

def _nonnull_coder_from_type(field_type):
    if False:
        print('Hello World!')
    type_info = field_type.WhichOneof('type_info')
    if type_info == 'atomic_type':
        if field_type.atomic_type in (schema_pb2.INT32, schema_pb2.INT64):
            return VarIntCoder()
        if field_type.atomic_type == schema_pb2.INT16:
            return BigEndianShortCoder()
        elif field_type.atomic_type == schema_pb2.FLOAT:
            return SinglePrecisionFloatCoder()
        elif field_type.atomic_type == schema_pb2.DOUBLE:
            return FloatCoder()
        elif field_type.atomic_type == schema_pb2.STRING:
            return StrUtf8Coder()
        elif field_type.atomic_type == schema_pb2.BOOLEAN:
            return BooleanCoder()
        elif field_type.atomic_type == schema_pb2.BYTES:
            return BytesCoder()
    elif type_info == 'array_type':
        return IterableCoder(_coder_from_type(field_type.array_type.element_type))
    elif type_info == 'map_type':
        return MapCoder(_coder_from_type(field_type.map_type.key_type), _coder_from_type(field_type.map_type.value_type))
    elif type_info == 'logical_type':
        if field_type.logical_type.urn == PYTHON_ANY_URN:
            return typecoders.registry.get_coder(object)
        elif field_type.logical_type.urn == common_urns.millis_instant.urn:
            return TimestampCoder()
        elif field_type.logical_type.urn == 'beam:logical_type:decimal:v1':
            return DecimalCoder()
        logical_type = LogicalType.from_runner_api(field_type.logical_type)
        return LogicalTypeCoder(logical_type, _coder_from_type(field_type.logical_type.representation))
    elif type_info == 'row_type':
        return RowCoder(field_type.row_type.schema)
    raise ValueError('Encountered a type that is not currently supported by RowCoder: %s' % field_type)

class LogicalTypeCoder(FastCoder):

    def __init__(self, logical_type, representation_coder):
        if False:
            for i in range(10):
                print('nop')
        self.logical_type = logical_type
        self.representation_coder = representation_coder

    def _create_impl(self):
        if False:
            i = 10
            return i + 15
        return LogicalTypeCoderImpl(self.logical_type, self.representation_coder)

    def is_deterministic(self):
        if False:
            for i in range(10):
                print('nop')
        return self.representation_coder.is_deterministic()

    def to_type_hint(self):
        if False:
            while True:
                i = 10
        return self.logical_type.language_type()