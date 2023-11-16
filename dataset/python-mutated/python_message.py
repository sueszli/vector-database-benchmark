"""Contains a metaclass and helper functions used to create
protocol message classes from Descriptor objects at runtime.

Recall that a metaclass is the "type" of a class.
(A class is to a metaclass what an instance is to a class.)

In this case, we use the GeneratedProtocolMessageType metaclass
to inject all the useful functionality into the classes
output by the protocol compiler at compile-time.

The upshot of all this is that the real implementation
details for ALL pure-Python protocol buffers are *here in
this file*.
"""
__author__ = 'robinson@google.com (Will Robinson)'
from io import BytesIO
import sys
import struct
import weakref
import six
try:
    import six.moves.copyreg as copyreg
except ImportError:
    import copy_reg as copyreg
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import symbol_database
from google.protobuf import text_format
_FieldDescriptor = descriptor_mod.FieldDescriptor
_AnyFullTypeName = 'google.protobuf.Any'

class GeneratedProtocolMessageType(type):
    """Metaclass for protocol message classes created at runtime from Descriptors.

  We add implementations for all methods described in the Message class.  We
  also create properties to allow getting/setting all fields in the protocol
  message.  Finally, we create slots to prevent users from accidentally
  "setting" nonexistent fields in the protocol message, which then wouldn't get
  serialized / deserialized properly.

  The protocol compiler currently uses this metaclass to create protocol
  message classes at runtime.  Clients can also manually create their own
  classes at runtime, as in this example:

  mydescriptor = Descriptor(.....)
  class MyProtoClass(Message):
    __metaclass__ = GeneratedProtocolMessageType
    DESCRIPTOR = mydescriptor
  myproto_instance = MyProtoClass()
  myproto.foo_field = 23
  ...

  The above example will not work for nested types. If you wish to include them,
  use reflection.MakeClass() instead of manually instantiating the class in
  order to create the appropriate class structure.
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'

    def __new__(cls, name, bases, dictionary):
        if False:
            print('Hello World!')
        "Custom allocation for runtime-generated class types.\n\n    We override __new__ because this is apparently the only place\n    where we can meaningfully set __slots__ on the class we're creating(?).\n    (The interplay between metaclasses and slots is not very well-documented).\n\n    Args:\n      name: Name of the class (ignored, but required by the\n        metaclass protocol).\n      bases: Base classes of the class we're constructing.\n        (Should be message.Message).  We ignore this field, but\n        it's required by the metaclass protocol\n      dictionary: The class dictionary of the class we're\n        constructing.  dictionary[_DESCRIPTOR_KEY] must contain\n        a Descriptor object describing this protocol message\n        type.\n\n    Returns:\n      Newly-allocated class.\n    "
        descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]
        if descriptor.full_name in well_known_types.WKTBASES:
            bases += (well_known_types.WKTBASES[descriptor.full_name],)
        _AddClassAttributesForNestedExtensions(descriptor, dictionary)
        _AddSlots(descriptor, dictionary)
        superclass = super(GeneratedProtocolMessageType, cls)
        new_class = superclass.__new__(cls, name, bases, dictionary)
        return new_class

    def __init__(cls, name, bases, dictionary):
        if False:
            return 10
        "Here we perform the majority of our work on the class.\n    We add enum getters, an __init__ method, implementations\n    of all Message methods, and properties for all fields\n    in the protocol type.\n\n    Args:\n      name: Name of the class (ignored, but required by the\n        metaclass protocol).\n      bases: Base classes of the class we're constructing.\n        (Should be message.Message).  We ignore this field, but\n        it's required by the metaclass protocol\n      dictionary: The class dictionary of the class we're\n        constructing.  dictionary[_DESCRIPTOR_KEY] must contain\n        a Descriptor object describing this protocol message\n        type.\n    "
        descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]
        cls._decoders_by_tag = {}
        cls._extensions_by_name = {}
        cls._extensions_by_number = {}
        if descriptor.has_options and descriptor.GetOptions().message_set_wire_format:
            cls._decoders_by_tag[decoder.MESSAGE_SET_ITEM_TAG] = (decoder.MessageSetItemDecoder(cls._extensions_by_number), None)
        for field in descriptor.fields:
            _AttachFieldHelpers(cls, field)
        descriptor._concrete_class = cls
        _AddEnumValues(descriptor, cls)
        _AddInitMethod(descriptor, cls)
        _AddPropertiesForFields(descriptor, cls)
        _AddPropertiesForExtensions(descriptor, cls)
        _AddStaticMethods(cls)
        _AddMessageMethods(descriptor, cls)
        _AddPrivateHelperMethods(descriptor, cls)
        copyreg.pickle(cls, lambda obj: (cls, (), obj.__getstate__()))
        superclass = super(GeneratedProtocolMessageType, cls)
        superclass.__init__(name, bases, dictionary)

def _PropertyName(proto_field_name):
    if False:
        print('Hello World!')
    'Returns the name of the public property attribute which\n  clients can use to get and (in some cases) set the value\n  of a protocol message field.\n\n  Args:\n    proto_field_name: The protocol message field name, exactly\n      as it appears (or would appear) in a .proto file.\n  '
    return proto_field_name

def _VerifyExtensionHandle(message, extension_handle):
    if False:
        print('Hello World!')
    'Verify that the given extension handle is valid.'
    if not isinstance(extension_handle, _FieldDescriptor):
        raise KeyError('HasExtension() expects an extension handle, got: %s' % extension_handle)
    if not extension_handle.is_extension:
        raise KeyError('"%s" is not an extension.' % extension_handle.full_name)
    if not extension_handle.containing_type:
        raise KeyError('"%s" is missing a containing_type.' % extension_handle.full_name)
    if extension_handle.containing_type is not message.DESCRIPTOR:
        raise KeyError('Extension "%s" extends message type "%s", but this message is of type "%s".' % (extension_handle.full_name, extension_handle.containing_type.full_name, message.DESCRIPTOR.full_name))

def _AddSlots(message_descriptor, dictionary):
    if False:
        while True:
            i = 10
    "Adds a __slots__ entry to dictionary, containing the names of all valid\n  attributes for this message type.\n\n  Args:\n    message_descriptor: A Descriptor instance describing this message type.\n    dictionary: Class dictionary to which we'll add a '__slots__' entry.\n  "
    dictionary['__slots__'] = ['_cached_byte_size', '_cached_byte_size_dirty', '_fields', '_unknown_fields', '_is_present_in_parent', '_listener', '_listener_for_children', '__weakref__', '_oneofs']

def _IsMessageSetExtension(field):
    if False:
        while True:
            i = 10
    return field.is_extension and field.containing_type.has_options and field.containing_type.GetOptions().message_set_wire_format and (field.type == _FieldDescriptor.TYPE_MESSAGE) and (field.label == _FieldDescriptor.LABEL_OPTIONAL)

def _IsMapField(field):
    if False:
        return 10
    return field.type == _FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options and field.message_type.GetOptions().map_entry

def _IsMessageMapField(field):
    if False:
        i = 10
        return i + 15
    value_type = field.message_type.fields_by_name['value']
    return value_type.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE

def _AttachFieldHelpers(cls, field_descriptor):
    if False:
        while True:
            i = 10
    is_repeated = field_descriptor.label == _FieldDescriptor.LABEL_REPEATED
    is_packable = is_repeated and wire_format.IsTypePackable(field_descriptor.type)
    if not is_packable:
        is_packed = False
    elif field_descriptor.containing_type.syntax == 'proto2':
        is_packed = field_descriptor.has_options and field_descriptor.GetOptions().packed
    else:
        has_packed_false = field_descriptor.has_options and field_descriptor.GetOptions().HasField('packed') and (field_descriptor.GetOptions().packed == False)
        is_packed = not has_packed_false
    is_map_entry = _IsMapField(field_descriptor)
    if is_map_entry:
        field_encoder = encoder.MapEncoder(field_descriptor)
        sizer = encoder.MapSizer(field_descriptor)
    elif _IsMessageSetExtension(field_descriptor):
        field_encoder = encoder.MessageSetItemEncoder(field_descriptor.number)
        sizer = encoder.MessageSetItemSizer(field_descriptor.number)
    else:
        field_encoder = type_checkers.TYPE_TO_ENCODER[field_descriptor.type](field_descriptor.number, is_repeated, is_packed)
        sizer = type_checkers.TYPE_TO_SIZER[field_descriptor.type](field_descriptor.number, is_repeated, is_packed)
    field_descriptor._encoder = field_encoder
    field_descriptor._sizer = sizer
    field_descriptor._default_constructor = _DefaultValueConstructorForField(field_descriptor)

    def AddDecoder(wiretype, is_packed):
        if False:
            while True:
                i = 10
        tag_bytes = encoder.TagBytes(field_descriptor.number, wiretype)
        decode_type = field_descriptor.type
        if decode_type == _FieldDescriptor.TYPE_ENUM and type_checkers.SupportsOpenEnums(field_descriptor):
            decode_type = _FieldDescriptor.TYPE_INT32
        oneof_descriptor = None
        if field_descriptor.containing_oneof is not None:
            oneof_descriptor = field_descriptor
        if is_map_entry:
            is_message_map = _IsMessageMapField(field_descriptor)
            field_decoder = decoder.MapDecoder(field_descriptor, _GetInitializeDefaultForMap(field_descriptor), is_message_map)
        else:
            field_decoder = type_checkers.TYPE_TO_DECODER[decode_type](field_descriptor.number, is_repeated, is_packed, field_descriptor, field_descriptor._default_constructor)
        cls._decoders_by_tag[tag_bytes] = (field_decoder, oneof_descriptor)
    AddDecoder(type_checkers.FIELD_TYPE_TO_WIRE_TYPE[field_descriptor.type], False)
    if is_repeated and wire_format.IsTypePackable(field_descriptor.type):
        AddDecoder(wire_format.WIRETYPE_LENGTH_DELIMITED, True)

def _AddClassAttributesForNestedExtensions(descriptor, dictionary):
    if False:
        for i in range(10):
            print('nop')
    extension_dict = descriptor.extensions_by_name
    for (extension_name, extension_field) in extension_dict.items():
        assert extension_name not in dictionary
        dictionary[extension_name] = extension_field

def _AddEnumValues(descriptor, cls):
    if False:
        while True:
            i = 10
    "Sets class-level attributes for all enum fields defined in this message.\n\n  Also exporting a class-level object that can name enum values.\n\n  Args:\n    descriptor: Descriptor object for this message type.\n    cls: Class we're constructing for this message type.\n  "
    for enum_type in descriptor.enum_types:
        setattr(cls, enum_type.name, enum_type_wrapper.EnumTypeWrapper(enum_type))
        for enum_value in enum_type.values:
            setattr(cls, enum_value.name, enum_value.number)

def _GetInitializeDefaultForMap(field):
    if False:
        return 10
    if field.label != _FieldDescriptor.LABEL_REPEATED:
        raise ValueError('map_entry set on non-repeated field %s' % field.name)
    fields_by_name = field.message_type.fields_by_name
    key_checker = type_checkers.GetTypeChecker(fields_by_name['key'])
    value_field = fields_by_name['value']
    if _IsMessageMapField(field):

        def MakeMessageMapDefault(message):
            if False:
                for i in range(10):
                    print('nop')
            return containers.MessageMap(message._listener_for_children, value_field.message_type, key_checker)
        return MakeMessageMapDefault
    else:
        value_checker = type_checkers.GetTypeChecker(value_field)

        def MakePrimitiveMapDefault(message):
            if False:
                for i in range(10):
                    print('nop')
            return containers.ScalarMap(message._listener_for_children, key_checker, value_checker)
        return MakePrimitiveMapDefault

def _DefaultValueConstructorForField(field):
    if False:
        for i in range(10):
            print('nop')
    'Returns a function which returns a default value for a field.\n\n  Args:\n    field: FieldDescriptor object for this field.\n\n  The returned function has one argument:\n    message: Message instance containing this field, or a weakref proxy\n      of same.\n\n  That function in turn returns a default value for this field.  The default\n    value may refer back to |message| via a weak reference.\n  '
    if _IsMapField(field):
        return _GetInitializeDefaultForMap(field)
    if field.label == _FieldDescriptor.LABEL_REPEATED:
        if field.has_default_value and field.default_value != []:
            raise ValueError('Repeated field default value not empty list: %s' % field.default_value)
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            message_type = field.message_type

            def MakeRepeatedMessageDefault(message):
                if False:
                    while True:
                        i = 10
                return containers.RepeatedCompositeFieldContainer(message._listener_for_children, field.message_type)
            return MakeRepeatedMessageDefault
        else:
            type_checker = type_checkers.GetTypeChecker(field)

            def MakeRepeatedScalarDefault(message):
                if False:
                    print('Hello World!')
                return containers.RepeatedScalarFieldContainer(message._listener_for_children, type_checker)
            return MakeRepeatedScalarDefault
    if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        message_type = field.message_type

        def MakeSubMessageDefault(message):
            if False:
                while True:
                    i = 10
            result = message_type._concrete_class()
            result._SetListener(_OneofListener(message, field) if field.containing_oneof is not None else message._listener_for_children)
            return result
        return MakeSubMessageDefault

    def MakeScalarDefault(message):
        if False:
            while True:
                i = 10
        return field.default_value
    return MakeScalarDefault

def _ReraiseTypeErrorWithFieldName(message_name, field_name):
    if False:
        for i in range(10):
            print('nop')
    'Re-raise the currently-handled TypeError with the field name added.'
    exc = sys.exc_info()[1]
    if len(exc.args) == 1 and type(exc) is TypeError:
        exc = TypeError('%s for field %s.%s' % (str(exc), message_name, field_name))
    six.reraise(type(exc), exc, sys.exc_info()[2])

def _AddInitMethod(message_descriptor, cls):
    if False:
        return 10
    'Adds an __init__ method to cls.'

    def _GetIntegerEnumValue(enum_type, value):
        if False:
            return 10
        "Convert a string or integer enum value to an integer.\n\n    If the value is a string, it is converted to the enum value in\n    enum_type with the same name.  If the value is not a string, it's\n    returned as-is.  (No conversion or bounds-checking is done.)\n    "
        if isinstance(value, six.string_types):
            try:
                return enum_type.values_by_name[value].number
            except KeyError:
                raise ValueError('Enum type %s: unknown label "%s"' % (enum_type.full_name, value))
        return value

    def init(self, **kwargs):
        if False:
            print('Hello World!')
        self._cached_byte_size = 0
        self._cached_byte_size_dirty = len(kwargs) > 0
        self._fields = {}
        self._oneofs = {}
        self._unknown_fields = ()
        self._is_present_in_parent = False
        self._listener = message_listener_mod.NullMessageListener()
        self._listener_for_children = _Listener(self)
        for (field_name, field_value) in kwargs.items():
            field = _GetFieldByName(message_descriptor, field_name)
            if field is None:
                raise TypeError("%s() got an unexpected keyword argument '%s'" % (message_descriptor.name, field_name))
            if field_value is None:
                continue
            if field.label == _FieldDescriptor.LABEL_REPEATED:
                copy = field._default_constructor(self)
                if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
                    if _IsMapField(field):
                        if _IsMessageMapField(field):
                            for key in field_value:
                                copy[key].MergeFrom(field_value[key])
                        else:
                            copy.update(field_value)
                    else:
                        for val in field_value:
                            if isinstance(val, dict):
                                copy.add(**val)
                            else:
                                copy.add().MergeFrom(val)
                else:
                    if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
                        field_value = [_GetIntegerEnumValue(field.enum_type, val) for val in field_value]
                    copy.extend(field_value)
                self._fields[field] = copy
            elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
                copy = field._default_constructor(self)
                new_val = field_value
                if isinstance(field_value, dict):
                    new_val = field.message_type._concrete_class(**field_value)
                try:
                    copy.MergeFrom(new_val)
                except TypeError:
                    _ReraiseTypeErrorWithFieldName(message_descriptor.name, field_name)
                self._fields[field] = copy
            else:
                if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
                    field_value = _GetIntegerEnumValue(field.enum_type, field_value)
                try:
                    setattr(self, field_name, field_value)
                except TypeError:
                    _ReraiseTypeErrorWithFieldName(message_descriptor.name, field_name)
    init.__module__ = None
    init.__doc__ = None
    cls.__init__ = init

def _GetFieldByName(message_descriptor, field_name):
    if False:
        return 10
    'Returns a field descriptor by field name.\n\n  Args:\n    message_descriptor: A Descriptor describing all fields in message.\n    field_name: The name of the field to retrieve.\n  Returns:\n    The field descriptor associated with the field name.\n  '
    try:
        return message_descriptor.fields_by_name[field_name]
    except KeyError:
        raise ValueError('Protocol message %s has no "%s" field.' % (message_descriptor.name, field_name))

def _AddPropertiesForFields(descriptor, cls):
    if False:
        print('Hello World!')
    'Adds properties for all fields in this protocol message type.'
    for field in descriptor.fields:
        _AddPropertiesForField(field, cls)
    if descriptor.is_extendable:
        cls.Extensions = property(lambda self: _ExtensionDict(self))

def _AddPropertiesForField(field, cls):
    if False:
        i = 10
        return i + 15
    "Adds a public property for a protocol message field.\n  Clients can use this property to get and (in the case\n  of non-repeated scalar fields) directly set the value\n  of a protocol message field.\n\n  Args:\n    field: A FieldDescriptor for this field.\n    cls: The class we're constructing.\n  "
    assert _FieldDescriptor.MAX_CPPTYPE == 10
    constant_name = field.name.upper() + '_FIELD_NUMBER'
    setattr(cls, constant_name, field.number)
    if field.label == _FieldDescriptor.LABEL_REPEATED:
        _AddPropertiesForRepeatedField(field, cls)
    elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        _AddPropertiesForNonRepeatedCompositeField(field, cls)
    else:
        _AddPropertiesForNonRepeatedScalarField(field, cls)

def _AddPropertiesForRepeatedField(field, cls):
    if False:
        print('Hello World!')
    'Adds a public property for a "repeated" protocol message field.  Clients\n  can use this property to get the value of the field, which will be either a\n  _RepeatedScalarFieldContainer or _RepeatedCompositeFieldContainer (see\n  below).\n\n  Note that when clients add values to these containers, we perform\n  type-checking in the case of repeated scalar fields, and we also set any\n  necessary "has" bits as a side-effect.\n\n  Args:\n    field: A FieldDescriptor for this field.\n    cls: The class we\'re constructing.\n  '
    proto_field_name = field.name
    property_name = _PropertyName(proto_field_name)

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        field_value = self._fields.get(field)
        if field_value is None:
            field_value = field._default_constructor(self)
            field_value = self._fields.setdefault(field, field_value)
        return field_value
    getter.__module__ = None
    getter.__doc__ = 'Getter for %s.' % proto_field_name

    def setter(self, new_value):
        if False:
            print('Hello World!')
        raise AttributeError('Assignment not allowed to repeated field "%s" in protocol message object.' % proto_field_name)
    doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
    setattr(cls, property_name, property(getter, setter, doc=doc))

def _AddPropertiesForNonRepeatedScalarField(field, cls):
    if False:
        while True:
            i = 10
    'Adds a public property for a nonrepeated, scalar protocol message field.\n  Clients can use this property to get and directly set the value of the field.\n  Note that when the client sets the value of a field by using this property,\n  all necessary "has" bits are set as a side-effect, and we also perform\n  type-checking.\n\n  Args:\n    field: A FieldDescriptor for this field.\n    cls: The class we\'re constructing.\n  '
    proto_field_name = field.name
    property_name = _PropertyName(proto_field_name)
    type_checker = type_checkers.GetTypeChecker(field)
    default_value = field.default_value
    valid_values = set()
    is_proto3 = field.containing_type.syntax == 'proto3'

    def getter(self):
        if False:
            print('Hello World!')
        return self._fields.get(field, default_value)
    getter.__module__ = None
    getter.__doc__ = 'Getter for %s.' % proto_field_name
    clear_when_set_to_default = is_proto3 and (not field.containing_oneof)

    def field_setter(self, new_value):
        if False:
            for i in range(10):
                print('nop')
        new_value = type_checker.CheckValue(new_value)
        if clear_when_set_to_default and (not new_value):
            self._fields.pop(field, None)
        else:
            self._fields[field] = new_value
        if not self._cached_byte_size_dirty:
            self._Modified()
    if field.containing_oneof:

        def setter(self, new_value):
            if False:
                i = 10
                return i + 15
            field_setter(self, new_value)
            self._UpdateOneofState(field)
    else:
        setter = field_setter
    setter.__module__ = None
    setter.__doc__ = 'Setter for %s.' % proto_field_name
    doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
    setattr(cls, property_name, property(getter, setter, doc=doc))

def _AddPropertiesForNonRepeatedCompositeField(field, cls):
    if False:
        while True:
            i = 10
    'Adds a public property for a nonrepeated, composite protocol message field.\n  A composite field is a "group" or "message" field.\n\n  Clients can use this property to get the value of the field, but cannot\n  assign to the property directly.\n\n  Args:\n    field: A FieldDescriptor for this field.\n    cls: The class we\'re constructing.\n  '
    proto_field_name = field.name
    property_name = _PropertyName(proto_field_name)

    def getter(self):
        if False:
            while True:
                i = 10
        field_value = self._fields.get(field)
        if field_value is None:
            field_value = field._default_constructor(self)
            field_value = self._fields.setdefault(field, field_value)
        return field_value
    getter.__module__ = None
    getter.__doc__ = 'Getter for %s.' % proto_field_name

    def setter(self, new_value):
        if False:
            return 10
        raise AttributeError('Assignment not allowed to composite field "%s" in protocol message object.' % proto_field_name)
    doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
    setattr(cls, property_name, property(getter, setter, doc=doc))

def _AddPropertiesForExtensions(descriptor, cls):
    if False:
        print('Hello World!')
    'Adds properties for all fields in this protocol message type.'
    extension_dict = descriptor.extensions_by_name
    for (extension_name, extension_field) in extension_dict.items():
        constant_name = extension_name.upper() + '_FIELD_NUMBER'
        setattr(cls, constant_name, extension_field.number)

def _AddStaticMethods(cls):
    if False:
        for i in range(10):
            print('nop')

    def RegisterExtension(extension_handle):
        if False:
            while True:
                i = 10
        extension_handle.containing_type = cls.DESCRIPTOR
        _AttachFieldHelpers(cls, extension_handle)
        actual_handle = cls._extensions_by_number.setdefault(extension_handle.number, extension_handle)
        if actual_handle is not extension_handle:
            raise AssertionError('Extensions "%s" and "%s" both try to extend message type "%s" with field number %d.' % (extension_handle.full_name, actual_handle.full_name, cls.DESCRIPTOR.full_name, extension_handle.number))
        cls._extensions_by_name[extension_handle.full_name] = extension_handle
        handle = extension_handle
        if _IsMessageSetExtension(handle):
            cls._extensions_by_name[extension_handle.message_type.full_name] = extension_handle
    cls.RegisterExtension = staticmethod(RegisterExtension)

    def FromString(s):
        if False:
            return 10
        message = cls()
        message.MergeFromString(s)
        return message
    cls.FromString = staticmethod(FromString)

def _IsPresent(item):
    if False:
        return 10
    'Given a (FieldDescriptor, value) tuple from _fields, return true if the\n  value should be included in the list returned by ListFields().'
    if item[0].label == _FieldDescriptor.LABEL_REPEATED:
        return bool(item[1])
    elif item[0].cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        return item[1]._is_present_in_parent
    else:
        return True

def _AddListFieldsMethod(message_descriptor, cls):
    if False:
        print('Hello World!')
    'Helper for _AddMessageMethods().'

    def ListFields(self):
        if False:
            i = 10
            return i + 15
        all_fields = [item for item in self._fields.items() if _IsPresent(item)]
        all_fields.sort(key=lambda item: item[0].number)
        return all_fields
    cls.ListFields = ListFields
_Proto3HasError = 'Protocol message has no non-repeated submessage field "%s"'
_Proto2HasError = 'Protocol message has no non-repeated field "%s"'

def _AddHasFieldMethod(message_descriptor, cls):
    if False:
        return 10
    'Helper for _AddMessageMethods().'
    is_proto3 = message_descriptor.syntax == 'proto3'
    error_msg = _Proto3HasError if is_proto3 else _Proto2HasError
    hassable_fields = {}
    for field in message_descriptor.fields:
        if field.label == _FieldDescriptor.LABEL_REPEATED:
            continue
        if is_proto3 and field.cpp_type != _FieldDescriptor.CPPTYPE_MESSAGE and (not field.containing_oneof):
            continue
        hassable_fields[field.name] = field
    if not is_proto3:
        for oneof in message_descriptor.oneofs:
            hassable_fields[oneof.name] = oneof

    def HasField(self, field_name):
        if False:
            print('Hello World!')
        try:
            field = hassable_fields[field_name]
        except KeyError:
            raise ValueError(error_msg % field_name)
        if isinstance(field, descriptor_mod.OneofDescriptor):
            try:
                return HasField(self, self._oneofs[field].name)
            except KeyError:
                return False
        elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            value = self._fields.get(field)
            return value is not None and value._is_present_in_parent
        else:
            return field in self._fields
    cls.HasField = HasField

def _AddClearFieldMethod(message_descriptor, cls):
    if False:
        print('Hello World!')
    'Helper for _AddMessageMethods().'

    def ClearField(self, field_name):
        if False:
            while True:
                i = 10
        try:
            field = message_descriptor.fields_by_name[field_name]
        except KeyError:
            try:
                field = message_descriptor.oneofs_by_name[field_name]
                if field in self._oneofs:
                    field = self._oneofs[field]
                else:
                    return
            except KeyError:
                raise ValueError('Protocol message %s() has no "%s" field.' % (message_descriptor.name, field_name))
        if field in self._fields:
            if hasattr(self._fields[field], 'InvalidateIterators'):
                self._fields[field].InvalidateIterators()
            del self._fields[field]
            if self._oneofs.get(field.containing_oneof, None) is field:
                del self._oneofs[field.containing_oneof]
        self._Modified()
    cls.ClearField = ClearField

def _AddClearExtensionMethod(cls):
    if False:
        while True:
            i = 10
    'Helper for _AddMessageMethods().'

    def ClearExtension(self, extension_handle):
        if False:
            while True:
                i = 10
        _VerifyExtensionHandle(self, extension_handle)
        if extension_handle in self._fields:
            del self._fields[extension_handle]
        self._Modified()
    cls.ClearExtension = ClearExtension

def _AddHasExtensionMethod(cls):
    if False:
        i = 10
        return i + 15
    'Helper for _AddMessageMethods().'

    def HasExtension(self, extension_handle):
        if False:
            return 10
        _VerifyExtensionHandle(self, extension_handle)
        if extension_handle.label == _FieldDescriptor.LABEL_REPEATED:
            raise KeyError('"%s" is repeated.' % extension_handle.full_name)
        if extension_handle.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            value = self._fields.get(extension_handle)
            return value is not None and value._is_present_in_parent
        else:
            return extension_handle in self._fields
    cls.HasExtension = HasExtension

def _InternalUnpackAny(msg):
    if False:
        print('Hello World!')
    'Unpacks Any message and returns the unpacked message.\n\n  This internal method is differnt from public Any Unpack method which takes\n  the target message as argument. _InternalUnpackAny method does not have\n  target message type and need to find the message type in descriptor pool.\n\n  Args:\n    msg: An Any message to be unpacked.\n\n  Returns:\n    The unpacked message.\n  '
    type_url = msg.type_url
    db = symbol_database.Default()
    if not type_url:
        return None
    type_name = type_url.split('/')[-1]
    descriptor = db.pool.FindMessageTypeByName(type_name)
    if descriptor is None:
        return None
    message_class = db.GetPrototype(descriptor)
    message = message_class()
    message.ParseFromString(msg.value)
    return message

def _AddEqualsMethod(message_descriptor, cls):
    if False:
        for i in range(10):
            print('nop')
    'Helper for _AddMessageMethods().'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, message_mod.Message) or other.DESCRIPTOR != self.DESCRIPTOR:
            return False
        if self is other:
            return True
        if self.DESCRIPTOR.full_name == _AnyFullTypeName:
            any_a = _InternalUnpackAny(self)
            any_b = _InternalUnpackAny(other)
            if any_a and any_b:
                return any_a == any_b
        if not self.ListFields() == other.ListFields():
            return False
        unknown_fields = list(self._unknown_fields)
        unknown_fields.sort()
        other_unknown_fields = list(other._unknown_fields)
        other_unknown_fields.sort()
        return unknown_fields == other_unknown_fields
    cls.__eq__ = __eq__

def _AddStrMethod(message_descriptor, cls):
    if False:
        print('Hello World!')
    'Helper for _AddMessageMethods().'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return text_format.MessageToString(self)
    cls.__str__ = __str__

def _AddReprMethod(message_descriptor, cls):
    if False:
        print('Hello World!')
    'Helper for _AddMessageMethods().'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return text_format.MessageToString(self)
    cls.__repr__ = __repr__

def _AddUnicodeMethod(unused_message_descriptor, cls):
    if False:
        for i in range(10):
            print('nop')
    'Helper for _AddMessageMethods().'

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        return text_format.MessageToString(self, as_utf8=True).decode('utf-8')
    cls.__unicode__ = __unicode__

def _BytesForNonRepeatedElement(value, field_number, field_type):
    if False:
        for i in range(10):
            print('nop')
    "Returns the number of bytes needed to serialize a non-repeated element.\n  The returned byte count includes space for tag information and any\n  other additional space associated with serializing value.\n\n  Args:\n    value: Value we're serializing.\n    field_number: Field number of this value.  (Since the field number\n      is stored as part of a varint-encoded tag, this has an impact\n      on the total bytes required to serialize the value).\n    field_type: The type of the field.  One of the TYPE_* constants\n      within FieldDescriptor.\n  "
    try:
        fn = type_checkers.TYPE_TO_BYTE_SIZE_FN[field_type]
        return fn(field_number, value)
    except KeyError:
        raise message_mod.EncodeError('Unrecognized field type: %d' % field_type)

def _AddByteSizeMethod(message_descriptor, cls):
    if False:
        i = 10
        return i + 15
    'Helper for _AddMessageMethods().'

    def ByteSize(self):
        if False:
            i = 10
            return i + 15
        if not self._cached_byte_size_dirty:
            return self._cached_byte_size
        size = 0
        for (field_descriptor, field_value) in self.ListFields():
            size += field_descriptor._sizer(field_value)
        for (tag_bytes, value_bytes) in self._unknown_fields:
            size += len(tag_bytes) + len(value_bytes)
        self._cached_byte_size = size
        self._cached_byte_size_dirty = False
        self._listener_for_children.dirty = False
        return size
    cls.ByteSize = ByteSize

def _AddSerializeToStringMethod(message_descriptor, cls):
    if False:
        for i in range(10):
            print('nop')
    'Helper for _AddMessageMethods().'

    def SerializeToString(self):
        if False:
            for i in range(10):
                print('nop')
        errors = []
        if not self.IsInitialized():
            raise message_mod.EncodeError('Message %s is missing required fields: %s' % (self.DESCRIPTOR.full_name, ','.join(self.FindInitializationErrors())))
        return self.SerializePartialToString()
    cls.SerializeToString = SerializeToString

def _AddSerializePartialToStringMethod(message_descriptor, cls):
    if False:
        while True:
            i = 10
    'Helper for _AddMessageMethods().'

    def SerializePartialToString(self):
        if False:
            return 10
        out = BytesIO()
        self._InternalSerialize(out.write)
        return out.getvalue()
    cls.SerializePartialToString = SerializePartialToString

    def InternalSerialize(self, write_bytes):
        if False:
            return 10
        for (field_descriptor, field_value) in self.ListFields():
            field_descriptor._encoder(write_bytes, field_value)
        for (tag_bytes, value_bytes) in self._unknown_fields:
            write_bytes(tag_bytes)
            write_bytes(value_bytes)
    cls._InternalSerialize = InternalSerialize

def _AddMergeFromStringMethod(message_descriptor, cls):
    if False:
        for i in range(10):
            print('nop')
    'Helper for _AddMessageMethods().'

    def MergeFromString(self, serialized):
        if False:
            print('Hello World!')
        length = len(serialized)
        try:
            if self._InternalParse(serialized, 0, length) != length:
                raise message_mod.DecodeError('Unexpected end-group tag.')
        except (IndexError, TypeError):
            raise message_mod.DecodeError('Truncated message.')
        except struct.error as e:
            raise message_mod.DecodeError(e)
        return length
    cls.MergeFromString = MergeFromString
    local_ReadTag = decoder.ReadTag
    local_SkipField = decoder.SkipField
    decoders_by_tag = cls._decoders_by_tag
    is_proto3 = message_descriptor.syntax == 'proto3'

    def InternalParse(self, buffer, pos, end):
        if False:
            print('Hello World!')
        self._Modified()
        field_dict = self._fields
        unknown_field_list = self._unknown_fields
        while pos != end:
            (tag_bytes, new_pos) = local_ReadTag(buffer, pos)
            (field_decoder, field_desc) = decoders_by_tag.get(tag_bytes, (None, None))
            if field_decoder is None:
                value_start_pos = new_pos
                new_pos = local_SkipField(buffer, new_pos, end, tag_bytes)
                if new_pos == -1:
                    return pos
                if not is_proto3:
                    if not unknown_field_list:
                        unknown_field_list = self._unknown_fields = []
                    unknown_field_list.append((tag_bytes, buffer[value_start_pos:new_pos]))
                pos = new_pos
            else:
                pos = field_decoder(buffer, new_pos, end, self, field_dict)
                if field_desc:
                    self._UpdateOneofState(field_desc)
        return pos
    cls._InternalParse = InternalParse

def _AddIsInitializedMethod(message_descriptor, cls):
    if False:
        for i in range(10):
            print('nop')
    'Adds the IsInitialized and FindInitializationError methods to the\n  protocol message class.'
    required_fields = [field for field in message_descriptor.fields if field.label == _FieldDescriptor.LABEL_REQUIRED]

    def IsInitialized(self, errors=None):
        if False:
            return 10
        'Checks if all required fields of a message are set.\n\n    Args:\n      errors:  A list which, if provided, will be populated with the field\n               paths of all missing required fields.\n\n    Returns:\n      True iff the specified message has all required fields set.\n    '
        for field in required_fields:
            if field not in self._fields or (field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE and (not self._fields[field]._is_present_in_parent)):
                if errors is not None:
                    errors.extend(self.FindInitializationErrors())
                return False
        for (field, value) in list(self._fields.items()):
            if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
                if field.label == _FieldDescriptor.LABEL_REPEATED:
                    if field.message_type.has_options and field.message_type.GetOptions().map_entry:
                        continue
                    for element in value:
                        if not element.IsInitialized():
                            if errors is not None:
                                errors.extend(self.FindInitializationErrors())
                            return False
                elif value._is_present_in_parent and (not value.IsInitialized()):
                    if errors is not None:
                        errors.extend(self.FindInitializationErrors())
                    return False
        return True
    cls.IsInitialized = IsInitialized

    def FindInitializationErrors(self):
        if False:
            i = 10
            return i + 15
        'Finds required fields which are not initialized.\n\n    Returns:\n      A list of strings.  Each string is a path to an uninitialized field from\n      the top-level message, e.g. "foo.bar[5].baz".\n    '
        errors = []
        for field in required_fields:
            if not self.HasField(field.name):
                errors.append(field.name)
        for (field, value) in self.ListFields():
            if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
                if field.is_extension:
                    name = '(%s)' % field.full_name
                else:
                    name = field.name
                if _IsMapField(field):
                    if _IsMessageMapField(field):
                        for key in value:
                            element = value[key]
                            prefix = '%s[%s].' % (name, key)
                            sub_errors = element.FindInitializationErrors()
                            errors += [prefix + error for error in sub_errors]
                    else:
                        pass
                elif field.label == _FieldDescriptor.LABEL_REPEATED:
                    for i in range(len(value)):
                        element = value[i]
                        prefix = '%s[%d].' % (name, i)
                        sub_errors = element.FindInitializationErrors()
                        errors += [prefix + error for error in sub_errors]
                else:
                    prefix = name + '.'
                    sub_errors = value.FindInitializationErrors()
                    errors += [prefix + error for error in sub_errors]
        return errors
    cls.FindInitializationErrors = FindInitializationErrors

def _AddMergeFromMethod(cls):
    if False:
        i = 10
        return i + 15
    LABEL_REPEATED = _FieldDescriptor.LABEL_REPEATED
    CPPTYPE_MESSAGE = _FieldDescriptor.CPPTYPE_MESSAGE

    def MergeFrom(self, msg):
        if False:
            return 10
        if not isinstance(msg, cls):
            raise TypeError('Parameter to MergeFrom() must be instance of same class: expected %s got %s.' % (cls.__name__, type(msg).__name__))
        assert msg is not self
        self._Modified()
        fields = self._fields
        for (field, value) in msg._fields.items():
            if field.label == LABEL_REPEATED:
                field_value = fields.get(field)
                if field_value is None:
                    field_value = field._default_constructor(self)
                    fields[field] = field_value
                field_value.MergeFrom(value)
            elif field.cpp_type == CPPTYPE_MESSAGE:
                if value._is_present_in_parent:
                    field_value = fields.get(field)
                    if field_value is None:
                        field_value = field._default_constructor(self)
                        fields[field] = field_value
                    field_value.MergeFrom(value)
            else:
                self._fields[field] = value
                if field.containing_oneof:
                    self._UpdateOneofState(field)
        if msg._unknown_fields:
            if not self._unknown_fields:
                self._unknown_fields = []
            self._unknown_fields.extend(msg._unknown_fields)
    cls.MergeFrom = MergeFrom

def _AddWhichOneofMethod(message_descriptor, cls):
    if False:
        return 10

    def WhichOneof(self, oneof_name):
        if False:
            i = 10
            return i + 15
        'Returns the name of the currently set field inside a oneof, or None.'
        try:
            field = message_descriptor.oneofs_by_name[oneof_name]
        except KeyError:
            raise ValueError('Protocol message has no oneof "%s" field.' % oneof_name)
        nested_field = self._oneofs.get(field, None)
        if nested_field is not None and self.HasField(nested_field.name):
            return nested_field.name
        else:
            return None
    cls.WhichOneof = WhichOneof

def _Clear(self):
    if False:
        while True:
            i = 10
    self._fields = {}
    self._unknown_fields = ()
    self._oneofs = {}
    self._Modified()

def _DiscardUnknownFields(self):
    if False:
        return 10
    self._unknown_fields = []
    for (field, value) in self.ListFields():
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            if field.label == _FieldDescriptor.LABEL_REPEATED:
                for sub_message in value:
                    sub_message.DiscardUnknownFields()
            else:
                value.DiscardUnknownFields()

def _SetListener(self, listener):
    if False:
        for i in range(10):
            print('nop')
    if listener is None:
        self._listener = message_listener_mod.NullMessageListener()
    else:
        self._listener = listener

def _AddMessageMethods(message_descriptor, cls):
    if False:
        while True:
            i = 10
    'Adds implementations of all Message methods to cls.'
    _AddListFieldsMethod(message_descriptor, cls)
    _AddHasFieldMethod(message_descriptor, cls)
    _AddClearFieldMethod(message_descriptor, cls)
    if message_descriptor.is_extendable:
        _AddClearExtensionMethod(cls)
        _AddHasExtensionMethod(cls)
    _AddEqualsMethod(message_descriptor, cls)
    _AddStrMethod(message_descriptor, cls)
    _AddReprMethod(message_descriptor, cls)
    _AddUnicodeMethod(message_descriptor, cls)
    _AddByteSizeMethod(message_descriptor, cls)
    _AddSerializeToStringMethod(message_descriptor, cls)
    _AddSerializePartialToStringMethod(message_descriptor, cls)
    _AddMergeFromStringMethod(message_descriptor, cls)
    _AddIsInitializedMethod(message_descriptor, cls)
    _AddMergeFromMethod(cls)
    _AddWhichOneofMethod(message_descriptor, cls)
    cls.Clear = _Clear
    cls.DiscardUnknownFields = _DiscardUnknownFields
    cls._SetListener = _SetListener

def _AddPrivateHelperMethods(message_descriptor, cls):
    if False:
        while True:
            i = 10
    'Adds implementation of private helper methods to cls.'

    def Modified(self):
        if False:
            print('Hello World!')
        'Sets the _cached_byte_size_dirty bit to true,\n    and propagates this to our listener iff this was a state change.\n    '
        if not self._cached_byte_size_dirty:
            self._cached_byte_size_dirty = True
            self._listener_for_children.dirty = True
            self._is_present_in_parent = True
            self._listener.Modified()

    def _UpdateOneofState(self, field):
        if False:
            print('Hello World!')
        'Sets field as the active field in its containing oneof.\n\n    Will also delete currently active field in the oneof, if it is different\n    from the argument. Does not mark the message as modified.\n    '
        other_field = self._oneofs.setdefault(field.containing_oneof, field)
        if other_field is not field:
            del self._fields[other_field]
            self._oneofs[field.containing_oneof] = field
    cls._Modified = Modified
    cls.SetInParent = Modified
    cls._UpdateOneofState = _UpdateOneofState

class _Listener(object):
    """MessageListener implementation that a parent message registers with its
  child message.

  In order to support semantics like:

    foo.bar.baz.qux = 23
    assert foo.HasField('bar')

  ...child objects must have back references to their parents.
  This helper class is at the heart of this support.
  """

    def __init__(self, parent_message):
        if False:
            return 10
        'Args:\n      parent_message: The message whose _Modified() method we should call when\n        we receive Modified() messages.\n    '
        if isinstance(parent_message, weakref.ProxyType):
            self._parent_message_weakref = parent_message
        else:
            self._parent_message_weakref = weakref.proxy(parent_message)
        self.dirty = False

    def Modified(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dirty:
            return
        try:
            self._parent_message_weakref._Modified()
        except ReferenceError:
            pass

class _OneofListener(_Listener):
    """Special listener implementation for setting composite oneof fields."""

    def __init__(self, parent_message, field):
        if False:
            return 10
        'Args:\n      parent_message: The message whose _Modified() method we should call when\n        we receive Modified() messages.\n      field: The descriptor of the field being set in the parent message.\n    '
        super(_OneofListener, self).__init__(parent_message)
        self._field = field

    def Modified(self):
        if False:
            i = 10
            return i + 15
        'Also updates the state of the containing oneof in the parent message.'
        try:
            self._parent_message_weakref._UpdateOneofState(self._field)
            super(_OneofListener, self).Modified()
        except ReferenceError:
            pass

class _ExtensionDict(object):
    """Dict-like container for supporting an indexable "Extensions"
  field on proto instances.

  Note that in all cases we expect extension handles to be
  FieldDescriptors.
  """

    def __init__(self, extended_message):
        if False:
            print('Hello World!')
        'extended_message: Message instance for which we are the Extensions dict.\n    '
        self._extended_message = extended_message

    def __getitem__(self, extension_handle):
        if False:
            i = 10
            return i + 15
        'Returns the current value of the given extension handle.'
        _VerifyExtensionHandle(self._extended_message, extension_handle)
        result = self._extended_message._fields.get(extension_handle)
        if result is not None:
            return result
        if extension_handle.label == _FieldDescriptor.LABEL_REPEATED:
            result = extension_handle._default_constructor(self._extended_message)
        elif extension_handle.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            result = extension_handle.message_type._concrete_class()
            try:
                result._SetListener(self._extended_message._listener_for_children)
            except ReferenceError:
                pass
        else:
            return extension_handle.default_value
        result = self._extended_message._fields.setdefault(extension_handle, result)
        return result

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, self.__class__):
            return False
        my_fields = self._extended_message.ListFields()
        other_fields = other._extended_message.ListFields()
        my_fields = [field for field in my_fields if field.is_extension]
        other_fields = [field for field in other_fields if field.is_extension]
        return my_fields == other_fields

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def __hash__(self):
        if False:
            print('Hello World!')
        raise TypeError('unhashable object')

    def __setitem__(self, extension_handle, value):
        if False:
            print('Hello World!')
        'If extension_handle specifies a non-repeated, scalar extension\n    field, sets the value of that field.\n    '
        _VerifyExtensionHandle(self._extended_message, extension_handle)
        if extension_handle.label == _FieldDescriptor.LABEL_REPEATED or extension_handle.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            raise TypeError('Cannot assign to extension "%s" because it is a repeated or composite type.' % extension_handle.full_name)
        type_checker = type_checkers.GetTypeChecker(extension_handle)
        self._extended_message._fields[extension_handle] = type_checker.CheckValue(value)
        self._extended_message._Modified()

    def _FindExtensionByName(self, name):
        if False:
            print('Hello World!')
        'Tries to find a known extension with the specified name.\n\n    Args:\n      name: Extension full name.\n\n    Returns:\n      Extension field descriptor.\n    '
        return self._extended_message._extensions_by_name.get(name, None)

    def _FindExtensionByNumber(self, number):
        if False:
            return 10
        'Tries to find a known extension with the field number.\n\n    Args:\n      number: Extension field number.\n\n    Returns:\n      Extension field descriptor.\n    '
        return self._extended_message._extensions_by_number.get(number, None)