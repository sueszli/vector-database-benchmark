"""Dynamic Protobuf class creator."""
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import hashlib
import os
from google.protobuf import descriptor_pb2
from google.protobuf import message_factory

def _GetMessageFromFactory(factory, full_name):
    if False:
        i = 10
        return i + 15
    "Get a proto class from the MessageFactory by name.\n\n  Args:\n    factory: a MessageFactory instance.\n    full_name: str, the fully qualified name of the proto type.\n  Returns:\n    A class, for the type identified by full_name.\n  Raises:\n    KeyError, if the proto is not found in the factory's descriptor pool.\n  "
    proto_descriptor = factory.pool.FindMessageTypeByName(full_name)
    proto_cls = factory.GetPrototype(proto_descriptor)
    return proto_cls

def MakeSimpleProtoClass(fields, full_name=None, pool=None):
    if False:
        for i in range(10):
            print('nop')
    "Create a Protobuf class whose fields are basic types.\n\n  Note: this doesn't validate field names!\n\n  Args:\n    fields: dict of {name: field_type} mappings for each field in the proto. If\n        this is an OrderedDict the order will be maintained, otherwise the\n        fields will be sorted by name.\n    full_name: optional str, the fully-qualified name of the proto type.\n    pool: optional DescriptorPool instance.\n  Returns:\n    a class, the new protobuf class with a FileDescriptor.\n  "
    factory = message_factory.MessageFactory(pool=pool)
    if full_name is not None:
        try:
            proto_cls = _GetMessageFromFactory(factory, full_name)
            return proto_cls
        except KeyError:
            pass
    field_items = fields.items()
    if not isinstance(fields, OrderedDict):
        field_items = sorted(field_items)
    fields_hash = hashlib.sha1()
    for (f_name, f_type) in field_items:
        fields_hash.update(f_name.encode('utf-8'))
        fields_hash.update(str(f_type).encode('utf-8'))
    proto_file_name = fields_hash.hexdigest() + '.proto'
    if full_name is None:
        full_name = 'net.proto2.python.public.proto_builder.AnonymousProto_' + fields_hash.hexdigest()
        try:
            proto_cls = _GetMessageFromFactory(factory, full_name)
            return proto_cls
        except KeyError:
            pass
    factory.pool.Add(_MakeFileDescriptorProto(proto_file_name, full_name, field_items))
    return _GetMessageFromFactory(factory, full_name)

def _MakeFileDescriptorProto(proto_file_name, full_name, field_items):
    if False:
        print('Hello World!')
    "Populate FileDescriptorProto for MessageFactory's DescriptorPool."
    (package, name) = full_name.rsplit('.', 1)
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = os.path.join(package.replace('.', '/'), proto_file_name)
    file_proto.package = package
    desc_proto = file_proto.message_type.add()
    desc_proto.name = name
    for (f_number, (f_name, f_type)) in enumerate(field_items, 1):
        field_proto = desc_proto.field.add()
        field_proto.name = f_name
        field_proto.number = f_number
        field_proto.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field_proto.type = f_type
    return file_proto