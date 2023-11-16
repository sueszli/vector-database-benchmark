"""Provides a factory class for generating dynamic messages.

The easiest way to use this class is if you have access to the FileDescriptor
protos containing the messages you want to create you can just do the following:

message_classes = message_factory.GetMessages(iterable_of_file_descriptors)
my_proto_instance = message_classes['some.proto.package.MessageName']()
"""
__author__ = 'matthewtoia@google.com (Matt Toia)'
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import reflection

class MessageFactory(object):
    """Factory for creating Proto2 messages from descriptors in a pool."""

    def __init__(self, pool=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a new factory.'
        self.pool = pool or descriptor_pool.DescriptorPool()
        self._classes = {}

    def GetPrototype(self, descriptor):
        if False:
            print('Hello World!')
        'Builds a proto2 message class based on the passed in descriptor.\n\n    Passing a descriptor with a fully qualified name matching a previous\n    invocation will cause the same class to be returned.\n\n    Args:\n      descriptor: The descriptor to build from.\n\n    Returns:\n      A class describing the passed in descriptor.\n    '
        if descriptor.full_name not in self._classes:
            descriptor_name = descriptor.name
            if str is bytes:
                descriptor_name = descriptor.name.encode('ascii', 'ignore')
            result_class = reflection.GeneratedProtocolMessageType(descriptor_name, (message.Message,), {'DESCRIPTOR': descriptor, '__module__': None})
            self._classes[descriptor.full_name] = result_class
            for field in descriptor.fields:
                if field.message_type:
                    self.GetPrototype(field.message_type)
            for extension in result_class.DESCRIPTOR.extensions:
                if extension.containing_type.full_name not in self._classes:
                    self.GetPrototype(extension.containing_type)
                extended_class = self._classes[extension.containing_type.full_name]
                extended_class.RegisterExtension(extension)
        return self._classes[descriptor.full_name]

    def GetMessages(self, files):
        if False:
            return 10
        'Gets all the messages from a specified file.\n\n    This will find and resolve dependencies, failing if the descriptor\n    pool cannot satisfy them.\n\n    Args:\n      files: The file names to extract messages from.\n\n    Returns:\n      A dictionary mapping proto names to the message classes. This will include\n      any dependent messages as well as any messages defined in the same file as\n      a specified message.\n    '
        result = {}
        for file_name in files:
            file_desc = self.pool.FindFileByName(file_name)
            for (name, msg) in file_desc.message_types_by_name.items():
                if file_desc.package:
                    full_name = '.'.join([file_desc.package, name])
                else:
                    full_name = msg.name
                result[full_name] = self.GetPrototype(self.pool.FindMessageTypeByName(full_name))
            for (name, extension) in file_desc.extensions_by_name.items():
                if extension.containing_type.full_name not in self._classes:
                    self.GetPrototype(extension.containing_type)
                extended_class = self._classes[extension.containing_type.full_name]
                extended_class.RegisterExtension(extension)
        return result
_FACTORY = MessageFactory()

def GetMessages(file_protos):
    if False:
        while True:
            i = 10
    'Builds a dictionary of all the messages available in a set of files.\n\n  Args:\n    file_protos: A sequence of file protos to build messages out of.\n\n  Returns:\n    A dictionary mapping proto names to the message classes. This will include\n    any dependent messages as well as any messages defined in the same file as\n    a specified message.\n  '
    for file_proto in file_protos:
        _FACTORY.pool.Add(file_proto)
    return _FACTORY.GetMessages([file_proto.name for file_proto in file_protos])