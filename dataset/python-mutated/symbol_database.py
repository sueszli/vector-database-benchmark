"""A database of Python protocol buffer generated symbols.

SymbolDatabase makes it easy to create new instances of a registered type, given
only the type's protocol buffer symbol name. Once all symbols are registered,
they can be accessed using either the MessageFactory interface which
SymbolDatabase exposes, or the DescriptorPool interface of the underlying
pool.

Example usage:

  db = symbol_database.SymbolDatabase()

  # Register symbols of interest, from one or multiple files.
  db.RegisterFileDescriptor(my_proto_pb2.DESCRIPTOR)
  db.RegisterMessage(my_proto_pb2.MyMessage)
  db.RegisterEnumDescriptor(my_proto_pb2.MyEnum.DESCRIPTOR)

  # The database can be used as a MessageFactory, to generate types based on
  # their name:
  types = db.GetMessages(['my_proto.proto'])
  my_message_instance = types['MyMessage']()

  # The database's underlying descriptor pool can be queried, so it's not
  # necessary to know a type's filename to be able to generate it:
  filename = db.pool.FindFileContainingSymbol('MyMessage')
  my_message_instance = db.GetMessages([filename])['MyMessage']()

  # This functionality is also provided directly via a convenience method:
  my_message_instance = db.GetSymbol('MyMessage')()
"""
from google.protobuf import descriptor_pool

class SymbolDatabase(object):
    """A database of Python generated symbols.

  SymbolDatabase also models message_factory.MessageFactory.

  The symbol database can be used to keep a global registry of all protocol
  buffer types used within a program.
  """

    def __init__(self, pool=None):
        if False:
            i = 10
            return i + 15
        'Constructor.'
        self._symbols = {}
        self._symbols_by_file = {}
        self.pool = pool or descriptor_pool.Default()

    def RegisterMessage(self, message):
        if False:
            print('Hello World!')
        'Registers the given message type in the local database.\n\n    Args:\n      message: a message.Message, to be registered.\n\n    Returns:\n      The provided message.\n    '
        desc = message.DESCRIPTOR
        self._symbols[desc.full_name] = message
        if desc.file.name not in self._symbols_by_file:
            self._symbols_by_file[desc.file.name] = {}
        self._symbols_by_file[desc.file.name][desc.full_name] = message
        self.pool.AddDescriptor(desc)
        return message

    def RegisterEnumDescriptor(self, enum_descriptor):
        if False:
            i = 10
            return i + 15
        'Registers the given enum descriptor in the local database.\n\n    Args:\n      enum_descriptor: a descriptor.EnumDescriptor.\n\n    Returns:\n      The provided descriptor.\n    '
        self.pool.AddEnumDescriptor(enum_descriptor)
        return enum_descriptor

    def RegisterFileDescriptor(self, file_descriptor):
        if False:
            return 10
        'Registers the given file descriptor in the local database.\n\n    Args:\n      file_descriptor: a descriptor.FileDescriptor.\n\n    Returns:\n      The provided descriptor.\n    '
        self.pool.AddFileDescriptor(file_descriptor)

    def GetSymbol(self, symbol):
        if False:
            i = 10
            return i + 15
        'Tries to find a symbol in the local database.\n\n    Currently, this method only returns message.Message instances, however, if\n    may be extended in future to support other symbol types.\n\n    Args:\n      symbol: A str, a protocol buffer symbol.\n\n    Returns:\n      A Python class corresponding to the symbol.\n\n    Raises:\n      KeyError: if the symbol could not be found.\n    '
        return self._symbols[symbol]

    def GetPrototype(self, descriptor):
        if False:
            for i in range(10):
                print('nop')
        'Builds a proto2 message class based on the passed in descriptor.\n\n    Passing a descriptor with a fully qualified name matching a previous\n    invocation will cause the same class to be returned.\n\n    Args:\n      descriptor: The descriptor to build from.\n\n    Returns:\n      A class describing the passed in descriptor.\n    '
        return self.GetSymbol(descriptor.full_name)

    def GetMessages(self, files):
        if False:
            while True:
                i = 10
        'Gets all the messages from a specified file.\n\n    This will find and resolve dependencies, failing if they are not registered\n    in the symbol database.\n\n\n    Args:\n      files: The file names to extract messages from.\n\n    Returns:\n      A dictionary mapping proto names to the message classes. This will include\n      any dependent messages as well as any messages defined in the same file as\n      a specified message.\n\n    Raises:\n      KeyError: if a file could not be found.\n    '
        result = {}
        for f in files:
            result.update(self._symbols_by_file[f])
        return result
_DEFAULT = SymbolDatabase(pool=descriptor_pool.Default())

def Default():
    if False:
        while True:
            i = 10
    'Returns the default SymbolDatabase.'
    return _DEFAULT