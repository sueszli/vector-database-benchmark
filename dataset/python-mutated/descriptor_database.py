"""Provides a container for DescriptorProtos."""
__author__ = 'matthewtoia@google.com (Matt Toia)'

class Error(Exception):
    pass

class DescriptorDatabaseConflictingDefinitionError(Error):
    """Raised when a proto is added with the same name & different descriptor."""

class DescriptorDatabase(object):
    """A container accepting FileDescriptorProtos and maps DescriptorProtos."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._file_desc_protos_by_file = {}
        self._file_desc_protos_by_symbol = {}

    def Add(self, file_desc_proto):
        if False:
            i = 10
            return i + 15
        'Adds the FileDescriptorProto and its types to this database.\n\n    Args:\n      file_desc_proto: The FileDescriptorProto to add.\n    Raises:\n      DescriptorDatabaseException: if an attempt is made to add a proto\n        with the same name but different definition than an exisiting\n        proto in the database.\n    '
        proto_name = file_desc_proto.name
        if proto_name not in self._file_desc_protos_by_file:
            self._file_desc_protos_by_file[proto_name] = file_desc_proto
        elif self._file_desc_protos_by_file[proto_name] != file_desc_proto:
            raise DescriptorDatabaseConflictingDefinitionError('%s already added, but with different descriptor.' % proto_name)
        package = file_desc_proto.package
        for message in file_desc_proto.message_type:
            self._file_desc_protos_by_symbol.update(((name, file_desc_proto) for name in _ExtractSymbols(message, package)))
        for enum in file_desc_proto.enum_type:
            self._file_desc_protos_by_symbol['.'.join((package, enum.name))] = file_desc_proto
        for extension in file_desc_proto.extension:
            self._file_desc_protos_by_symbol['.'.join((package, extension.name))] = file_desc_proto

    def FindFileByName(self, name):
        if False:
            return 10
        'Finds the file descriptor proto by file name.\n\n    Typically the file name is a relative path ending to a .proto file. The\n    proto with the given name will have to have been added to this database\n    using the Add method or else an error will be raised.\n\n    Args:\n      name: The file name to find.\n\n    Returns:\n      The file descriptor proto matching the name.\n\n    Raises:\n      KeyError if no file by the given name was added.\n    '
        return self._file_desc_protos_by_file[name]

    def FindFileContainingSymbol(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        "Finds the file descriptor proto containing the specified symbol.\n\n    The symbol should be a fully qualified name including the file descriptor's\n    package and any containing messages. Some examples:\n\n    'some.package.name.Message'\n    'some.package.name.Message.NestedEnum'\n\n    The file descriptor proto containing the specified symbol must be added to\n    this database using the Add method or else an error will be raised.\n\n    Args:\n      symbol: The fully qualified symbol name.\n\n    Returns:\n      The file descriptor proto containing the symbol.\n\n    Raises:\n      KeyError if no file contains the specified symbol.\n    "
        return self._file_desc_protos_by_symbol[symbol]

def _ExtractSymbols(desc_proto, package):
    if False:
        i = 10
        return i + 15
    'Pulls out all the symbols from a descriptor proto.\n\n  Args:\n    desc_proto: The proto to extract symbols from.\n    package: The package containing the descriptor type.\n\n  Yields:\n    The fully qualified name found in the descriptor.\n  '
    message_name = '.'.join((package, desc_proto.name))
    yield message_name
    for nested_type in desc_proto.nested_type:
        for symbol in _ExtractSymbols(nested_type, message_name):
            yield symbol
    for enum_type in desc_proto.enum_type:
        yield '.'.join((message_name, enum_type.name))