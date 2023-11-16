"""A simple wrapper around enum types to expose utility functions.

Instances are created as properties with the same name as the enum they wrap
on proto classes.  For usage, see:
  reflection_test.py
"""
__author__ = 'rabsatt@google.com (Kevin Rabsatt)'

class EnumTypeWrapper(object):
    """A utility for finding the names of enum values."""
    DESCRIPTOR = None

    def __init__(self, enum_type):
        if False:
            while True:
                i = 10
        'Inits EnumTypeWrapper with an EnumDescriptor.'
        self._enum_type = enum_type
        self.DESCRIPTOR = enum_type

    def Name(self, number):
        if False:
            while True:
                i = 10
        'Returns a string containing the name of an enum value.'
        if number in self._enum_type.values_by_number:
            return self._enum_type.values_by_number[number].name
        raise ValueError('Enum %s has no name defined for value %d' % (self._enum_type.name, number))

    def Value(self, name):
        if False:
            print('Hello World!')
        'Returns the value coresponding to the given enum name.'
        if name in self._enum_type.values_by_name:
            return self._enum_type.values_by_name[name].number
        raise ValueError('Enum %s has no value defined for name %s' % (self._enum_type.name, name))

    def keys(self):
        if False:
            i = 10
            return i + 15
        'Return a list of the string names in the enum.\n\n    These are returned in the order they were defined in the .proto file.\n    '
        return [value_descriptor.name for value_descriptor in self._enum_type.values]

    def values(self):
        if False:
            print('Hello World!')
        'Return a list of the integer values in the enum.\n\n    These are returned in the order they were defined in the .proto file.\n    '
        return [value_descriptor.number for value_descriptor in self._enum_type.values]

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of the (name, value) pairs of the enum.\n\n    These are returned in the order they were defined in the .proto file.\n    '
        return [(value_descriptor.name, value_descriptor.number) for value_descriptor in self._enum_type.values]