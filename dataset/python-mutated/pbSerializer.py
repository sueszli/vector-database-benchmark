from __future__ import print_function
import sys
from .Switch import Switch

class PBSerializer(object):

    def __init__(self, file_path=None, encoding=None, file_type=None):
        if False:
            i = 10
            return i + 15
        self.string_encoding = encoding
        self.file_path = file_path
        self.file_type = file_type

    def write(self, obj=None):
        if False:
            return 10
        for case in Switch(self.file_type):
            if case('ascii'):
                try:
                    file_descriptor = open(self.file_path, 'w')
                    self.__writeObject(file_descriptor, obj)
                    file_descriptor.close()
                except IOError as exception:
                    print('I/O error({0}): {1}'.format(exception.errno, exception.strerror))
                except:
                    print('Unexpected error:' + str(sys.exc_info()[0]))
                    raise
                break
            if case('binary'):
                import biplist
                biplist.writePlist(obj, self.file_path)
                break
            if case('xml'):
                import plistlib
                plistlib.writePlist(obj, self.file_path)
                break
            if case():
                break

    def __writeObject(self, file_descriptor=None, obj=None):
        if False:
            return 10
        if file_descriptor is None:
            message = 'Fatal error, file descriptor is None'
            raise TypeError(message)
        if self.string_encoding is not None:
            file_descriptor.write('// !$*' + self.string_encoding + '*$!\n')
        if obj is not None:
            (write_string, indent_level) = obj.writeString()
            _ = indent_level
            file_descriptor.write(write_string)