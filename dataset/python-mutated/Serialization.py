""" Write and read constants data files and provide identifiers.

"""
import os
import pickle
import sys
from nuitka import OutputDirectories
from nuitka.__past__ import BaseExceptionGroup, ExceptionGroup, GenericAlias, UnionType, basestring, to_byte, xrange
from nuitka.Builtins import builtin_anon_codes, builtin_anon_values, builtin_exception_values_list
from nuitka.code_generation.Namify import namifyConstant
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.PythonVersions import python_version
from nuitka.utils.FileOperations import openTextFile

class BuiltinAnonValue(object):
    """Used to pickle anonymous values."""
    anon_values = tuple(builtin_anon_values.values())

    def __init__(self, anon_name):
        if False:
            for i in range(10):
                print('nop')
        self.anon_name = anon_name

    def getStreamValueByte(self):
        if False:
            for i in range(10):
                print('nop')
        'Return byte value, encoding the anon built-in value.'
        return to_byte(self.anon_values.index(self.anon_name))

class BuiltinGenericAliasValue(object):
    """For transporting GenericAlias values through pickler."""

    def __init__(self, origin, args):
        if False:
            while True:
                i = 10
        self.origin = origin
        self.args = args

class BuiltinUnionTypeValue(object):
    """For transporting UnionType values through pickler."""

    def __init__(self, args):
        if False:
            i = 10
            return i + 15
        self.args = args

class BuiltinSpecialValue(object):
    """Used to pickle special values."""

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def getStreamValueByte(self):
        if False:
            print('Hello World!')
        'Return byte value, encoding the special built-in value.'
        if self.value == 'Ellipsis':
            return to_byte(0)
        elif self.value == 'NotImplemented':
            return to_byte(1)
        elif self.value == 'Py_SysVersionInfo':
            return to_byte(2)
        else:
            assert False, self.value

class BlobData(object):
    """Used to pickle bytes to become raw pointers."""
    __slots__ = ('data', 'name')

    def __init__(self, data, name):
        if False:
            for i in range(10):
                print('nop')
        self.data = data
        self.name = name

    def getData(self):
        if False:
            print('Hello World!')
        return self.data

    def __repr__(self):
        if False:
            return 10
        return '<nuitka.Serialization.BlobData %s>' % self.name

def _pickleAnonValues(pickler, value):
    if False:
        print('Hello World!')
    if value in builtin_anon_values:
        pickler.save(BuiltinAnonValue(builtin_anon_values[value]))
    elif value is Ellipsis:
        pickler.save(BuiltinSpecialValue('Ellipsis'))
    elif value is NotImplemented:
        pickler.save(BuiltinSpecialValue('NotImplemented'))
    elif value is sys.version_info:
        pickler.save(BuiltinSpecialValue('Py_SysVersionInfo'))
    else:
        pickler.save_global(value)

def _pickleGenericAlias(pickler, value):
    if False:
        while True:
            i = 10
    pickler.save(BuiltinGenericAliasValue(origin=value.__origin__, args=value.__args__))

def _pickleUnionType(pickler, value):
    if False:
        while True:
            i = 10
    pickler.save(BuiltinUnionTypeValue(args=value.__args__))

class ConstantStreamWriter(object):
    """Write constants to a stream and return numbers for them."""

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        self.count = 0
        filename = os.path.join(OutputDirectories.getSourceDirectoryPath(), filename)
        self.file = openTextFile(filename, 'wb')
        if python_version < 768:
            self.pickle = pickle.Pickler(self.file, -1)
        else:
            self.pickle = pickle._Pickler(self.file, -1)
        self.pickle.dispatch[type] = _pickleAnonValues
        self.pickle.dispatch[type(Ellipsis)] = _pickleAnonValues
        self.pickle.dispatch[type(NotImplemented)] = _pickleAnonValues
        if type(sys.version_info) is not tuple:
            self.pickle.dispatch[type(sys.version_info)] = _pickleAnonValues
        if python_version >= 912:
            self.pickle.dispatch[GenericAlias] = _pickleGenericAlias
        if python_version >= 928:
            self.pickle.dispatch[UnionType] = _pickleUnionType

    def addConstantValue(self, constant_value):
        if False:
            for i in range(10):
                print('nop')
        self.pickle.dump(constant_value)
        self.count += 1

    def addBlobData(self, data, name):
        if False:
            i = 10
            return i + 15
        self.pickle.dump(BlobData(data, name))
        self.count += 1

    def close(self):
        if False:
            print('Hello World!')
        self.file.close()

class ConstantStreamReader(object):

    def __init__(self, const_file):
        if False:
            print('Hello World!')
        self.count = 0
        self.pickle = pickle.Unpickler(const_file)

    def readConstantValue(self):
        if False:
            print('Hello World!')
        return self.pickle.load()

class ConstantAccessor(object):

    def __init__(self, data_filename, top_level_name):
        if False:
            i = 10
            return i + 15
        self.constants = OrderedSet()
        self.constants_writer = ConstantStreamWriter(data_filename)
        self.top_level_name = top_level_name

    def getConstantCode(self, constant):
        if False:
            while True:
                i = 10
        if constant is None:
            key = 'Py_None'
        elif constant is True:
            key = 'Py_True'
        elif constant is False:
            key = 'Py_False'
        elif constant is Ellipsis:
            key = 'Py_Ellipsis'
        elif constant is NotImplemented:
            key = 'Py_NotImplemented'
        elif constant is sys.version_info:
            key = 'Py_SysVersionInfo'
        elif type(constant) is type:
            if constant is None:
                key = '(PyObject *)Py_TYPE(Py_None)'
            elif constant is object:
                key = '(PyObject *)&PyBaseObject_Type'
            elif constant is staticmethod:
                key = '(PyObject *)&PyStaticMethod_Type'
            elif constant is classmethod:
                key = '(PyObject *)&PyClassMethod_Type'
            elif constant is bytearray:
                key = '(PyObject *)&PyByteArray_Type'
            elif constant is enumerate:
                key = '(PyObject *)&PyEnum_Type'
            elif constant is frozenset:
                key = '(PyObject *)&PyFrozenSet_Type'
            elif python_version >= 624 and constant is memoryview:
                key = '(PyObject *)&PyMemoryView_Type'
            elif python_version < 768 and constant is basestring:
                key = '(PyObject *)&PyBaseString_Type'
            elif python_version < 768 and constant is xrange:
                key = '(PyObject *)&PyRange_Type'
            elif constant in builtin_anon_values:
                key = '(PyObject *)' + builtin_anon_codes[builtin_anon_values[constant]]
            elif constant in builtin_exception_values_list:
                key = '(PyObject *)PyExc_%s' % constant.__name__
            elif constant is ExceptionGroup:
                key = '(PyObject *)_PyInterpreterState_GET()->exc_state.PyExc_ExceptionGroup'
            elif constant is BaseExceptionGroup:
                key = '(PyObject *)PyExc_BaseExceptionGroup'
            else:
                type_name = constant.__name__
                if constant is int and python_version >= 768:
                    type_name = 'long'
                elif constant is str:
                    type_name = 'string' if python_version < 768 else 'unicode'
                key = '(PyObject *)&Py%s_Type' % type_name.capitalize()
        else:
            key = 'const_' + namifyConstant(constant)
            if key not in self.constants:
                self.constants.add(key)
                self.constants_writer.addConstantValue(constant)
            key = '%s[%d]' % (self.top_level_name, self.constants.index(key))
        return key

    def getBlobDataCode(self, data, name):
        if False:
            for i in range(10):
                print('nop')
        key = 'blob_' + namifyConstant(data)
        if key not in self.constants:
            self.constants.add(key)
            self.constants_writer.addBlobData(data=data, name=name)
        key = '%s[%d]' % (self.top_level_name, self.constants.index(key))
        return key

    def getConstantsCount(self):
        if False:
            print('Hello World!')
        self.constants_writer.close()
        return len(self.constants)