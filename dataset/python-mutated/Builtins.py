""" Built-ins module. Information about built-ins of the running Python.

"""
import sys
from types import BuiltinFunctionType, FunctionType, GeneratorType, ModuleType
from nuitka.__past__ import GenericAlias, builtins
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.PythonVersions import python_version

def _getBuiltinExceptionNames():
    if False:
        print('Hello World!')

    def isExceptionName(builtin_name):
        if False:
            while True:
                i = 10
        if builtin_name.endswith('Error') or builtin_name.endswith('Exception'):
            return True
        elif builtin_name in ('StopIteration', 'GeneratorExit', 'SystemExit', 'NotImplemented', 'KeyboardInterrupt', 'StopAsyncIteration'):
            return True
        else:
            return False
    exceptions = OrderedDict()
    if python_version < 768:
        import exceptions as builtin_exceptions
        for key in sorted(dir(builtin_exceptions)):
            name = str(key)
            if isExceptionName(name):
                exceptions[name] = getattr(builtin_exceptions, key)
        for key in sorted(dir(builtins)):
            name = str(key)
            if isExceptionName(name):
                exceptions[name] = getattr(builtins, key)
    else:
        for key in sorted(dir(builtins)):
            if isExceptionName(key):
                exceptions[key] = getattr(builtins, key)
    return (list(exceptions.keys()), exceptions)
(builtin_exception_names, builtin_exception_values) = _getBuiltinExceptionNames()
builtin_exception_values_list = tuple(builtin_exception_values.values())
assert 'TypeError' in builtin_exception_names
assert 'ValueError' in builtin_exception_names
assert 'StopIteration' in builtin_exception_names
assert 'GeneratorExit' in builtin_exception_names
assert 'AssertionError' in builtin_exception_names
assert 'BaseException' in builtin_exception_names
assert 'Exception' in builtin_exception_names
assert 'NotImplemented' in builtin_exception_names
assert 'StopAsyncIteration' in builtin_exception_names or python_version < 848

def _getBuiltinNames():
    if False:
        return 10
    names = [str(x) for x in dir(builtins)]
    names.sort()
    for builtin_exception_name in builtin_exception_names:
        if builtin_exception_name in names:
            names.remove(builtin_exception_name)
    names.remove('__doc__')
    names.remove('__name__')
    names.remove('__package__')
    if '__loader__' in names:
        names.remove('__loader__')
    if '__spec__' in names:
        names.remove('__spec__')
    warnings = []
    for builtin_name in names:
        if builtin_name.endswith('Warning'):
            warnings.append(builtin_name)
    for builtin_name in warnings:
        names.remove(builtin_name)
    return (names, warnings)
(builtin_names, builtin_warnings) = _getBuiltinNames()
builtin_named_values = dict(((getattr(builtins, x), x) for x in builtin_names))
builtin_named_values_list = tuple(builtin_named_values)
assert type in builtin_named_values
assert '__import__' in builtin_names
assert 'int' in builtin_names
assert 'type' in builtin_names
assert '__doc__' not in builtin_names
assert 'sys' not in builtin_names
builtin_all_names = builtin_names + builtin_exception_names + builtin_warnings

def getBuiltinTypeNames():
    if False:
        return 10
    result = []
    for builtin_name in builtin_names:
        if isinstance(getattr(builtins, builtin_name), type):
            result.append(builtin_name)
    return tuple(sorted(result))
builtin_type_names = getBuiltinTypeNames()

def _getAnonBuiltins():
    if False:
        return 10
    anon_names = OrderedDict()
    anon_codes = OrderedDict()
    anon_names['NoneType'] = type(None)
    anon_codes['NoneType'] = 'Py_TYPE(Py_None)'
    anon_names['ellipsis'] = type(Ellipsis)
    anon_codes['ellipsis'] = '&PyEllipsis_Type'
    anon_names['NotImplementedType'] = type(NotImplemented)
    anon_codes['NotImplementedType'] = 'Py_TYPE(Py_NotImplemented)'
    anon_names['function'] = FunctionType
    anon_codes['function'] = '&PyFunction_Type'
    anon_names['generator'] = GeneratorType
    anon_codes['generator'] = '&PyGenerator_Type'
    anon_names['builtin_function_or_method'] = BuiltinFunctionType
    anon_codes['builtin_function_or_method'] = '&PyCFunction_Type'
    anon_names['code'] = type(_getAnonBuiltins.__code__)
    anon_codes['code'] = '&PyCode_Type'
    anon_names['module'] = ModuleType
    anon_codes['module'] = '&PyModule_Type'
    if python_version < 768:
        from types import ClassType, InstanceType, MethodType
        with open(sys.executable) as any_file:
            anon_names['file'] = type(any_file)
        anon_codes['file'] = '&PyFile_Type'
        anon_names['classobj'] = ClassType
        anon_codes['classobj'] = '&PyClass_Type'
        anon_names['instance'] = InstanceType
        anon_codes['instance'] = '&PyInstance_Type'
        anon_names['instancemethod'] = MethodType
        anon_codes['instancemethod'] = '&PyMethod_Type'
    if python_version >= 624:
        anon_names['version_info'] = type(sys.version_info)
        anon_codes['version_info'] = 'Py_TYPE(Nuitka_SysGetObject("version_info"))'
    if python_version >= 912:
        assert GenericAlias is not None
        anon_names['GenericAlias'] = GenericAlias
        anon_codes['GenericAlias'] = '&Py_GenericAliasType'
    if python_version >= 928:
        anon_names['UnionType'] = type(int | str)
        anon_codes['UnionType'] = 'Nuitka_PyUnion_Type'
    return (anon_names, anon_codes)
(builtin_anon_names, builtin_anon_codes) = _getAnonBuiltins()
builtin_anon_values = OrderedDict(((b, a) for (a, b) in builtin_anon_names.items()))
builtin_anon_value_list = tuple(builtin_anon_values)