"""A visitor class that generates protobufs for each python object."""
import enum
import re
import sys
from google.protobuf import message
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.api.lib import api_objects_pb2
_CORNER_CASES = {'': {'tools': {}}, 'test.TestCase': {}, 'test.TestCase.failureException': {}, 'train.NanLossDuringTrainingError': {'message': {}}, 'estimator.NanLossDuringTrainingError': {'message': {}}, 'train.LooperThread': {'isAlive': {}, 'join': {}, 'native_id': {}}}
if sys.version_info.major == 3:
    _NORMALIZE_TYPE = {}
    for t in ('property', 'object', 'getset_descriptor', 'int', 'str', 'type', 'tuple', 'module', 'collections.defaultdict', 'set', 'dict', 'NoneType', 'frozenset', 'member_descriptor'):
        _NORMALIZE_TYPE["<class '%s'>" % t] = "<type '%s'>" % t
    for e in ('Exception', 'RuntimeError'):
        _NORMALIZE_TYPE["<class '%s'>" % e] = "<type 'exceptions.%s'>" % e
    _NORMALIZE_TYPE["<class 'abc.ABCMeta'>"] = "<type 'type'>"
    _NORMALIZE_ISINSTANCE = {"<class 'tensorflow.lite.python.op_hint.OpHint.OpHintArgumentTracker'>": "<class 'tensorflow.lite.python.op_hint.OpHintArgumentTracker'>", "<class 'tensorflow.python.training.monitored_session._MonitoredSession.StepContext'>": "<class 'tensorflow.python.training.monitored_session.StepContext'>", "<class 'tensorflow.python.ops.variables.Variable.SaveSliceInfo'>": "<class 'tensorflow.python.ops.variables.SaveSliceInfo'>"}

    def _SkipMember(cls, member):
        if False:
            i = 10
            return i + 15
        return member == 'with_traceback' or (member in ('name', 'value') and isinstance(cls, type) and issubclass(cls, enum.Enum))
else:
    _NORMALIZE_TYPE = {"<class 'abc.ABCMeta'>": "<type 'type'>", "<class 'pybind11_type'>": "<class 'pybind11_builtins.pybind11_type'>"}
    _NORMALIZE_ISINSTANCE = {"<class 'pybind11_object'>": "<class 'pybind11_builtins.pybind11_object'>"}

    def _SkipMember(cls, member):
        if False:
            for i in range(10):
                print('nop')
        return False
_NORMALIZE_TYPE['tensorflow.python.framework.tensor.Tensor'] = "<class 'tensorflow.python.framework.tensor.Tensor'>"
_NORMALIZE_TYPE['typing.Generic'] = "<class 'typing.Generic'>"
_NORMALIZE_TYPE["<class 'typing._GenericAlias'>"] = 'typing.Union'
_NORMALIZE_TYPE["<class 'typing._UnionGenericAlias'>"] = 'typing.Union'
_NORMALIZE_TYPE["<class 'typing_extensions._ProtocolMeta'>"] = "<class 'typing._ProtocolMeta'>"
_NORMALIZE_TYPE["<class 'typing_extensions.Protocol'>"] = "<class 'typing.Protocol'>"
if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    _NORMALIZE_TYPE["<class '_collections._tuplegetter'>"] = "<type 'property'>"

def _NormalizeType(ty):
    if False:
        for i in range(10):
            print('nop')
    return _NORMALIZE_TYPE.get(ty, ty)

def _NormalizeIsInstance(ty):
    if False:
        return 10
    return _NORMALIZE_ISINSTANCE.get(ty, ty)

def _SanitizedArgSpec(obj):
    if False:
        return 10
    'Get an ArgSpec string that is free of addresses.\n\n  We have callables as function arg defaults. This results in addresses in\n  getargspec output. This function returns a sanitized string list of base\n  classes.\n\n  Args:\n    obj: A python routine for us the create the sanitized arspec of.\n\n  Returns:\n    string, a string representation of the argspec.\n  '
    output_string = ''
    unsanitized_arg_spec = tf_inspect.getargspec(obj)
    for clean_attr in ('args', 'varargs', 'keywords'):
        output_string += '%s=%s, ' % (clean_attr, getattr(unsanitized_arg_spec, clean_attr))
    if unsanitized_arg_spec.defaults:
        sanitized_defaults = []
        for val in unsanitized_arg_spec.defaults:
            str_val = str(val)
            if ' at 0x' in str_val:
                sanitized_defaults.append('%s instance>' % str_val.split(' at ')[0])
            else:
                sanitized_defaults.append(str_val)
        output_string += 'defaults=%s, ' % sanitized_defaults
    else:
        output_string += 'defaults=None'
    return output_string

def _GenerateArgsSpec(doc):
    if False:
        for i in range(10):
            print('nop')
    'Generate args spec from a method docstring.'
    args_spec = []
    doc = re.search('\\(.*\\)', doc)
    if not doc:
        return None
    doc = doc.group().strip('(').strip(')')
    doc_split = doc.split(',')
    for s in doc_split:
        arg = re.search('\\w+', s)
        if not arg:
            return None
        args_spec.append(f"'{arg.group()}'")
    return ', '.join(args_spec)

def _ParseDocstringArgSpec(doc):
    if False:
        i = 10
        return i + 15
    'Get an ArgSpec string from a method docstring.\n\n  This method is used to generate argspec for C extension functions that follow\n  pybind11 DocString format function signature. For example:\n  `foo_function(a: int, b: string) -> None...`\n\n  Args:\n    doc: A python string which starts with function signature.\n\n  Returns:\n    string: a argspec string representation if successful. If not, return None.\n\n  Raises:\n    ValueError: Raised when failed to parse the input docstring.\n  '
    match = re.search('^\\w+\\(.*\\)', doc)
    args_spec = _GenerateArgsSpec(doc)
    if not match or args_spec is None:
        raise ValueError(f'Failed to parse argspec from docstring: {doc}')
    output_string = f'args=[{args_spec}], varargs=None, keywords=None, defaults=None'
    return output_string

def _SanitizedMRO(obj):
    if False:
        print('Hello World!')
    'Get a list of superclasses with minimal amount of non-TF classes.\n\n  Based on many parameters like python version, OS, protobuf implementation\n  or changes in google core libraries the list of superclasses of a class\n  can change. We only return the first non-TF class to be robust to non API\n  affecting changes. The Method Resolution Order returned by `tf_inspect.getmro`\n  is still maintained in the return value.\n\n  Args:\n    obj: A python routine for us the create the sanitized arspec of.\n\n  Returns:\n    list of strings, string representation of the class names.\n  '
    return_list = []
    for cls in tf_inspect.getmro(obj):
        if cls.__name__ == '_NewClass':
            continue
        str_repr = _NormalizeType(str(cls))
        return_list.append(str_repr)
        if 'tensorflow' not in str_repr and 'keras' not in str_repr:
            break
        if 'StubOutForTesting' in str_repr:
            break
    return return_list

def _IsProtoClass(obj):
    if False:
        while True:
            i = 10
    'Returns whether the passed obj is a Protocol Buffer class.'
    return isinstance(obj, type) and issubclass(obj, message.Message)

class PythonObjectToProtoVisitor:
    """A visitor that summarizes given python objects as protobufs."""

    def __init__(self, default_path='tensorflow'):
        if False:
            print('Hello World!')
        self._protos = {}
        self._default_path = default_path

    def GetProtos(self):
        if False:
            while True:
                i = 10
        'Return the list of protos stored.'
        return self._protos

    def __call__(self, path, parent, children):
        if False:
            i = 10
            return i + 15
        lib_path = self._default_path + '.' + path if path else self._default_path
        (_, parent) = tf_decorator.unwrap(parent)

        def _AddMember(member_name, member_obj, proto):
            if False:
                return 10
            'Add the child object to the object being constructed.'
            (_, member_obj) = tf_decorator.unwrap(member_obj)
            if _SkipMember(parent, member_name) or isinstance(member_obj, deprecation.HiddenTfApiAttribute):
                return
            if member_name == '__init__' or not member_name.startswith('_'):
                if tf_inspect.isroutine(member_obj):
                    new_method = proto.member_method.add()
                    new_method.name = member_name
                    if hasattr(member_obj, '__code__'):
                        new_method.argspec = _SanitizedArgSpec(member_obj)
                    elif member_name != '__init__' and hasattr(member_obj, '__doc__'):
                        doc = member_obj.__doc__
                        try:
                            spec_str = _ParseDocstringArgSpec(doc)
                        except ValueError:
                            pass
                        else:
                            new_method.argspec = spec_str
                else:
                    new_member = proto.member.add()
                    new_member.name = member_name
                    if tf_inspect.ismodule(member_obj):
                        new_member.mtype = "<type 'module'>"
                    else:
                        new_member.mtype = _NormalizeType(str(type(member_obj)))
        parent_corner_cases = _CORNER_CASES.get(path, {})
        if path not in _CORNER_CASES or parent_corner_cases:
            if tf_inspect.ismodule(parent):
                module_obj = api_objects_pb2.TFAPIModule()
                for (name, child) in children:
                    if name in parent_corner_cases:
                        if parent_corner_cases[name]:
                            module_obj.member.add(**parent_corner_cases[name])
                    else:
                        _AddMember(name, child, module_obj)
                self._protos[lib_path] = api_objects_pb2.TFAPIObject(path=lib_path, tf_module=module_obj)
            elif _IsProtoClass(parent):
                proto_obj = api_objects_pb2.TFAPIProto()
                parent.DESCRIPTOR.CopyToProto(proto_obj.descriptor)
                self._protos[lib_path] = api_objects_pb2.TFAPIObject(path=lib_path, tf_proto=proto_obj)
            elif tf_inspect.isclass(parent):
                class_obj = api_objects_pb2.TFAPIClass()
                class_obj.is_instance.extend((_NormalizeIsInstance(i) for i in _SanitizedMRO(parent)))
                for (name, child) in children:
                    if name in parent_corner_cases:
                        if parent_corner_cases[name]:
                            class_obj.member.add(**parent_corner_cases[name])
                    else:
                        _AddMember(name, child, class_obj)
                self._protos[lib_path] = api_objects_pb2.TFAPIObject(path=lib_path, tf_class=class_obj)
            else:
                logging.error('Illegal call to ApiProtoDump::_py_obj_to_proto.Object is neither a module nor a class: %s', path)