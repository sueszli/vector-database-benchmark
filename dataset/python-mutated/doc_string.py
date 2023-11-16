"""Allows us to create and absorb changes (aka Deltas) to elements."""
import ast
import contextlib
import inspect
import re
import types
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import Final
import streamlit
from streamlit.logger import get_logger
from streamlit.proto.DocString_pb2 import DocString as DocStringProto
from streamlit.proto.DocString_pb2 import Member as MemberProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_runner import __file__ as SCRIPTRUNNER_FILENAME
from streamlit.runtime.secrets import Secrets
from streamlit.string_util import is_mem_address_str
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
LOGGER: Final = get_logger(__name__)
CONFUSING_STREAMLIT_SIG_PREFIXES: Final = ('(element, ',)

class HelpMixin:

    @gather_metrics('help')
    def help(self, obj: Any=streamlit) -> 'DeltaGenerator':
        if False:
            print('Hello World!')
        "Display help and other information for a given object.\n\n        Depending on the type of object that is passed in, this displays the\n        object's name, type, value, signature, docstring, and member variables,\n        methods — as well as the values/docstring of members and methods.\n\n        Parameters\n        ----------\n        obj : any\n            The object whose information should be displayed. If left\n            unspecified, this call will display help for Streamlit itself.\n\n        Example\n        -------\n\n        Don't remember how to initialize a dataframe? Try this:\n\n        >>> import streamlit as st\n        >>> import pandas\n        >>>\n        >>> st.help(pandas.DataFrame)\n\n        .. output::\n            https://doc-string.streamlit.app/\n            height: 700px\n\n        Want to quickly check what data type is output by a certain function?\n        Try:\n\n        >>> import streamlit as st\n        >>>\n        >>> x = my_poorly_documented_function()\n        >>> st.help(x)\n\n        Want to quickly inspect an object? No sweat:\n\n        >>> class Dog:\n        >>>   '''A typical dog.'''\n        >>>\n        >>>   def __init__(self, breed, color):\n        >>>     self.breed = breed\n        >>>     self.color = color\n        >>>\n        >>>   def bark(self):\n        >>>     return 'Woof!'\n        >>>\n        >>>\n        >>> fido = Dog('poodle', 'white')\n        >>>\n        >>> st.help(fido)\n\n        .. output::\n            https://doc-string1.streamlit.app/\n            height: 300px\n\n        And if you're using Magic, you can get help for functions, classes,\n        and modules without even typing ``st.help``:\n\n        >>> import streamlit as st\n        >>> import pandas\n        >>>\n        >>> # Get help for Pandas read_csv:\n        >>> pandas.read_csv\n        >>>\n        >>> # Get help for Streamlit itself:\n        >>> st\n\n        .. output::\n            https://doc-string2.streamlit.app/\n            height: 700px\n        "
        doc_string_proto = DocStringProto()
        _marshall(doc_string_proto, obj)
        return self.dg._enqueue('doc_string', doc_string_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            return 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def _marshall(doc_string_proto: DocStringProto, obj: Any) -> None:
    if False:
        i = 10
        return i + 15
    'Construct a DocString object.\n\n    See DeltaGenerator.help for docs.\n    '
    var_name = _get_variable_name()
    if var_name is not None:
        doc_string_proto.name = var_name
    obj_type = _get_type_as_str(obj)
    doc_string_proto.type = obj_type
    obj_docs = _get_docstring(obj)
    if obj_docs is not None:
        doc_string_proto.doc_string = obj_docs
    obj_value = _get_value(obj, var_name)
    if obj_value is not None:
        doc_string_proto.value = obj_value
    doc_string_proto.members.extend(_get_members(obj))

def _get_name(obj):
    if False:
        return 10
    name = getattr(obj, '__qualname__', None)
    if name:
        return name
    return getattr(obj, '__name__', None)

def _get_module(obj):
    if False:
        while True:
            i = 10
    return getattr(obj, '__module__', None)

def _get_signature(obj):
    if False:
        for i in range(10):
            print('nop')
    if not inspect.isclass(obj) and (not callable(obj)):
        return None
    sig = ''
    try:
        sig = str(inspect.signature(obj))
    except ValueError:
        sig = '(...)'
    except TypeError:
        return None
    is_delta_gen = False
    with contextlib.suppress(AttributeError):
        is_delta_gen = obj.__module__ == 'streamlit.delta_generator'
    if is_delta_gen:
        for prefix in CONFUSING_STREAMLIT_SIG_PREFIXES:
            if sig.startswith(prefix):
                sig = sig.replace(prefix, '(')
                break
    return sig

def _get_docstring(obj):
    if False:
        print('Hello World!')
    doc_string = inspect.getdoc(obj)
    if doc_string is None:
        obj_type = type(obj)
        if obj_type is not type and obj_type is not types.ModuleType and (not inspect.isfunction(obj)) and (not inspect.ismethod(obj)):
            doc_string = inspect.getdoc(obj_type)
    if doc_string:
        return doc_string.strip()
    return None

def _get_variable_name():
    if False:
        i = 10
        return i + 15
    'Try to get the name of the variable in the current line, as set by the user.\n\n    For example:\n    foo = bar.Baz(123)\n    st.help(foo)\n\n    The name is "foo"\n    '
    code = _get_current_line_of_code_as_str()
    if code is None:
        return None
    return _get_variable_name_from_code_str(code)

def _get_variable_name_from_code_str(code):
    if False:
        while True:
            i = 10
    tree = ast.parse(code)
    if not _is_stcommand(tree, command_name='help') and (not _is_stcommand(tree, command_name='write')):
        if code.endswith(','):
            code = code[:-1]
        return code
    arg_node = _get_stcommand_arg(tree)
    if not arg_node:
        return None
    elif type(arg_node) is ast.NamedExpr:
        if type(arg_node.target) is ast.Name:
            return arg_node.target.id
    elif type(arg_node) in (ast.Constant, ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis):
        return None
    code_lines = code.split('\n')
    is_multiline = len(code_lines) > 1
    start_offset = arg_node.col_offset
    if is_multiline:
        first_lineno = arg_node.lineno - 1
        first_line = code_lines[first_lineno]
        end_offset = None
    else:
        first_line = code_lines[0]
        end_offset = getattr(arg_node, 'end_col_offset', -1)
    return first_line[start_offset:end_offset]
_NEWLINES = re.compile('[\\n\\r]+')

def _get_current_line_of_code_as_str():
    if False:
        print('Hello World!')
    scriptrunner_frame = _get_scriptrunner_frame()
    if scriptrunner_frame is None:
        return None
    code_context = scriptrunner_frame.code_context
    if not code_context:
        return None
    code_as_string = ''.join(code_context)
    return re.sub(_NEWLINES, '', code_as_string.strip())

def _get_scriptrunner_frame():
    if False:
        return 10
    prev_frame = None
    scriptrunner_frame = None
    for frame in inspect.stack():
        if frame.code_context is None:
            return None
        if frame.filename == SCRIPTRUNNER_FILENAME:
            scriptrunner_frame = prev_frame
            break
        prev_frame = frame
    return scriptrunner_frame

def _is_stcommand(tree, command_name):
    if False:
        return 10
    'Checks whether the AST in tree is a call for command_name.'
    root_node = tree.body[0].value
    if not type(root_node) is ast.Call:
        return False
    return getattr(root_node.func, 'id', None) == command_name or getattr(root_node.func, 'attr', None) == command_name

def _get_stcommand_arg(tree):
    if False:
        print('Hello World!')
    'Gets the argument node for the st command in tree (AST).'
    root_node = tree.body[0].value
    if root_node.args:
        return root_node.args[0]
    return None

def _get_type_as_str(obj):
    if False:
        return 10
    if inspect.isclass(obj):
        return 'class'
    return str(type(obj).__name__)

def _get_first_line(text):
    if False:
        print('Hello World!')
    if not text:
        return ''
    (left, _, _) = text.partition('\n')
    return left

def _get_weight(value):
    if False:
        i = 10
        return i + 15
    if inspect.ismodule(value):
        return 3
    if inspect.isclass(value):
        return 2
    if callable(value):
        return 1
    return 0

def _get_value(obj, var_name):
    if False:
        while True:
            i = 10
    obj_value = _get_human_readable_value(obj)
    if obj_value is not None:
        return obj_value
    name = _get_name(obj)
    if name:
        name_obj = obj
    else:
        name_obj = type(obj)
        name = _get_name(name_obj)
    module = _get_module(name_obj)
    sig = _get_signature(name_obj) or ''
    if name:
        if module:
            obj_value = f'{module}.{name}{sig}'
        else:
            obj_value = f'{name}{sig}'
    if obj_value == var_name:
        obj_value = None
    return obj_value

def _get_human_readable_value(value):
    if False:
        print('Hello World!')
    if isinstance(value, Secrets):
        return None
    if inspect.isclass(value) or inspect.ismodule(value) or callable(value):
        return None
    value_str = repr(value)
    if isinstance(value, str):
        return _shorten(value_str)
    if is_mem_address_str(value_str):
        return None
    return _shorten(value_str)

def _shorten(s, length=300):
    if False:
        print('Hello World!')
    s = s.strip()
    return s[:length] + '...' if len(s) > length else s

def _is_computed_property(obj, attr_name):
    if False:
        while True:
            i = 10
    obj_class = getattr(obj, '__class__', None)
    if not obj_class:
        return False
    for parent_class in inspect.getmro(obj_class):
        class_attr = getattr(parent_class, attr_name, None)
        if class_attr is None:
            continue
        if isinstance(class_attr, property) or inspect.isgetsetdescriptor(class_attr):
            return True
    return False

def _get_members(obj):
    if False:
        print('Hello World!')
    members_for_sorting = []
    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue
        is_computed_value = _is_computed_property(obj, attr_name)
        if is_computed_value:
            parent_attr = getattr(obj.__class__, attr_name)
            member_type = 'property'
            weight = 0
            member_docs = _get_docstring(parent_attr)
            member_value = None
        else:
            attr_value = getattr(obj, attr_name)
            weight = _get_weight(attr_value)
            human_readable_value = _get_human_readable_value(attr_value)
            member_type = _get_type_as_str(attr_value)
            if human_readable_value is None:
                member_docs = _get_docstring(attr_value)
                member_value = None
            else:
                member_docs = None
                member_value = human_readable_value
        if member_type == 'module':
            continue
        member = MemberProto()
        member.name = attr_name
        member.type = member_type
        if member_docs is not None:
            member.doc_string = _get_first_line(member_docs)
        if member_value is not None:
            member.value = member_value
        members_for_sorting.append((weight, member))
    if members_for_sorting:
        sorted_members = sorted(members_for_sorting, key=lambda x: (x[0], x[1].name))
        return [m for (_, m) in sorted_members]
    return []