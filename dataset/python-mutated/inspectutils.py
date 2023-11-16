"""Inspection utility functions for Python Fire."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
if six.PY34:
    import asyncio

class FullArgSpec(object):
    """The arguments of a function, as in Python 3's inspect.FullArgSpec."""

    def __init__(self, args=None, varargs=None, varkw=None, defaults=None, kwonlyargs=None, kwonlydefaults=None, annotations=None):
        if False:
            while True:
                i = 10
        "Constructs a FullArgSpec with each provided attribute, or the default.\n\n    Args:\n      args: A list of the argument names accepted by the function.\n      varargs: The name of the *varargs argument or None if there isn't one.\n      varkw: The name of the **kwargs argument or None if there isn't one.\n      defaults: A tuple of the defaults for the arguments that accept defaults.\n      kwonlyargs: A list of argument names that must be passed with a keyword.\n      kwonlydefaults: A dictionary of keyword only arguments and their defaults.\n      annotations: A dictionary of arguments and their annotated types.\n    "
        self.args = args or []
        self.varargs = varargs
        self.varkw = varkw
        self.defaults = defaults or ()
        self.kwonlyargs = kwonlyargs or []
        self.kwonlydefaults = kwonlydefaults or {}
        self.annotations = annotations or {}

def _GetArgSpecInfo(fn):
    if False:
        while True:
            i = 10
    "Gives information pertaining to computing the ArgSpec of fn.\n\n  Determines if the first arg is supplied automatically when fn is called.\n  This arg will be supplied automatically if fn is a bound method or a class\n  with an __init__ method.\n\n  Also returns the function who's ArgSpec should be used for determining the\n  calling parameters for fn. This may be different from fn itself if fn is a\n  class with an __init__ method.\n\n  Args:\n    fn: The function or class of interest.\n  Returns:\n    A tuple with the following two items:\n      fn: The function to use for determining the arg spec of this function.\n      skip_arg: Whether the first argument will be supplied automatically, and\n        hence should be skipped when supplying args from a Fire command.\n  "
    skip_arg = False
    if inspect.isclass(fn):
        skip_arg = True
        if six.PY2 and hasattr(fn, '__init__'):
            fn = fn.__init__
    elif inspect.ismethod(fn):
        skip_arg = fn.__self__ is not None
    elif inspect.isbuiltin(fn):
        if not isinstance(fn.__self__, types.ModuleType):
            skip_arg = True
    elif not inspect.isfunction(fn):
        skip_arg = True
    return (fn, skip_arg)

def Py2GetArgSpec(fn):
    if False:
        for i in range(10):
            print('nop')
    'A wrapper around getargspec that tries both fn and fn.__call__.'
    try:
        return inspect.getargspec(fn)
    except TypeError:
        if hasattr(fn, '__call__'):
            return inspect.getargspec(fn.__call__)
        raise

def Py3GetFullArgSpec(fn):
    if False:
        print('Hello World!')
    'A alternative to the builtin getfullargspec.\n\n  The builtin inspect.getfullargspec uses:\n  `skip_bound_args=False, follow_wrapped_chains=False`\n  in order to be backwards compatible.\n\n  This function instead skips bound args (self) and follows wrapped chains.\n\n  Args:\n    fn: The function or class of interest.\n  Returns:\n    An inspect.FullArgSpec namedtuple with the full arg spec of the function.\n  '
    try:
        sig = inspect._signature_from_callable(fn, skip_bound_arg=True, follow_wrapper_chains=True, sigcls=inspect.Signature)
    except Exception:
        raise TypeError('Unsupported callable.')
    args = []
    varargs = None
    varkw = None
    kwonlyargs = []
    defaults = ()
    annotations = {}
    defaults = ()
    kwdefaults = {}
    if sig.return_annotation is not sig.empty:
        annotations['return'] = sig.return_annotation
    for param in sig.parameters.values():
        kind = param.kind
        name = param.name
        if kind is inspect._POSITIONAL_ONLY:
            args.append(name)
        elif kind is inspect._POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)
        elif kind is inspect._VAR_POSITIONAL:
            varargs = name
        elif kind is inspect._KEYWORD_ONLY:
            kwonlyargs.append(name)
            if param.default is not param.empty:
                kwdefaults[name] = param.default
        elif kind is inspect._VAR_KEYWORD:
            varkw = name
        if param.annotation is not param.empty:
            annotations[name] = param.annotation
    if not kwdefaults:
        kwdefaults = None
    if not defaults:
        defaults = None
    return inspect.FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwdefaults, annotations)

def GetFullArgSpec(fn):
    if False:
        return 10
    'Returns a FullArgSpec describing the given callable.'
    original_fn = fn
    (fn, skip_arg) = _GetArgSpecInfo(fn)
    try:
        if sys.version_info[0:2] >= (3, 5):
            (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations) = Py3GetFullArgSpec(fn)
        elif six.PY3:
            (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations) = inspect.getfullargspec(fn)
        else:
            (args, varargs, varkw, defaults) = Py2GetArgSpec(fn)
            kwonlyargs = kwonlydefaults = None
            annotations = getattr(fn, '__annotations__', None)
    except TypeError:
        if inspect.isbuiltin(fn):
            return FullArgSpec(varargs='vars', varkw='kwargs')
        fields = getattr(original_fn, '_fields', None)
        if fields is not None:
            return FullArgSpec(args=list(fields))
        return FullArgSpec()
    skip_arg_required = six.PY2 or sys.version_info[0:2] == (3, 4)
    if skip_arg_required and skip_arg and args:
        args.pop(0)
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)

def GetFileAndLine(component):
    if False:
        while True:
            i = 10
    'Returns the filename and line number of component.\n\n  Args:\n    component: A component to find the source information for, usually a class\n        or routine.\n  Returns:\n    filename: The name of the file where component is defined.\n    lineno: The line number where component is defined.\n  '
    if inspect.isbuiltin(component):
        return (None, None)
    try:
        filename = inspect.getsourcefile(component)
    except TypeError:
        return (None, None)
    try:
        (unused_code, lineindex) = inspect.findsource(component)
        lineno = lineindex + 1
    except (IOError, IndexError):
        lineno = None
    return (filename, lineno)

def Info(component):
    if False:
        return 10
    'Returns a dict with information about the given component.\n\n  The dict will have at least some of the following fields.\n    type_name: The type of `component`.\n    string_form: A string representation of `component`.\n    file: The file in which `component` is defined.\n    line: The line number at which `component` is defined.\n    docstring: The docstring of `component`.\n    init_docstring: The init docstring of `component`.\n    class_docstring: The class docstring of `component`.\n    call_docstring: The call docstring of `component`.\n    length: The length of `component`.\n\n  Args:\n    component: The component to analyze.\n  Returns:\n    A dict with information about the component.\n  '
    try:
        from IPython.core import oinspect
        inspector = oinspect.Inspector()
        info = inspector.info(component)
        if info['docstring'] == '<no docstring>':
            info['docstring'] = None
    except ImportError:
        info = _InfoBackup(component)
    try:
        (unused_code, lineindex) = inspect.findsource(component)
        info['line'] = lineindex + 1
    except (TypeError, IOError):
        info['line'] = None
    if 'docstring' in info:
        info['docstring_info'] = docstrings.parse(info['docstring'])
    return info

def _InfoBackup(component):
    if False:
        for i in range(10):
            print('nop')
    "Returns a dict with information about the given component.\n\n  This function is to be called only in the case that IPython's\n  oinspect module is not available. The info dict it produces may\n  contain less information that contained in the info dict produced\n  by oinspect.\n\n  Args:\n    component: The component to analyze.\n  Returns:\n    A dict with information about the component.\n  "
    info = {}
    info['type_name'] = type(component).__name__
    info['string_form'] = str(component)
    (filename, lineno) = GetFileAndLine(component)
    info['file'] = filename
    info['line'] = lineno
    info['docstring'] = inspect.getdoc(component)
    try:
        info['length'] = str(len(component))
    except (TypeError, AttributeError):
        pass
    return info

def IsNamedTuple(component):
    if False:
        return 10
    'Return true if the component is a namedtuple.\n\n  Unfortunately, Python offers no native way to check for a namedtuple type.\n  Instead, we need to use a simple hack which should suffice for our case.\n  namedtuples are internally implemented as tuples, therefore we need to:\n    1. Check if the component is an instance of tuple.\n    2. Check if the component has a _fields attribute which regular tuples do\n       not have.\n\n  Args:\n    component: The component to analyze.\n  Returns:\n    True if the component is a namedtuple or False otherwise.\n  '
    if not isinstance(component, tuple):
        return False
    has_fields = bool(getattr(component, '_fields', None))
    return has_fields

def GetClassAttrsDict(component):
    if False:
        return 10
    'Gets the attributes of the component class, as a dict with name keys.'
    if not inspect.isclass(component):
        return None
    class_attrs_list = inspect.classify_class_attrs(component)
    return {class_attr.name: class_attr for class_attr in class_attrs_list}

def IsCoroutineFunction(fn):
    if False:
        while True:
            i = 10
    try:
        return six.PY34 and asyncio.iscoroutinefunction(fn)
    except:
        return False