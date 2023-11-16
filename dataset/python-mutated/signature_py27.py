"""Function signature objects for callables

Back port of Python 3.3's function signature tools from the inspect module,
modified to be compatible with Python 2.7 and 3.2+.
"""
from __future__ import absolute_import, division, print_function
import functools
import itertools
import re
import types
from collections import OrderedDict
__version__ = '0.4'
__all__ = ['BoundArguments', 'Parameter', 'Signature', 'signature']
_WrapperDescriptor = type(type.__call__)
_MethodWrapper = type(all.__call__)
_NonUserDefinedCallables = (_WrapperDescriptor, _MethodWrapper, types.BuiltinFunctionType)

def formatannotation(annotation, base_module=None):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', '__builtin__', base_module):
            return annotation.__name__
        return annotation.__module__ + '.' + annotation.__name__
    return repr(annotation)

def _get_user_defined_method(cls, method_name, *nested):
    if False:
        print('Hello World!')
    try:
        if cls is type:
            return
        meth = getattr(cls, method_name)
        for name in nested:
            meth = getattr(meth, name, meth)
    except AttributeError:
        return
    else:
        if not isinstance(meth, _NonUserDefinedCallables):
            return meth

def signature(obj):
    if False:
        while True:
            i = 10
    'Get a signature object for the passed callable.'
    if not callable(obj):
        raise TypeError('{0!r} is not a callable object'.format(obj))
    if isinstance(obj, types.MethodType):
        sig = signature(obj.__func__)
        if obj.__self__ is None:
            if sig.parameters:
                first = sig.parameters.values()[0].replace(kind=_POSITIONAL_ONLY)
                return sig.replace(parameters=(first,) + tuple(sig.parameters.values())[1:])
            else:
                return sig
        else:
            return sig.replace(parameters=tuple(sig.parameters.values())[1:])
    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            return sig
    try:
        wrapped = obj.__wrapped__
    except AttributeError:
        pass
    else:
        return signature(wrapped)
    if isinstance(obj, types.FunctionType):
        return Signature.from_function(obj)
    if isinstance(obj, functools.partial):
        sig = signature(obj.func)
        new_params = OrderedDict(sig.parameters.items())
        partial_args = obj.args or ()
        partial_keywords = obj.keywords or {}
        try:
            ba = sig.bind_partial(*partial_args, **partial_keywords)
        except TypeError:
            msg = 'partial object {0!r} has incorrect arguments'.format(obj)
            raise ValueError(msg)
        for (arg_name, arg_value) in ba.arguments.items():
            param = new_params[arg_name]
            if arg_name in partial_keywords:
                new_params[arg_name] = param.replace(default=arg_value, _partial_kwarg=True)
            elif param.kind not in (_VAR_KEYWORD, _VAR_POSITIONAL) and (not param._partial_kwarg):
                new_params.pop(arg_name)
        return sig.replace(parameters=new_params.values())
    sig = None
    if isinstance(obj, type):
        call = _get_user_defined_method(type(obj), '__call__')
        if call is not None:
            sig = signature(call)
        else:
            new = _get_user_defined_method(obj, '__new__')
            if new is not None:
                sig = signature(new)
            else:
                init = _get_user_defined_method(obj, '__init__')
                if init is not None:
                    sig = signature(init)
    elif not isinstance(obj, _NonUserDefinedCallables):
        call = _get_user_defined_method(type(obj), '__call__', 'im_func')
        if call is not None:
            sig = signature(call)
    if sig is not None:
        return sig.replace(parameters=tuple(sig.parameters.values())[1:])
    if isinstance(obj, types.BuiltinFunctionType):
        msg = 'no signature found for builtin function {0!r}'.format(obj)
        raise ValueError(msg)
    raise ValueError('callable {0!r} is not supported by signature'.format(obj))

class _void(object):
    """A private marker - used in Parameter & Signature"""

class _empty(object):
    pass

class _ParameterKind(int):

    def __new__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        obj = int.__new__(self, *args)
        obj._name = kwargs['name']
        return obj

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<_ParameterKind: {0!r}>'.format(self._name)
_POSITIONAL_ONLY = _ParameterKind(0, name='POSITIONAL_ONLY')
_POSITIONAL_OR_KEYWORD = _ParameterKind(1, name='POSITIONAL_OR_KEYWORD')
_VAR_POSITIONAL = _ParameterKind(2, name='VAR_POSITIONAL')
_KEYWORD_ONLY = _ParameterKind(3, name='KEYWORD_ONLY')
_VAR_KEYWORD = _ParameterKind(4, name='VAR_KEYWORD')

class Parameter(object):
    """Represents a parameter in a function signature.

    Has the following public attributes:

    * name : str
        The name of the parameter as a string.
    * default : object
        The default value for the parameter if specified.  If the
        parameter has no default value, this attribute is not set.
    * annotation
        The annotation for the parameter if specified.  If the
        parameter has no annotation, this attribute is not set.
    * kind : str
        Describes how argument values are bound to the parameter.
        Possible values: `Parameter.POSITIONAL_ONLY`,
        `Parameter.POSITIONAL_OR_KEYWORD`, `Parameter.VAR_POSITIONAL`,
        `Parameter.KEYWORD_ONLY`, `Parameter.VAR_KEYWORD`.
    """
    __slots__ = ('_name', '_kind', '_default', '_annotation', '_partial_kwarg')
    POSITIONAL_ONLY = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = _POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = _VAR_POSITIONAL
    KEYWORD_ONLY = _KEYWORD_ONLY
    VAR_KEYWORD = _VAR_KEYWORD
    empty = _empty

    def __init__(self, name, kind, default=_empty, annotation=_empty, _partial_kwarg=False):
        if False:
            for i in range(10):
                print('nop')
        if kind not in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _VAR_POSITIONAL, _KEYWORD_ONLY, _VAR_KEYWORD):
            raise ValueError("invalid value for 'Parameter.kind' attribute")
        self._kind = kind
        if default is not _empty:
            if kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
                msg = '{0} parameters cannot have default values'.format(kind)
                raise ValueError(msg)
        self._default = default
        self._annotation = annotation
        if name is None:
            if kind != _POSITIONAL_ONLY:
                raise ValueError('None is not a valid name for a non-positional-only parameter')
            self._name = name
        else:
            name = str(name)
            if kind != _POSITIONAL_ONLY and (not re.match('[a-z_]\\w*$', name, re.I)):
                msg = '{0!r} is not a valid parameter name'.format(name)
                raise ValueError(msg)
            self._name = name
        self._partial_kwarg = _partial_kwarg

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self._name

    @property
    def default(self):
        if False:
            i = 10
            return i + 15
        return self._default

    @property
    def annotation(self):
        if False:
            i = 10
            return i + 15
        return self._annotation

    @property
    def kind(self):
        if False:
            return 10
        return self._kind

    def replace(self, name=_void, kind=_void, annotation=_void, default=_void, _partial_kwarg=_void):
        if False:
            print('Hello World!')
        'Creates a customized copy of the Parameter.'
        if name is _void:
            name = self._name
        if kind is _void:
            kind = self._kind
        if annotation is _void:
            annotation = self._annotation
        if default is _void:
            default = self._default
        if _partial_kwarg is _void:
            _partial_kwarg = self._partial_kwarg
        return type(self)(name, kind, default=default, annotation=annotation, _partial_kwarg=_partial_kwarg)

    def __str__(self):
        if False:
            while True:
                i = 10
        kind = self.kind
        formatted = self._name
        if kind == _POSITIONAL_ONLY:
            if formatted is None:
                formatted = ''
            formatted = '<{0}>'.format(formatted)
        if self._annotation is not _empty:
            formatted = '{0}:{1}'.format(formatted, formatannotation(self._annotation))
        if self._default is not _empty:
            formatted = '{0}={1}'.format(formatted, repr(self._default))
        if kind == _VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == _VAR_KEYWORD:
            formatted = '**' + formatted
        return formatted

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<{0} at {1:#x} {2!r}>'.format(self.__class__.__name__, id(self), self.name)

    def __hash__(self):
        if False:
            while True:
                i = 10
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return issubclass(other.__class__, Parameter) and self._name == other._name and (self._kind == other._kind) and (self._default == other._default) and (self._annotation == other._annotation)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

class BoundArguments(object):
    """Result of `Signature.bind` call.  Holds the mapping of arguments
    to the function's parameters.

    Has the following public attributes:

    * arguments : OrderedDict
        An ordered mutable mapping of parameters' names to arguments' values.
        Does not contain arguments' default values.
    * signature : Signature
        The Signature object that created this instance.
    * args : tuple
        Tuple of positional arguments values.
    * kwargs : dict
        Dict of keyword arguments values.
    """

    def __init__(self, signature, arguments):
        if False:
            i = 10
            return i + 15
        self.arguments = arguments
        self._signature = signature

    @property
    def signature(self):
        if False:
            while True:
                i = 10
        return self._signature

    @property
    def args(self):
        if False:
            i = 10
            return i + 15
        args = []
        for (param_name, param) in self._signature.parameters.items():
            if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY) or param._partial_kwarg:
                break
            try:
                arg = self.arguments[param_name]
            except KeyError:
                break
            else:
                if param.kind == _VAR_POSITIONAL:
                    args.extend(arg)
                else:
                    args.append(arg)
        return tuple(args)

    @property
    def kwargs(self):
        if False:
            i = 10
            return i + 15
        kwargs = {}
        kwargs_started = False
        for (param_name, param) in self._signature.parameters.items():
            if not kwargs_started:
                if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY) or param._partial_kwarg:
                    kwargs_started = True
                elif param_name not in self.arguments:
                    kwargs_started = True
                    continue
            if not kwargs_started:
                continue
            try:
                arg = self.arguments[param_name]
            except KeyError:
                pass
            else:
                if param.kind == _VAR_KEYWORD:
                    kwargs.update(arg)
                else:
                    kwargs[param_name] = arg
        return kwargs

    def __hash__(self):
        if False:
            while True:
                i = 10
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return issubclass(other.__class__, BoundArguments) and self.signature == other.signature and (self.arguments == other.arguments)

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

class Signature(object):
    """A Signature object represents the overall signature of a function.
    It stores a Parameter object for each parameter accepted by the
    function, as well as information specific to the function itself.

    A Signature object has the following public attributes and methods:

    * parameters : OrderedDict
        An ordered mapping of parameters' names to the corresponding
        Parameter objects (keyword-only arguments are in the same order
        as listed in `code.co_varnames`).
    * return_annotation : object
        The annotation for the return type of the function if specified.
        If the function has no annotation for its return type, this
        attribute is not set.
    * bind(*args, **kwargs) -> BoundArguments
        Creates a mapping from positional and keyword arguments to
        parameters.
    * bind_partial(*args, **kwargs) -> BoundArguments
        Creates a partial mapping from positional and keyword arguments
        to parameters (simulating 'functools.partial' behavior.)
    """
    __slots__ = ('_return_annotation', '_parameters')
    _parameter_cls = Parameter
    _bound_arguments_cls = BoundArguments
    empty = _empty

    def __init__(self, parameters=None, return_annotation=_empty, __validate_parameters__=True):
        if False:
            print('Hello World!')
        "Constructs Signature from the given list of Parameter\n        objects and 'return_annotation'.  All arguments are optional.\n        "
        if parameters is None:
            params = OrderedDict()
        elif __validate_parameters__:
            params = OrderedDict()
            top_kind = _POSITIONAL_ONLY
            for (idx, param) in enumerate(parameters):
                kind = param.kind
                if kind < top_kind:
                    msg = 'wrong parameter order: {0} before {1}'
                    msg = msg.format(top_kind, param.kind)
                    raise ValueError(msg)
                else:
                    top_kind = kind
                name = param.name
                if name is None:
                    name = str(idx)
                    param = param.replace(name=name)
                if name in params:
                    msg = 'duplicate parameter name: {0!r}'.format(name)
                    raise ValueError(msg)
                params[name] = param
        else:
            params = OrderedDict(((param.name, param) for param in parameters))
        self._parameters = params
        self._return_annotation = return_annotation

    @classmethod
    def from_function(cls, func):
        if False:
            i = 10
            return i + 15
        'Constructs Signature for the given python function'
        if not isinstance(func, types.FunctionType):
            raise TypeError('{0!r} is not a Python function'.format(func))
        Parameter = cls._parameter_cls
        func_code = func.__code__
        pos_count = func_code.co_argcount
        arg_names = func_code.co_varnames
        positional = tuple(arg_names[:pos_count])
        keyword_only_count = getattr(func_code, 'co_kwonlyargcount', 0)
        keyword_only = arg_names[pos_count:pos_count + keyword_only_count]
        annotations = getattr(func, '__annotations__', {})
        defaults = func.__defaults__
        kwdefaults = getattr(func, '__kwdefaults__', None)
        if defaults:
            pos_default_count = len(defaults)
        else:
            pos_default_count = 0
        parameters = []
        non_default_count = pos_count - pos_default_count
        for name in positional[:non_default_count]:
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_POSITIONAL_OR_KEYWORD))
        for (offset, name) in enumerate(positional[non_default_count:]):
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_POSITIONAL_OR_KEYWORD, default=defaults[offset]))
        if func_code.co_flags & 4:
            name = arg_names[pos_count + keyword_only_count]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_VAR_POSITIONAL))
        for name in keyword_only:
            default = _empty
            if kwdefaults is not None:
                default = kwdefaults.get(name, _empty)
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_KEYWORD_ONLY, default=default))
        if func_code.co_flags & 8:
            index = pos_count + keyword_only_count
            if func_code.co_flags & 4:
                index += 1
            name = arg_names[index]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_VAR_KEYWORD))
        return cls(parameters, return_annotation=annotations.get('return', _empty), __validate_parameters__=False)

    @property
    def parameters(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return types.MappingProxyType(self._parameters)
        except AttributeError:
            return OrderedDict(self._parameters.items())

    @property
    def return_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        return self._return_annotation

    def replace(self, parameters=_void, return_annotation=_void):
        if False:
            i = 10
            return i + 15
        "Creates a customized copy of the Signature.\n        Pass 'parameters' and/or 'return_annotation' arguments\n        to override them in the new copy.\n        "
        if parameters is _void:
            parameters = self.parameters.values()
        if return_annotation is _void:
            return_annotation = self._return_annotation
        return type(self)(parameters, return_annotation=return_annotation)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        if False:
            return 10
        if not issubclass(type(other), Signature) or self.return_annotation != other.return_annotation or len(self.parameters) != len(other.parameters):
            return False
        other_positions = dict(((param, idx) for (idx, param) in enumerate(other.parameters.keys())))
        for (idx, (param_name, param)) in enumerate(self.parameters.items()):
            if param.kind == _KEYWORD_ONLY:
                try:
                    other_param = other.parameters[param_name]
                except KeyError:
                    return False
                else:
                    if param != other_param:
                        return False
            else:
                try:
                    other_idx = other_positions[param_name]
                except KeyError:
                    return False
                else:
                    if idx != other_idx or param != other.parameters[param_name]:
                        return False
        return True

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def _bind(self, args, kwargs, partial=False):
        if False:
            while True:
                i = 10
        "Private method.  Don't use directly."
        arguments = OrderedDict()
        parameters = iter(self.parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)
        if partial:
            for (param_name, param) in self.parameters.items():
                if param._partial_kwarg and param_name not in kwargs:
                    kwargs[param_name] = param.default
        while True:
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                try:
                    param = next(parameters)
                except StopIteration:
                    break
                else:
                    if param.kind == _VAR_POSITIONAL:
                        break
                    elif param.name in kwargs:
                        if param.kind == _POSITIONAL_ONLY:
                            msg = '{arg!r} parameter is positional only, but was passed as a keyword'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg)
                        parameters_ex = (param,)
                        break
                    elif param.kind == _VAR_KEYWORD or param.default is not _empty:
                        parameters_ex = (param,)
                        break
                    elif partial:
                        parameters_ex = (param,)
                        break
                    else:
                        msg = '{arg!r} parameter lacking default value'
                        msg = msg.format(arg=param.name)
                        raise TypeError(msg)
            else:
                try:
                    param = next(parameters)
                except StopIteration:
                    raise TypeError('too many positional arguments')
                else:
                    if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                        raise TypeError('too many positional arguments')
                    if param.kind == _VAR_POSITIONAL:
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break
                    if param.name in kwargs:
                        raise TypeError('multiple values for argument {arg!r}'.format(arg=param.name))
                    arguments[param.name] = arg_val
        kwargs_param = None
        for param in itertools.chain(parameters_ex, parameters):
            if param.kind == _POSITIONAL_ONLY:
                raise TypeError('{arg!r} parameter is positional only, but was passed as a keyword'.format(arg=param.name))
            if param.kind == _VAR_KEYWORD:
                kwargs_param = param
                continue
            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                if not partial and param.kind != _VAR_POSITIONAL and (param.default is _empty):
                    raise TypeError('{arg!r} parameter lacking default value'.format(arg=param_name))
            else:
                arguments[param_name] = arg_val
        if kwargs:
            if kwargs_param is not None:
                arguments[kwargs_param.name] = kwargs
            else:
                raise TypeError('too many keyword arguments')
        return self._bound_arguments_cls(self, arguments)

    def bind(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "Get a BoundArguments object, that maps the passed `args`\n        and `kwargs` to the function's signature.  Raises `TypeError`\n        if the passed arguments can not be bound.\n        "
        return self._bind(args, kwargs)

    def bind_partial(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Get a BoundArguments object, that partially maps the\n        passed `args` and `kwargs` to the function's signature.\n        Raises `TypeError` if the passed arguments can not be bound.\n        "
        return self._bind(args, kwargs, partial=True)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        result = []
        render_kw_only_separator = True
        for (idx, param) in enumerate(self.parameters.values()):
            formatted = str(param)
            kind = param.kind
            if kind == _VAR_POSITIONAL:
                render_kw_only_separator = False
            elif kind == _KEYWORD_ONLY and render_kw_only_separator:
                result.append('*')
                render_kw_only_separator = False
            result.append(formatted)
        rendered = '({0})'.format(', '.join(result))
        if self.return_annotation is not _empty:
            anno = formatannotation(self.return_annotation)
            rendered += ' -> {0}'.format(anno)
        return rendered