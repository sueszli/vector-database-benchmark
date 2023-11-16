from abc import ABC, abstractmethod
from inspect import isclass, signature, Parameter
from typing import get_type_hints
from robot.errors import DataError
from robot.utils import is_string, split_from_equals
from robot.variables import is_assign, is_scalar_assign
from .argumentspec import ArgumentSpec

class ArgumentParser(ABC):

    def __init__(self, type='Keyword', error_reporter=None):
        if False:
            while True:
                i = 10
        self._type = type
        self._error_reporter = error_reporter

    @abstractmethod
    def parse(self, source, name=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _report_error(self, error):
        if False:
            for i in range(10):
                print('nop')
        if self._error_reporter:
            self._error_reporter(error)
        else:
            raise DataError(f'Invalid argument specification: {error}')

class PythonArgumentParser(ArgumentParser):

    def parse(self, handler, name=None):
        if False:
            for i in range(10):
                print('nop')
        spec = ArgumentSpec(name, self._type)
        self._set_args(spec, handler)
        self._set_types(spec, handler)
        return spec

    def _set_args(self, spec, handler):
        if False:
            return 10
        try:
            sig = signature(handler)
        except ValueError:
            spec.var_positional = 'args'
            return
        except TypeError as err:
            raise DataError(str(err))
        parameters = list(sig.parameters.values())
        if getattr(handler, '__name__', None) == '__init__':
            parameters = parameters[1:]
        setters = {Parameter.POSITIONAL_ONLY: spec.positional_only.append, Parameter.POSITIONAL_OR_KEYWORD: spec.positional_or_named.append, Parameter.VAR_POSITIONAL: lambda name: setattr(spec, 'var_positional', name), Parameter.KEYWORD_ONLY: spec.named_only.append, Parameter.VAR_KEYWORD: lambda name: setattr(spec, 'var_named', name)}
        for param in parameters:
            setters[param.kind](param.name)
            if param.default is not param.empty:
                spec.defaults[param.name] = param.default

    def _set_types(self, spec, handler):
        if False:
            i = 10
            return i + 15
        types = self._get_types(handler)
        if isinstance(types, dict) and 'return' in types:
            spec.return_type = types.pop('return')
        spec.types = types

    def _get_types(self, handler):
        if False:
            i = 10
            return i + 15
        if isclass(handler):
            handler = handler.__init__
        types = getattr(handler, 'robot_types', ())
        if types or types is None:
            return types
        try:
            return get_type_hints(handler)
        except Exception:
            return getattr(handler, '__annotations__', {})

class ArgumentSpecParser(ArgumentParser):

    def parse(self, argspec, name=None):
        if False:
            for i in range(10):
                print('nop')
        spec = ArgumentSpec(name, self._type)
        named_only = positional_only_separator_seen = False
        for arg in argspec:
            arg = self._validate_arg(arg)
            if spec.var_named:
                self._report_error('Only last argument can be kwargs.')
            elif self._is_positional_only_separator(arg):
                if named_only:
                    self._report_error('Positional-only separator must be before named-only arguments.')
                if positional_only_separator_seen:
                    self._report_error('Too many positional-only separators.')
                spec.positional_only = spec.positional_or_named
                spec.positional_or_named = []
                positional_only_separator_seen = True
            elif isinstance(arg, tuple):
                (arg, default) = arg
                arg = self._add_arg(spec, arg, named_only)
                spec.defaults[arg] = default
            elif self._is_var_named(arg):
                spec.var_named = self._format_var_named(arg)
            elif self._is_var_positional(arg):
                if named_only:
                    self._report_error('Cannot have multiple varargs.')
                if not self._is_named_only_separator(arg):
                    spec.var_positional = self._format_var_positional(arg)
                named_only = True
            elif spec.defaults and (not named_only):
                self._report_error('Non-default argument after default arguments.')
            else:
                self._add_arg(spec, arg, named_only)
        return spec

    @abstractmethod
    def _validate_arg(self, arg):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def _is_var_named(self, arg):
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def _format_var_named(self, kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abstractmethod
    def _is_positional_only_separator(self, arg):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def _is_named_only_separator(self, arg):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def _is_var_positional(self, arg):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @abstractmethod
    def _format_var_positional(self, varargs):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _format_arg(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return arg

    def _add_arg(self, spec, arg, named_only=False):
        if False:
            return 10
        arg = self._format_arg(arg)
        target = spec.positional_or_named if not named_only else spec.named_only
        target.append(arg)
        return arg

class DynamicArgumentParser(ArgumentSpecParser):

    def _validate_arg(self, arg):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(arg, tuple):
            if self._is_invalid_tuple(arg):
                self._report_error(f'Invalid argument "{arg}".')
            if len(arg) == 1:
                return arg[0]
            return arg
        if '=' in arg:
            return tuple(arg.split('=', 1))
        return arg

    def _is_invalid_tuple(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return len(arg) > 2 or not is_string(arg[0]) or (arg[0].startswith('*') and len(arg) > 1)

    def _is_var_named(self, arg):
        if False:
            i = 10
            return i + 15
        return arg[:2] == '**'

    def _format_var_named(self, kwargs):
        if False:
            while True:
                i = 10
        return kwargs[2:]

    def _is_var_positional(self, arg):
        if False:
            while True:
                i = 10
        return arg and arg[0] == '*'

    def _is_positional_only_separator(self, arg):
        if False:
            i = 10
            return i + 15
        return arg == '/'

    def _is_named_only_separator(self, arg):
        if False:
            i = 10
            return i + 15
        return arg == '*'

    def _format_var_positional(self, varargs):
        if False:
            i = 10
            return i + 15
        return varargs[1:]

class UserKeywordArgumentParser(ArgumentSpecParser):

    def _validate_arg(self, arg):
        if False:
            while True:
                i = 10
        (arg, default) = split_from_equals(arg)
        if not (is_assign(arg) or arg == '@{}'):
            self._report_error(f"Invalid argument syntax '{arg}'.")
        if default is None:
            return arg
        if not is_scalar_assign(arg):
            typ = 'list' if arg[0] == '@' else 'dictionary'
            self._report_error(f"Only normal arguments accept default values, {typ} arguments like '{arg}' do not.")
        return (arg, default)

    def _is_var_named(self, arg):
        if False:
            while True:
                i = 10
        return arg and arg[0] == '&'

    def _format_var_named(self, kwargs):
        if False:
            i = 10
            return i + 15
        return kwargs[2:-1]

    def _is_var_positional(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return arg and arg[0] == '@'

    def _is_positional_only_separator(self, arg):
        if False:
            i = 10
            return i + 15
        return False

    def _is_named_only_separator(self, arg):
        if False:
            i = 10
            return i + 15
        return arg == '@{}'

    def _format_var_positional(self, varargs):
        if False:
            i = 10
            return i + 15
        return varargs[2:-1]

    def _format_arg(self, arg):
        if False:
            while True:
                i = 10
        return arg[2:-1]