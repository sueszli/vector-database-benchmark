from typing import TYPE_CHECKING
from robot.errors import DataError
from robot.utils import is_string, is_dict_like, split_from_equals
from robot.variables import is_dict_variable
from .argumentvalidator import ArgumentValidator
if TYPE_CHECKING:
    from .argumentspec import ArgumentSpec

class ArgumentResolver:

    def __init__(self, spec: 'ArgumentSpec', resolve_named: bool=True, resolve_variables_until: 'int|None'=None, dict_to_kwargs: bool=False):
        if False:
            i = 10
            return i + 15
        self.named_resolver = NamedArgumentResolver(spec) if resolve_named else NullNamedArgumentResolver()
        self.variable_replacer = VariableReplacer(spec, resolve_variables_until)
        self.dict_to_kwargs = DictToKwargs(spec, dict_to_kwargs)
        self.argument_validator = ArgumentValidator(spec)

    def resolve(self, arguments, variables=None):
        if False:
            while True:
                i = 10
        (positional, named) = self.named_resolver.resolve(arguments, variables)
        (positional, named) = self.variable_replacer.replace(positional, named, variables)
        (positional, named) = self.dict_to_kwargs.handle(positional, named)
        self.argument_validator.validate(positional, named, dryrun=variables is None)
        return (positional, named)

class NamedArgumentResolver:

    def __init__(self, spec: 'ArgumentSpec'):
        if False:
            i = 10
            return i + 15
        self.spec = spec

    def resolve(self, arguments, variables=None):
        if False:
            return 10
        named = []
        for arg in arguments[len(self.spec.embedded):]:
            if is_dict_variable(arg):
                named.append(arg)
            elif self._is_named(arg, named, variables):
                named.append(split_from_equals(arg))
            elif named:
                self._raise_positional_after_named()
        positional = arguments[:-len(named)] if named else arguments
        return (positional, named)

    def _is_named(self, arg, previous_named, variables=None):
        if False:
            i = 10
            return i + 15
        (name, value) = split_from_equals(arg)
        if value is None:
            return False
        if variables:
            try:
                name = variables.replace_scalar(name)
            except DataError:
                return False
        return bool(previous_named or self.spec.var_named or name in self.spec.named)

    def _raise_positional_after_named(self):
        if False:
            for i in range(10):
                print('nop')
        raise DataError(f"{self.spec.type.capitalize()} '{self.spec.name}' got positional argument after named arguments.")

class NullNamedArgumentResolver:

    def resolve(self, arguments, variables=None):
        if False:
            print('Hello World!')
        return (arguments, {})

class DictToKwargs:

    def __init__(self, spec: 'ArgumentSpec', enabled: bool=False):
        if False:
            print('Hello World!')
        self.maxargs = spec.maxargs
        self.enabled = enabled and bool(spec.var_named)

    def handle(self, positional, named):
        if False:
            for i in range(10):
                print('nop')
        if self.enabled and self._extra_arg_has_kwargs(positional, named):
            named = positional.pop().items()
        return (positional, named)

    def _extra_arg_has_kwargs(self, positional, named):
        if False:
            i = 10
            return i + 15
        if named or len(positional) != self.maxargs + 1:
            return False
        return is_dict_like(positional[-1])

class VariableReplacer:

    def __init__(self, spec: 'ArgumentSpec', resolve_until: 'int|None'=None):
        if False:
            return 10
        self.spec = spec
        self.resolve_until = resolve_until

    def replace(self, positional, named, variables=None):
        if False:
            return 10
        if variables:
            if self.spec.embedded:
                embedded = len(self.spec.embedded)
                positional = [variables.replace_scalar(emb) for emb in positional[:embedded]] + variables.replace_list(positional[embedded:])
            else:
                positional = variables.replace_list(positional, self.resolve_until)
            named = list(self._replace_named(named, variables.replace_scalar))
        else:
            named = [var if isinstance(var, tuple) else (var, var) for var in named]
        return (positional, named)

    def _replace_named(self, named, replace_scalar):
        if False:
            while True:
                i = 10
        for item in named:
            for (name, value) in self._get_replaced_named(item, replace_scalar):
                if not is_string(name):
                    raise DataError('Argument names must be strings.')
                yield (name, value)

    def _get_replaced_named(self, item, replace_scalar):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, tuple):
            return replace_scalar(item).items()
        (name, value) = item
        return [(replace_scalar(name), replace_scalar(value))]