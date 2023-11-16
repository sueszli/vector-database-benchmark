from typing import TYPE_CHECKING
from robot.errors import DataError
from robot.utils import plural_or_not as s, seq2str
from robot.variables import is_dict_variable, is_list_variable
if TYPE_CHECKING:
    from .argumentspec import ArgumentSpec

class ArgumentValidator:

    def __init__(self, arg_spec: 'ArgumentSpec'):
        if False:
            return 10
        self.arg_spec = arg_spec

    def validate(self, positional, named, dryrun=False):
        if False:
            while True:
                i = 10
        named = set((name for (name, value) in named))
        if dryrun and (any((is_list_variable(arg) for arg in positional)) or any((is_dict_variable(arg) for arg in named))):
            return
        self._validate_no_multiple_values(positional, named, self.arg_spec)
        self._validate_no_positional_only_as_named(named, self.arg_spec)
        self._validate_positional_limits(positional, named, self.arg_spec)
        self._validate_no_mandatory_missing(positional, named, self.arg_spec)
        self._validate_no_named_only_missing(named, self.arg_spec)
        self._validate_no_extra_named(named, self.arg_spec)

    def _validate_no_multiple_values(self, positional, named, spec):
        if False:
            return 10
        for name in spec.positional[:len(positional) - len(spec.embedded)]:
            if name in named and name not in spec.positional_only:
                self._raise_error(f"got multiple values for argument '{name}'")

    def _raise_error(self, message):
        if False:
            for i in range(10):
                print('nop')
        spec = self.arg_spec
        name = f"'{spec.name}' " if spec.name else ''
        raise DataError(f'{spec.type.capitalize()} {name}{message}.')

    def _validate_no_positional_only_as_named(self, named, spec):
        if False:
            while True:
                i = 10
        if not spec.var_named:
            for name in named:
                if name in spec.positional_only:
                    self._raise_error(f"does not accept argument '{name}' as named argument")

    def _validate_positional_limits(self, positional, named, spec):
        if False:
            for i in range(10):
                print('nop')
        count = len(positional) + self._named_positionals(named, spec)
        if not spec.minargs <= count <= spec.maxargs:
            self._raise_wrong_count(count, spec)

    def _named_positionals(self, named, spec):
        if False:
            return 10
        return sum((1 for n in named if n in spec.positional_or_named))

    def _raise_wrong_count(self, count, spec):
        if False:
            while True:
                i = 10
        embedded = len(spec.embedded)
        minargs = spec.minargs - embedded
        maxargs = spec.maxargs - embedded
        if minargs == maxargs:
            expected = f'{minargs} argument{s(minargs)}'
        elif not spec.var_positional:
            expected = f'{minargs} to {maxargs} arguments'
        else:
            expected = f'at least {minargs} argument{s(minargs)}'
        if spec.var_named or spec.named_only:
            expected = expected.replace('argument', 'non-named argument')
        self._raise_error(f'expected {expected}, got {count - embedded}')

    def _validate_no_mandatory_missing(self, positional, named, spec):
        if False:
            return 10
        for name in spec.positional[len(positional):]:
            if name not in spec.defaults and name not in named:
                self._raise_error(f"missing value for argument '{name}'")

    def _validate_no_named_only_missing(self, named, spec):
        if False:
            for i in range(10):
                print('nop')
        defined = set(named) | set(spec.defaults)
        missing = [arg for arg in spec.named_only if arg not in defined]
        if missing:
            self._raise_error(f'missing named-only argument{s(missing)} {seq2str(sorted(missing))}')

    def _validate_no_extra_named(self, named, spec):
        if False:
            print('Hello World!')
        if not spec.var_named:
            extra = set(named) - set(spec.positional_or_named) - set(spec.named_only)
            if extra:
                self._raise_error(f'got unexpected named argument{s(extra)} {seq2str(sorted(extra))}')