from typing import TYPE_CHECKING
from robot.variables import contains_variable
from .typeinfo import TypeInfo
if TYPE_CHECKING:
    from robot.conf import LanguagesLike
    from .argumentspec import ArgumentSpec
    from .customconverters import CustomArgumentConverters

class ArgumentConverter:

    def __init__(self, arg_spec: 'ArgumentSpec', custom_converters: 'CustomArgumentConverters', dry_run: bool=False, languages: 'LanguagesLike'=None):
        if False:
            return 10
        self.spec = arg_spec
        self.custom_converters = custom_converters
        self.dry_run = dry_run
        self.languages = languages

    def convert(self, positional, named):
        if False:
            return 10
        return (self._convert_positional(positional), self._convert_named(named))

    def _convert_positional(self, positional):
        if False:
            i = 10
            return i + 15
        names = self.spec.positional
        converted = [self._convert(name, value) for (name, value) in zip(names, positional)]
        if self.spec.var_positional:
            converted.extend((self._convert(self.spec.var_positional, value) for value in positional[len(names):]))
        return converted

    def _convert_named(self, named):
        if False:
            while True:
                i = 10
        names = set(self.spec.positional) | set(self.spec.named_only)
        var_named = self.spec.var_named
        return [(name, self._convert(name if name in names else var_named, value)) for (name, value) in named]

    def _convert(self, name, value):
        if False:
            i = 10
            return i + 15
        spec = self.spec
        if spec.types is None or (self.dry_run and contains_variable(value, identifiers='$@&%')):
            return value
        conversion_error = None
        if value is None and name in spec.defaults and (spec.defaults[name] is None):
            return value
        if name in spec.types:
            info: TypeInfo = spec.types[name]
            try:
                return info.convert(value, name, self.custom_converters, self.languages)
            except ValueError as err:
                conversion_error = err
            except TypeError:
                pass
        if name in spec.defaults:
            typ = type(spec.defaults[name])
            if typ == str:
                info = TypeInfo()
            elif typ == int:
                info = TypeInfo.from_sequence([int, float])
            else:
                info = TypeInfo.from_type(typ)
            try:
                return info.convert(value, name, languages=self.languages)
            except (ValueError, TypeError):
                pass
        if conversion_error:
            raise conversion_error
        return value