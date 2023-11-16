import random as random_module
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Optional, Type, Union
from .typing import SeedType
if TYPE_CHECKING:
    from .providers import BaseProvider
_re_token = re.compile('\\{\\{\\s*(\\w+)(:\\s*\\w+?)?\\s*\\}\\}')
random = random_module.Random()
mod_random = random
Sentinel = object()

class Generator:
    __config: Dict[str, Dict[Hashable, Any]] = {'arguments': {}}
    _is_seeded = False
    _global_seed = Sentinel

    def __init__(self, **config: Dict) -> None:
        if False:
            i = 10
            return i + 15
        self.providers: List['BaseProvider'] = []
        self.__config = dict(list(self.__config.items()) + list(config.items()))
        self.__random = random

    def add_provider(self, provider: Union['BaseProvider', Type['BaseProvider']]) -> None:
        if False:
            return 10
        if isinstance(provider, type):
            provider = provider(self)
        self.providers.insert(0, provider)
        for method_name in dir(provider):
            if method_name.startswith('_'):
                continue
            faker_function = getattr(provider, method_name)
            if callable(faker_function):
                self.set_formatter(method_name, faker_function)

    def provider(self, name: str) -> Optional['BaseProvider']:
        if False:
            while True:
                i = 10
        try:
            lst = [p for p in self.get_providers() if hasattr(p, '__provider__') and p.__provider__ == name.lower()]
            return lst[0]
        except IndexError:
            return None

    def get_providers(self) -> List['BaseProvider']:
        if False:
            i = 10
            return i + 15
        'Returns added providers.'
        return self.providers

    @property
    def random(self) -> random_module.Random:
        if False:
            for i in range(10):
                print('nop')
        return self.__random

    @random.setter
    def random(self, value: random_module.Random) -> None:
        if False:
            while True:
                i = 10
        self.__random = value

    def seed_instance(self, seed: Optional[SeedType]=None) -> 'Generator':
        if False:
            print('Hello World!')
        'Calls random.seed'
        if self.__random == random:
            self.__random = random_module.Random()
        self.__random.seed(seed)
        self._is_seeded = True
        return self

    @classmethod
    def seed(cls, seed: Optional[SeedType]=None) -> None:
        if False:
            while True:
                i = 10
        random.seed(seed)
        cls._global_seed = seed
        cls._is_seeded = True

    def format(self, formatter: str, *args: Any, **kwargs: Any) -> str:
        if False:
            while True:
                i = 10
        '\n        This is a secure way to make a fake from another Provider.\n        '
        return self.get_formatter(formatter)(*args, **kwargs)

    def get_formatter(self, formatter: str) -> Callable:
        if False:
            print('Hello World!')
        try:
            return getattr(self, formatter)
        except AttributeError:
            if 'locale' in self.__config:
                msg = f"Unknown formatter {formatter!r} with locale {self.__config['locale']!r}"
            else:
                raise AttributeError(f'Unknown formatter {formatter!r}')
            raise AttributeError(msg)

    def set_formatter(self, name: str, formatter: Callable) -> None:
        if False:
            print('Hello World!')
        '\n        This method adds a provider method to generator.\n        Override this method to add some decoration or logging stuff.\n        '
        setattr(self, name, formatter)

    def set_arguments(self, group: str, argument: str, value: Optional[Any]=None) -> None:
        if False:
            while True:
                i = 10
        "\n        Creates an argument group, with an individual argument or a dictionary\n        of arguments. The argument groups is used to apply arguments to tokens,\n        when using the generator.parse() method. To further manage argument\n        groups, use get_arguments() and del_arguments() methods.\n\n        generator.set_arguments('small', 'max_value', 10)\n        generator.set_arguments('small', {'min_value': 5, 'max_value': 10})\n        "
        if group not in self.__config['arguments']:
            self.__config['arguments'][group] = {}
        if isinstance(argument, dict):
            self.__config['arguments'][group] = argument
        elif not isinstance(argument, str):
            raise ValueError('Arguments must be either a string or dictionary')
        else:
            self.__config['arguments'][group][argument] = value

    def get_arguments(self, group: str, argument: Optional[str]=None) -> Any:
        if False:
            i = 10
            return i + 15
        "\n        Get the value of an argument configured within a argument group, or\n        the entire group as a dictionary. Used in conjunction with the\n        set_arguments() method.\n\n        generator.get_arguments('small', 'max_value')\n        generator.get_arguments('small')\n        "
        if group in self.__config['arguments'] and argument:
            result = self.__config['arguments'][group].get(argument)
        else:
            result = self.__config['arguments'].get(group)
        return result

    def del_arguments(self, group: str, argument: Optional[str]=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        "\n        Delete an argument from an argument group or the entire argument group.\n        Used in conjunction with the set_arguments() method.\n\n        generator.del_arguments('small')\n        generator.del_arguments('small', 'max_value')\n        "
        if group in self.__config['arguments']:
            if argument:
                result = self.__config['arguments'][group].pop(argument)
            else:
                result = self.__config['arguments'].pop(group)
        else:
            result = None
        return result

    def parse(self, text: str) -> str:
        if False:
            print('Hello World!')
        "\n        Replaces tokens like '{{ tokenName }}' or '{{tokenName}}' in a string with\n        the result from the token method call. Arguments can be parsed by using an\n        argument group. For more information on the use of argument groups, please\n        refer to the set_arguments() method.\n\n        Example:\n\n        generator.set_arguments('red_rgb', {'hue': 'red', 'color_format': 'rgb'})\n        generator.set_arguments('small', 'max_value', 10)\n\n        generator.parse('{{ color:red_rgb }} - {{ pyint:small }}')\n        "
        return _re_token.sub(self.__format_token, text)

    def __format_token(self, matches):
        if False:
            return 10
        (formatter, argument_group) = list(matches.groups())
        argument_group = argument_group.lstrip(':').strip() if argument_group else ''
        if argument_group:
            try:
                arguments = self.__config['arguments'][argument_group]
            except KeyError:
                raise AttributeError(f'Unknown argument group {argument_group!r}')
            formatted = str(self.format(formatter, **arguments))
        else:
            formatted = str(self.format(formatter))
        return ''.join(formatted)