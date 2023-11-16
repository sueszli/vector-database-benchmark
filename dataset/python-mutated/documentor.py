import inspect
import warnings
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from .generator import Generator
from .providers import BaseProvider
from .proxy import Faker

class FakerEnum(Enum):
    """Required for faker.providers.enum"""
    A = auto
    B = auto

class Documentor:

    def __init__(self, generator: Union[Generator, Faker]) -> None:
        if False:
            return 10
        '\n        :param generator: a localized Generator with providers filled,\n                          for which to write the documentation\n        :type generator: faker.Generator()\n        '
        self.generator = generator
        self.max_name_len: int = 0
        self.already_generated: List[str] = []

    def get_formatters(self, locale: Optional[str]=None, excludes: Optional[List[str]]=None, **kwargs: Any) -> List[Tuple[BaseProvider, Dict[str, str]]]:
        if False:
            while True:
                i = 10
        self.max_name_len = 0
        self.already_generated = [] if excludes is None else excludes[:]
        formatters = []
        providers: List[BaseProvider] = self.generator.get_providers()
        for provider in providers[::-1]:
            if locale and provider.__lang__ and (provider.__lang__ != locale):
                continue
            formatters.append((provider, self.get_provider_formatters(provider, **kwargs)))
        return formatters

    def get_provider_formatters(self, provider: BaseProvider, prefix: str='fake.', with_args: bool=True, with_defaults: bool=True) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        formatters = {}
        for (name, method) in inspect.getmembers(provider, inspect.ismethod):
            if name.startswith('_') or name in self.already_generated:
                continue
            arguments = []
            faker_args: List[Union[str, Type[Enum]]] = []
            faker_kwargs = {}
            if name == 'binary':
                faker_kwargs['length'] = 1024
            elif name in ['zip', 'tar']:
                faker_kwargs.update({'uncompressed_size': 1024, 'min_file_size': 512})
            if name == 'enum':
                faker_args = [FakerEnum]
            if with_args:
                argspec = inspect.getfullargspec(method)
                lst = [x for x in argspec.args if x not in ['self', 'cls']]
                for (i, arg) in enumerate(lst):
                    if argspec.defaults and with_defaults:
                        try:
                            default = argspec.defaults[i]
                            if isinstance(default, str):
                                default = repr(default)
                            else:
                                default = f'{default}'
                            arg = f'{arg}={default}'
                        except IndexError:
                            pass
                    arguments.append(arg)
                    if with_args == 'first':
                        break
                if with_args != 'first':
                    if argspec.varargs:
                        arguments.append('*' + argspec.varargs)
                    if argspec.varkw:
                        arguments.append('**' + argspec.varkw)
            signature = f"{prefix}{name}({', '.join(arguments)})"
            try:
                example = self.generator.format(name, *faker_args, **faker_kwargs)
            except (AttributeError, ValueError) as e:
                warnings.warn(str(e))
                continue
            formatters[signature] = example
            self.max_name_len = max(self.max_name_len, *(len(part) for part in signature.split()))
            self.already_generated.append(name)
        return formatters

    @staticmethod
    def get_provider_name(provider_class: BaseProvider) -> str:
        if False:
            while True:
                i = 10
        return provider_class.__provider__