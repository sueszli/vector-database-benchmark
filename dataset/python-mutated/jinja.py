import ast
from typing import Any, Optional, Tuple, Type
from airbyte_cdk.sources.declarative.interpolation.filters import filters
from airbyte_cdk.sources.declarative.interpolation.interpolation import Interpolation
from airbyte_cdk.sources.declarative.interpolation.macros import macros
from airbyte_cdk.sources.declarative.types import Config
from jinja2 import meta
from jinja2.exceptions import UndefinedError
from jinja2.sandbox import Environment

class JinjaInterpolation(Interpolation):
    """
    Interpolation strategy using the Jinja2 template engine.

    If the input string is a raw string, the interpolated string will be the same.
    `eval("hello world") -> "hello world"`

    The engine will evaluate the content passed within {{}}, interpolating the keys from the config and context-specific arguments.
    `eval("hello {{ name }}", name="airbyte") -> "hello airbyte")`
    `eval("hello {{ config.name }}", config={"name": "airbyte"}) -> "hello airbyte")`

    In additional to passing additional values through the kwargs argument, macros can be called from within the string interpolation.
    For example,
    "{{ max(2, 3) }}" will return 3

    Additional information on jinja templating can be found at https://jinja.palletsprojects.com/en/3.1.x/templates/#
    """
    ALIASES = {'stream_interval': 'stream_slice', 'stream_partition': 'stream_slice'}
    RESTRICTED_EXTENSIONS = ['jinja2.ext.loopcontrols']
    RESTRICTED_BUILTIN_FUNCTIONS = ['range']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._environment = Environment()
        self._environment.filters.update(**filters)
        self._environment.globals.update(**macros)
        for extension in self.RESTRICTED_EXTENSIONS:
            self._environment.extensions.pop(extension, None)
        for builtin in self.RESTRICTED_BUILTIN_FUNCTIONS:
            self._environment.globals.pop(builtin, None)

    def eval(self, input_str: str, config: Config, default: Optional[str]=None, valid_types: Optional[Tuple[Type[Any]]]=None, **additional_parameters):
        if False:
            while True:
                i = 10
        context = {'config': config, **additional_parameters}
        for (alias, equivalent) in self.ALIASES.items():
            if alias in context:
                raise ValueError(f'Found reserved keyword {alias} in interpolation context. This is unexpected and indicative of a bug in the CDK.')
            elif equivalent in context:
                context[alias] = context[equivalent]
        try:
            if isinstance(input_str, str):
                result = self._eval(input_str, context)
                if result:
                    return self._literal_eval(result, valid_types)
            else:
                raise Exception(f'Expected a string. got {input_str}')
        except UndefinedError:
            pass
        return self._literal_eval(self._eval(default, context), valid_types)

    def _literal_eval(self, result, valid_types: Optional[Tuple[Type[Any]]]):
        if False:
            while True:
                i = 10
        try:
            evaluated = ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return result
        if not valid_types or (valid_types and isinstance(evaluated, valid_types)):
            return evaluated
        return result

    def _eval(self, s: str, context):
        if False:
            for i in range(10):
                print('nop')
        try:
            ast = self._environment.parse(s)
            undeclared = meta.find_undeclared_variables(ast)
            undeclared_not_in_context = {var for var in undeclared if var not in context}
            if undeclared_not_in_context:
                raise ValueError(f'Jinja macro has undeclared variables: {undeclared_not_in_context}. Context: {context}')
            return self._environment.from_string(s).render(context)
        except TypeError:
            return s