from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Sequence
import click
from click import Context, Parameter, ParamType
from airflow_breeze.utils.cache import check_if_values_allowed, read_and_validate_value_from_cache, read_from_cache_file, write_to_cache_file
from airflow_breeze.utils.coertions import coerce_bool_value
from airflow_breeze.utils.console import get_console
from airflow_breeze.utils.recording import generating_command_images
from airflow_breeze.utils.shared_options import set_dry_run, set_forced_answer, set_verbose

class BetterChoice(click.Choice):
    """
    Nicer formatted choice class for click. We have a lot of parameters sometimes, and formatting
    them without spaces causes ugly artifacts as the words are broken. This one adds spaces so
    that when the long list of choices does not wrap on words.
    """
    name = 'BetterChoice'

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        super().__init__(*args)
        self.all_choices: Sequence[str] = self.choices

    def get_metavar(self, param) -> str:
        if False:
            while True:
                i = 10
        choices_str = ' | '.join(self.all_choices)
        if param.required and param.param_type_name == 'argument':
            return f'{{{choices_str}}}'
        if param.param_type_name == 'argument' and param.nargs == -1:
            return f'{choices_str}'
        return f'[{choices_str}]'

class NotVerifiedBetterChoice(BetterChoice):
    """
    This parameter allows to pass parameters that do not pass verification by choice. This is
    useful to keep autocomplete working but also to allow some extra parameters that are dynamic,
    for example allowing glob in package names for docs building.
    """
    name = 'NotVerifiedBetterChoice'

    def convert(self, value: Any, param: Parameter | None, ctx: Context | None) -> Any:
        if False:
            i = 10
            return i + 15
        normed_value = value
        normed_choices = {choice: choice for choice in self.choices}
        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(value)
            normed_choices = {ctx.token_normalize_func(normed_choice): original for (normed_choice, original) in normed_choices.items()}
        if not self.case_sensitive:
            normed_value = normed_value.casefold()
            normed_choices = {normed_choice.casefold(): original for (normed_choice, original) in normed_choices.items()}
        if normed_value in normed_choices:
            return normed_choices[normed_value]
        return normed_value

class AnswerChoice(BetterChoice):
    """
    Stores forced answer if it has been selected
    """
    name = 'AnswerChoice'

    def convert(self, value, param, ctx):
        if False:
            return 10
        set_forced_answer(value)
        return super().convert(value, param, ctx)

class VerboseOption(ParamType):
    """
    Stores and allows to retrieve verbose option
    """
    name = 'VerboseOption'

    def convert(self, value, param, ctx):
        if False:
            print('Hello World!')
        set_verbose(coerce_bool_value(value))
        return super().convert(value, param, ctx)

class DryRunOption(ParamType):
    """
    Stores and allows to retrieve dry_run option
    """
    name = 'DryRunOption'

    def convert(self, value, param, ctx):
        if False:
            print('Hello World!')
        set_dry_run(coerce_bool_value(value))
        return super().convert(value, param, ctx)

@dataclass
class CacheableDefault:
    value: Any

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.value

class CacheableChoice(click.Choice):
    """
    This class implements caching of values from the last use.
    """

    def convert(self, value, param, ctx):
        if False:
            return 10
        param_name = param.envvar if param.envvar else param.name.upper()
        if isinstance(value, CacheableDefault):
            (is_cached, new_value) = read_and_validate_value_from_cache(param_name, value.value)
            if not is_cached:
                get_console().print(f'\n[info]Default value of {param.name} parameter {new_value} used.[/]\n')
        else:
            (allowed, allowed_values) = check_if_values_allowed(param_name, value)
            if allowed:
                new_value = value
                write_to_cache_file(param_name, new_value, check_allowed_values=False)
            else:
                new_value = allowed_values[0]
                get_console().print(f'\n[warning]The value {value} is not allowed for parameter {param.name}. Setting default value to {new_value}')
                write_to_cache_file(param_name, new_value, check_allowed_values=False)
        return super().convert(new_value, param, ctx)

    def get_metavar(self, param) -> str:
        if False:
            for i in range(10):
                print('nop')
        param_name = param.envvar if param.envvar else param.name.upper()
        current_value = read_from_cache_file(param_name) if not generating_command_images() else param.default.value
        if not current_value:
            current_choices = self.choices
        else:
            current_choices = [f'>{choice}<' if choice == current_value else choice for choice in self.choices]
        choices_str = ' | '.join(current_choices)
        if param.required and param.param_type_name == 'argument':
            return f'{{{choices_str}}}'
        if param.param_type_name == 'argument' and param.nargs == -1:
            return f'{choices_str}'
        return f'[{choices_str}]'

    def __init__(self, choices, case_sensitive: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(choices=choices, case_sensitive=case_sensitive)

class MySQLBackendVersionType(CacheableChoice):

    def convert(self, value, param, ctx):
        if False:
            while True:
                i = 10
        if isinstance(value, CacheableDefault):
            param_name = param.envvar if param.envvar else param.name.upper()
            mysql_version = read_from_cache_file(param_name)
            if mysql_version == '8':
                value = '8.0'
                get_console().print(f'\n[warning]Found outdated cached value {mysql_version} for parameter {param.name}. Replaced by {value}')
                write_to_cache_file(param_name, '8.0', check_allowed_values=False)
        elif value == '8':
            value = '8.0'
            get_console().print(f'\n[warning]Provided outdated value {8} for parameter {param.name}. Will use {value} instead')
        return super().convert(value, param, ctx)

class UseAirflowVersionType(BetterChoice):
    """Extends choice with dynamic version number."""

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.all_choices = [*self.choices, '<airflow_version>']

    def convert(self, value, param, ctx):
        if False:
            while True:
                i = 10
        if re.match('^\\d*\\.\\d*\\.\\d*\\S*$', value):
            return value
        return super().convert(value, param, ctx)