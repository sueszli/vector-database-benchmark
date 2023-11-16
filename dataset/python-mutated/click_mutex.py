"""
Module to check mutually exclusive cli parameters
"""
from typing import Any, List, Mapping, Tuple, cast
import click
from samcli.commands._utils.custom_options.replace_help_option import ReplaceHelpSummaryOption

class ClickMutex(ReplaceHelpSummaryOption):
    """
    Preprocessing checks for mutually exclusive or required parameters as supported by click api.

    required_param_lists: List[List[str]]
        List of lists with each supported combination of params
        Ex:
        With option = "a" and required_param_lists = [["b", "c"], ["c", "d"]]
        It is valid to specify --a --b --c or --a --c --d
        but not --a --b --d

    required_params_hint: str
        String to be appended after default missing required params prompt

    incompatible_params: List[str]
        List of incompatible parameters

    incompatible_params_hint: str
        String to be appended after default incompatible params prompt
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.required_param_lists: List[List[str]] = kwargs.pop('required_param_lists', [])
        self.required_params_hint: str = kwargs.pop('required_params_hint', '')
        self.incompatible_params: List[str] = kwargs.pop('incompatible_params', [])
        self.incompatible_params_hint: str = kwargs.pop('incompatible_params_hint', '')
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx: click.Context, opts: Mapping[str, Any], args: List[str]) -> Tuple[Any, List[str]]:
        if False:
            print('Hello World!')
        '\n        Checks whether any option is in self.incompatible_params\n        If one is found, prompt and throw an UsageError\n\n        Then checks any combination in self.required_param_lists is satisfied.\n        With option = "a" and required_param_lists = [["b", "c"], ["c", "d"]]\n        It is valid to specify --a --b --c, --a --c --d, or --a --b --c --d\n        but not --a --b --d\n        '
        if self.name not in opts:
            return super().handle_parse_result(ctx, opts, args)
        option_name: str = cast(str, self.name)
        for incompatible_param in self.incompatible_params:
            if incompatible_param in opts:
                msg = f'You must not provide both the {ClickMutex._to_param_name(option_name)} and {ClickMutex._to_param_name(incompatible_param)} parameters.\n'
                msg += self.incompatible_params_hint
                raise click.UsageError(msg)
        if self.required_param_lists:
            for required_params in self.required_param_lists:
                has_all_required_params = False not in [required_param in opts for required_param in required_params]
                if has_all_required_params:
                    break
            else:
                msg = f"Missing required parameters, with --{option_name.replace('_', '-')} set.\nMust provide one of the following required parameter combinations:\n"
                for required_params in self.required_param_lists:
                    msg += '\t'
                    msg += ', '.join((ClickMutex._to_param_name(param) for param in required_params))
                    msg += '\n'
                msg += self.required_params_hint
                raise click.UsageError(msg)
            self.prompt = ''
        return super().handle_parse_result(ctx, opts, args)

    @staticmethod
    def _to_param_name(param: str):
        if False:
            while True:
                i = 10
        return f"--{param.replace('_', '-')}"