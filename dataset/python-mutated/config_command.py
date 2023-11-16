"""Config sub-commands."""
from __future__ import annotations
from io import StringIO
import pygments
from pygments.lexers.configs import IniLexer
from airflow.configuration import conf
from airflow.utils.cli import should_use_colors
from airflow.utils.code_utils import get_terminal_formatter
from airflow.utils.providers_configuration_loader import providers_configuration_loaded

@providers_configuration_loaded
def show_config(args):
    if False:
        print('Hello World!')
    'Show current application configuration.'
    with StringIO() as output:
        conf.write(output, section=args.section, include_examples=args.include_examples or args.defaults, include_descriptions=args.include_descriptions or args.defaults, include_sources=args.include_sources and (not args.defaults), include_env_vars=args.include_env_vars or args.defaults, include_providers=not args.exclude_providers, comment_out_everything=args.comment_out_everything or args.defaults, only_defaults=args.defaults)
        code = output.getvalue()
    if should_use_colors(args):
        code = pygments.highlight(code=code, formatter=get_terminal_formatter(), lexer=IniLexer())
    print(code)

@providers_configuration_loaded
def get_value(args):
    if False:
        return 10
    'Get one value from configuration.'
    if not conf.has_option(args.section, args.option):
        raise SystemExit(f'The option [{args.section}/{args.option}] is not found in config.')
    value = conf.get(args.section, args.option)
    print(value)