"""Conda command line interface parsers."""
from __future__ import annotations
import argparse
import os
import sys
from argparse import SUPPRESS, RawDescriptionHelpFormatter
from argparse import ArgumentParser as ArgumentParserBase
from importlib import import_module
from logging import getLogger
from subprocess import Popen
from .. import __version__
from ..auxlib.compat import isiterable
from ..auxlib.ish import dals
from ..base.context import context, sys_rc_path, user_rc_path
from ..common.compat import on_win
from ..common.constants import NULL
from ..deprecations import deprecated
from .actions import ExtendConstAction, NullCountAction
from .find_commands import find_commands, find_executable
from .helpers import add_output_and_prompt_options, add_parser_channels, add_parser_create_install_update, add_parser_default_packages, add_parser_help, add_parser_json, add_parser_known, add_parser_networking, add_parser_package_install_options, add_parser_platform, add_parser_prefix, add_parser_prune, add_parser_pscheck, add_parser_show_channel_urls, add_parser_solver, add_parser_solver_mode, add_parser_update_modifiers, add_parser_verbose
from .main_clean import configure_parser as configure_parser_clean
from .main_compare import configure_parser as configure_parser_compare
from .main_config import configure_parser as configure_parser_config
from .main_create import configure_parser as configure_parser_create
from .main_info import configure_parser as configure_parser_info
from .main_init import configure_parser as configure_parser_init
from .main_install import configure_parser as configure_parser_install
from .main_list import configure_parser as configure_parser_list
from .main_mock_activate import configure_parser as configure_parser_mock_activate
from .main_mock_deactivate import configure_parser as configure_parser_mock_deactivate
from .main_notices import configure_parser as configure_parser_notices
from .main_package import configure_parser as configure_parser_package
from .main_remove import configure_parser as configure_parser_remove
from .main_rename import configure_parser as configure_parser_rename
from .main_run import configure_parser as configure_parser_run
from .main_search import configure_parser as configure_parser_search
from .main_update import configure_parser as configure_parser_update
log = getLogger(__name__)
escaped_user_rc_path = user_rc_path.replace('%', '%%')
escaped_sys_rc_path = sys_rc_path.replace('%', '%%')
BUILTIN_COMMANDS = {'activate', 'clean', 'compare', 'config', 'create', 'deactivate', 'info', 'init', 'install', 'list', 'package', 'remove', 'rename', 'run', 'search', 'update', 'upgrade', 'notices'}

def generate_pre_parser(**kwargs) -> ArgumentParser:
    if False:
        print('Hello World!')
    pre_parser = ArgumentParser(description='conda is a tool for managing and deploying applications, environments and packages.', **kwargs)
    add_parser_verbose(pre_parser)
    pre_parser.add_argument('--json', action='store_true', default=NULL, help=SUPPRESS)
    pre_parser.add_argument('--no-plugins', action='store_true', default=NULL, help='Disable all plugins that are not built into conda.')
    return pre_parser

def generate_parser(**kwargs) -> ArgumentParser:
    if False:
        print('Hello World!')
    parser = generate_pre_parser(**kwargs)
    parser.add_argument('-V', '--version', action='version', version='conda %s' % __version__, help='Show the conda version number and exit.')
    sub_parsers = parser.add_subparsers(metavar='COMMAND', title='commands', description='The following built-in and plugins subcommands are available.', dest='cmd', action=_GreedySubParsersAction, required=True)
    configure_parser_mock_activate(sub_parsers)
    configure_parser_mock_deactivate(sub_parsers)
    configure_parser_clean(sub_parsers)
    configure_parser_compare(sub_parsers)
    configure_parser_config(sub_parsers)
    configure_parser_create(sub_parsers)
    configure_parser_info(sub_parsers)
    configure_parser_init(sub_parsers)
    configure_parser_install(sub_parsers)
    configure_parser_list(sub_parsers)
    configure_parser_notices(sub_parsers)
    configure_parser_package(sub_parsers)
    configure_parser_remove(sub_parsers, aliases=['uninstall'])
    configure_parser_rename(sub_parsers)
    configure_parser_run(sub_parsers)
    configure_parser_search(sub_parsers)
    configure_parser_update(sub_parsers, aliases=['upgrade'])
    configure_parser_plugins(sub_parsers)
    return parser

def do_call(args: argparse.Namespace, parser: ArgumentParser):
    if False:
        for i in range(10):
            print('nop')
    '\n    Serves as the primary entry point for commands referred to in this file and for\n    all registered plugin subcommands.\n    '
    if (plugin_subcommand := getattr(args, '_plugin_subcommand', None)):
        context.plugin_manager.invoke_pre_commands(plugin_subcommand.name)
        result = plugin_subcommand.action(getattr(args, '_args', args))
        context.plugin_manager.invoke_post_commands(plugin_subcommand.name)
    elif (name := getattr(args, '_executable', None)):
        deprecated.topic('23.3', '24.3', topic='Loading conda subcommands via executables', addendum='Use the plugin system instead.')
        executable = find_executable(f'conda-{name}')
        if not executable:
            from ..exceptions import CommandNotFoundError
            raise CommandNotFoundError(name)
        return _exec([executable, *args._args], os.environ)
    else:
        (module_name, func_name) = args.func.rsplit('.', 1)
        module = import_module(module_name)
        command = module_name.split('.')[-1].replace('main_', '')
        context.plugin_manager.invoke_pre_commands(command)
        result = getattr(module, func_name)(args, parser)
        context.plugin_manager.invoke_post_commands(command)
    return result

def find_builtin_commands(parser):
    if False:
        i = 10
        return i + 15
    return tuple(parser._subparsers._group_actions[0].choices.keys())

class ArgumentParser(ArgumentParserBase):

    def __init__(self, *args, add_help=True, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.setdefault('formatter_class', RawDescriptionHelpFormatter)
        super().__init__(*args, add_help=False, **kwargs)
        if add_help:
            add_parser_help(self)

    def _check_value(self, action, value):
        if False:
            print('Hello World!')
        if action.choices is not None and isiterable(value):
            for element in value:
                super()._check_value(action, element)
        else:
            super()._check_value(action, value)

    def parse_args(self, *args, override_args=None, **kwargs):
        if False:
            return 10
        parsed_args = super().parse_args(*args, **kwargs)
        for (name, value) in (override_args or {}).items():
            if value is not NULL and getattr(parsed_args, name, NULL) is NULL:
                setattr(parsed_args, name, value)
        return parsed_args

class _GreedySubParsersAction(argparse._SubParsersAction):
    """A custom subparser action to conditionally act as a greedy consumer.

    This is a workaround since argparse.REMAINDER does not work as expected,
    see https://github.com/python/cpython/issues/61252.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if False:
            i = 10
            return i + 15
        super().__call__(parser, namespace, values, option_string)
        parser = self._name_parser_map[values[0]]
        if getattr(parser, 'greedy', False):
            try:
                unknown = getattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR)
                delattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR)
            except AttributeError:
                unknown = ()
            namespace._args = tuple(unknown)

    def _get_subactions(self):
        if False:
            for i in range(10):
                print('nop')
        'Sort actions for subcommands to appear alphabetically in help blurb.'
        return sorted(self._choices_actions, key=lambda action: action.dest)

def _exec(executable_args, env_vars):
    if False:
        i = 10
        return i + 15
    return (_exec_win if on_win else _exec_unix)(executable_args, env_vars)

def _exec_win(executable_args, env_vars):
    if False:
        for i in range(10):
            print('nop')
    p = Popen(executable_args, env=env_vars)
    try:
        p.communicate()
    except KeyboardInterrupt:
        p.wait()
    finally:
        sys.exit(p.returncode)

def _exec_unix(executable_args, env_vars):
    if False:
        i = 10
        return i + 15
    os.execvpe(executable_args[0], executable_args, env_vars)

def configure_parser_plugins(sub_parsers) -> None:
    if False:
        return 10
    "\n    For each of the provided plugin-based subcommands, we'll create\n    a new subparser for an improved help printout and calling the\n    :meth:`~conda.plugins.types.CondaSubcommand.configure_parser`\n    with the newly created subcommand specific argument parser.\n    "
    plugin_subcommands = context.plugin_manager.get_subcommands()
    for (name, plugin_subcommand) in plugin_subcommands.items():
        if name in BUILTIN_COMMANDS:
            log.error(dals(f"\n                    The plugin '{name}' is trying to override the built-in command\n                    with the same name, which is not allowed.\n\n                    Please uninstall the plugin to stop seeing this error message.\n                    "))
            continue
        parser = sub_parsers.add_parser(name, description=plugin_subcommand.summary, help=plugin_subcommand.summary, add_help=False)
        if plugin_subcommand.configure_parser:
            plugin_subcommand.configure_parser(parser)
            try:
                add_parser_help(parser)
            except argparse.ArgumentError:
                pass
        else:
            parser.greedy = True
        parser.set_defaults(_plugin_subcommand=plugin_subcommand)
    legacy = ['env'] if context.no_plugins else set(find_commands()).difference(plugin_subcommands)
    for name in legacy:
        if name in BUILTIN_COMMANDS:
            log.error(dals(f"\n                    The (legacy) plugin '{name}' is trying to override the built-in command\n                    with the same name, which is not allowed.\n\n                    Please uninstall the plugin to stop seeing this error message.\n                    "))
            continue
        parser = sub_parsers.add_parser(name, description=f'See `conda {name} --help`.', help=f'See `conda {name} --help`.', add_help=False)
        parser.greedy = True
        parser.set_defaults(_executable=name)