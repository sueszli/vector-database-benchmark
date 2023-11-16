"""A single place for constructing and exposing the main parser
"""
import os
import subprocess
import sys
from typing import List, Optional, Tuple
from pip._internal.build_env import get_runnable_pip
from pip._internal.cli import cmdoptions
from pip._internal.cli.parser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip._internal.commands import commands_dict, get_similar_commands
from pip._internal.exceptions import CommandError
from pip._internal.utils.misc import get_pip_version, get_prog
__all__ = ['create_main_parser', 'parse_command']

def create_main_parser() -> ConfigOptionParser:
    if False:
        while True:
            i = 10
    "Creates and returns the main parser for pip's CLI"
    parser = ConfigOptionParser(usage='\n%prog <command> [options]', add_help_option=False, formatter=UpdatingDefaultsHelpFormatter(), name='global', prog=get_prog())
    parser.disable_interspersed_args()
    parser.version = get_pip_version()
    gen_opts = cmdoptions.make_option_group(cmdoptions.general_group, parser)
    parser.add_option_group(gen_opts)
    parser.main = True
    description = [''] + [f'{name:27} {command_info.summary}' for (name, command_info) in commands_dict.items()]
    parser.description = '\n'.join(description)
    return parser

def identify_python_interpreter(python: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    if os.path.exists(python):
        if os.path.isdir(python):
            for exe in ('bin/python', 'Scripts/python.exe'):
                py = os.path.join(python, exe)
                if os.path.exists(py):
                    return py
        else:
            return python
    return None

def parse_command(args: List[str]) -> Tuple[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    parser = create_main_parser()
    (general_options, args_else) = parser.parse_args(args)
    if general_options.python and '_PIP_RUNNING_IN_SUBPROCESS' not in os.environ:
        interpreter = identify_python_interpreter(general_options.python)
        if interpreter is None:
            raise CommandError(f'Could not locate Python interpreter {general_options.python}')
        pip_cmd = [interpreter, get_runnable_pip()]
        pip_cmd.extend(args)
        os.environ['_PIP_RUNNING_IN_SUBPROCESS'] = '1'
        returncode = 0
        try:
            proc = subprocess.run(pip_cmd)
            returncode = proc.returncode
        except (subprocess.SubprocessError, OSError) as exc:
            raise CommandError(f'Failed to run pip under {interpreter}: {exc}')
        sys.exit(returncode)
    if general_options.version:
        sys.stdout.write(parser.version)
        sys.stdout.write(os.linesep)
        sys.exit()
    if not args_else or (args_else[0] == 'help' and len(args_else) == 1):
        parser.print_help()
        sys.exit()
    cmd_name = args_else[0]
    if cmd_name not in commands_dict:
        guess = get_similar_commands(cmd_name)
        msg = [f'unknown command "{cmd_name}"']
        if guess:
            msg.append(f'maybe you meant "{guess}"')
        raise CommandError(' - '.join(msg))
    cmd_args = args[:]
    cmd_args.remove(cmd_name)
    return (cmd_name, cmd_args)