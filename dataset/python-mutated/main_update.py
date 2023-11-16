"""CLI implementation for `conda-env update`.

Updates the conda environments with the specified packages.
"""
import os
import sys
import textwrap
from argparse import RawDescriptionHelpFormatter
from conda.base.context import context, determine_target_prefix
from conda.cli.conda_argparse import add_parser_json, add_parser_prefix, add_parser_solver
from conda.core.prefix_data import PrefixData
from conda.exceptions import CondaEnvException
from conda.misc import touch_nonadmin
from conda.notices import notices
from .. import specs as install_specs
from ..installers.base import InvalidInstaller, get_installer
from .common import get_filename, print_result
description = '\nUpdate the current environment based on environment file\n'
example = '\nexamples:\n    conda env update\n    conda env update -n=foo\n    conda env update -f=/path/to/environment.yml\n    conda env update --name=foo --file=environment.yml\n    conda env update vader/deathstar\n'

def configure_parser(sub_parsers):
    if False:
        while True:
            i = 10
    p = sub_parsers.add_parser('update', formatter_class=RawDescriptionHelpFormatter, description=description, help=description, epilog=example)
    add_parser_prefix(p)
    p.add_argument('-f', '--file', action='store', help='environment definition (default: environment.yml)', default='environment.yml')
    p.add_argument('--prune', action='store_true', default=False, help='remove installed packages not defined in environment.yml')
    p.add_argument('remote_definition', help='remote environment definition / IPython notebook', action='store', default=None, nargs='?')
    add_parser_json(p)
    add_parser_solver(p)
    p.set_defaults(func='.main_update.execute')

@notices
def execute(args, parser):
    if False:
        print('Hello World!')
    spec = install_specs.detect(name=args.name, filename=get_filename(args.file), directory=os.getcwd(), remote_definition=args.remote_definition)
    env = spec.environment
    if not (args.name or args.prefix):
        if not env.name:
            name = os.environ.get('CONDA_DEFAULT_ENV', False)
            if not name:
                msg = 'Unable to determine environment\n\n'
                msg += textwrap.dedent('\n                    Please re-run this command with one of the following options:\n\n                    * Provide an environment name via --name or -n\n                    * Re-run this command inside an activated conda environment.').lstrip()
                raise CondaEnvException(msg)
        args.name = env.name
    prefix = determine_target_prefix(context, args)
    installers = {}
    for installer_type in env.dependencies:
        try:
            installers[installer_type] = get_installer(installer_type)
        except InvalidInstaller:
            sys.stderr.write(textwrap.dedent('\n                Unable to install package for {0}.\n\n                Please double check and ensure you dependencies file has\n                the correct spelling.  You might also try installing the\n                conda-env-{0} package to see if provides the required\n                installer.\n                ').lstrip().format(installer_type))
            return -1
    result = {'conda': None, 'pip': None}
    for (installer_type, specs) in env.dependencies.items():
        installer = installers[installer_type]
        result[installer_type] = installer.install(prefix, specs, args, env)
    if env.variables:
        pd = PrefixData(prefix)
        pd.set_environment_env_vars(env.variables)
    touch_nonadmin(prefix)
    print_result(args, prefix, result)