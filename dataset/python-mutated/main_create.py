"""CLI implementation for `conda-env create`.

Creates new conda environments with the specified packages.
"""
import json
import os
import sys
import textwrap
from argparse import RawDescriptionHelpFormatter, _StoreTrueAction
from conda.base.context import context, determine_target_prefix
from conda.cli import install as cli_install
from conda.cli.conda_argparse import add_output_and_prompt_options, add_parser_default_packages, add_parser_networking, add_parser_platform, add_parser_prefix, add_parser_solver
from conda.core.prefix_data import PrefixData
from conda.deprecations import deprecated
from conda.gateways.disk.delete import rm_rf
from conda.misc import touch_nonadmin
from conda.notices import notices
from .. import specs
from ..installers.base import InvalidInstaller, get_installer
from .common import get_filename, print_result
description = "\nCreate an environment based on an environment definition file.\n\nIf using an environment.yml file (the default), you can name the\nenvironment in the first line of the file with 'name: envname' or\nyou can specify the environment name in the CLI command using the\n-n/--name argument. The name specified in the CLI will override\nthe name specified in the environment.yml file.\n\nUnless you are in the directory containing the environment definition\nfile, use -f to specify the file path of the environment definition\nfile you want to use.\n"
example = '\nexamples:\n    conda env create\n    conda env create -n envname\n    conda env create folder/envname\n    conda env create -f /path/to/environment.yml\n    conda env create -f /path/to/requirements.txt -n envname\n    conda env create -f /path/to/requirements.txt -p /home/user/envname\n'

def configure_parser(sub_parsers):
    if False:
        for i in range(10):
            print('nop')
    p = sub_parsers.add_parser('create', formatter_class=RawDescriptionHelpFormatter, description=description, help=description, epilog=example)
    p.add_argument('-f', '--file', action='store', help='Environment definition file (default: environment.yml)', default='environment.yml')
    add_parser_prefix(p)
    add_parser_networking(p)
    p.add_argument('remote_definition', help='Remote environment definition / IPython notebook', action='store', default=None, nargs='?')
    p.add_argument('--force', dest='yes', action=deprecated.action('23.9', '24.3', _StoreTrueAction, addendum='Use `--yes` instead.'), default=False)
    add_parser_default_packages(p)
    add_parser_platform(p)
    add_output_and_prompt_options(p)
    add_parser_solver(p)
    p.set_defaults(func='.main_create.execute')

@notices
def execute(args, parser):
    if False:
        return 10
    spec = specs.detect(name=args.name, filename=get_filename(args.file), directory=os.getcwd(), remote_definition=args.remote_definition)
    env = spec.environment
    if args.prefix is None and args.name is None:
        args.name = env.name
    prefix = determine_target_prefix(context, args)
    if args.yes and prefix != context.root_prefix and os.path.exists(prefix):
        rm_rf(prefix)
    cli_install.check_prefix(prefix, json=args.json)
    result = {'conda': None, 'pip': None}
    args_packages = context.create_default_packages if not args.no_default_packages else []
    if args.dry_run:
        installer_type = 'conda'
        installer = get_installer(installer_type)
        pkg_specs = env.dependencies.get(installer_type, [])
        pkg_specs.extend(args_packages)
        solved_env = installer.dry_run(pkg_specs, args, env)
        if args.json:
            print(json.dumps(solved_env.to_dict(), indent=2))
        else:
            print(solved_env.to_yaml(), end='')
    else:
        if args_packages:
            installer_type = 'conda'
            installer = get_installer(installer_type)
            result[installer_type] = installer.install(prefix, args_packages, args, env)
        if len(env.dependencies.items()) == 0:
            installer_type = 'conda'
            pkg_specs = []
            installer = get_installer(installer_type)
            result[installer_type] = installer.install(prefix, pkg_specs, args, env)
        else:
            for (installer_type, pkg_specs) in env.dependencies.items():
                try:
                    installer = get_installer(installer_type)
                    result[installer_type] = installer.install(prefix, pkg_specs, args, env)
                except InvalidInstaller:
                    sys.stderr.write(textwrap.dedent('\n                        Unable to install package for {0}.\n\n                        Please double check and ensure your dependencies file has\n                        the correct spelling.  You might also try installing the\n                        conda-env-{0} package to see if provides the required\n                        installer.\n                        ').lstrip().format(installer_type))
                    return -1
        if env.variables:
            pd = PrefixData(prefix)
            pd.set_environment_env_vars(env.variables)
        touch_nonadmin(prefix)
        print_result(args, prefix, result)