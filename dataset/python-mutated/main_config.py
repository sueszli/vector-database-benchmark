"""CLI implementation for `conda-env config`.

Allows for programmatically interacting with conda-env's configuration files (e.g., `~/.condarc`).
"""
from argparse import RawDescriptionHelpFormatter
from .main_vars import configure_parser as configure_vars_parser
config_description = '\nConfigure a conda environment\n'
config_example = '\nexamples:\n    conda env config vars list\n    conda env config --append channels conda-forge\n'

def configure_parser(sub_parsers):
    if False:
        print('Hello World!')
    config_parser = sub_parsers.add_parser('config', formatter_class=RawDescriptionHelpFormatter, description=config_description, help=config_description, epilog=config_example)
    config_parser.set_defaults(func='.main_config.execute')
    config_subparser = config_parser.add_subparsers()
    configure_vars_parser(config_subparser)

def execute(args, parser):
    if False:
        while True:
            i = 10
    parser.parse_args(['config', '--help'])