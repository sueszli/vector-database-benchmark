"""CLI implementation for `conda-env remove`.

Removes the specified conda environment.
"""
from argparse import Namespace, RawDescriptionHelpFormatter
from conda.cli.conda_argparse import add_output_and_prompt_options, add_parser_prefix, add_parser_solver
_help = 'Remove an environment'
_description = _help + '\n\nRemoves a provided environment.  You must deactivate the existing\nenvironment before you can remove it.\n'.lstrip()
_example = '\n\nExamples:\n\n    conda env remove --name FOO\n    conda env remove -n FOO\n'

def configure_parser(sub_parsers):
    if False:
        print('Hello World!')
    p = sub_parsers.add_parser('remove', formatter_class=RawDescriptionHelpFormatter, description=_description, help=_help, epilog=_example)
    add_parser_prefix(p)
    add_parser_solver(p)
    add_output_and_prompt_options(p)
    p.set_defaults(func='.main_remove.execute')

def execute(args, parser):
    if False:
        i = 10
        return i + 15
    import conda.cli.main_remove
    args = vars(args)
    args.update({'all': True, 'channel': None, 'features': None, 'override_channels': None, 'use_local': None, 'use_cache': None, 'offline': None, 'force': True, 'pinned': None})
    args = Namespace(**args)
    from conda.base.context import context
    context.__init__(argparse_args=args)
    conda.cli.main_remove.execute(args, parser)