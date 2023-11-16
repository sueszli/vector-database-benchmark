"""CLI implementation for `conda rename`.

Renames an existing environment by cloning it and then removing the original environment.
"""
from __future__ import annotations
import os
from argparse import ArgumentParser, Namespace, _SubParsersAction
from functools import partial
from pathlib import Path
from ..deprecations import deprecated

def configure_parser(sub_parsers: _SubParsersAction, **kwargs) -> ArgumentParser:
    if False:
        return 10
    from ..auxlib.ish import dals
    from .helpers import add_parser_prefix
    summary = 'Rename an existing environment.'
    description = dals(f'\n        {summary}\n\n        This command renames a conda environment via its name (-n/--name) or\n        its prefix (-p/--prefix).\n\n        The base environment and the currently-active environment cannot be renamed.\n        ')
    epilog = dals('\n        Examples::\n\n            conda rename -n test123 test321\n\n            conda rename --name test123 test321\n\n            conda rename -p path/to/test123 test321\n\n            conda rename --prefix path/to/test123 test321\n\n        ')
    p = sub_parsers.add_parser('rename', help=summary, description=description, epilog=epilog, **kwargs)
    add_parser_prefix(p)
    p.add_argument('destination', help='New name for the conda environment.')
    p.add_argument('--force', help='Force rename of an environment.', action='store_true', default=False)
    p.add_argument('-d', '--dry-run', help='Only display what would have been done by the current command, arguments, and other flags.', action='store_true', default=False)
    p.set_defaults(func='conda.cli.main_rename.execute')
    return p

@deprecated.argument('24.3', '24.9', 'name')
@deprecated.argument('24.3', '24.9', 'prefix')
def validate_src() -> str:
    if False:
        return 10
    '\n    Validate that we are receiving at least one value for --name or --prefix\n    and ensure that the "base" environment is not being renamed\n    '
    from ..base.context import context
    from ..exceptions import CondaEnvException
    if Path(context.target_prefix).samefile(context.root_prefix):
        raise CondaEnvException("The 'base' environment cannot be renamed")
    if Path(context.target_prefix).samefile(context.active_prefix):
        raise CondaEnvException('Cannot rename the active environment')
    return context.target_prefix

def validate_destination(dest: str, force: bool=False) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Ensure that our destination does not exist'
    from ..base.context import context, validate_prefix_name
    from ..common.path import expand
    from ..exceptions import CondaEnvException
    if os.sep in dest:
        dest = expand(dest)
    else:
        dest = validate_prefix_name(dest, ctx=context, allow_base=False)
    if not force and os.path.exists(dest):
        env_name = os.path.basename(os.path.normpath(dest))
        raise CondaEnvException(f"The environment '{env_name}' already exists. Override with --force.")
    return dest

def execute(args: Namespace, parser: ArgumentParser) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Executes the command for renaming an existing environment.'
    from ..base.constants import DRY_RUN_PREFIX
    from ..base.context import context
    from ..cli import install
    from ..gateways.disk.delete import rm_rf
    from ..gateways.disk.update import rename_context
    source = validate_src()
    destination = validate_destination(args.destination, force=args.force)

    def clone_and_remove() -> None:
        if False:
            i = 10
            return i + 15
        actions: tuple[partial, ...] = (partial(install.clone, source, destination, quiet=context.quiet, json=context.json), partial(rm_rf, source))
        for func in actions:
            if args.dry_run:
                print(f"{DRY_RUN_PREFIX} {func.func.__name__} {','.join(func.args)}")
            else:
                func()
    if args.force:
        with rename_context(destination, dry_run=args.dry_run):
            clone_and_remove()
    else:
        clone_and_remove()
    return 0