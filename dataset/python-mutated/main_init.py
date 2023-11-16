"""CLI implementation for `conda init`.

Prepares the user's profile for running conda, and sets up the conda shell interface.
"""
from argparse import SUPPRESS, ArgumentParser, Namespace, _SubParsersAction
from logging import getLogger
log = getLogger(__name__)

def configure_parser(sub_parsers: _SubParsersAction, **kwargs) -> ArgumentParser:
    if False:
        for i in range(10):
            print('nop')
    from ..auxlib.ish import dals
    from ..base.constants import COMPATIBLE_SHELLS
    from ..common.compat import on_win
    from ..common.constants import NULL
    from .helpers import add_parser_json
    summary = 'Initialize conda for shell interaction.'
    description = summary
    epilog = dals("\n        Key parts of conda's functionality require that it interact directly with the shell\n        within which conda is being invoked. The `conda activate` and `conda deactivate` commands\n        specifically are shell-level commands. That is, they affect the state (e.g. environment\n        variables) of the shell context being interacted with. Other core commands, like\n        `conda create` and `conda install`, also necessarily interact with the shell environment.\n        They're therefore implemented in ways specific to each shell. Each shell must be configured\n        to make use of them.\n\n        This command makes changes to your system that are specific and customized for each shell.\n        To see the specific files and locations on your system that will be affected before, use\n        the '--dry-run' flag.  To see the exact changes that are being or will be made to each\n        location, use the '--verbose' flag.\n\n        IMPORTANT: After running `conda init`, most shells will need to be closed and restarted for\n        changes to take effect.\n\n        ")
    p = sub_parsers.add_parser('init', help=summary, description=description, epilog=epilog, **kwargs)
    p.add_argument('--dev', action='store_true', help=SUPPRESS, default=NULL)
    p.add_argument('--all', action='store_true', help='Initialize all currently available shells.', default=NULL)
    setup_type_group = p.add_argument_group('setup type')
    setup_type_group.add_argument('--install', action='store_true', help=SUPPRESS, default=NULL)
    setup_type_group.add_argument('--user', action='store_true', dest='user', help='Initialize conda for the current user (default).', default=True)
    setup_type_group.add_argument('--no-user', action='store_false', dest='user', help="Don't initialize conda for the current user.")
    setup_type_group.add_argument('--system', action='store_true', help='Initialize conda for all users on the system.', default=NULL)
    setup_type_group.add_argument('--reverse', action='store_true', help='Undo effects of last conda init.', default=NULL)
    p.add_argument('shells', nargs='*', choices=COMPATIBLE_SHELLS, metavar='SHELLS', help=f"One or more shells to be initialized. If not given, the default value is 'bash' on unix and 'cmd.exe' & 'powershell' on Windows. Use the '--all' flag to initialize all shells. Available shells: {sorted(COMPATIBLE_SHELLS)}", default=['cmd.exe', 'powershell'] if on_win else ['bash'])
    if on_win:
        p.add_argument('--anaconda-prompt', action='store_true', help="Add an 'Anaconda Prompt' icon to your desktop.", default=NULL)
    add_parser_json(p)
    p.add_argument('-d', '--dry-run', action='store_true', help='Only display what would have been done.')
    p.set_defaults(func='conda.cli.main_init.execute')
    return p

def execute(args: Namespace, parser: ArgumentParser) -> int:
    if False:
        i = 10
        return i + 15
    from ..base.constants import COMPATIBLE_SHELLS
    from ..base.context import context
    from ..common.compat import on_win
    from ..core.initialize import initialize, initialize_dev, install
    from ..exceptions import ArgumentError
    if args.install:
        return install(context.conda_prefix)
    selected_shells: tuple[str, ...]
    if args.all:
        selected_shells = COMPATIBLE_SHELLS
    else:
        selected_shells = tuple(args.shells)
    if args.dev:
        if len(selected_shells) != 1:
            raise ArgumentError('--dev can only handle one shell at a time right now')
        return initialize_dev(selected_shells[0])
    else:
        for_user = args.user and (not args.system)
        anaconda_prompt = on_win and args.anaconda_prompt
        return initialize(context.conda_prefix, selected_shells, for_user, args.system, anaconda_prompt, args.reverse)