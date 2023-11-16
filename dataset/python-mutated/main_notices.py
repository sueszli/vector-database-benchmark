"""CLI implementation for `conda notices`.

Manually retrieves channel notifications, caches them and displays them.
"""
from argparse import ArgumentParser, Namespace, _SubParsersAction

def configure_parser(sub_parsers: _SubParsersAction, **kwargs) -> ArgumentParser:
    if False:
        while True:
            i = 10
    from ..auxlib.ish import dals
    from .helpers import add_parser_channels
    summary = 'Retrieve latest channel notifications.'
    description = dals(f'\n        {summary}\n\n        Conda channel maintainers have the option of setting messages that\n        users will see intermittently. Some of these notices are informational\n        while others are messages concerning the stability of the channel.\n\n        ')
    epilog = dals('\n        Examples::\n\n        conda notices\n\n        conda notices -c defaults\n\n        ')
    p = sub_parsers.add_parser('notices', help=summary, description=description, epilog=epilog, **kwargs)
    add_parser_channels(p)
    p.set_defaults(func='conda.cli.main_notices.execute')
    return p

def execute(args: Namespace, parser: ArgumentParser) -> int:
    if False:
        i = 10
        return i + 15
    'Command that retrieves channel notifications, caches them and displays them.'
    from ..exceptions import CondaError
    from ..notices import core as notices
    try:
        channel_notice_set = notices.retrieve_notices()
    except OSError as exc:
        raise CondaError(f'Unable to retrieve notices: {exc}')
    notices.display_notices(channel_notice_set)
    return 0