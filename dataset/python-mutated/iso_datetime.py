"""jc - JSON Convert ISO 8601 Datetime string parser

This parser has been renamed to datetime-iso (cli) or datetime_iso (module).

This parser will be removed in a future version, so please start using
the new parser name.
"""
from jc.parsers import datetime_iso
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.1'
    description = 'Deprecated - please use datetime-iso'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    details = 'Deprecated - please use datetime-iso'
    compatible = ['linux', 'aix', 'freebsd', 'darwin', 'win32', 'cygwin']
    tags = ['standard', 'string']
    deprecated = True
__version__ = info.version

def parse(data, raw=False, quiet=False):
    if False:
        return 10
    '\n    This parser is deprecated and calls datetime_iso. Please use datetime_iso\n    directly. This parser will be removed in the future.\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.warning_message(['iso-datetime parser is deprecated. Please use datetime-iso instead.'])
    return datetime_iso.parse(data, raw=raw, quiet=quiet)