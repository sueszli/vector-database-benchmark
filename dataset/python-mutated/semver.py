"""jc - JSON Convert Semantic Version string parser

This parser conforms to the specification at https://semver.org/

See Also: `ver` parser.

Usage (cli):

    $ echo 1.2.3-rc.1+44837 | jc --semver

Usage (module):

    import jc
    result = jc.parse('semver', semver_string)

Schema:

Strings that do not strictly conform to the specification will return an
empty object.

    {
      "major":                  integer,
      "minor":                  integer,
      "patch":                  integer,
      "prerelease":             string/null,
      "build":                  string/null
    }

Examples:

    $ echo 1.2.3-rc.1+44837 | jc --semver -p
    {
      "major": 1,
      "minor": 2,
      "patch": 3,
      "prerelease": "rc.1",
      "build": "44837"
    }

    $ echo 1.2.3-rc.1+44837 | jc --semver -p -r
    {
      "major": "1",
      "minor": "2",
      "patch": "3",
      "prerelease": "rc.1",
      "build": "44837"
    }
"""
import re
from typing import Set, Dict
from jc.jc_types import JSONDictType
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = 'Semantic Version string parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['standard', 'string']
__version__ = info.version

def _process(proc_data: JSONDictType) -> JSONDictType:
    if False:
        return 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured to conform to the schema.\n    '
    int_list: Set[str] = {'major', 'minor', 'patch'}
    for item in int_list:
        if item in proc_data:
            proc_data[item] = int(proc_data[item])
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> JSONDictType:
    if False:
        print('Hello World!')
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: Dict = {}
    semver_pattern = re.compile('\n        ^(?P<major>0|[1-9]\\d*)\\.\n        (?P<minor>0|[1-9]\\d*)\\.\n        (?P<patch>0|[1-9]\\d*)\n        (?:-(?P<prerelease>(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\\.(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?\n        (?:\\+(?P<build>[0-9a-zA-Z-]+(?:\\.[0-9a-zA-Z-]+)*))?$\n    ', re.VERBOSE)
    if jc.utils.has_data(data):
        semver_match = re.match(semver_pattern, data)
        if semver_match:
            raw_output = semver_match.groupdict()
    return raw_output if raw else _process(raw_output)