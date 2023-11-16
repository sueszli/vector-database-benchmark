"""jc - JSON Convert `/proc/version` file parser

> Note: This parser will parse `/proc/version` files that follow the
> common format used by most popular linux distributions.

Usage (cli):

    $ cat /proc/version | jc --proc

or

    $ jc /proc/version

or

    $ cat /proc/version | jc --proc-version

Usage (module):

    import jc
    result = jc.parse('proc', proc_version_file)

or

    import jc
    result = jc.parse('proc_version', proc_version_file)

Schema:

    {
      "version":                  string,
      "email":                    string,
      "gcc":                      string,
      "build":                    string,
      "flags":                    string/null,
      "date":                     string
    }

Examples:

    $ cat /proc/version | jc --proc -p
    {
      "version": "5.8.0-63-generic",
      "email": "buildd@lcy01-amd64-028",
      "gcc": "gcc (Ubuntu 10.3.0-1ubuntu1~20.10) 10.3.0, GNU ld (GNU Binutils for Ubuntu) 2.35.1",
      "build": "#71-Ubuntu",
      "flags": "SMP",
      "date": "Tue Jul 13 15:59:12 UTC 2021"
    }
"""
import re
from typing import Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/proc/version` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    tags = ['file']
    hidden = True
__version__ = info.version
version_pattern = re.compile('\n    Linux\\ version\\ (?P<version>\\S+)\\s\n    \\((?P<email>\\S+?)\\)\\s\n    \\((?P<gcc>gcc.+)\\)\\s\n    (?P<build>\\#\\d+(\\S+)?)\\s\n    (?P<flags>.*)?\n    (?P<date>(Sun|Mon|Tue|Wed|Thu|Fri|Sat).+)\n    ', re.VERBOSE)

def _process(proc_data: Dict) -> Dict:
    if False:
        i = 10
        return i + 15
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured to conform to the schema.\n    '
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> Dict:
    if False:
        return 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: Dict = {}
    if jc.utils.has_data(data):
        version_match = version_pattern.match(data)
        if version_match:
            ver_dict = version_match.groupdict()
            raw_output = {x: y.strip() or None for (x, y) in ver_dict.items()}
    return raw_output if raw else _process(raw_output)