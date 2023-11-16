"""jc - JSON Convert `/proc/<pid>/io` file parser

Usage (cli):

    $ cat /proc/1/io | jc --proc

or

    $ jc /proc/1/io

or

    $ cat /proc/1/io | jc --proc-pid-io

Usage (module):

    import jc
    result = jc.parse('proc', proc_pid_io_file)

or

    import jc
    result = jc.parse('proc_pid_io', proc_pid_io_file)

Schema:

All values are integers.

    {
      <keyName>             integer
    }

Examples:

    $ cat /proc/1/io | jc --proc -p
    {
      "rchar": 4699288382,
      "wchar": 2931802997,
      "syscr": 661897,
      "syscw": 890910,
      "read_bytes": 168468480,
      "write_bytes": 27357184,
      "cancelled_write_bytes": 16883712
    }
"""
from typing import Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/proc/<pid>/io` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    tags = ['file']
    hidden = True
__version__ = info.version

def _process(proc_data: Dict) -> Dict:
    if False:
        print('Hello World!')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured to conform to the schema.\n    '
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> Dict:
    if False:
        print('Hello World!')
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: Dict = {}
    if jc.utils.has_data(data):
        for line in filter(None, data.splitlines()):
            (key, val) = line.split(':', maxsplit=1)
            raw_output[key] = int(val)
    return raw_output if raw else _process(raw_output)