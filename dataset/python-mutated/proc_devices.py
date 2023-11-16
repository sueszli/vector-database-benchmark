"""jc - JSON Convert `/proc/devices` file parser

Usage (cli):

    $ cat /proc/devices | jc --proc

or

    $ jc /proc/devices

or

    $ cat /proc/devices | jc --proc-devices

Usage (module):

    import jc
    result = jc.parse('proc', proc_devices_file)

or

    import jc
    result = jc.parse('proc_devices', proc_devices_file)

Schema:

Since devices can be members of multiple groups, the value for each device
is a list.

    {
      "character": {
        "<device number>": [
                                    string
        ]
      },
      "block": {
        "<device number>": [
                                    string
        ]
      }
    }

Examples:

    $ cat /proc/devices | jc --proc -p
    {
      "character": {
        "1": [
          "mem"
        ],
        "4": [
          "/dev/vc/0",
          "tty",
          "ttyS"
        ],
        "5": [
          "/dev/tty",
          "/dev/console",
          "/dev/ptmx",
          "ttyprintk"
        ],
      "block": {
        "7": [
          "loop"
        ],
        "8": [
          "sd"
        ],
        "9": [
          "md"
        ]
      }
    }
"""
from typing import Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/proc/devices` file parser'
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
        for i in range(10):
            print('nop')
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: Dict = {}
    character: Dict = {}
    block: Dict = {}
    section = ''
    if jc.utils.has_data(data):
        for line in filter(None, data.splitlines()):
            if 'Character devices:' in line:
                section = 'character'
                continue
            if 'Block devices:' in line:
                section = 'block'
                continue
            (devnum, group) = line.split()
            if section == 'character':
                if not devnum in character:
                    character[devnum] = []
                character[devnum].append(group)
                continue
            if section == 'block':
                if not devnum in block:
                    block[devnum] = []
                block[devnum].append(group)
                continue
    if character or block:
        raw_output = {'character': character, 'block': block}
    return raw_output if raw else _process(raw_output)