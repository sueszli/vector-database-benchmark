"""jc - JSON Convert `update-alternatives --get-selections` command output parser

Usage (cli):

    $ update-alternatives --get-selections | jc --update-alt-gs

or

    $ jc update-alternatives --get-selections

Usage (module):

    import jc
    result = jc.parse('update-alt-gs',
                      update_alternatives_get_selections_command_output)

Schema:

    [
      {
        "name":     string,
        "status":   string,
        "current":  string
      }
    ]

Examples:

    $ update-alternatives --get-selections | jc --update-alt-gs -p
    [
      {
        "name": "arptables",
        "status": "auto",
        "current": "/usr/sbin/arptables-nft"
      },
      {
        "name": "awk",
        "status": "auto",
        "current": "/usr/bin/gawk"
      }
    ]
"""
from typing import List, Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`update-alternatives --get-selections` command parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    magic_commands = ['update-alternatives --get-selections']
    tags = ['command']
__version__ = info.version

def _process(proc_data: List[Dict]) -> List[Dict]:
    if False:
        i = 10
        return i + 15
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured to conform to the schema.\n    '
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> List[Dict]:
    if False:
        i = 10
        return i + 15
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: List = []
    output_line = {}
    if jc.utils.has_data(data):
        for line in filter(None, data.splitlines()):
            line_list = line.split(maxsplit=2)
            output_line = {'name': line_list[0], 'status': line_list[1], 'current': line_list[2]}
            raw_output.append(output_line)
    return raw_output if raw else _process(raw_output)