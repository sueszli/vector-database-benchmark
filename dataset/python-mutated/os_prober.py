"""jc - JSON Convert `os-prober` command output parser

Usage (cli):

    $ os-prober | jc --os-prober

or

    $ jc os-prober

Usage (module):

    import jc
    result = jc.parse('os_prober', os_prober_command_output)

Schema:

    {
      "partition":              string,
      "efi_bootmgr":            string,  # [0]
      "name":                   string,
      "short_name":             string,
      "type":                   string
    }

    [0] only exists if an EFI boot manager is detected

Examples:

    $ os-prober | jc --os-prober -p
    {
      "partition": "/dev/sda1",
      "name": "Windows 10",
      "short_name": "Windows",
      "type": "chain"
    }
"""
from typing import Dict
from jc.jc_types import JSONDictType
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.1'
    description = '`os-prober` command parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    magic_commands = ['os-prober']
    tags = ['command']
__version__ = info.version

def _process(proc_data: JSONDictType) -> JSONDictType:
    if False:
        print('Hello World!')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured to conform to the schema.\n    '
    if 'partition' in proc_data and '@' in proc_data['partition']:
        (new_part, efi_bootmgr) = proc_data['partition'].split('@', maxsplit=1)
        proc_data['partition'] = new_part
        proc_data['efi_bootmgr'] = efi_bootmgr
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> JSONDictType:
    if False:
        while True:
            i = 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        Dictionary. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: Dict = {}
    if jc.utils.has_data(data):
        (partition, name, short_name, type_) = data.split(':')
        raw_output = {'partition': partition.strip(), 'name': name.strip(), 'short_name': short_name.strip(), 'type': type_.strip()}
    return raw_output if raw else _process(raw_output)