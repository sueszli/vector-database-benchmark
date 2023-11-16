"""jc - JSON Convert `/proc/ioports` file parser

Usage (cli):

    $ cat /proc/ioports | jc --proc

or

    $ jc /proc/ioports

or

    $ cat /proc/ioports | jc --proc-ioports

Usage (module):

    import jc
    result = jc.parse('proc', proc_ioports_file)

or

    import jc
    result = jc.parse('proc_ioports', proc_ioports_file)

Schema:

    [
      {
        "start":                   string,
        "end":                     string,
        "device":                  string
      }
    ]

Examples:

    $ cat /proc/ioports | jc --proc -p
    [
      {
        "start": "0000",
        "end": "0cf7",
        "device": "PCI Bus 0000:00"
      },
      {
        "start": "0000",
        "end": "001f",
        "device": "dma1"
      },
      {
        "start": "0020",
        "end": "0021",
        "device": "PNP0001:00"
      },
      ...
    ]
"""
from typing import List, Dict
import jc.utils
import jc.parsers.proc_iomem as proc_iomem

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/proc/ioports` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    tags = ['file']
    hidden = True
__version__ = info.version

def _process(proc_data: List[Dict]) -> List[Dict]:
    if False:
        while True:
            i = 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured to conform to the schema.\n    '
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> List[Dict]:
    if False:
        i = 10
        return i + 15
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: List = proc_iomem.parse(data, quiet=True, raw=raw)
    return raw_output if raw else _process(raw_output)