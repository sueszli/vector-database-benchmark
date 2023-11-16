"""jc - JSON Convert `/proc/filesystems` file parser

Usage (cli):

    $ cat /proc/filesystems | jc --proc

or

    $ jc /proc/filesystems

or

    $ cat /proc/filesystems | jc --proc-filesystems

Usage (module):

    import jc
    result = jc.parse('proc', proc_filesystems_file)

or

    import jc
    result = jc.parse('proc_filesystems', proc_filesystems_file)

Schema:

    [
      {
        "filesystem":               string,
        "nodev":                    boolean
      }
    ]

Examples:

    $ cat /proc/filesystems | jc --proc -p
    [
      {
          "filesystem": "sysfs",
          "nodev": true
      },
      {
          "filesystem": "tmpfs",
          "nodev": true
      },
      {
          "filesystem": "bdev",
          "nodev": true
      },
      ...
    ]
"""
from typing import List, Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/proc/filesystems` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    tags = ['file']
    hidden = True
__version__ = info.version

def _process(proc_data: List[Dict]) -> List[Dict]:
    if False:
        print('Hello World!')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured to conform to the schema.\n    '
    return proc_data

def parse(data: str, raw: bool=False, quiet: bool=False) -> List[Dict]:
    if False:
        while True:
            i = 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: List = []
    if jc.utils.has_data(data):
        for line in filter(None, data.splitlines()):
            split_line = line.split()
            output_line = {'filesystem': split_line[-1]}
            if len(split_line) == 2:
                output_line['nodev'] = True
            else:
                output_line['nodev'] = False
            raw_output.append(output_line)
    return raw_output if raw else _process(raw_output)