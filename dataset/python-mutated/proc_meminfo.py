"""jc - JSON Convert `/proc/meminfo` file parser

Usage (cli):

    $ cat /proc/meminfo | jc --proc

or

    $ jc /proc/meminfo

or

    $ cat /proc/meminfo | jc --proc-meminfo

Usage (module):

    import jc
    result = jc.parse('proc', proc_meminfo_file)

or

    import jc
    result = jc.parse('proc_meminfo', proc_meminfo_file)

Schema:

All values are integers.

    {
      <keyName>             integer
    }

Examples:

    $ cat /proc/meminfo | jc --proc -p
    {
      "MemTotal": 3997272,
      "MemFree": 2760316,
      "MemAvailable": 3386876,
      "Buffers": 40452,
      "Cached": 684856,
      "SwapCached": 0,
      "Active": 475816,
      "Inactive": 322064,
      "Active(anon)": 70216,
      "Inactive(anon)": 148,
      "Active(file)": 405600,
      "Inactive(file)": 321916,
      "Unevictable": 19476,
      "Mlocked": 19476,
      "SwapTotal": 3996668,
      "SwapFree": 3996668,
      "Dirty": 152,
      "Writeback": 0,
      "AnonPages": 92064,
      "Mapped": 79464,
      "Shmem": 1568,
      "KReclaimable": 188216,
      "Slab": 288096,
      "SReclaimable": 188216,
      "SUnreclaim": 99880,
      "KernelStack": 5872,
      "PageTables": 1812,
      "NFS_Unstable": 0,
      "Bounce": 0,
      "WritebackTmp": 0,
      "CommitLimit": 5995304,
      "Committed_AS": 445240,
      "VmallocTotal": 34359738367,
      "VmallocUsed": 21932,
      "VmallocChunk": 0,
      "Percpu": 107520,
      "HardwareCorrupted": 0,
      "AnonHugePages": 0,
      "ShmemHugePages": 0,
      "ShmemPmdMapped": 0,
      "FileHugePages": 0,
      "FilePmdMapped": 0,
      "HugePages_Total": 0,
      "HugePages_Free": 0,
      "HugePages_Rsvd": 0,
      "HugePages_Surp": 0,
      "Hugepagesize": 2048,
      "Hugetlb": 0,
      "DirectMap4k": 192320,
      "DirectMap2M": 4001792,
      "DirectMap1G": 2097152
    }
"""
from typing import Dict
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`/proc/meminfo` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    tags = ['file']
    hidden = True
__version__ = info.version

def _process(proc_data: Dict) -> Dict:
    if False:
        while True:
            i = 10
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
            (key, val, *_) = line.replace(':', '').split()
            raw_output[key] = int(val)
    return raw_output if raw else _process(raw_output)