"""jc - JSON Convert `systemctl` command output parser

Usage (cli):

    $ systemctl | jc --systemctl

or

    $ jc systemctl

Usage (module):

    import jc
    result = jc.parse('systemctl', systemctl_command_output)

Schema:

    [
      {
        "unit":          string,
        "load":          string,
        "active":        string,
        "sub":           string,
        "description":   string
      }
    ]

Examples:

    $ systemctl -a | jc --systemctl -p
    [
      {
        "unit": "proc-sys-fs-binfmt_misc.automount",
        "load": "loaded",
        "active": "active",
        "sub": "waiting",
        "description": "Arbitrary Executable File Formats File System ..."
      },
      {
        "unit": "dev-block-8:2.device",
        "load": "loaded",
        "active": "active",
        "sub": "plugged",
        "description": "LVM PV 3klkIj-w1qk-DkJi-0XBJ-y3o7-i2Ac-vHqWBM o..."
      },
      {
        "unit": "dev-cdrom.device",
        "load": "loaded",
        "active": "active",
        "sub": "plugged",
        "description": "VMware_Virtual_IDE_CDROM_Drive"
      },
      ...
    ]
"""
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.5'
    description = '`systemctl` command parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    magic_commands = ['systemctl']
    tags = ['command']
__version__ = info.version

def _process(proc_data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured data to conform to the schema.\n    '
    return proc_data

def parse(data, raw=False, quiet=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    linedata = list(filter(None, data.splitlines()))
    raw_output = []
    if jc.utils.has_data(data):
        cleandata = []
        for entry in linedata:
            cleandata.append(entry.encode('ascii', errors='ignore').decode())
        header_text = cleandata[0]
        header_list = header_text.lower().split()
        raw_output = []
        for entry in cleandata[1:]:
            if 'LOAD   = ' in entry:
                break
            else:
                entry_list = entry.rstrip().split(maxsplit=4)
                output_line = dict(zip(header_list, entry_list))
                raw_output.append(output_line)
    if raw:
        return raw_output
    else:
        return _process(raw_output)