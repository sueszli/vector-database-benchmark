"""jc - JSON Convert `pip-list` command output parser

Usage (cli):

    $ pip list | jc --pip-list

or

    $ jc pip list

Usage (module):

    import jc
    result = jc.parse('pip_list', pip_list_command_output)

Schema:

    [
      {
        "package":     string,
        "version":     string,
        "location":    string
      }
    ]

Examples:

    $ pip list | jc --pip-list -p
    [
      {
        "package": "ansible",
        "version": "2.8.5"
      },
      {
        "package": "antlr4-python3-runtime",
        "version": "4.7.2"
      },
      {
        "package": "asn1crypto",
        "version": "0.24.0"
      },
      ...
    ]
"""
import jc.utils
import jc.parsers.universal

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.5'
    description = '`pip list` command parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    magic_commands = ['pip list', 'pip3 list']
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
        i = 10
        return i + 15
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output = []
    cleandata = list(filter(None, data.splitlines()))
    if jc.utils.has_data(data):
        if ' (' in cleandata[0]:
            for row in cleandata:
                raw_output.append({'package': row.split(' (')[0], 'version': row.split(' (')[1].rstrip(')')})
        else:
            for (i, line) in reversed(list(enumerate(cleandata))):
                if '---' in line:
                    cleandata.pop(i)
            cleandata[0] = cleandata[0].lower()
            if cleandata:
                raw_output = jc.parsers.universal.simple_table_parse(cleandata)
    if raw:
        return raw_output
    else:
        return _process(raw_output)