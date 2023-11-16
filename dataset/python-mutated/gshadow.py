"""jc - JSON Convert `/etc/gshadow` file parser

Usage (cli):

    $ cat /etc/gshadow | jc --gshadow

Usage (module):

    import jc
    result = jc.parse('gshadow', gshadow_file_output)

Schema:

    [
      {
        "group_name":       string,
        "password":         string,
        "administrators": [
                            string
        ],
        "members": [
                            string
        ]
      }
    ]

Examples:

    $ cat /etc/gshadow | jc --gshadow -p
    [
      {
        "group_name": "root",
        "password": "*",
        "administrators": [],
        "members": []
      },
      {
        "group_name": "adm",
        "password": "*",
        "administrators": [],
        "members": [
          "syslog",
          "joeuser"
        ]
      },
      ...
    ]

    $ cat /etc/gshadow | jc --gshadow -p -r
    [
      {
        "group_name": "root",
        "password": "*",
        "administrators": [
          ""
        ],
        "members": [
          ""
        ]
      },
      {
        "group_name": "adm",
        "password": "*",
        "administrators": [
          ""
        ],
        "members": [
          "syslog",
          "joeuser"
        ]
      },
      ...
    ]
"""
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.3'
    description = '`/etc/gshadow` file parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux', 'aix', 'freebsd']
    tags = ['file']
__version__ = info.version

def _process(proc_data):
    if False:
        while True:
            i = 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured data to conform to the schema.\n    '
    for entry in proc_data:
        if entry['administrators'] == ['']:
            entry['administrators'] = []
        if entry['members'] == ['']:
            entry['members'] = []
    return proc_data

def parse(data, raw=False, quiet=False):
    if False:
        i = 10
        return i + 15
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output = []
    cleandata = data.splitlines()
    cleandata = list(filter(None, cleandata))
    if jc.utils.has_data(data):
        for entry in cleandata:
            if entry.startswith('#'):
                continue
            output_line = {}
            fields = entry.split(':')
            output_line['group_name'] = fields[0]
            output_line['password'] = fields[1]
            output_line['administrators'] = fields[2].split(',')
            output_line['members'] = fields[3].split(',')
            raw_output.append(output_line)
    if raw:
        return raw_output
    else:
        return _process(raw_output)