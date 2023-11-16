"""jc - JSON Convert `systemctl list-jobs` command output parser

Usage (cli):

    $ systemctl list-jobs | jc --systemctl-lj

or

    $ jc systemctl list-jobs

Usage (module):

    import jc
    result = jc.parse('systemctl_lj', systemctl_lj_command_output)

Schema:

    [
      {
        "job":      integer,
        "unit":     string,
        "type":     string,
        "state":    string
      }
    ]

Examples:

    $ systemctl list-jobs| jc --systemctl-lj -p
    [
      {
        "job": 3543,
        "unit": "nginxAfterGlusterfs.service",
        "type": "start",
        "state": "waiting"
      },
      {
        "job": 3545,
        "unit": "glusterReadyForLocalhostMount.service",
        "type": "start",
        "state": "running"
      },
      {
        "job": 3506,
        "unit": "nginx.service",
        "type": "start",
        "state": "waiting"
      }
    ]

    $ systemctl list-jobs| jc --systemctl-lj -p -r
    [
      {
        "job": "3543",
        "unit": "nginxAfterGlusterfs.service",
        "type": "start",
        "state": "waiting"
      },
      {
        "job": "3545",
        "unit": "glusterReadyForLocalhostMount.service",
        "type": "start",
        "state": "running"
      },
      {
        "job": "3506",
        "unit": "nginx.service",
        "type": "start",
        "state": "waiting"
      }
    ]
"""
import jc.utils

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.7'
    description = '`systemctl list-jobs` command parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux']
    magic_commands = ['systemctl list-jobs']
    tags = ['command']
__version__ = info.version

def _process(proc_data):
    if False:
        return 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured data to conform to the schema.\n    '
    int_list = {'job'}
    for entry in proc_data:
        for key in entry:
            if key in int_list:
                entry[key] = jc.utils.convert_to_int(entry[key])
    return proc_data

def parse(data, raw=False, quiet=False):
    if False:
        print('Hello World!')
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
        header_text = header_text.lower()
        header_list = header_text.split()
        raw_output = []
        for entry in cleandata[1:]:
            if 'No jobs running.' in entry or 'jobs listed.' in entry:
                break
            else:
                entry_list = entry.split(maxsplit=4)
                output_line = dict(zip(header_list, entry_list))
                raw_output.append(output_line)
    if raw:
        return raw_output
    else:
        return _process(raw_output)