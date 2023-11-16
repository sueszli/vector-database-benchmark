"""jc - JSON Convert Syslog RFC 5424 string streaming parser

> This streaming parser outputs JSON Lines (cli) or returns an Iterable of
> Dictionaries (module)

This parser accepts a single syslog line string or multiple syslog lines
separated by newlines. A warning message to `STDERR` will be printed if an
unparsable line is found unless `--quiet` or `quiet=True` is used.

The `timestamp_epoch` calculated timestamp field is naive. (i.e. based on
the local time of the system the parser is run on)

The `timestamp_epoch_utc` calculated timestamp field is timezone-aware and
is only available if the timezone field is UTC.

Usage (cli):

    $ echo <165>1 2003-08-24T05:14:15.000003-07:00 192.0... | jc --syslog-s

Usage (module):

    import jc

    result = jc.parse('syslog_s', syslog_command_output.splitlines())
    for item in result:
        # do something

Schema:

Blank values converted to `null`/`None`.

    {
      "priority":                   integer,
      "version":                    integer,
      "timestamp":                  string,
      "timestamp_epoch":            integer,  # [0]
      "timestamp_epoch_utc":        integer,  # [1]
      "hostname":                   string,
      "appname":                    string,
      "proc_id":                    integer,
      "msg_id":                     string,
      "structured_data": [
        {
          "identity":               string,
          "parameters": {
            "<key>":                string
          }
        }
      ],
      "message":                    string,
      "unparsable":                 string  # [2]

      # below object only exists if using -qq or ignore_exceptions=True
      "_jc_meta": {
        "success":      boolean,     # false if error parsing
        "error":        string,      # exists if "success" is false
        "line":         string       # exists if "success" is false
      }
    }

    [0] naive timestamp if "timestamp" field is parsable, else null
    [1] timezone aware timestamp available for UTC, else null
    [2] this field exists if the syslog line is not parsable. The value
        is the original syslog line.

Examples:

    $ cat syslog.txt | jc --syslog-s -p
    {"priority":165,"version":1,"timestamp":"2003-08-24T05:14:15.000003-...}
    {"priority":165,"version":1,"timestamp":"2003-08-24T05:14:16.000003-...}
    ...

    $ cat syslog.txt | jc --syslog-s -p -r
    {"priority":"165","version":"1","timestamp":"2003-08-24T05:14:15.000...}
    {"priority":"165","version":"1","timestamp":"2003-08-24T05:15:15.000...}
    ...
"""
from typing import List, Dict, Iterable, Union, Optional
import re
import jc.utils
from jc.streaming import add_jc_meta, streaming_input_type_check, streaming_line_input_type_check, raise_or_yield
from jc.exceptions import ParseError

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = 'Syslog RFC 5424 string streaming parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['standard', 'file', 'string']
    streaming = True
__version__ = info.version
escape_map = {'\\\\': '\\', '\\"': '"', '\\]': ']'}

def _extract_structs(structs_string: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    struct_match = re.compile('(?P<eachstruct>\\[.+?(?<!\\\\)\\])')
    each_struct = struct_match.findall(structs_string)
    my_structs = []
    if each_struct:
        for structured in each_struct:
            my_structs.append(structured)
    return my_structs

def _extract_ident(struct_string) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    ident = re.compile('\\[(?P<ident>[^\\[\\=\\x22\\]\\x20]{1,32})\\s')
    ident_match = ident.search(struct_string)
    if ident_match:
        return ident_match.group('ident')
    return None

def _extract_kv(struct_string) -> List[Dict]:
    if False:
        for i in range(10):
            print('nop')
    key_vals = re.compile('(?P<key>\\w+)=(?P<val>\\"[^\\"]*\\")')
    key_vals_match = key_vals.findall(struct_string)
    kv_list = []
    if key_vals_match:
        for kv in key_vals_match:
            (key, val) = kv
            for (esc, esc_sub) in escape_map.items():
                val = val.replace(esc, esc_sub)
            kv_list.append({key: val[1:-1]})
    return kv_list

def _process(proc_data: Dict) -> Dict:
    if False:
        print('Hello World!')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured data to conform to the schema.\n    '
    int_list = {'priority', 'version', 'proc_id'}
    for (key, value) in proc_data.items():
        if proc_data[key]:
            proc_data[key] = value.strip()
    if 'timestamp' in proc_data and proc_data['timestamp']:
        format = (1300, 1310)
        dt = jc.utils.timestamp(proc_data['timestamp'], format)
        proc_data['timestamp_epoch'] = dt.naive
        proc_data['timestamp_epoch_utc'] = dt.utc
    if 'message' in proc_data and proc_data['message']:
        for (esc, esc_sub) in escape_map.items():
            proc_data['message'] = proc_data['message'].replace(esc, esc_sub)
    if 'structured_data' in proc_data and proc_data['structured_data']:
        structs_list = []
        structs = _extract_structs(proc_data['structured_data'])
        for a_struct in structs:
            struct_obj = {'identity': _extract_ident(a_struct)}
            my_values = {}
            for val_obj in _extract_kv(a_struct):
                my_values.update(val_obj)
            struct_obj.update({'parameters': my_values})
            structs_list.append(struct_obj)
        proc_data['structured_data'] = structs_list
    for key in proc_data:
        if key in int_list:
            proc_data[key] = jc.utils.convert_to_int(proc_data[key])
    return proc_data

@add_jc_meta
def parse(data: Iterable[str], raw: bool=False, quiet: bool=False, ignore_exceptions: bool=False) -> Union[Iterable[Dict], tuple]:
    if False:
        return 10
    '\n    Main text parsing generator function. Returns an iterable object.\n\n    Parameters:\n\n        data:              (iterable)  line-based text data to parse\n                                       (e.g. sys.stdin or str.splitlines())\n\n        raw:               (boolean)   unprocessed output if True\n        quiet:             (boolean)   suppress warning messages if True\n        ignore_exceptions: (boolean)   ignore parsing exceptions if True\n\n\n    Returns:\n\n        Iterable of Dictionaries\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    streaming_input_type_check(data)
    syslog = re.compile('\n        (?P<priority><(\\d|\\d{2}|1[1-8]\\d|19[01])>)?\n        (?P<version>\\d{1,2})?\\s*\n        (?P<timestamp>-|\n            (?P<fullyear>[12]\\d{3})-\n            (?P<month>0\\d|[1][012])-\n            (?P<mday>[012]\\d|3[01])T\n            (?P<hour>[01]\\d|2[0-4]):\n            (?P<minute>[0-5]\\d):\n            (?P<second>[0-5]\\d|60)(?#60seconds can be used for leap year!)(?:\\.\n            (?P<secfrac>\\d{1,6}))?\n            (?P<numoffset>Z|[+-]\\d{2}:\\d{2})(?#=timezone))\\s\n        (?P<hostname>[\\S]{1,255})\\s\n        (?P<appname>[\\S]{1,48})\\s\n        (?P<procid>[\\S]{1,128})\\s\n        (?P<msgid>[\\S]{1,32})\\s\n        (?P<structureddata>-|(?:\\[.+?(?<!\\\\)\\])+)\n        (?:\\s(?P<msg>.+))?\n        ', re.VERBOSE)
    for line in data:
        try:
            streaming_line_input_type_check(line)
            output_line: Dict = {}
            if not line.strip():
                continue
            syslog_match = syslog.match(line)
            if syslog_match:
                syslog_dict = syslog_match.groupdict()
                for item in syslog_dict:
                    if syslog_dict[item] == '-':
                        syslog_dict[item] = None
                priority = None
                if syslog_dict['priority']:
                    priority = syslog_dict['priority'][1:-1]
                output_line = {'priority': priority, 'version': syslog_dict['version'], 'timestamp': syslog_dict['timestamp'], 'hostname': syslog_dict['hostname'], 'appname': syslog_dict['appname'], 'proc_id': syslog_dict['procid'], 'msg_id': syslog_dict['msgid'], 'structured_data': syslog_dict['structureddata'], 'message': syslog_dict['msg']}
            else:
                output_line = {'unparsable': line.rstrip()}
                if not quiet:
                    jc.utils.warning_message([f'Unparsable line found: {line.rstrip()}'])
            if output_line:
                yield (output_line if raw else _process(output_line))
        except Exception as e:
            yield raise_or_yield(ignore_exceptions, e, line)