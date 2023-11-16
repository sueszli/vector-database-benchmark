"""jc - JSON Convert `csv` file streaming parser

> This streaming parser outputs JSON Lines (cli) or returns an Iterable of
> Dictionaries (module)

The `csv` streaming parser will attempt to automatically detect the
delimiter character. If the delimiter cannot be detected it will default
to comma. The first row of the file must be a header row.

> Note: The first 100 rows are read into memory to enable delimiter
> detection, then the rest of the rows are loaded lazily.

Usage (cli):

    $ cat file.csv | jc --csv-s

Usage (module):

    import jc

    result = jc.parse('csv_s', csv_output.splitlines())
    for item in result:
        # do something

Schema:

CSV file converted to a Dictionary:
https://docs.python.org/3/library/csv.html

    {
      "column_name1":     string,
      "column_name2":     string,

      # below object only exists if using -qq or ignore_exceptions=True
      "_jc_meta": {
        "success":        boolean,     # false if error parsing
        "error":          string,      # exists if "success" is false
        "line":           string       # exists if "success" is false
      }
    }

Examples:

    $ cat homes.csv
    "Sell", "List", "Living", "Rooms", "Beds", "Baths", "Age", "Acres"...
    142, 160, 28, 10, 5, 3,  60, 0.28,  3167
    175, 180, 18,  8, 4, 1,  12, 0.43,  4033
    129, 132, 13,  6, 3, 1,  41, 0.33,  1471
    ...

    $ cat homes.csv | jc --csv-s
    {"Sell":"142","List":"160","Living":"28","Rooms":"10","Beds":"5"...}
    {"Sell":"175","List":"180","Living":"18","Rooms":"8","Beds":"4"...}
    {"Sell":"129","List":"132","Living":"13","Rooms":"6","Beds":"3"...}
    ...
"""
import itertools
import csv
import jc.utils
from jc.streaming import streaming_input_type_check, add_jc_meta, raise_or_yield
from jc.exceptions import ParseError

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.4'
    description = 'CSV file streaming parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    details = 'Using the python standard csv library'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['standard', 'file', 'string']
    streaming = True
__version__ = info.version

def _process(proc_data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Each Dictionary represents a row in the csv\n        file.\n    '
    return proc_data

@add_jc_meta
def parse(data, raw=False, quiet=False, ignore_exceptions=False):
    if False:
        return 10
    '\n    Main text parsing generator function. Returns an iterable object.\n\n    Parameters:\n\n        data:              (iterable)  line-based text data to parse\n                                       (e.g. sys.stdin or str.splitlines())\n\n        raw:               (boolean)   unprocessed output if True\n        quiet:             (boolean)   suppress warning messages if True\n        ignore_exceptions: (boolean)   ignore parsing exceptions if True\n\n    Returns:\n\n        Iterable of Dictionaries\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    streaming_input_type_check(data)
    data = iter(data)
    temp_list = []
    for line in itertools.islice(data, 100):
        temp_list.append(line.rstrip())
    if len(temp_list) == 1:
        raise ParseError('Unable to detect line endings. Please try the non-streaming CSV parser instead.')
    if temp_list:
        if isinstance(temp_list[0], str):
            temp_list[0] = temp_list[0].encode('utf-8')
        temp_list[0] = temp_list[0].decode('utf-8-sig')
    sniffdata = '\r\n'.join(temp_list)[:1024]
    dialect = 'excel'
    try:
        dialect = csv.Sniffer().sniff(sniffdata)
        if '""' in sniffdata:
            dialect.doublequote = True
    except Exception:
        pass
    new_data = itertools.chain(temp_list, data)
    reader = csv.DictReader(new_data, dialect=dialect)
    for row in reader:
        try:
            yield (row if raw else _process(row))
        except Exception as e:
            yield raise_or_yield(ignore_exceptions, e, str(row))