"""jc - JSON Convert `foo` command output streaming parser

> This streaming parser outputs JSON Lines (cli) or returns an Iterable of
> Dictionaries (module)

<<Short foo description and caveats>>

Usage (cli):

    $ foo | jc --foo-s

Usage (module):

    import jc

    result = jc.parse('foo_s', foo_command_output.splitlines())
    for item in result:
        # do something

Schema:

    {
      "foo":            string,

      # below object only exists if using -qq or ignore_exceptions=True
      "_jc_meta": {
        "success":      boolean,     # false if error parsing
        "error":        string,      # exists if "success" is false
        "line":         string       # exists if "success" is false
      }
    }

Examples:

    $ foo | jc --foo-s
    {example output}
    ...

    $ foo | jc --foo-s -r
    {example output}
    ...
"""
from typing import Dict, Iterable
import jc.utils
from jc.streaming import add_jc_meta, streaming_input_type_check, streaming_line_input_type_check, raise_or_yield
from jc.jc_types import JSONDictType, StreamingOutputType
from jc.exceptions import ParseError

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.0'
    description = '`foo` command streaming parser'
    author = 'John Doe'
    author_email = 'johndoe@gmail.com'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['command']
    streaming = True
__version__ = info.version

def _process(proc_data: JSONDictType) -> JSONDictType:
    if False:
        while True:
            i = 10
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured data to conform to the schema.\n    '
    return proc_data

@add_jc_meta
def parse(data: Iterable[str], raw: bool=False, quiet: bool=False, ignore_exceptions: bool=False) -> StreamingOutputType:
    if False:
        i = 10
        return i + 15
    '\n    Main text parsing generator function. Returns an iterable object.\n\n    Parameters:\n\n        data:              (iterable)  line-based text data to parse\n                                       (e.g. sys.stdin or str.splitlines())\n\n        raw:               (boolean)   unprocessed output if True\n        quiet:             (boolean)   suppress warning messages if True\n        ignore_exceptions: (boolean)   ignore parsing exceptions if True\n\n\n    Returns:\n\n        Iterable of Dictionaries\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    streaming_input_type_check(data)
    for line in data:
        try:
            streaming_line_input_type_check(line)
            output_line: Dict = {}
            if not line.strip():
                continue
            if output_line:
                yield (output_line if raw else _process(output_line))
            else:
                raise ParseError('Not foo data')
        except Exception as e:
            yield raise_or_yield(ignore_exceptions, e, line)