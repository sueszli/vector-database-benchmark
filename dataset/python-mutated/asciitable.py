"""jc - JSON Convert `asciitable` parser

This parser converts ASCII and Unicode text tables with single-line rows.

Column headers must be at least two spaces apart from each other and must
be unique. For best results, column headers should be left-justified. If
column separators are present, then non-left-justified headers will be fixed
automatically.

Row separators are optional and are ignored. Each non-row-separator line is
considered a separate row in the table.

For example:

    ╒══════════╤═════════╤════════╕
    │ foo      │ bar     │ baz    │
    ╞══════════╪═════════╪════════╡
    │ good day │         │ 12345  │
    ├──────────┼─────────┼────────┤
    │ hi there │ abc def │ 3.14   │
    ╘══════════╧═════════╧════════╛

or

    +-----------------------------+
    | foo        bar       baz    |
    +-----------------------------+
    | good day             12345  |
    | hi there   abc def   3.14   |
    +-----------------------------+

or

    | foo      | bar     | baz    |
    |----------|---------|--------|
    | good day |         | 12345  |
    | hi there | abc def | 3.14   |

or

    foo        bar       baz
    ---------  --------  ------
    good day             12345
    hi there   abc def   3.14

or

    foo        bar       baz
    good day             12345
    hi there   abc def   3.14

etc...

Headers (keys) are converted to snake-case. All values are returned as
strings, except empty strings, which are converted to None/null.

> Note: To preserve the case of the keys use the `-r` cli option or
> `raw=True` argument in `parse()`.

Usage (cli):

    $ cat table.txt | jc --asciitable

Usage (module):

    import jc
    result = jc.parse('asciitable', asciitable_string)

Schema:

    [
      {
        "column_name1":     string,    # empty string is null
        "column_name2":     string     # empty string is null
      }
    ]

Examples:

    $ echo '
    >     ╒══════════╤═════════╤════════╕
    >     │ foo      │ bar     │ baz    │
    >     ╞══════════╪═════════╪════════╡
    >     │ good day │         │ 12345  │
    >     ├──────────┼─────────┼────────┤
    >     │ hi there │ abc def │ 3.14   │
    >     ╘══════════╧═════════╧════════╛' | jc --asciitable -p
    [
      {
        "foo": "good day",
        "bar": null,
        "baz": "12345"
      },
      {
        "foo": "hi there",
        "bar": "abc def",
        "baz": "3.14"
      }
    ]

    $ echo '
    >     foo        bar       baz
    >     ---------  --------  ------
    >     good day             12345
    >     hi there   abc def   3.14'  | jc --asciitable -p
    [
      {
        "foo": "good day",
        "bar": null,
        "baz": "12345"
      },
      {
        "foo": "hi there",
        "bar": "abc def",
        "baz": "3.14"
      }
    ]
"""
import re
from functools import lru_cache
from typing import List, Dict
import jc.utils
from jc.parsers.universal import sparse_table_parse

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.2'
    description = 'ASCII and Unicode table parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['generic', 'string']
__version__ = info.version

def _process(proc_data: List[Dict]) -> List[Dict]:
    if False:
        print('Hello World!')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured to conform to the schema.\n    '
    for item in proc_data:
        for key in item.copy():
            k_new = key.lower()
            item[k_new] = item.pop(key)
    return proc_data

def _remove_ansi(string: str) -> str:
    if False:
        print('Hello World!')
    ansi_escape = re.compile('(\\x9B|\\x1B\\[)[0-?]*[ -\\/]*[@-~]')
    return ansi_escape.sub('', string)

def _lstrip(string: str) -> str:
    if False:
        print('Hello World!')
    'find the leftmost non-whitespace character and lstrip to that index'
    lstrip_list = [x for x in string.splitlines() if not len(x.strip()) == 0]
    start_points = (len(x) - len(x.lstrip()) for x in lstrip_list)
    min_point = min(start_points)
    new_lstrip_list = (x[min_point:] for x in lstrip_list)
    return '\n'.join(new_lstrip_list)

def _rstrip(string: str) -> str:
    if False:
        i = 10
        return i + 15
    'find the rightmost non-whitespace character and rstrip and pad to that index'
    rstrip_list = [x for x in string.splitlines() if not len(x.strip()) == 0]
    end_points = (len(x.rstrip()) for x in rstrip_list)
    max_point = max(end_points)
    new_rstrip_list = ((x + ' ' * max_point)[:max_point] for x in rstrip_list)
    return '\n'.join(new_rstrip_list)

def _strip(string: str) -> str:
    if False:
        print('Hello World!')
    string = _lstrip(string)
    string = _rstrip(string)
    return string

@lru_cache(maxsize=32)
def _is_separator(line: str) -> bool:
    if False:
        i = 10
        return i + 15
    'returns true if a table separator line is found'
    strip_line = line.strip()
    if any((strip_line.startswith('|-') and strip_line.endswith('-|'), strip_line.startswith('━━') and strip_line.endswith('━━'), strip_line.startswith('──') and strip_line.endswith('──'), strip_line.startswith('┄┄') and strip_line.endswith('┄┄'), strip_line.startswith('┅┅') and strip_line.endswith('┅┅'), strip_line.startswith('┈┈') and strip_line.endswith('┈┈'), strip_line.startswith('┉┉') and strip_line.endswith('┉┉'), strip_line.startswith('══') and strip_line.endswith('══'), strip_line.startswith('--') and strip_line.endswith('--'), strip_line.startswith('==') and strip_line.endswith('=='), strip_line.startswith('+=') and strip_line.endswith('=+'), strip_line.startswith('+-') and strip_line.endswith('-+'), strip_line.startswith('╒') and strip_line.endswith('╕'), strip_line.startswith('╞') and strip_line.endswith('╡'), strip_line.startswith('╘') and strip_line.endswith('╛'), strip_line.startswith('┏') and strip_line.endswith('┓'), strip_line.startswith('┣') and strip_line.endswith('┫'), strip_line.startswith('┗') and strip_line.endswith('┛'), strip_line.startswith('┡') and strip_line.endswith('┩'), strip_line.startswith('┢') and strip_line.endswith('┪'), strip_line.startswith('┟') and strip_line.endswith('┧'), strip_line.startswith('┞') and strip_line.endswith('┦'), strip_line.startswith('┠') and strip_line.endswith('┨'), strip_line.startswith('┝') and strip_line.endswith('┥'), strip_line.startswith('┍') and strip_line.endswith('┑'), strip_line.startswith('┕') and strip_line.endswith('┙'), strip_line.startswith('┎') and strip_line.endswith('┒'), strip_line.startswith('┖') and strip_line.endswith('┚'), strip_line.startswith('╓') and strip_line.endswith('╖'), strip_line.startswith('╟') and strip_line.endswith('╢'), strip_line.startswith('╙') and strip_line.endswith('╜'), strip_line.startswith('╔') and strip_line.endswith('╗'), strip_line.startswith('╠') and strip_line.endswith('╣'), strip_line.startswith('╚') and strip_line.endswith('╝'), strip_line.startswith('┌') and strip_line.endswith('┐'), strip_line.startswith('├') and strip_line.endswith('┤'), strip_line.startswith('└') and strip_line.endswith('┘'), strip_line.startswith('╭') and strip_line.endswith('╮'), strip_line.startswith('╰') and strip_line.endswith('╯'))):
        return True
    return False

def _snake_case(line: str) -> str:
    if False:
        return 10
    '\n    Replace spaces between words and special characters with an underscore.\n    Ignore the replacement char (�) used for header padding.\n    '
    line = re.sub('[^a-zA-Z0-9� ]', '_', line)
    line = re.sub('\\b \\b', '_', line)
    return line

def _normalize_rows(table: str) -> List[str]:
    if False:
        print('Hello World!')
    '\n    returns a List of row strings. Header is snake-cased\n    '
    result: List[str] = []
    for line in table.splitlines():
        if not line.strip():
            continue
        if _is_separator(line):
            continue
        if not result:
            line = line.replace('│', '|').replace('┃', '|').replace('┆', '|').replace('┇', '|').replace('┊', '|').replace('┋', '|').replace('╎', '|').replace('╏', '|').replace('║', '|')
            problem_header_pattern = re.compile('(?:\\| )( +)([^|]+)')
            problem_headers = problem_header_pattern.findall(line)
            if problem_headers:
                for p_header in problem_headers:
                    old_header = p_header[0] + p_header[1]
                    sub_chars = '�' * len(p_header[0])
                    new_header = sub_chars + p_header[1]
                    line = line.replace(old_header, new_header)
            line = line.replace('|', ' ')
            result.append(_snake_case(line))
            continue
        line = line.replace('|', ' ').replace('│', ' ').replace('┃', ' ').replace('┆', ' ').replace('┇', ' ').replace('┊', ' ').replace('┋', ' ').replace('╎', ' ').replace('╏', ' ').replace('║', ' ')
        result.append(line)
    return result

def _fixup_headers(table: List[Dict]) -> List[Dict]:
    if False:
        for i in range(10):
            print('nop')
    'remove consecutive underscores and any trailing underscores'
    new_table = []
    for row in table:
        new_row = row.copy()
        for k in row:
            k_new = k.replace('�', '')
            k_new = re.sub('__+', '_', k_new)
            k_new = re.sub('_+$', '', k_new)
            new_row[k_new] = new_row.pop(k)
        new_table.append(new_row)
    return new_table

def parse(data: str, raw: bool=False, quiet: bool=False) -> List[Dict]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output: List = []
    if jc.utils.has_data(data):
        data = _remove_ansi(data)
        data = _strip(data)
        data_list = _normalize_rows(data)
        raw_table = sparse_table_parse(data_list)
        raw_output = _fixup_headers(raw_table)
    return raw_output if raw else _process(raw_output)