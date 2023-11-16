import re
from .robottypes import is_string
_CONTROL_WORDS = frozenset(('ELSE', 'ELSE IF', 'AND', 'WITH NAME', 'AS'))
_SEQUENCES_TO_BE_ESCAPED = ('\\', '${', '@{', '%{', '&{', '*{', '=')

def escape(item):
    if False:
        return 10
    if not is_string(item):
        return item
    if item in _CONTROL_WORDS:
        return '\\' + item
    for seq in _SEQUENCES_TO_BE_ESCAPED:
        if seq in item:
            item = item.replace(seq, '\\' + seq)
    return item

def glob_escape(item):
    if False:
        print('Hello World!')
    for char in '[*?':
        if char in item:
            item = item.replace(char, '[%s]' % char)
    return item

class Unescaper:
    _escape_sequences = re.compile('\n        (\\\\+)                # escapes\n        (n|r|t            # n, r, or t\n         |x[0-9a-fA-F]{2}    # x+HH\n         |u[0-9a-fA-F]{4}    # u+HHHH\n         |U[0-9a-fA-F]{8}    # U+HHHHHHHH\n        )?                   # optionally\n    ', re.VERBOSE)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._escape_handlers = {'': lambda value: value, 'n': lambda value: '\n', 'r': lambda value: '\r', 't': lambda value: '\t', 'x': self._hex_to_unichr, 'u': self._hex_to_unichr, 'U': self._hex_to_unichr}

    def _hex_to_unichr(self, value):
        if False:
            print('Hello World!')
        ordinal = int(value, 16)
        if ordinal > 1114111:
            return 'U' + value
        if ordinal > 65535:
            return eval("'\\U%08x'" % ordinal)
        return chr(ordinal)

    def unescape(self, item):
        if False:
            print('Hello World!')
        if not (is_string(item) and '\\' in item):
            return item
        return self._escape_sequences.sub(self._handle_escapes, item)

    def _handle_escapes(self, match):
        if False:
            return 10
        (escapes, text) = match.groups()
        (half, is_escaped) = divmod(len(escapes), 2)
        escapes = escapes[:half]
        text = text or ''
        if is_escaped:
            (marker, value) = (text[:1], text[1:])
            text = self._escape_handlers[marker](value)
        return escapes + text
unescape = Unescaper().unescape

def split_from_equals(string):
    if False:
        i = 10
        return i + 15
    from robot.variables import VariableMatches
    if not is_string(string) or '=' not in string:
        return (string, None)
    matches = VariableMatches(string, ignore_errors=True)
    if not matches and '\\' not in string:
        return tuple(string.split('=', 1))
    try:
        index = _find_split_index(string, matches)
    except ValueError:
        return (string, None)
    return (string[:index], string[index + 1:])

def _find_split_index(string, matches):
    if False:
        return 10
    remaining = string
    relative_index = 0
    for match in matches:
        try:
            return _find_split_index_from_part(match.before) + relative_index
        except ValueError:
            remaining = match.after
            relative_index += match.end
    return _find_split_index_from_part(remaining) + relative_index

def _find_split_index_from_part(string):
    if False:
        while True:
            i = 10
    index = 0
    while '=' in string[index:]:
        index += string[index:].index('=')
        if _not_escaping(string[:index]):
            return index
        index += 1
    raise ValueError

def _not_escaping(name):
    if False:
        i = 10
        return i + 15
    backslashes = len(name) - len(name.rstrip('\\'))
    return backslashes % 2 == 0