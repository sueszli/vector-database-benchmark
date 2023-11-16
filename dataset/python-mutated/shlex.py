import re
import shlex
_find_unsafe = re.compile('[^\\w@%+=:,./-]').search

def _shlex_quote(s):
    if False:
        for i in range(10):
            print('nop')
    'Return a shell-escaped version of the string *s*.'
    if not s:
        return "''"
    if _find_unsafe(s) is None:
        return s
    return "'" + s.replace("'", '\'"\'"\'') + "'"
if not hasattr(shlex, 'quote'):
    quote = _shlex_quote
else:
    quote = shlex.quote
split = shlex.split