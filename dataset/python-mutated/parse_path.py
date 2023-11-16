from __future__ import absolute_import
from six.moves.urllib.parse import unquote
from trashcli.parse_trashinfo.parser_error import ParseError

def parse_path(contents):
    if False:
        i = 10
        return i + 15
    for line in contents.split('\n'):
        if line.startswith('Path='):
            return unquote(line[len('Path='):])
    raise ParseError('Unable to parse Path')