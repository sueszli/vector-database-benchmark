import re
import string
import sys
import time
PY2 = sys.version_info[0] == 2
if not PY2:
    text_type = str
    string_types = (str,)
    unichr = chr
else:
    text_type = unicode
    string_types = (str, unicode)
    unichr = unichr
COMMENT = object()

def load_properties(fh, mapping=dict):
    if False:
        while True:
            i = 10
    '\n    Reads properties from a Java .properties file.\n\n    Returns a dict (or provided mapping) of properties.\n\n    :param fh: a readable file-like object\n    :param mapping: mapping type to load properties into\n  '
    return mapping(iter_properties(fh))

def store_properties(fh, props, comment=None, timestamp=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Writes properties to the file in Java properties format.\n\n    :param fh: a writable file-like object\n    :param props: a mapping (dict) or iterable of key/value pairs\n    :param comment: comment to write to the beginning of the file\n    :param timestamp: boolean indicating whether to write a timestamp comment\n  '
    if comment is not None:
        write_comment(fh, comment)
    if timestamp:
        write_comment(fh, time.strftime('%a %b %d %H:%M:%S %Z %Y'))
    if hasattr(props, 'keys'):
        for key in props:
            write_property(fh, key, props[key])
    else:
        for (key, value) in props:
            write_property(fh, key, value)

def write_comment(fh, comment):
    if False:
        while True:
            i = 10
    '\n    Writes a comment to the file in Java properties format.\n\n    Newlines in the comment text are automatically turned into a continuation\n    of the comment by adding a "#" to the beginning of each line.\n\n    :param fh: a writable file-like object\n    :param comment: comment string to write\n  '
    _require_string(comment, 'comments')
    fh.write(_escape_comment(comment))
    fh.write(b'\n')

def _require_string(value, name):
    if False:
        print('Hello World!')
    if isinstance(value, string_types):
        return
    valid_types = ' or '.join((cls.__name__ for cls in string_types))
    raise TypeError('%s must be %s, but got: %s %r' % (name, valid_types, type(value), value))

def write_property(fh, key, value):
    if False:
        print('Hello World!')
    '\n    Write a single property to the file in Java properties format.\n\n    :param fh: a writable file-like object\n    :param key: the key to write\n    :param value: the value to write\n  '
    if key is COMMENT:
        write_comment(fh, value)
        return
    _require_string(key, 'keys')
    _require_string(value, 'values')
    fh.write(_escape_key(key))
    fh.write(b'=')
    fh.write(_escape_value(value))
    fh.write(b'\n')

def iter_properties(fh, comments=False):
    if False:
        while True:
            i = 10
    '\n    Incrementally read properties from a Java .properties file.\n\n    Yields tuples of key/value pairs.\n\n    If ``comments`` is `True`, comments will be included with ``jprops.COMMENT``\n    in place of the key.\n\n    :param fh: a readable file-like object\n    :param comments: should include comments (default: False)\n  '
    for line in _property_lines(fh):
        (key, value) = _split_key_value(line)
        if key is not COMMENT:
            key = _unescape(key)
        elif not comments:
            continue
        yield (key, _unescape(value))
_COMMENT_CHARS = '#!'
_COMMENT_CHARS_BYTES = bytearray(_COMMENT_CHARS, 'ascii')
_LINE_PATTERN = re.compile(b'^\\s*(?P<body>.*?)(?P<backslashes>\\\\*)$')
_KEY_TERMINATORS_EXPLICIT = '=:'
_KEY_TERMINATORS = _KEY_TERMINATORS_EXPLICIT + string.whitespace
_KEY_TERMINATORS_EXPLICIT_BYTES = bytearray(_KEY_TERMINATORS_EXPLICIT, 'ascii')
_KEY_TERMINATORS_BYTES = bytearray(_KEY_TERMINATORS, 'ascii')
_escapes = {'t': '\t', 'n': '\n', 'f': '\x0c', 'r': '\r'}
_escapes_rev = dict(((v, '\\' + k) for (k, v) in _escapes.items()))
for c in '\\' + _COMMENT_CHARS + _KEY_TERMINATORS_EXPLICIT:
    _escapes_rev.setdefault(c, '\\' + c)

def _unescape(value):
    if False:
        print('Hello World!')
    value = value.decode('latin-1')
    if not isinstance(value, str):
        try:
            value = value.encode('ascii')
        except UnicodeEncodeError:
            pass

    def unirepl(m):
        if False:
            print('Hello World!')
        backslashes = m.group(1)
        charcode = m.group(2)
        if len(backslashes) % 2 == 0:
            return m.group(0)
        c = unichr(int(charcode, 16))
        if c == '\\':
            c = u'\\\\'
        return backslashes + c
    value = re.sub('(\\\\+)u([0-9a-fA-F]{4})', unirepl, value)

    def bslashrepl(m):
        if False:
            return 10
        code = m.group(1)
        return _escapes.get(code, code)
    return re.sub('\\\\(.)', bslashrepl, value)

def _escape_comment(comment):
    if False:
        while True:
            i = 10
    comment = comment.replace('\r\n', '\n').replace('\r', '\n')
    comment = re.sub('\\n(?![#!])', '\n#', comment)
    if isinstance(comment, text_type):
        comment = re.sub(u'[Ā-\uffff]', _unicode_replace, comment)
        comment = comment.encode('latin-1')
    return b'#' + comment

def _escape_key(key):
    if False:
        return 10
    return _escape(key, _KEY_TERMINATORS)

def _escape_value(value):
    if False:
        while True:
            i = 10
    tail = value.lstrip()
    if len(tail) == len(value):
        return _escape(value)
    if tail:
        head = value[:-len(tail)]
    else:
        head = value
    return _escape(head, string.whitespace) + _escape(tail)

def _escape(value, chars=''):
    if False:
        i = 10
        return i + 15
    escape_chars = set(_escapes_rev)
    escape_chars.update(chars)
    escape_pattern = '[%s]' % re.escape(''.join(escape_chars))

    def esc(m):
        if False:
            for i in range(10):
                print('nop')
        c = m.group(0)
        return _escapes_rev.get(c) or '\\' + c
    value = re.sub(escape_pattern, esc, value)
    value = re.sub(u'[\x00-\x19\x7f-\uffff]', _unicode_replace, value)
    return value.encode('latin-1')

def _unicode_replace(m):
    if False:
        print('Hello World!')
    c = m.group(0)
    return '\\u%.4x' % ord(c)

def _split_key_value(line):
    if False:
        print('Hello World!')
    if line[0] in _COMMENT_CHARS_BYTES:
        return (COMMENT, line[1:])
    escaped = False
    key_buf = bytearray()
    line_orig = line
    line = bytearray(line)
    for (idx, c) in enumerate(line):
        if not escaped and c in _KEY_TERMINATORS_BYTES:
            key_terminated_fully = c in _KEY_TERMINATORS_EXPLICIT_BYTES
            break
        key_buf.append(c)
        escaped = c == ord('\\')
    else:
        return (line_orig, b'')
    value = line[idx + 1:].lstrip()
    if not key_terminated_fully and value[:1] in _KEY_TERMINATORS_EXPLICIT_BYTES:
        value = value[1:].lstrip()
    return (bytes(key_buf), bytes(value))

def _universal_newlines(fp):
    if False:
        i = 10
        return i + 15
    '\n    Wrap a file to convert newlines regardless of whether the file was opened\n    with the "universal newlines" option or not.\n  '
    if 'U' in getattr(fp, 'mode', ''):
        for line in fp:
            yield line
    else:
        for line in fp:
            line = line.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            for piece in line.split(b'\n'):
                yield piece

def _property_lines(fp):
    if False:
        return 10
    buf = bytearray()
    for line in _universal_newlines(fp):
        m = _LINE_PATTERN.match(line)
        body = m.group('body')
        backslashes = m.group('backslashes')
        if len(backslashes) % 2 == 0:
            body += backslashes
            continuation = False
        else:
            body += backslashes[:-1]
            continuation = True
        if not body:
            continue
        buf.extend(body)
        if not continuation:
            yield bytes(buf)
            buf = bytearray()