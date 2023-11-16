"""
A partial parser for WebVTT segments. Interprets enough of the WebVTT stream
to be able to assemble a single stand-alone subtitle file, suitably adjusting
timestamps on the way, while everything else is passed through unmodified.

Regular expressions based on the W3C WebVTT specification
<https://www.w3.org/TR/webvtt1/>. The X-TIMESTAMP-MAP extension is described
in RFC 8216 §3.5 <https://tools.ietf.org/html/rfc8216#section-3.5>.
"""
import io
import re
from .utils import int_or_none, timetuple_from_msec

class _MatchParser:
    """
    An object that maintains the current parsing position and allows
    conveniently advancing it as syntax elements are successfully parsed.
    """

    def __init__(self, string):
        if False:
            i = 10
            return i + 15
        self._data = string
        self._pos = 0

    def match(self, r):
        if False:
            while True:
                i = 10
        if isinstance(r, re.Pattern):
            return r.match(self._data, self._pos)
        if isinstance(r, str):
            if self._data.startswith(r, self._pos):
                return len(r)
            return None
        raise ValueError(r)

    def advance(self, by):
        if False:
            return 10
        if by is None:
            amt = 0
        elif isinstance(by, re.Match):
            amt = len(by.group(0))
        elif isinstance(by, str):
            amt = len(by)
        elif isinstance(by, int):
            amt = by
        else:
            raise ValueError(by)
        self._pos += amt
        return by

    def consume(self, r):
        if False:
            i = 10
            return i + 15
        return self.advance(self.match(r))

    def child(self):
        if False:
            for i in range(10):
                print('nop')
        return _MatchChildParser(self)

class _MatchChildParser(_MatchParser):
    """
    A child parser state, which advances through the same data as
    its parent, but has an independent position. This is useful when
    advancing through syntax elements we might later want to backtrack
    from.
    """

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super().__init__(parent._data)
        self.__parent = parent
        self._pos = parent._pos

    def commit(self):
        if False:
            print('Hello World!')
        '\n        Advance the parent state to the current position of this child state.\n        '
        self.__parent._pos = self._pos
        return self.__parent

class ParseError(Exception):

    def __init__(self, parser):
        if False:
            i = 10
            return i + 15
        super().__init__('Parse error at position %u (near %r)' % (parser._pos, parser._data[parser._pos:parser._pos + 20]))
_REGEX_TS = re.compile('(?x)\n    (?:([0-9]{1,}):)?\n    ([0-9]{2}):\n    ([0-9]{2})\\.\n    ([0-9]{3})?\n')
_REGEX_EOF = re.compile('\\Z')
_REGEX_NL = re.compile('(?:\\r\\n|[\\r\\n]|$)')
_REGEX_BLANK = re.compile('(?:\\r\\n|[\\r\\n])+')

def _parse_ts(ts):
    if False:
        while True:
            i = 10
    '\n    Convert a parsed WebVTT timestamp (a re.Match obtained from _REGEX_TS)\n    into an MPEG PES timestamp: a tick counter at 90 kHz resolution.\n    '
    return 90 * sum((int(part or 0) * mult for (part, mult) in zip(ts.groups(), (3600000, 60000, 1000, 1))))

def _format_ts(ts):
    if False:
        i = 10
        return i + 15
    '\n    Convert an MPEG PES timestamp into a WebVTT timestamp.\n    This will lose sub-millisecond precision.\n    '
    return '%02u:%02u:%02u.%03u' % timetuple_from_msec(int((ts + 45) // 90))

class Block:
    """
    An abstract WebVTT block.
    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        for (key, val) in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def parse(cls, parser):
        if False:
            for i in range(10):
                print('nop')
        m = parser.match(cls._REGEX)
        if not m:
            return None
        parser.advance(m)
        return cls(raw=m.group(0))

    def write_into(self, stream):
        if False:
            i = 10
            return i + 15
        stream.write(self.raw)

class HeaderBlock(Block):
    """
    A WebVTT block that may only appear in the header part of the file,
    i.e. before any cue blocks.
    """
    pass

class Magic(HeaderBlock):
    _REGEX = re.compile('\\ufeff?WEBVTT([ \\t][^\\r\\n]*)?(?:\\r\\n|[\\r\\n])')
    _REGEX_TSMAP = re.compile('X-TIMESTAMP-MAP=')
    _REGEX_TSMAP_LOCAL = re.compile('LOCAL:')
    _REGEX_TSMAP_MPEGTS = re.compile('MPEGTS:([0-9]+)')
    _REGEX_TSMAP_SEP = re.compile('[ \\t]*,[ \\t]*')
    _REGEX_META = re.compile('(?:(?!-->)[^\\r\\n])+:(?:(?!-->)[^\\r\\n])+(?:\\r\\n|[\\r\\n])')

    @classmethod
    def __parse_tsmap(cls, parser):
        if False:
            while True:
                i = 10
        parser = parser.child()
        while True:
            m = parser.consume(cls._REGEX_TSMAP_LOCAL)
            if m:
                m = parser.consume(_REGEX_TS)
                if m is None:
                    raise ParseError(parser)
                local = _parse_ts(m)
                if local is None:
                    raise ParseError(parser)
            else:
                m = parser.consume(cls._REGEX_TSMAP_MPEGTS)
                if m:
                    mpegts = int_or_none(m.group(1))
                    if mpegts is None:
                        raise ParseError(parser)
                else:
                    raise ParseError(parser)
            if parser.consume(cls._REGEX_TSMAP_SEP):
                continue
            if parser.consume(_REGEX_NL):
                break
            raise ParseError(parser)
        parser.commit()
        return (local, mpegts)

    @classmethod
    def parse(cls, parser):
        if False:
            while True:
                i = 10
        parser = parser.child()
        m = parser.consume(cls._REGEX)
        if not m:
            raise ParseError(parser)
        extra = m.group(1)
        (local, mpegts, meta) = (None, None, '')
        while not parser.consume(_REGEX_NL):
            if parser.consume(cls._REGEX_TSMAP):
                (local, mpegts) = cls.__parse_tsmap(parser)
                continue
            m = parser.consume(cls._REGEX_META)
            if m:
                meta += m.group(0)
                continue
            raise ParseError(parser)
        parser.commit()
        return cls(extra=extra, mpegts=mpegts, local=local, meta=meta)

    def write_into(self, stream):
        if False:
            for i in range(10):
                print('nop')
        stream.write('WEBVTT')
        if self.extra is not None:
            stream.write(self.extra)
        stream.write('\n')
        if self.local or self.mpegts:
            stream.write('X-TIMESTAMP-MAP=LOCAL:')
            stream.write(_format_ts(self.local if self.local is not None else 0))
            stream.write(',MPEGTS:')
            stream.write(str(self.mpegts if self.mpegts is not None else 0))
            stream.write('\n')
        if self.meta:
            stream.write(self.meta)
        stream.write('\n')

class StyleBlock(HeaderBlock):
    _REGEX = re.compile('(?x)\n        STYLE[\\ \\t]*(?:\\r\\n|[\\r\\n])\n        ((?:(?!-->)[^\\r\\n])+(?:\\r\\n|[\\r\\n]))*\n        (?:\\r\\n|[\\r\\n])\n    ')

class RegionBlock(HeaderBlock):
    _REGEX = re.compile('(?x)\n        REGION[\\ \\t]*\n        ((?:(?!-->)[^\\r\\n])+(?:\\r\\n|[\\r\\n]))*\n        (?:\\r\\n|[\\r\\n])\n    ')

class CommentBlock(Block):
    _REGEX = re.compile('(?x)\n        NOTE(?:\\r\\n|[\\ \\t\\r\\n])\n        ((?:(?!-->)[^\\r\\n])+(?:\\r\\n|[\\r\\n]))*\n        (?:\\r\\n|[\\r\\n])\n    ')

class CueBlock(Block):
    """
    A cue block. The payload is not interpreted.
    """
    _REGEX_ID = re.compile('((?:(?!-->)[^\\r\\n])+)(?:\\r\\n|[\\r\\n])')
    _REGEX_ARROW = re.compile('[ \\t]+-->[ \\t]+')
    _REGEX_SETTINGS = re.compile('[ \\t]+((?:(?!-->)[^\\r\\n])+)')
    _REGEX_PAYLOAD = re.compile('[^\\r\\n]+(?:\\r\\n|[\\r\\n])?')

    @classmethod
    def parse(cls, parser):
        if False:
            while True:
                i = 10
        parser = parser.child()
        id = None
        m = parser.consume(cls._REGEX_ID)
        if m:
            id = m.group(1)
        m0 = parser.consume(_REGEX_TS)
        if not m0:
            return None
        if not parser.consume(cls._REGEX_ARROW):
            return None
        m1 = parser.consume(_REGEX_TS)
        if not m1:
            return None
        m2 = parser.consume(cls._REGEX_SETTINGS)
        if not parser.consume(_REGEX_NL):
            return None
        start = _parse_ts(m0)
        end = _parse_ts(m1)
        settings = m2.group(1) if m2 is not None else None
        text = io.StringIO()
        while True:
            m = parser.consume(cls._REGEX_PAYLOAD)
            if not m:
                break
            text.write(m.group(0))
        parser.commit()
        return cls(id=id, start=start, end=end, settings=settings, text=text.getvalue())

    def write_into(self, stream):
        if False:
            for i in range(10):
                print('nop')
        if self.id is not None:
            stream.write(self.id)
            stream.write('\n')
        stream.write(_format_ts(self.start))
        stream.write(' --> ')
        stream.write(_format_ts(self.end))
        if self.settings is not None:
            stream.write(' ')
            stream.write(self.settings)
        stream.write('\n')
        stream.write(self.text)
        stream.write('\n')

    @property
    def as_json(self):
        if False:
            return 10
        return {'id': self.id, 'start': self.start, 'end': self.end, 'text': self.text, 'settings': self.settings}

    def __eq__(self, other):
        if False:
            return 10
        return self.as_json == other.as_json

    @classmethod
    def from_json(cls, json):
        if False:
            i = 10
            return i + 15
        return cls(id=json['id'], start=json['start'], end=json['end'], text=json['text'], settings=json['settings'])

    def hinges(self, other):
        if False:
            return 10
        if self.text != other.text:
            return False
        if self.settings != other.settings:
            return False
        return self.start <= self.end == other.start <= other.end

def parse_fragment(frag_content):
    if False:
        i = 10
        return i + 15
    '\n    A generator that yields (partially) parsed WebVTT blocks when given\n    a bytes object containing the raw contents of a WebVTT file.\n    '
    parser = _MatchParser(frag_content.decode())
    yield Magic.parse(parser)
    while not parser.match(_REGEX_EOF):
        if parser.consume(_REGEX_BLANK):
            continue
        block = RegionBlock.parse(parser)
        if block:
            yield block
            continue
        block = StyleBlock.parse(parser)
        if block:
            yield block
            continue
        block = CommentBlock.parse(parser)
        if block:
            yield block
            continue
        break
    while not parser.match(_REGEX_EOF):
        if parser.consume(_REGEX_BLANK):
            continue
        block = CommentBlock.parse(parser)
        if block:
            yield block
            continue
        block = CueBlock.parse(parser)
        if block:
            yield block
            continue
        raise ParseError(parser)