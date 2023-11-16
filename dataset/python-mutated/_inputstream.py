from __future__ import absolute_import, division, unicode_literals
from pip._vendor.six import text_type, binary_type
from pip._vendor.six.moves import http_client, urllib
import codecs
import re
from pip._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
from io import StringIO
try:
    from io import BytesIO
except ImportError:
    BytesIO = StringIO
spaceCharactersBytes = frozenset([item.encode('ascii') for item in spaceCharacters])
asciiLettersBytes = frozenset([item.encode('ascii') for item in asciiLetters])
asciiUppercaseBytes = frozenset([item.encode('ascii') for item in asciiUppercase])
spacesAngleBrackets = spaceCharactersBytes | frozenset([b'>', b'<'])
invalid_unicode_no_surrogate = '[\x01-\x08\x0b\x0e-\x1f\x7f-\x9f\ufdd0-\ufdef\ufffe\uffff\U0001fffe\U0001ffff\U0002fffe\U0002ffff\U0003fffe\U0003ffff\U0004fffe\U0004ffff\U0005fffe\U0005ffff\U0006fffe\U0006ffff\U0007fffe\U0007ffff\U0008fffe\U0008ffff\U0009fffe\U0009ffff\U000afffe\U000affff\U000bfffe\U000bffff\U000cfffe\U000cffff\U000dfffe\U000dffff\U000efffe\U000effff\U000ffffe\U000fffff\U0010fffe\U0010ffff]'
if _utils.supports_lone_surrogates:
    assert invalid_unicode_no_surrogate[-1] == ']' and invalid_unicode_no_surrogate.count(']') == 1
    invalid_unicode_re = re.compile(invalid_unicode_no_surrogate[:-1] + eval('"\\uD800-\\uDFFF"') + ']')
else:
    invalid_unicode_re = re.compile(invalid_unicode_no_surrogate)
non_bmp_invalid_codepoints = set([131070, 131071, 196606, 196607, 262142, 262143, 327678, 327679, 393214, 393215, 458750, 458751, 524286, 524287, 589822, 589823, 655358, 655359, 720894, 720895, 786430, 786431, 851966, 851967, 917502, 917503, 983038, 983039, 1048574, 1048575, 1114110, 1114111])
ascii_punctuation_re = re.compile('[\t-\r -/:-@\\[-`{-~]')
charsUntilRegEx = {}

class BufferedStream(object):
    """Buffering for streams that do not have buffering of their own

    The buffer is implemented as a list of chunks on the assumption that
    joining many strings will be slow since it is O(n**2)
    """

    def __init__(self, stream):
        if False:
            i = 10
            return i + 15
        self.stream = stream
        self.buffer = []
        self.position = [-1, 0]

    def tell(self):
        if False:
            i = 10
            return i + 15
        pos = 0
        for chunk in self.buffer[:self.position[0]]:
            pos += len(chunk)
        pos += self.position[1]
        return pos

    def seek(self, pos):
        if False:
            return 10
        assert pos <= self._bufferedBytes()
        offset = pos
        i = 0
        while len(self.buffer[i]) < offset:
            offset -= len(self.buffer[i])
            i += 1
        self.position = [i, offset]

    def read(self, bytes):
        if False:
            for i in range(10):
                print('nop')
        if not self.buffer:
            return self._readStream(bytes)
        elif self.position[0] == len(self.buffer) and self.position[1] == len(self.buffer[-1]):
            return self._readStream(bytes)
        else:
            return self._readFromBuffer(bytes)

    def _bufferedBytes(self):
        if False:
            for i in range(10):
                print('nop')
        return sum([len(item) for item in self.buffer])

    def _readStream(self, bytes):
        if False:
            while True:
                i = 10
        data = self.stream.read(bytes)
        self.buffer.append(data)
        self.position[0] += 1
        self.position[1] = len(data)
        return data

    def _readFromBuffer(self, bytes):
        if False:
            return 10
        remainingBytes = bytes
        rv = []
        bufferIndex = self.position[0]
        bufferOffset = self.position[1]
        while bufferIndex < len(self.buffer) and remainingBytes != 0:
            assert remainingBytes > 0
            bufferedData = self.buffer[bufferIndex]
            if remainingBytes <= len(bufferedData) - bufferOffset:
                bytesToRead = remainingBytes
                self.position = [bufferIndex, bufferOffset + bytesToRead]
            else:
                bytesToRead = len(bufferedData) - bufferOffset
                self.position = [bufferIndex, len(bufferedData)]
                bufferIndex += 1
            rv.append(bufferedData[bufferOffset:bufferOffset + bytesToRead])
            remainingBytes -= bytesToRead
            bufferOffset = 0
        if remainingBytes:
            rv.append(self._readStream(remainingBytes))
        return b''.join(rv)

def HTMLInputStream(source, **kwargs):
    if False:
        print('Hello World!')
    if isinstance(source, http_client.HTTPResponse) or (isinstance(source, urllib.response.addbase) and isinstance(source.fp, http_client.HTTPResponse)):
        isUnicode = False
    elif hasattr(source, 'read'):
        isUnicode = isinstance(source.read(0), text_type)
    else:
        isUnicode = isinstance(source, text_type)
    if isUnicode:
        encodings = [x for x in kwargs if x.endswith('_encoding')]
        if encodings:
            raise TypeError('Cannot set an encoding with a unicode input, set %r' % encodings)
        return HTMLUnicodeInputStream(source, **kwargs)
    else:
        return HTMLBinaryInputStream(source, **kwargs)

class HTMLUnicodeInputStream(object):
    """Provides a unicode stream of characters to the HTMLTokenizer.

    This class takes care of character encoding and removing or replacing
    incorrect byte-sequences and also provides column and line tracking.

    """
    _defaultChunkSize = 10240

    def __init__(self, source):
        if False:
            print('Hello World!')
        'Initialises the HTMLInputStream.\n\n        HTMLInputStream(source, [encoding]) -> Normalized stream from source\n        for use by html5lib.\n\n        source can be either a file-object, local filename or a string.\n\n        The optional encoding parameter must be a string that indicates\n        the encoding.  If specified, that encoding will be used,\n        regardless of any BOM or later declaration (such as in a meta\n        element)\n\n        '
        if not _utils.supports_lone_surrogates:
            self.reportCharacterErrors = None
        elif len('\U0010ffff') == 1:
            self.reportCharacterErrors = self.characterErrorsUCS4
        else:
            self.reportCharacterErrors = self.characterErrorsUCS2
        self.newLines = [0]
        self.charEncoding = (lookupEncoding('utf-8'), 'certain')
        self.dataStream = self.openStream(source)
        self.reset()

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.chunk = ''
        self.chunkSize = 0
        self.chunkOffset = 0
        self.errors = []
        self.prevNumLines = 0
        self.prevNumCols = 0
        self._bufferedCharacter = None

    def openStream(self, source):
        if False:
            return 10
        'Produces a file object from source.\n\n        source can be either a file object, local filename or a string.\n\n        '
        if hasattr(source, 'read'):
            stream = source
        else:
            stream = StringIO(source)
        return stream

    def _position(self, offset):
        if False:
            print('Hello World!')
        chunk = self.chunk
        nLines = chunk.count('\n', 0, offset)
        positionLine = self.prevNumLines + nLines
        lastLinePos = chunk.rfind('\n', 0, offset)
        if lastLinePos == -1:
            positionColumn = self.prevNumCols + offset
        else:
            positionColumn = offset - (lastLinePos + 1)
        return (positionLine, positionColumn)

    def position(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns (line, col) of the current position in the stream.'
        (line, col) = self._position(self.chunkOffset)
        return (line + 1, col)

    def char(self):
        if False:
            while True:
                i = 10
        ' Read one character from the stream or queue if available. Return\n            EOF when EOF is reached.\n        '
        if self.chunkOffset >= self.chunkSize:
            if not self.readChunk():
                return EOF
        chunkOffset = self.chunkOffset
        char = self.chunk[chunkOffset]
        self.chunkOffset = chunkOffset + 1
        return char

    def readChunk(self, chunkSize=None):
        if False:
            for i in range(10):
                print('nop')
        if chunkSize is None:
            chunkSize = self._defaultChunkSize
        (self.prevNumLines, self.prevNumCols) = self._position(self.chunkSize)
        self.chunk = ''
        self.chunkSize = 0
        self.chunkOffset = 0
        data = self.dataStream.read(chunkSize)
        if self._bufferedCharacter:
            data = self._bufferedCharacter + data
            self._bufferedCharacter = None
        elif not data:
            return False
        if len(data) > 1:
            lastv = ord(data[-1])
            if lastv == 13 or 55296 <= lastv <= 56319:
                self._bufferedCharacter = data[-1]
                data = data[:-1]
        if self.reportCharacterErrors:
            self.reportCharacterErrors(data)
        data = data.replace('\r\n', '\n')
        data = data.replace('\r', '\n')
        self.chunk = data
        self.chunkSize = len(data)
        return True

    def characterErrorsUCS4(self, data):
        if False:
            i = 10
            return i + 15
        for _ in range(len(invalid_unicode_re.findall(data))):
            self.errors.append('invalid-codepoint')

    def characterErrorsUCS2(self, data):
        if False:
            i = 10
            return i + 15
        skip = False
        for match in invalid_unicode_re.finditer(data):
            if skip:
                continue
            codepoint = ord(match.group())
            pos = match.start()
            if _utils.isSurrogatePair(data[pos:pos + 2]):
                char_val = _utils.surrogatePairToCodepoint(data[pos:pos + 2])
                if char_val in non_bmp_invalid_codepoints:
                    self.errors.append('invalid-codepoint')
                skip = True
            elif codepoint >= 55296 and codepoint <= 57343 and (pos == len(data) - 1):
                self.errors.append('invalid-codepoint')
            else:
                skip = False
                self.errors.append('invalid-codepoint')

    def charsUntil(self, characters, opposite=False):
        if False:
            print('Hello World!')
        " Returns a string of characters from the stream up to but not\n        including any character in 'characters' or EOF. 'characters' must be\n        a container that supports the 'in' method and iteration over its\n        characters.\n        "
        try:
            chars = charsUntilRegEx[characters, opposite]
        except KeyError:
            if __debug__:
                for c in characters:
                    assert ord(c) < 128
            regex = ''.join(['\\x%02x' % ord(c) for c in characters])
            if not opposite:
                regex = '^%s' % regex
            chars = charsUntilRegEx[characters, opposite] = re.compile('[%s]+' % regex)
        rv = []
        while True:
            m = chars.match(self.chunk, self.chunkOffset)
            if m is None:
                if self.chunkOffset != self.chunkSize:
                    break
            else:
                end = m.end()
                if end != self.chunkSize:
                    rv.append(self.chunk[self.chunkOffset:end])
                    self.chunkOffset = end
                    break
            rv.append(self.chunk[self.chunkOffset:])
            if not self.readChunk():
                break
        r = ''.join(rv)
        return r

    def unget(self, char):
        if False:
            i = 10
            return i + 15
        if char is not None:
            if self.chunkOffset == 0:
                self.chunk = char + self.chunk
                self.chunkSize += 1
            else:
                self.chunkOffset -= 1
                assert self.chunk[self.chunkOffset] == char

class HTMLBinaryInputStream(HTMLUnicodeInputStream):
    """Provides a unicode stream of characters to the HTMLTokenizer.

    This class takes care of character encoding and removing or replacing
    incorrect byte-sequences and also provides column and line tracking.

    """

    def __init__(self, source, override_encoding=None, transport_encoding=None, same_origin_parent_encoding=None, likely_encoding=None, default_encoding='windows-1252', useChardet=True):
        if False:
            for i in range(10):
                print('nop')
        'Initialises the HTMLInputStream.\n\n        HTMLInputStream(source, [encoding]) -> Normalized stream from source\n        for use by html5lib.\n\n        source can be either a file-object, local filename or a string.\n\n        The optional encoding parameter must be a string that indicates\n        the encoding.  If specified, that encoding will be used,\n        regardless of any BOM or later declaration (such as in a meta\n        element)\n\n        '
        self.rawStream = self.openStream(source)
        HTMLUnicodeInputStream.__init__(self, self.rawStream)
        self.numBytesMeta = 1024
        self.numBytesChardet = 100
        self.override_encoding = override_encoding
        self.transport_encoding = transport_encoding
        self.same_origin_parent_encoding = same_origin_parent_encoding
        self.likely_encoding = likely_encoding
        self.default_encoding = default_encoding
        self.charEncoding = self.determineEncoding(useChardet)
        assert self.charEncoding[0] is not None
        self.reset()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.dataStream = self.charEncoding[0].codec_info.streamreader(self.rawStream, 'replace')
        HTMLUnicodeInputStream.reset(self)

    def openStream(self, source):
        if False:
            i = 10
            return i + 15
        'Produces a file object from source.\n\n        source can be either a file object, local filename or a string.\n\n        '
        if hasattr(source, 'read'):
            stream = source
        else:
            stream = BytesIO(source)
        try:
            stream.seek(stream.tell())
        except:
            stream = BufferedStream(stream)
        return stream

    def determineEncoding(self, chardet=True):
        if False:
            for i in range(10):
                print('nop')
        charEncoding = (self.detectBOM(), 'certain')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (lookupEncoding(self.override_encoding), 'certain')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (lookupEncoding(self.transport_encoding), 'certain')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (self.detectEncodingMeta(), 'tentative')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (lookupEncoding(self.same_origin_parent_encoding), 'tentative')
        if charEncoding[0] is not None and (not charEncoding[0].name.startswith('utf-16')):
            return charEncoding
        charEncoding = (lookupEncoding(self.likely_encoding), 'tentative')
        if charEncoding[0] is not None:
            return charEncoding
        if chardet:
            try:
                from pip._vendor.chardet.universaldetector import UniversalDetector
            except ImportError:
                pass
            else:
                buffers = []
                detector = UniversalDetector()
                while not detector.done:
                    buffer = self.rawStream.read(self.numBytesChardet)
                    assert isinstance(buffer, bytes)
                    if not buffer:
                        break
                    buffers.append(buffer)
                    detector.feed(buffer)
                detector.close()
                encoding = lookupEncoding(detector.result['encoding'])
                self.rawStream.seek(0)
                if encoding is not None:
                    return (encoding, 'tentative')
        charEncoding = (lookupEncoding(self.default_encoding), 'tentative')
        if charEncoding[0] is not None:
            return charEncoding
        return (lookupEncoding('windows-1252'), 'tentative')

    def changeEncoding(self, newEncoding):
        if False:
            for i in range(10):
                print('nop')
        assert self.charEncoding[1] != 'certain'
        newEncoding = lookupEncoding(newEncoding)
        if newEncoding is None:
            return
        if newEncoding.name in ('utf-16be', 'utf-16le'):
            newEncoding = lookupEncoding('utf-8')
            assert newEncoding is not None
        elif newEncoding == self.charEncoding[0]:
            self.charEncoding = (self.charEncoding[0], 'certain')
        else:
            self.rawStream.seek(0)
            self.charEncoding = (newEncoding, 'certain')
            self.reset()
            raise _ReparseException('Encoding changed from %s to %s' % (self.charEncoding[0], newEncoding))

    def detectBOM(self):
        if False:
            i = 10
            return i + 15
        'Attempts to detect at BOM at the start of the stream. If\n        an encoding can be determined from the BOM return the name of the\n        encoding otherwise return None'
        bomDict = {codecs.BOM_UTF8: 'utf-8', codecs.BOM_UTF16_LE: 'utf-16le', codecs.BOM_UTF16_BE: 'utf-16be', codecs.BOM_UTF32_LE: 'utf-32le', codecs.BOM_UTF32_BE: 'utf-32be'}
        string = self.rawStream.read(4)
        assert isinstance(string, bytes)
        encoding = bomDict.get(string[:3])
        seek = 3
        if not encoding:
            encoding = bomDict.get(string)
            seek = 4
            if not encoding:
                encoding = bomDict.get(string[:2])
                seek = 2
        if encoding:
            self.rawStream.seek(seek)
            return lookupEncoding(encoding)
        else:
            self.rawStream.seek(0)
            return None

    def detectEncodingMeta(self):
        if False:
            i = 10
            return i + 15
        'Report the encoding declared by the meta element\n        '
        buffer = self.rawStream.read(self.numBytesMeta)
        assert isinstance(buffer, bytes)
        parser = EncodingParser(buffer)
        self.rawStream.seek(0)
        encoding = parser.getEncoding()
        if encoding is not None and encoding.name in ('utf-16be', 'utf-16le'):
            encoding = lookupEncoding('utf-8')
        return encoding

class EncodingBytes(bytes):
    """String-like object with an associated position and various extra methods
    If the position is ever greater than the string length then an exception is
    raised"""

    def __new__(self, value):
        if False:
            return 10
        assert isinstance(value, bytes)
        return bytes.__new__(self, value.lower())

    def __init__(self, value):
        if False:
            return 10
        self._position = -1

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        p = self._position = self._position + 1
        if p >= len(self):
            raise StopIteration
        elif p < 0:
            raise TypeError
        return self[p:p + 1]

    def next(self):
        if False:
            return 10
        return self.__next__()

    def previous(self):
        if False:
            while True:
                i = 10
        p = self._position
        if p >= len(self):
            raise StopIteration
        elif p < 0:
            raise TypeError
        self._position = p = p - 1
        return self[p:p + 1]

    def setPosition(self, position):
        if False:
            for i in range(10):
                print('nop')
        if self._position >= len(self):
            raise StopIteration
        self._position = position

    def getPosition(self):
        if False:
            for i in range(10):
                print('nop')
        if self._position >= len(self):
            raise StopIteration
        if self._position >= 0:
            return self._position
        else:
            return None
    position = property(getPosition, setPosition)

    def getCurrentByte(self):
        if False:
            return 10
        return self[self.position:self.position + 1]
    currentByte = property(getCurrentByte)

    def skip(self, chars=spaceCharactersBytes):
        if False:
            while True:
                i = 10
        'Skip past a list of characters'
        p = self.position
        while p < len(self):
            c = self[p:p + 1]
            if c not in chars:
                self._position = p
                return c
            p += 1
        self._position = p
        return None

    def skipUntil(self, chars):
        if False:
            return 10
        p = self.position
        while p < len(self):
            c = self[p:p + 1]
            if c in chars:
                self._position = p
                return c
            p += 1
        self._position = p
        return None

    def matchBytes(self, bytes):
        if False:
            for i in range(10):
                print('nop')
        'Look for a sequence of bytes at the start of a string. If the bytes\n        are found return True and advance the position to the byte after the\n        match. Otherwise return False and leave the position alone'
        p = self.position
        data = self[p:p + len(bytes)]
        rv = data.startswith(bytes)
        if rv:
            self.position += len(bytes)
        return rv

    def jumpTo(self, bytes):
        if False:
            for i in range(10):
                print('nop')
        'Look for the next sequence of bytes matching a given sequence. If\n        a match is found advance the position to the last byte of the match'
        newPosition = self[self.position:].find(bytes)
        if newPosition > -1:
            if self._position == -1:
                self._position = 0
            self._position += newPosition + len(bytes) - 1
            return True
        else:
            raise StopIteration

class EncodingParser(object):
    """Mini parser for detecting character encoding from meta elements"""

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        'string - the data to work on for encoding detection'
        self.data = EncodingBytes(data)
        self.encoding = None

    def getEncoding(self):
        if False:
            for i in range(10):
                print('nop')
        methodDispatch = ((b'<!--', self.handleComment), (b'<meta', self.handleMeta), (b'</', self.handlePossibleEndTag), (b'<!', self.handleOther), (b'<?', self.handleOther), (b'<', self.handlePossibleStartTag))
        for _ in self.data:
            keepParsing = True
            for (key, method) in methodDispatch:
                if self.data.matchBytes(key):
                    try:
                        keepParsing = method()
                        break
                    except StopIteration:
                        keepParsing = False
                        break
            if not keepParsing:
                break
        return self.encoding

    def handleComment(self):
        if False:
            while True:
                i = 10
        'Skip over comments'
        return self.data.jumpTo(b'-->')

    def handleMeta(self):
        if False:
            return 10
        if self.data.currentByte not in spaceCharactersBytes:
            return True
        hasPragma = False
        pendingEncoding = None
        while True:
            attr = self.getAttribute()
            if attr is None:
                return True
            elif attr[0] == b'http-equiv':
                hasPragma = attr[1] == b'content-type'
                if hasPragma and pendingEncoding is not None:
                    self.encoding = pendingEncoding
                    return False
            elif attr[0] == b'charset':
                tentativeEncoding = attr[1]
                codec = lookupEncoding(tentativeEncoding)
                if codec is not None:
                    self.encoding = codec
                    return False
            elif attr[0] == b'content':
                contentParser = ContentAttrParser(EncodingBytes(attr[1]))
                tentativeEncoding = contentParser.parse()
                if tentativeEncoding is not None:
                    codec = lookupEncoding(tentativeEncoding)
                    if codec is not None:
                        if hasPragma:
                            self.encoding = codec
                            return False
                        else:
                            pendingEncoding = codec

    def handlePossibleStartTag(self):
        if False:
            i = 10
            return i + 15
        return self.handlePossibleTag(False)

    def handlePossibleEndTag(self):
        if False:
            return 10
        next(self.data)
        return self.handlePossibleTag(True)

    def handlePossibleTag(self, endTag):
        if False:
            return 10
        data = self.data
        if data.currentByte not in asciiLettersBytes:
            if endTag:
                data.previous()
                self.handleOther()
            return True
        c = data.skipUntil(spacesAngleBrackets)
        if c == b'<':
            data.previous()
        else:
            attr = self.getAttribute()
            while attr is not None:
                attr = self.getAttribute()
        return True

    def handleOther(self):
        if False:
            i = 10
            return i + 15
        return self.data.jumpTo(b'>')

    def getAttribute(self):
        if False:
            print('Hello World!')
        'Return a name,value pair for the next attribute in the stream,\n        if one is found, or None'
        data = self.data
        c = data.skip(spaceCharactersBytes | frozenset([b'/']))
        assert c is None or len(c) == 1
        if c in (b'>', None):
            return None
        attrName = []
        attrValue = []
        while True:
            if c == b'=' and attrName:
                break
            elif c in spaceCharactersBytes:
                c = data.skip()
                break
            elif c in (b'/', b'>'):
                return (b''.join(attrName), b'')
            elif c in asciiUppercaseBytes:
                attrName.append(c.lower())
            elif c is None:
                return None
            else:
                attrName.append(c)
            c = next(data)
        if c != b'=':
            data.previous()
            return (b''.join(attrName), b'')
        next(data)
        c = data.skip()
        if c in (b"'", b'"'):
            quoteChar = c
            while True:
                c = next(data)
                if c == quoteChar:
                    next(data)
                    return (b''.join(attrName), b''.join(attrValue))
                elif c in asciiUppercaseBytes:
                    attrValue.append(c.lower())
                else:
                    attrValue.append(c)
        elif c == b'>':
            return (b''.join(attrName), b'')
        elif c in asciiUppercaseBytes:
            attrValue.append(c.lower())
        elif c is None:
            return None
        else:
            attrValue.append(c)
        while True:
            c = next(data)
            if c in spacesAngleBrackets:
                return (b''.join(attrName), b''.join(attrValue))
            elif c in asciiUppercaseBytes:
                attrValue.append(c.lower())
            elif c is None:
                return None
            else:
                attrValue.append(c)

class ContentAttrParser(object):

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(data, bytes)
        self.data = data

    def parse(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.data.jumpTo(b'charset')
            self.data.position += 1
            self.data.skip()
            if not self.data.currentByte == b'=':
                return None
            self.data.position += 1
            self.data.skip()
            if self.data.currentByte in (b'"', b"'"):
                quoteMark = self.data.currentByte
                self.data.position += 1
                oldPosition = self.data.position
                if self.data.jumpTo(quoteMark):
                    return self.data[oldPosition:self.data.position]
                else:
                    return None
            else:
                oldPosition = self.data.position
                try:
                    self.data.skipUntil(spaceCharactersBytes)
                    return self.data[oldPosition:self.data.position]
                except StopIteration:
                    return self.data[oldPosition:]
        except StopIteration:
            return None

def lookupEncoding(encoding):
    if False:
        return 10
    "Return the python codec name corresponding to an encoding or None if the\n    string doesn't correspond to a valid encoding."
    if isinstance(encoding, binary_type):
        try:
            encoding = encoding.decode('ascii')
        except UnicodeDecodeError:
            return None
    if encoding is not None:
        try:
            return webencodings.lookup(encoding)
        except AttributeError:
            return None
    else:
        return None