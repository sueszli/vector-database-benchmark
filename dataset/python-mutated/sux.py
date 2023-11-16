"""
*S*mall, *U*ncomplicated *X*ML.

This is a very simple implementation of XML/HTML as a network
protocol.  It is not at all clever.  Its main features are that it
does not:

  - support namespaces
  - mung mnemonic entity references
  - validate
  - perform *any* external actions (such as fetching URLs or writing files)
    under *any* circumstances
  - has lots and lots of horrible hacks for supporting broken HTML (as an
    option, they're not on by default).
"""
from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
BEGIN_HANDLER = 0
DO_HANDLER = 1
END_HANDLER = 2
identChars = '.-_:'
lenientIdentChars = identChars + ';+#/%~'

def nop(*args, **kw):
    if False:
        i = 10
        return i + 15
    'Do nothing.'

def unionlist(*args):
    if False:
        print('Hello World!')
    l = []
    for x in args:
        l.extend(x)
    d = {x: 1 for x in l}
    return d.keys()

def zipfndict(*args, **kw):
    if False:
        print('Hello World!')
    default = kw.get('default', nop)
    d = {}
    for key in unionlist(*(fndict.keys() for fndict in args)):
        d[key] = tuple((x.get(key, default) for x in args))
    return d

def prefixedMethodClassDict(clazz, prefix):
    if False:
        return 10
    return {name: getattr(clazz, prefix + name) for name in prefixedMethodNames(clazz, prefix)}

def prefixedMethodObjDict(obj, prefix):
    if False:
        for i in range(10):
            print('nop')
    return {name: getattr(obj, prefix + name) for name in prefixedMethodNames(obj.__class__, prefix)}

class ParseError(Exception):

    def __init__(self, filename, line, col, message):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        self.line = line
        self.col = col
        self.message = message

    def __str__(self) -> str:
        if False:
            return 10
        return f'{self.filename}:{self.line}:{self.col}: {self.message}'

class XMLParser(Protocol):
    state = None
    encodings = None
    filename = '<xml />'
    beExtremelyLenient = 0
    _prepend = None
    _leadingBodyData = None

    def connectionMade(self):
        if False:
            return 10
        self.lineno = 1
        self.colno = 0
        self.encodings = []

    def saveMark(self):
        if False:
            while True:
                i = 10
        'Get the line number and column of the last character parsed'
        return (self.lineno, self.colno)

    def _parseError(self, message):
        if False:
            i = 10
            return i + 15
        raise ParseError(*(self.filename,) + self.saveMark() + (message,))

    def _buildStateTable(self):
        if False:
            print('Hello World!')
        'Return a dictionary of begin, do, end state function tuples'
        stateTable = getattr(self.__class__, '__stateTable', None)
        if stateTable is None:
            stateTable = self.__class__.__stateTable = zipfndict(*(prefixedMethodObjDict(self, prefix) for prefix in ('begin_', 'do_', 'end_')))
        return stateTable

    def _decode(self, data):
        if False:
            return 10
        if 'UTF-16' in self.encodings or 'UCS-2' in self.encodings:
            assert not len(data) & 1, 'UTF-16 must come in pairs for now'
        if self._prepend:
            data = self._prepend + data
        for encoding in self.encodings:
            data = str(data, encoding)
        return data

    def maybeBodyData(self):
        if False:
            for i in range(10):
                print('nop')
        if self.endtag:
            return 'bodydata'
        if self.tagName == 'script' and 'src' not in self.tagAttributes:
            self.begin_bodydata(None)
            return 'waitforendscript'
        return 'bodydata'

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        stateTable = self._buildStateTable()
        if not self.state:
            if data.startswith((b'\xff\xfe', b'\xfe\xff')):
                self._prepend = data[0:2]
                self.encodings.append('UTF-16')
                data = data[2:]
            self.state = 'begin'
        if self.encodings:
            data = self._decode(data)
        else:
            data = data.decode('utf-8')
        (lineno, colno) = (self.lineno, self.colno)
        curState = self.state
        _saveMark = self.saveMark

        def saveMark():
            if False:
                return 10
            return (lineno, colno)
        self.saveMark = saveMark
        (beginFn, doFn, endFn) = stateTable[curState]
        try:
            for byte in data:
                if byte == '\n':
                    lineno += 1
                    colno = 0
                else:
                    colno += 1
                newState = doFn(byte)
                if newState is not None and newState != curState:
                    endFn()
                    curState = newState
                    (beginFn, doFn, endFn) = stateTable[curState]
                    beginFn(byte)
        finally:
            self.saveMark = _saveMark
            (self.lineno, self.colno) = (lineno, colno)
        self.state = curState

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        '\n        End the last state we were in.\n        '
        stateTable = self._buildStateTable()
        stateTable[self.state][END_HANDLER]()

    def do_begin(self, byte):
        if False:
            print('Hello World!')
        if byte.isspace():
            return
        if byte != '<':
            if self.beExtremelyLenient:
                self._leadingBodyData = byte
                return 'bodydata'
            self._parseError(f"First char of document [{byte!r}] wasn't <")
        return 'tagstart'

    def begin_comment(self, byte):
        if False:
            return 10
        self.commentbuf = ''

    def do_comment(self, byte):
        if False:
            for i in range(10):
                print('nop')
        self.commentbuf += byte
        if self.commentbuf.endswith('-->'):
            self.gotComment(self.commentbuf[:-3])
            return 'bodydata'

    def begin_tagstart(self, byte):
        if False:
            i = 10
            return i + 15
        self.tagName = ''
        self.tagAttributes = {}
        self.termtag = 0
        self.endtag = 0

    def do_tagstart(self, byte):
        if False:
            while True:
                i = 10
        if byte.isalnum() or byte in identChars:
            self.tagName += byte
            if self.tagName == '!--':
                return 'comment'
        elif byte.isspace():
            if self.tagName:
                if self.endtag:
                    return 'waitforgt'
                return 'attrs'
            else:
                self._parseError('Whitespace before tag-name')
        elif byte == '>':
            if self.endtag:
                self.gotTagEnd(self.tagName)
                return 'bodydata'
            else:
                self.gotTagStart(self.tagName, {})
                return not self.beExtremelyLenient and 'bodydata' or self.maybeBodyData()
        elif byte == '/':
            if self.tagName:
                return 'afterslash'
            else:
                self.endtag = 1
        elif byte in '!?':
            if self.tagName:
                if not self.beExtremelyLenient:
                    self._parseError('Invalid character in tag-name')
            else:
                self.tagName += byte
                self.termtag = 1
        elif byte == '[':
            if self.tagName == '!':
                return 'expectcdata'
            else:
                self._parseError("Invalid '[' in tag-name")
        else:
            if self.beExtremelyLenient:
                self.bodydata = '<'
                return 'unentity'
            self._parseError('Invalid tag character: %r' % byte)

    def begin_unentity(self, byte):
        if False:
            i = 10
            return i + 15
        self.bodydata += byte

    def do_unentity(self, byte):
        if False:
            i = 10
            return i + 15
        self.bodydata += byte
        return 'bodydata'

    def end_unentity(self):
        if False:
            print('Hello World!')
        self.gotText(self.bodydata)

    def begin_expectcdata(self, byte):
        if False:
            i = 10
            return i + 15
        self.cdatabuf = byte

    def do_expectcdata(self, byte):
        if False:
            print('Hello World!')
        self.cdatabuf += byte
        cdb = self.cdatabuf
        cd = '[CDATA['
        if len(cd) > len(cdb):
            if cd.startswith(cdb):
                return
            elif self.beExtremelyLenient:
                return 'waitforgt'
            else:
                self._parseError('Mal-formed CDATA header')
        if cd == cdb:
            self.cdatabuf = ''
            return 'cdata'
        self._parseError('Mal-formed CDATA header')

    def do_cdata(self, byte):
        if False:
            for i in range(10):
                print('nop')
        self.cdatabuf += byte
        if self.cdatabuf.endswith(']]>'):
            self.cdatabuf = self.cdatabuf[:-3]
            return 'bodydata'

    def end_cdata(self):
        if False:
            for i in range(10):
                print('nop')
        self.gotCData(self.cdatabuf)
        self.cdatabuf = ''

    def do_attrs(self, byte):
        if False:
            print('Hello World!')
        if byte.isalnum() or byte in identChars:
            if self.tagName == '!DOCTYPE':
                return 'doctype'
            if self.tagName[0] in '!?':
                return 'waitforgt'
            return 'attrname'
        elif byte.isspace():
            return
        elif byte == '>':
            self.gotTagStart(self.tagName, self.tagAttributes)
            return not self.beExtremelyLenient and 'bodydata' or self.maybeBodyData()
        elif byte == '/':
            return 'afterslash'
        elif self.beExtremelyLenient:
            return
        self._parseError('Unexpected character: %r' % byte)

    def begin_doctype(self, byte):
        if False:
            print('Hello World!')
        self.doctype = byte

    def do_doctype(self, byte):
        if False:
            print('Hello World!')
        if byte == '>':
            return 'bodydata'
        self.doctype += byte

    def end_doctype(self):
        if False:
            while True:
                i = 10
        self.gotDoctype(self.doctype)
        self.doctype = None

    def do_waitforgt(self, byte):
        if False:
            return 10
        if byte == '>':
            if self.endtag or not self.beExtremelyLenient:
                return 'bodydata'
            return self.maybeBodyData()

    def begin_attrname(self, byte):
        if False:
            for i in range(10):
                print('nop')
        self.attrname = byte
        self._attrname_termtag = 0

    def do_attrname(self, byte):
        if False:
            return 10
        if byte.isalnum() or byte in identChars:
            self.attrname += byte
            return
        elif byte == '=':
            return 'beforeattrval'
        elif byte.isspace():
            return 'beforeeq'
        elif self.beExtremelyLenient:
            if byte in '"\'':
                return 'attrval'
            if byte in lenientIdentChars or byte.isalnum():
                self.attrname += byte
                return
            if byte == '/':
                self._attrname_termtag = 1
                return
            if byte == '>':
                self.attrval = 'True'
                self.tagAttributes[self.attrname] = self.attrval
                self.gotTagStart(self.tagName, self.tagAttributes)
                if self._attrname_termtag:
                    self.gotTagEnd(self.tagName)
                    return 'bodydata'
                return self.maybeBodyData()
            return
        self._parseError(f'Invalid attribute name: {self.attrname!r} {byte!r}')

    def do_beforeattrval(self, byte):
        if False:
            print('Hello World!')
        if byte in '"\'':
            return 'attrval'
        elif byte.isspace():
            return
        elif self.beExtremelyLenient:
            if byte in lenientIdentChars or byte.isalnum():
                return 'messyattr'
            if byte == '>':
                self.attrval = 'True'
                self.tagAttributes[self.attrname] = self.attrval
                self.gotTagStart(self.tagName, self.tagAttributes)
                return self.maybeBodyData()
            if byte == '\\':
                return
        self._parseError('Invalid initial attribute value: %r; Attribute values must be quoted.' % byte)
    attrname = ''
    attrval = ''

    def begin_beforeeq(self, byte):
        if False:
            print('Hello World!')
        self._beforeeq_termtag = 0

    def do_beforeeq(self, byte):
        if False:
            for i in range(10):
                print('nop')
        if byte == '=':
            return 'beforeattrval'
        elif byte.isspace():
            return
        elif self.beExtremelyLenient:
            if byte.isalnum() or byte in identChars:
                self.attrval = 'True'
                self.tagAttributes[self.attrname] = self.attrval
                return 'attrname'
            elif byte == '>':
                self.attrval = 'True'
                self.tagAttributes[self.attrname] = self.attrval
                self.gotTagStart(self.tagName, self.tagAttributes)
                if self._beforeeq_termtag:
                    self.gotTagEnd(self.tagName)
                    return 'bodydata'
                return self.maybeBodyData()
            elif byte == '/':
                self._beforeeq_termtag = 1
                return
        self._parseError('Invalid attribute')

    def begin_attrval(self, byte):
        if False:
            while True:
                i = 10
        self.quotetype = byte
        self.attrval = ''

    def do_attrval(self, byte):
        if False:
            for i in range(10):
                print('nop')
        if byte == self.quotetype:
            return 'attrs'
        self.attrval += byte

    def end_attrval(self):
        if False:
            return 10
        self.tagAttributes[self.attrname] = self.attrval
        self.attrname = self.attrval = ''

    def begin_messyattr(self, byte):
        if False:
            print('Hello World!')
        self.attrval = byte

    def do_messyattr(self, byte):
        if False:
            return 10
        if byte.isspace():
            return 'attrs'
        elif byte == '>':
            endTag = 0
            if self.attrval.endswith('/'):
                endTag = 1
                self.attrval = self.attrval[:-1]
            self.tagAttributes[self.attrname] = self.attrval
            self.gotTagStart(self.tagName, self.tagAttributes)
            if endTag:
                self.gotTagEnd(self.tagName)
                return 'bodydata'
            return self.maybeBodyData()
        else:
            self.attrval += byte

    def end_messyattr(self):
        if False:
            while True:
                i = 10
        if self.attrval:
            self.tagAttributes[self.attrname] = self.attrval

    def begin_afterslash(self, byte):
        if False:
            i = 10
            return i + 15
        self._after_slash_closed = 0

    def do_afterslash(self, byte):
        if False:
            while True:
                i = 10
        if self._after_slash_closed:
            self._parseError('Mal-formed')
        if byte != '>':
            if self.beExtremelyLenient:
                return
            else:
                self._parseError("No data allowed after '/'")
        self._after_slash_closed = 1
        self.gotTagStart(self.tagName, self.tagAttributes)
        self.gotTagEnd(self.tagName)
        return 'bodydata'

    def begin_bodydata(self, byte):
        if False:
            for i in range(10):
                print('nop')
        if self._leadingBodyData:
            self.bodydata = self._leadingBodyData
            del self._leadingBodyData
        else:
            self.bodydata = ''

    def do_bodydata(self, byte):
        if False:
            return 10
        if byte == '<':
            return 'tagstart'
        if byte == '&':
            return 'entityref'
        self.bodydata += byte

    def end_bodydata(self):
        if False:
            while True:
                i = 10
        self.gotText(self.bodydata)
        self.bodydata = ''

    def do_waitforendscript(self, byte):
        if False:
            print('Hello World!')
        if byte == '<':
            return 'waitscriptendtag'
        self.bodydata += byte

    def begin_waitscriptendtag(self, byte):
        if False:
            while True:
                i = 10
        self.temptagdata = ''
        self.tagName = ''
        self.endtag = 0

    def do_waitscriptendtag(self, byte):
        if False:
            print('Hello World!')
        self.temptagdata += byte
        if byte == '/':
            self.endtag = True
        elif not self.endtag:
            self.bodydata += '<' + self.temptagdata
            return 'waitforendscript'
        elif byte.isalnum() or byte in identChars:
            self.tagName += byte
            if not 'script'.startswith(self.tagName):
                self.bodydata += '<' + self.temptagdata
                return 'waitforendscript'
            elif self.tagName == 'script':
                self.gotText(self.bodydata)
                self.gotTagEnd(self.tagName)
                return 'waitforgt'
        elif byte.isspace():
            return 'waitscriptendtag'
        else:
            self.bodydata += '<' + self.temptagdata
            return 'waitforendscript'

    def begin_entityref(self, byte):
        if False:
            while True:
                i = 10
        self.erefbuf = ''
        self.erefextra = ''

    def do_entityref(self, byte):
        if False:
            i = 10
            return i + 15
        if byte.isspace() or byte == '<':
            if self.beExtremelyLenient:
                if self.erefbuf and self.erefbuf != 'amp':
                    self.erefextra = self.erefbuf
                self.erefbuf = 'amp'
                if byte == '<':
                    return 'tagstart'
                else:
                    self.erefextra += byte
                    return 'spacebodydata'
            self._parseError('Bad entity reference')
        elif byte != ';':
            self.erefbuf += byte
        else:
            return 'bodydata'

    def end_entityref(self):
        if False:
            return 10
        self.gotEntityReference(self.erefbuf)

    def begin_spacebodydata(self, byte):
        if False:
            for i in range(10):
                print('nop')
        self.bodydata = self.erefextra
        self.erefextra = None
    do_spacebodydata = do_bodydata
    end_spacebodydata = end_bodydata

    def gotTagStart(self, name, attributes):
        if False:
            return 10
        'Encountered an opening tag.\n\n        Default behaviour is to print.'
        print('begin', name, attributes)

    def gotText(self, data):
        if False:
            i = 10
            return i + 15
        'Encountered text\n\n        Default behaviour is to print.'
        print('text:', repr(data))

    def gotEntityReference(self, entityRef):
        if False:
            while True:
                i = 10
        'Encountered mnemonic entity reference\n\n        Default behaviour is to print.'
        print('entityRef: &%s;' % entityRef)

    def gotComment(self, comment):
        if False:
            while True:
                i = 10
        'Encountered comment.\n\n        Default behaviour is to ignore.'
        pass

    def gotCData(self, cdata):
        if False:
            return 10
        'Encountered CDATA\n\n        Default behaviour is to call the gotText method'
        self.gotText(cdata)

    def gotDoctype(self, doctype):
        if False:
            print('Hello World!')
        "Encountered DOCTYPE\n\n        This is really grotty: it basically just gives you everything between\n        '<!DOCTYPE' and '>' as an argument.\n        "
        print('!DOCTYPE', repr(doctype))

    def gotTagEnd(self, name):
        if False:
            i = 10
            return i + 15
        'Encountered closing tag\n\n        Default behaviour is to print.'
        print('end', name)