"""A utility class for a code container.

A code container is a class which holds source code for a debugger.  It knows how
to color the text, and also how to translate lines into offsets, and back.
"""
import os
import sys
import tokenize
import win32api
import winerror
from win32com.axdebug import axdebug, contexts
from win32com.axdebug.util import _wrap
from win32com.server.exception import Exception
_keywords = {}
for name in '\n and assert break class continue def del elif else except exec\n finally for from global if import in is lambda not\n or pass print raise return try while\n '.split():
    _keywords[name] = 1

class SourceCodeContainer:

    def __init__(self, text, fileName='<Remove Me!>', sourceContext=0, startLineNumber=0, site=None, debugDocument=None):
        if False:
            for i in range(10):
                print('nop')
        self.sourceContext = sourceContext
        self.text = text
        if text:
            self._buildlines()
        self.nextLineNo = 0
        self.fileName = fileName
        self.codeContexts = {}
        self.site = site
        self.startLineNumber = startLineNumber
        self.debugDocument = debugDocument

    def _Close(self):
        if False:
            print('Hello World!')
        self.text = self.lines = self.lineOffsets = None
        self.codeContexts = None
        self.debugDocument = None
        self.site = None
        self.sourceContext = None

    def GetText(self):
        if False:
            print('Hello World!')
        return self.text

    def GetName(self, dnt):
        if False:
            return 10
        assert 0, 'You must subclass this'

    def GetFileName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fileName

    def GetPositionOfLine(self, cLineNumber):
        if False:
            while True:
                i = 10
        self.GetText()
        try:
            return self.lineOffsets[cLineNumber]
        except IndexError:
            raise Exception(scode=winerror.S_FALSE)

    def GetLineOfPosition(self, charPos):
        if False:
            for i in range(10):
                print('nop')
        self.GetText()
        lastOffset = 0
        lineNo = 0
        for lineOffset in self.lineOffsets[1:]:
            if lineOffset > charPos:
                break
            lastOffset = lineOffset
            lineNo = lineNo + 1
        else:
            raise Exception(scode=winerror.S_FALSE)
        return (lineNo, charPos - lastOffset)

    def GetNextLine(self):
        if False:
            print('Hello World!')
        if self.nextLineNo >= len(self.lines):
            self.nextLineNo = 0
            return ''
        rc = self.lines[self.nextLineNo]
        self.nextLineNo = self.nextLineNo + 1
        return rc

    def GetLine(self, num):
        if False:
            i = 10
            return i + 15
        self.GetText()
        return self.lines[num]

    def GetNumChars(self):
        if False:
            return 10
        return len(self.GetText())

    def GetNumLines(self):
        if False:
            for i in range(10):
                print('nop')
        self.GetText()
        return len(self.lines)

    def _buildline(self, pos):
        if False:
            i = 10
            return i + 15
        i = self.text.find('\n', pos)
        if i < 0:
            newpos = len(self.text)
        else:
            newpos = i + 1
        r = self.text[pos:newpos]
        return (r, newpos)

    def _buildlines(self):
        if False:
            print('Hello World!')
        self.lines = []
        self.lineOffsets = [0]
        (line, pos) = self._buildline(0)
        while line:
            self.lines.append(line)
            self.lineOffsets.append(pos)
            (line, pos) = self._buildline(pos)

    def _ProcessToken(self, type, token, spos, epos, line):
        if False:
            i = 10
            return i + 15
        (srow, scol) = spos
        (erow, ecol) = epos
        self.GetText()
        linenum = srow - 1
        realCharPos = self.lineOffsets[linenum] + scol
        numskipped = realCharPos - self.lastPos
        if numskipped == 0:
            pass
        elif numskipped == 1:
            self.attrs.append(axdebug.SOURCETEXT_ATTR_COMMENT)
        else:
            self.attrs.append((axdebug.SOURCETEXT_ATTR_COMMENT, numskipped))
        kwSize = len(token)
        self.lastPos = realCharPos + kwSize
        attr = 0
        if type == tokenize.NAME:
            if token in _keywords:
                attr = axdebug.SOURCETEXT_ATTR_KEYWORD
        elif type == tokenize.STRING:
            attr = axdebug.SOURCETEXT_ATTR_STRING
        elif type == tokenize.NUMBER:
            attr = axdebug.SOURCETEXT_ATTR_NUMBER
        elif type == tokenize.OP:
            attr = axdebug.SOURCETEXT_ATTR_OPERATOR
        elif type == tokenize.COMMENT:
            attr = axdebug.SOURCETEXT_ATTR_COMMENT
        if kwSize == 0:
            pass
        elif kwSize == 1:
            self.attrs.append(attr)
        else:
            self.attrs.append((attr, kwSize))

    def GetSyntaxColorAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.lastPos = 0
        self.attrs = []
        try:
            tokenize.tokenize(self.GetNextLine, self._ProcessToken)
        except tokenize.TokenError:
            pass
        numAtEnd = len(self.GetText()) - self.lastPos
        if numAtEnd:
            self.attrs.append((axdebug.SOURCETEXT_ATTR_COMMENT, numAtEnd))
        return self.attrs

    def _MakeDebugCodeContext(self, lineNo, charPos, len):
        if False:
            return 10
        return _wrap(contexts.DebugCodeContext(lineNo, charPos, len, self, self.site), axdebug.IID_IDebugCodeContext)

    def _MakeContextAtPosition(self, charPos):
        if False:
            while True:
                i = 10
        (lineNo, offset) = self.GetLineOfPosition(charPos)
        try:
            endPos = self.GetPositionOfLine(lineNo + 1)
        except:
            endPos = charPos
        codecontext = self._MakeDebugCodeContext(lineNo, charPos, endPos - charPos)
        return codecontext

    def GetCodeContextAtPosition(self, charPos):
        if False:
            for i in range(10):
                print('nop')
        (lineNo, offset) = self.GetLineOfPosition(charPos)
        charPos = self.GetPositionOfLine(lineNo)
        try:
            cc = self.codeContexts[charPos]
        except KeyError:
            cc = self._MakeContextAtPosition(charPos)
            self.codeContexts[charPos] = cc
        return cc

class SourceModuleContainer(SourceCodeContainer):

    def __init__(self, module):
        if False:
            i = 10
            return i + 15
        self.module = module
        if hasattr(module, '__file__'):
            fname = self.module.__file__
            if fname[-1] in ['O', 'o', 'C', 'c', 'S', 's']:
                fname = fname[:-1]
            try:
                fname = win32api.GetFullPathName(fname)
            except win32api.error:
                pass
        elif module.__name__ == '__main__' and len(sys.argv) > 0:
            fname = sys.argv[0]
        else:
            fname = '<Unknown!>'
        SourceCodeContainer.__init__(self, None, fname)

    def GetText(self):
        if False:
            for i in range(10):
                print('nop')
        if self.text is None:
            fname = self.GetFileName()
            if fname:
                try:
                    self.text = open(fname, 'r').read()
                except OSError as details:
                    self.text = f'# Exception opening file\n# {repr(details)}'
            else:
                self.text = f"# No file available for module '{self.module}'"
            self._buildlines()
        return self.text

    def GetName(self, dnt):
        if False:
            print('Hello World!')
        name = self.module.__name__
        try:
            fname = win32api.GetFullPathName(self.module.__file__)
        except win32api.error:
            fname = self.module.__file__
        except AttributeError:
            fname = name
        if dnt == axdebug.DOCUMENTNAMETYPE_APPNODE:
            return name.split('.')[-1]
        elif dnt == axdebug.DOCUMENTNAMETYPE_TITLE:
            return fname
        elif dnt == axdebug.DOCUMENTNAMETYPE_FILE_TAIL:
            return os.path.split(fname)[1]
        elif dnt == axdebug.DOCUMENTNAMETYPE_URL:
            return f'file:{fname}'
        else:
            raise Exception(scode=winerror.E_UNEXPECTED)
if __name__ == '__main__':
    from Test import ttest
    sc = SourceModuleContainer(ttest)
    attrs = sc.GetSyntaxColorAttributes()
    attrlen = 0
    for attr in attrs:
        if isinstance(attr, tuple):
            attrlen = attrlen + attr[1]
        else:
            attrlen = attrlen + 1
    text = sc.GetText()
    if attrlen != len(text):
        print(f'Lengths dont match!!! ({attrlen}/{len(text)})')
    print('GetLineOfPos=', sc.GetLineOfPosition(0))
    print('GetLineOfPos=', sc.GetLineOfPosition(4))
    print('GetLineOfPos=', sc.GetLineOfPosition(10))