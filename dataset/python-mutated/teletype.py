"""
Class for handling whitespace properly in OpenDocument.

While it is possible to use getTextContent() and setTextContent()
to extract or create ODF content, these won't extract or create
the appropriate <text:s>, <text:tab>, or <text:line-break>
elements.  This module takes care of that problem.
"""
from .element import Node
from .text import S, LineBreak, Tab

class WhitespaceText:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.textBuffer = []
        self.spaceCount = 0

    def addTextToElement(self, odfElement, s):
        if False:
            for i in range(10):
                print('nop')
        " Process an input string, inserting\n            <text:tab> elements for '\t',\n            <text:line-break> elements for '\n', and\n            <text:s> elements for runs of more than one blank.\n            These will be added to the given element.\n        "
        i = 0
        ch = ' '
        while i < len(s):
            ch = s[i]
            if ch == '\t':
                self._emitTextBuffer(odfElement)
                odfElement.addElement(Tab())
                i += 1
            elif ch == '\n':
                self._emitTextBuffer(odfElement)
                odfElement.addElement(LineBreak())
                i += 1
            elif ch == ' ':
                self.textBuffer.append(' ')
                i += 1
                self.spaceCount = 0
                while i < len(s) and s[i] == ' ':
                    self.spaceCount += 1
                    i += 1
                if self.spaceCount > 0:
                    self._emitTextBuffer(odfElement)
                    self._emitSpaces(odfElement)
            else:
                self.textBuffer.append(ch)
                i += 1
        self._emitTextBuffer(odfElement)

    def _emitTextBuffer(self, odfElement):
        if False:
            return 10
        ' Creates a Text Node whose contents are the current textBuffer.\n            Side effect: clears the text buffer.\n        '
        if len(self.textBuffer) > 0:
            odfElement.addText(''.join(self.textBuffer))
        self.textBuffer = []

    def _emitSpaces(self, odfElement):
        if False:
            return 10
        ' Creates a <text:s> element for the current spaceCount.\n            Side effect: sets spaceCount back to zero\n        '
        if self.spaceCount > 0:
            spaceElement = S(c=self.spaceCount)
            odfElement.addElement(spaceElement)
        self.spaceCount = 0

def addTextToElement(odfElement, s):
    if False:
        while True:
            i = 10
    wst = WhitespaceText()
    wst.addTextToElement(odfElement, s)

def extractText(odfElement):
    if False:
        i = 10
        return i + 15
    ' Extract text content from an Element, with whitespace represented\n        properly. Returns the text, with tabs, spaces, and newlines\n        correctly evaluated. This method recursively descends through the\n        children of the given element, accumulating text and "unwrapping"\n        <text:s>, <text:tab>, and <text:line-break> elements along the way.\n    '
    result = []
    if len(odfElement.childNodes) != 0:
        for child in odfElement.childNodes:
            if child.nodeType == Node.TEXT_NODE:
                result.append(child.data)
            elif child.nodeType == Node.ELEMENT_NODE:
                subElement = child
                tagName = subElement.qname
                if tagName == ('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'line-break'):
                    result.append('\n')
                elif tagName == ('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 'tab'):
                    result.append('\t')
                elif tagName == ('urn:oasis:names:tc:opendocument:xmlns:text:1.0', 's'):
                    c = subElement.getAttribute('c')
                    if c:
                        spaceCount = int(c)
                    else:
                        spaceCount = 1
                    result.append(' ' * spaceCount)
                else:
                    result.append(extractText(subElement))
    return ''.join(result)