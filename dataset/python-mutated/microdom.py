"""
Micro Document Object Model: a partial DOM implementation with SUX.

This is an implementation of what we consider to be the useful subset of the
DOM.  The chief advantage of this library is that, not being burdened with
standards compliance, it can remain very stable between versions.  We can also
implement utility 'pythonic' ways to access and mutate the XML tree.

Since this has not subjected to a serious trial by fire, it is not recommended
to use this outside of Twisted applications.  However, it seems to work just
fine for the documentation generator, which parses a fairly representative
sample of XML.

Microdom mainly focuses on working with HTML and XHTML.

This module is now deprecated.
"""
from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
warningString = 'twisted.web.microdom was deprecated at {}'.format(getVersionString(Version('Twisted', 23, 10, 0)))
warnings.warn(warningString, DeprecationWarning, stacklevel=3)

def getElementsByTagName(iNode, name):
    if False:
        while True:
            i = 10
    '\n    Return a list of all child elements of C{iNode} with a name matching\n    C{name}.\n\n    Note that this implementation does not conform to the DOM Level 1 Core\n    specification because it may return C{iNode}.\n\n    @param iNode: An element at which to begin searching.  If C{iNode} has a\n        name matching C{name}, it will be included in the result.\n\n    @param name: A C{str} giving the name of the elements to return.\n\n    @return: A C{list} of direct or indirect child elements of C{iNode} with\n        the name C{name}.  This may include C{iNode}.\n    '
    matches = []
    matches_append = matches.append
    slice = [iNode]
    while len(slice) > 0:
        c = slice.pop(0)
        if c.nodeName == name:
            matches_append(c)
        slice[:0] = c.childNodes
    return matches

def getElementsByTagNameNoCase(iNode, name):
    if False:
        while True:
            i = 10
    name = name.lower()
    matches = []
    matches_append = matches.append
    slice = [iNode]
    while len(slice) > 0:
        c = slice.pop(0)
        if c.nodeName.lower() == name:
            matches_append(c)
        slice[:0] = c.childNodes
    return matches

def _streamWriteWrapper(stream):
    if False:
        for i in range(10):
            print('nop')
    if ioType(stream) == bytes:

        def w(s):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(s, str):
                s = s.encode('utf-8')
            stream.write(s)
    else:

        def w(s):
            if False:
                i = 10
                return i + 15
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            stream.write(s)
    return w
HTML_ESCAPE_CHARS = (('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;'), ('"', '&quot;'))
REV_HTML_ESCAPE_CHARS = list(HTML_ESCAPE_CHARS)
REV_HTML_ESCAPE_CHARS.reverse()
XML_ESCAPE_CHARS = HTML_ESCAPE_CHARS + (("'", '&apos;'),)
REV_XML_ESCAPE_CHARS = list(XML_ESCAPE_CHARS)
REV_XML_ESCAPE_CHARS.reverse()

def unescape(text, chars=REV_HTML_ESCAPE_CHARS):
    if False:
        return 10
    "\n    Perform the exact opposite of 'escape'.\n    "
    for (s, h) in chars:
        text = text.replace(h, s)
    return text

def escape(text, chars=HTML_ESCAPE_CHARS):
    if False:
        while True:
            i = 10
    '\n    Escape a few XML special chars with XML entities.\n    '
    for (s, h) in chars:
        text = text.replace(s, h)
    return text

class MismatchedTags(Exception):

    def __init__(self, filename, expect, got, endLine, endCol, begLine, begCol):
        if False:
            return 10
        (self.filename, self.expect, self.got, self.begLine, self.begCol, self.endLine, self.endCol) = (filename, expect, got, begLine, begCol, endLine, endCol)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'expected </%s>, got </%s> line: %s col: %s, began line: %s col: %s' % (self.expect, self.got, self.endLine, self.endCol, self.begLine, self.begCol)

class Node:
    nodeName = 'Node'

    def __init__(self, parentNode=None):
        if False:
            return 10
        self.parentNode = parentNode
        self.childNodes = []

    def isEqualToNode(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compare this node to C{other}.  If the nodes have the same number of\n        children and corresponding children are equal to each other, return\n        C{True}, otherwise return C{False}.\n\n        @type other: L{Node}\n        @rtype: C{bool}\n        '
        if len(self.childNodes) != len(other.childNodes):
            return False
        for (a, b) in zip(self.childNodes, other.childNodes):
            if not a.isEqualToNode(b):
                return False
        return True

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            return 10
        raise NotImplementedError()

    def toxml(self, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            return 10
        s = StringIO()
        self.writexml(s, indent, addindent, newl, strip, nsprefixes, namespace)
        rv = s.getvalue()
        return rv

    def writeprettyxml(self, stream, indent='', addindent=' ', newl='\n', strip=0):
        if False:
            return 10
        return self.writexml(stream, indent, addindent, newl, strip)

    def toprettyxml(self, indent='', addindent=' ', newl='\n', strip=0):
        if False:
            while True:
                i = 10
        return self.toxml(indent, addindent, newl, strip)

    def cloneNode(self, deep=0, parent=None):
        if False:
            return 10
        raise NotImplementedError()

    def hasChildNodes(self):
        if False:
            for i in range(10):
                print('nop')
        if self.childNodes:
            return 1
        else:
            return 0

    def appendChild(self, child):
        if False:
            return 10
        '\n        Make the given L{Node} the last child of this node.\n\n        @param child: The L{Node} which will become a child of this node.\n\n        @raise TypeError: If C{child} is not a C{Node} instance.\n        '
        if not isinstance(child, Node):
            raise TypeError('expected Node instance')
        self.childNodes.append(child)
        child.parentNode = self

    def insertBefore(self, new, ref):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make the given L{Node} C{new} a child of this node which comes before\n        the L{Node} C{ref}.\n\n        @param new: A L{Node} which will become a child of this node.\n\n        @param ref: A L{Node} which is already a child of this node which\n            C{new} will be inserted before.\n\n        @raise TypeError: If C{new} or C{ref} is not a C{Node} instance.\n\n        @return: C{new}\n        '
        if not isinstance(new, Node) or not isinstance(ref, Node):
            raise TypeError('expected Node instance')
        i = self.childNodes.index(ref)
        new.parentNode = self
        self.childNodes.insert(i, new)
        return new

    def removeChild(self, child):
        if False:
            print('Hello World!')
        "\n        Remove the given L{Node} from this node's children.\n\n        @param child: A L{Node} which is a child of this node which will no\n            longer be a child of this node after this method is called.\n\n        @raise TypeError: If C{child} is not a C{Node} instance.\n\n        @return: C{child}\n        "
        if not isinstance(child, Node):
            raise TypeError('expected Node instance')
        if child in self.childNodes:
            self.childNodes.remove(child)
            child.parentNode = None
        return child

    def replaceChild(self, newChild, oldChild):
        if False:
            i = 10
            return i + 15
        '\n        Replace a L{Node} which is already a child of this node with a\n        different node.\n\n        @param newChild: A L{Node} which will be made a child of this node.\n\n        @param oldChild: A L{Node} which is a child of this node which will\n            give up its position to C{newChild}.\n\n        @raise TypeError: If C{newChild} or C{oldChild} is not a C{Node}\n            instance.\n\n        @raise ValueError: If C{oldChild} is not a child of this C{Node}.\n        '
        if not isinstance(newChild, Node) or not isinstance(oldChild, Node):
            raise TypeError('expected Node instance')
        if oldChild.parentNode is not self:
            raise ValueError('oldChild is not a child of this node')
        self.childNodes[self.childNodes.index(oldChild)] = newChild
        oldChild.parentNode = None
        newChild.parentNode = self

    def lastChild(self):
        if False:
            for i in range(10):
                print('nop')
        return self.childNodes[-1]

    def firstChild(self):
        if False:
            i = 10
            return i + 15
        if len(self.childNodes):
            return self.childNodes[0]
        return None

class Document(Node):

    def __init__(self, documentElement=None):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self)
        if documentElement:
            self.appendChild(documentElement)

    def cloneNode(self, deep=0, parent=None):
        if False:
            i = 10
            return i + 15
        d = Document()
        d.doctype = self.doctype
        if deep:
            newEl = self.documentElement.cloneNode(1, self)
        else:
            newEl = self.documentElement
        d.appendChild(newEl)
        return d
    doctype: None | str = None

    def isEqualToDocument(self, n):
        if False:
            for i in range(10):
                print('nop')
        return self.doctype == n.doctype and Node.isEqualToNode(self, n)
    isEqualToNode = isEqualToDocument

    @property
    def documentElement(self):
        if False:
            return 10
        return self.childNodes[0]

    def appendChild(self, child):
        if False:
            i = 10
            return i + 15
        "\n        Make the given L{Node} the I{document element} of this L{Document}.\n\n        @param child: The L{Node} to make into this L{Document}'s document\n            element.\n\n        @raise ValueError: If this document already has a document element.\n        "
        if self.childNodes:
            raise ValueError('Only one element per document.')
        Node.appendChild(self, child)

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            print('Hello World!')
        w = _streamWriteWrapper(stream)
        w('<?xml version="1.0"?>' + newl)
        if self.doctype:
            w(f'<!DOCTYPE {self.doctype}>{newl}')
        self.documentElement.writexml(stream, indent, addindent, newl, strip, nsprefixes, namespace)

    def createElement(self, name, **kw):
        if False:
            i = 10
            return i + 15
        return Element(name, **kw)

    def createTextNode(self, text):
        if False:
            for i in range(10):
                print('nop')
        return Text(text)

    def createComment(self, text):
        if False:
            i = 10
            return i + 15
        return Comment(text)

    def getElementsByTagName(self, name):
        if False:
            while True:
                i = 10
        if self.documentElement.caseInsensitive:
            return getElementsByTagNameNoCase(self, name)
        return getElementsByTagName(self, name)

    def getElementById(self, id):
        if False:
            i = 10
            return i + 15
        childNodes = self.childNodes[:]
        while childNodes:
            node = childNodes.pop(0)
            if node.childNodes:
                childNodes.extend(node.childNodes)
            if hasattr(node, 'getAttribute') and node.getAttribute('id') == id:
                return node

class EntityReference(Node):

    def __init__(self, eref, parentNode=None):
        if False:
            while True:
                i = 10
        Node.__init__(self, parentNode)
        self.eref = eref
        self.nodeValue = self.data = '&' + eref + ';'

    def isEqualToEntityReference(self, n):
        if False:
            i = 10
            return i + 15
        if not isinstance(n, EntityReference):
            return 0
        return self.eref == n.eref and self.nodeValue == n.nodeValue
    isEqualToNode = isEqualToEntityReference

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            i = 10
            return i + 15
        w = _streamWriteWrapper(stream)
        w('' + self.nodeValue)

    def cloneNode(self, deep=0, parent=None):
        if False:
            for i in range(10):
                print('nop')
        return EntityReference(self.eref, parent)

class CharacterData(Node):

    def __init__(self, data, parentNode=None):
        if False:
            return 10
        Node.__init__(self, parentNode)
        self.value = self.data = self.nodeValue = data

    def isEqualToCharacterData(self, n):
        if False:
            print('Hello World!')
        return self.value == n.value
    isEqualToNode = isEqualToCharacterData

class Comment(CharacterData):
    """
    A comment node.
    """

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            return 10
        w = _streamWriteWrapper(stream)
        val = self.data
        w(f'<!--{val}-->')

    def cloneNode(self, deep=0, parent=None):
        if False:
            print('Hello World!')
        return Comment(self.nodeValue, parent)

class Text(CharacterData):

    def __init__(self, data, parentNode=None, raw=0):
        if False:
            print('Hello World!')
        CharacterData.__init__(self, data, parentNode)
        self.raw = raw

    def isEqualToNode(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compare this text to C{text}.  If the underlying values and the C{raw}\n        flag are the same, return C{True}, otherwise return C{False}.\n        '
        return CharacterData.isEqualToNode(self, other) and self.raw == other.raw

    def cloneNode(self, deep=0, parent=None):
        if False:
            print('Hello World!')
        return Text(self.nodeValue, parent, self.raw)

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            return 10
        w = _streamWriteWrapper(stream)
        if self.raw:
            val = self.nodeValue
            if not isinstance(val, str):
                val = str(self.nodeValue)
        else:
            v = self.nodeValue
            if not isinstance(v, str):
                v = str(v)
            if strip:
                v = ' '.join(v.split())
            val = escape(v)
        w(val)

    def __repr__(self) -> str:
        if False:
            return 10
        return 'Text(%s' % repr(self.nodeValue) + ')'

class CDATASection(CharacterData):

    def cloneNode(self, deep=0, parent=None):
        if False:
            print('Hello World!')
        return CDATASection(self.nodeValue, parent)

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            for i in range(10):
                print('nop')
        w = _streamWriteWrapper(stream)
        w('<![CDATA[')
        w('' + self.nodeValue)
        w(']]>')

def _genprefix():
    if False:
        return 10
    i = 0
    while True:
        yield ('p' + str(i))
        i = i + 1
genprefix = _genprefix()

class _Attr(CharacterData):
    """Support class for getAttributeNode."""

class Element(Node):
    preserveCase = 0
    caseInsensitive = 1
    nsprefixes = None

    def __init__(self, tagName, attributes=None, parentNode=None, filename=None, markpos=None, caseInsensitive=1, preserveCase=0, namespace=None):
        if False:
            print('Hello World!')
        Node.__init__(self, parentNode)
        self.preserveCase = preserveCase or not caseInsensitive
        self.caseInsensitive = caseInsensitive
        if not preserveCase:
            tagName = tagName.lower()
        if attributes is None:
            self.attributes = {}
        else:
            self.attributes = attributes
            for (k, v) in self.attributes.items():
                self.attributes[k] = unescape(v)
        if caseInsensitive:
            self.attributes = InsensitiveDict(self.attributes, preserve=preserveCase)
        self.endTagName = self.nodeName = self.tagName = tagName
        self._filename = filename
        self._markpos = markpos
        self.namespace = namespace

    def addPrefixes(self, pfxs):
        if False:
            for i in range(10):
                print('nop')
        if self.nsprefixes is None:
            self.nsprefixes = pfxs
        else:
            self.nsprefixes.update(pfxs)

    def endTag(self, endTagName):
        if False:
            for i in range(10):
                print('nop')
        if not self.preserveCase:
            endTagName = endTagName.lower()
        self.endTagName = endTagName

    def isEqualToElement(self, n):
        if False:
            print('Hello World!')
        if self.caseInsensitive:
            return self.attributes == n.attributes and self.nodeName.lower() == n.nodeName.lower()
        return self.attributes == n.attributes and self.nodeName == n.nodeName

    def isEqualToNode(self, other):
        if False:
            print('Hello World!')
        '\n        Compare this element to C{other}.  If the C{nodeName}, C{namespace},\n        C{attributes}, and C{childNodes} are all the same, return C{True},\n        otherwise return C{False}.\n        '
        return self.nodeName.lower() == other.nodeName.lower() and self.namespace == other.namespace and (self.attributes == other.attributes) and Node.isEqualToNode(self, other)

    def cloneNode(self, deep=0, parent=None):
        if False:
            return 10
        clone = Element(self.tagName, parentNode=parent, namespace=self.namespace, preserveCase=self.preserveCase, caseInsensitive=self.caseInsensitive)
        clone.attributes.update(self.attributes)
        if deep:
            clone.childNodes = [child.cloneNode(1, clone) for child in self.childNodes]
        else:
            clone.childNodes = []
        return clone

    def getElementsByTagName(self, name):
        if False:
            print('Hello World!')
        if self.caseInsensitive:
            return getElementsByTagNameNoCase(self, name)
        return getElementsByTagName(self, name)

    def hasAttributes(self):
        if False:
            i = 10
            return i + 15
        return 1

    def getAttribute(self, name, default=None):
        if False:
            print('Hello World!')
        return self.attributes.get(name, default)

    def getAttributeNS(self, ns, name, default=None):
        if False:
            print('Hello World!')
        nsk = (ns, name)
        if nsk in self.attributes:
            return self.attributes[nsk]
        if ns == self.namespace:
            return self.attributes.get(name, default)
        return default

    def getAttributeNode(self, name):
        if False:
            while True:
                i = 10
        return _Attr(self.getAttribute(name), self)

    def setAttribute(self, name, attr):
        if False:
            return 10
        self.attributes[name] = attr

    def removeAttribute(self, name):
        if False:
            print('Hello World!')
        if name in self.attributes:
            del self.attributes[name]

    def hasAttribute(self, name):
        if False:
            print('Hello World!')
        return name in self.attributes

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        if False:
            i = 10
            return i + 15
        '\n        Serialize this L{Element} to the given stream.\n\n        @param stream: A file-like object to which this L{Element} will be\n            written.\n\n        @param nsprefixes: A C{dict} mapping namespace URIs as C{str} to\n            prefixes as C{str}.  This defines the prefixes which are already in\n            scope in the document at the point at which this L{Element} exists.\n            This is essentially an implementation detail for namespace support.\n            Applications should not try to use it.\n\n        @param namespace: The namespace URI as a C{str} which is the default at\n            the point in the document at which this L{Element} exists.  This is\n            essentially an implementation detail for namespace support.\n            Applications should not try to use it.\n        '
        ALLOWSINGLETON = ('img', 'br', 'hr', 'base', 'meta', 'link', 'param', 'area', 'input', 'col', 'basefont', 'isindex', 'frame')
        BLOCKELEMENTS = ('html', 'head', 'body', 'noscript', 'ins', 'del', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'script', 'ul', 'ol', 'dl', 'pre', 'hr', 'blockquote', 'address', 'p', 'div', 'fieldset', 'table', 'tr', 'form', 'object', 'fieldset', 'applet', 'map')
        FORMATNICELY = ('tr', 'ul', 'ol', 'head')
        if not self.preserveCase:
            self.endTagName = self.tagName
        w = _streamWriteWrapper(stream)
        if self.nsprefixes:
            newprefixes = self.nsprefixes.copy()
            for ns in nsprefixes.keys():
                if ns in newprefixes:
                    del newprefixes[ns]
        else:
            newprefixes = {}
        begin = ['<']
        if self.tagName in BLOCKELEMENTS:
            begin = [newl, indent] + begin
        bext = begin.extend
        writeattr = lambda _atr, _val: bext((' ', _atr, '="', escape(_val), '"'))
        endTagName = self.endTagName
        if namespace != self.namespace and self.namespace is not None:
            if self.namespace in nsprefixes:
                prefix = nsprefixes[self.namespace]
                bext(prefix + ':' + self.tagName)
                endTagName = prefix + ':' + self.endTagName
            else:
                bext(self.tagName)
                writeattr('xmlns', self.namespace)
                namespace = self.namespace
        else:
            bext(self.tagName)
        j = ''.join
        for (attr, val) in sorted(self.attributes.items()):
            if isinstance(attr, tuple):
                (ns, key) = attr
                if ns in nsprefixes:
                    prefix = nsprefixes[ns]
                else:
                    prefix = next(genprefix)
                    newprefixes[ns] = prefix
                assert val is not None
                writeattr(prefix + ':' + key, val)
            else:
                assert val is not None
                writeattr(attr, val)
        if newprefixes:
            for (ns, prefix) in newprefixes.items():
                if prefix:
                    writeattr('xmlns:' + prefix, ns)
            newprefixes.update(nsprefixes)
            downprefixes = newprefixes
        else:
            downprefixes = nsprefixes
        w(j(begin))
        if self.childNodes:
            w('>')
            newindent = indent + addindent
            for child in self.childNodes:
                if self.tagName in BLOCKELEMENTS and self.tagName in FORMATNICELY:
                    w(j((newl, newindent)))
                child.writexml(stream, newindent, addindent, newl, strip, downprefixes, namespace)
            if self.tagName in BLOCKELEMENTS:
                w(j((newl, indent)))
            w(j(('</', endTagName, '>')))
        elif self.tagName.lower() not in ALLOWSINGLETON:
            w(j(('></', endTagName, '>')))
        else:
            w(' />')

    def __repr__(self) -> str:
        if False:
            return 10
        rep = 'Element(%s' % repr(self.nodeName)
        if self.attributes:
            rep += f', attributes={self.attributes!r}'
        if self._filename:
            rep += f', filename={self._filename!r}'
        if self._markpos:
            rep += f', markpos={self._markpos!r}'
        return rep + ')'

    def __str__(self) -> str:
        if False:
            return 10
        rep = '<' + self.nodeName
        if self._filename or self._markpos:
            rep += ' ('
        if self._filename:
            rep += repr(self._filename)
        if self._markpos:
            rep += ' line %s column %s' % self._markpos
        if self._filename or self._markpos:
            rep += ')'
        for item in self.attributes.items():
            rep += ' %s=%r' % item
        if self.hasChildNodes():
            rep += ' >...</%s>' % self.nodeName
        else:
            rep += ' />'
        return rep

def _unescapeDict(d):
    if False:
        print('Hello World!')
    dd = {}
    for (k, v) in d.items():
        dd[k] = unescape(v)
    return dd

def _reverseDict(d):
    if False:
        for i in range(10):
            print('nop')
    dd = {}
    for (k, v) in d.items():
        dd[v] = k
    return dd

class MicroDOMParser(XMLParser):
    soonClosers = 'area link br img hr input base meta'.split()
    laterClosers = {'p': ['p', 'dt'], 'dt': ['dt', 'dd'], 'dd': ['dt', 'dd'], 'li': ['li'], 'tbody': ['thead', 'tfoot', 'tbody'], 'thead': ['thead', 'tfoot', 'tbody'], 'tfoot': ['thead', 'tfoot', 'tbody'], 'colgroup': ['colgroup'], 'col': ['col'], 'tr': ['tr'], 'td': ['td'], 'th': ['th'], 'head': ['body'], 'title': ['head', 'body'], 'option': ['option']}

    def __init__(self, beExtremelyLenient=0, caseInsensitive=1, preserveCase=0, soonClosers=soonClosers, laterClosers=laterClosers):
        if False:
            while True:
                i = 10
        self.elementstack = []
        d = {'xmlns': 'xmlns', '': None}
        dr = _reverseDict(d)
        self.nsstack = [(d, None, dr)]
        self.documents = []
        self._mddoctype = None
        self.beExtremelyLenient = beExtremelyLenient
        self.caseInsensitive = caseInsensitive
        self.preserveCase = preserveCase or not caseInsensitive
        self.soonClosers = soonClosers
        self.laterClosers = laterClosers

    def shouldPreserveSpace(self):
        if False:
            while True:
                i = 10
        for edx in range(len(self.elementstack)):
            el = self.elementstack[-edx]
            if el.tagName == 'pre' or el.getAttribute('xml:space', '') == 'preserve':
                return 1
        return 0

    def _getparent(self):
        if False:
            return 10
        if self.elementstack:
            return self.elementstack[-1]
        else:
            return None
    COMMENT = re.compile('\\s*/[/*]\\s*')

    def _fixScriptElement(self, el):
        if False:
            i = 10
            return i + 15
        if not self.beExtremelyLenient or not len(el.childNodes) == 1:
            return
        c = el.firstChild()
        if isinstance(c, Text):
            prefix = ''
            oldvalue = c.value
            match = self.COMMENT.match(oldvalue)
            if match:
                prefix = match.group()
                oldvalue = oldvalue[len(prefix):]
            try:
                e = parseString('<a>%s</a>' % oldvalue).childNodes[0]
            except (ParseError, MismatchedTags):
                return
            if len(e.childNodes) != 1:
                return
            e = e.firstChild()
            if isinstance(e, (CDATASection, Comment)):
                el.childNodes = []
                if prefix:
                    el.childNodes.append(Text(prefix))
                el.childNodes.append(e)

    def gotDoctype(self, doctype):
        if False:
            while True:
                i = 10
        self._mddoctype = doctype

    def gotTagStart(self, name, attributes):
        if False:
            return 10
        parent = self._getparent()
        if self.beExtremelyLenient and isinstance(parent, Element):
            parentName = parent.tagName
            myName = name
            if self.caseInsensitive:
                parentName = parentName.lower()
                myName = myName.lower()
            if myName in self.laterClosers.get(parentName, []):
                self.gotTagEnd(parent.tagName)
                parent = self._getparent()
        attributes = _unescapeDict(attributes)
        namespaces = self.nsstack[-1][0]
        newspaces = {}
        keysToDelete = []
        for (k, v) in attributes.items():
            if k.startswith('xmlns'):
                spacenames = k.split(':', 1)
                if len(spacenames) == 2:
                    newspaces[spacenames[1]] = v
                else:
                    newspaces[''] = v
                keysToDelete.append(k)
        for k in keysToDelete:
            del attributes[k]
        if newspaces:
            namespaces = namespaces.copy()
            namespaces.update(newspaces)
        keysToDelete = []
        for (k, v) in attributes.items():
            ksplit = k.split(':', 1)
            if len(ksplit) == 2:
                (pfx, tv) = ksplit
                if pfx != 'xml' and pfx in namespaces:
                    attributes[namespaces[pfx], tv] = v
                    keysToDelete.append(k)
        for k in keysToDelete:
            del attributes[k]
        el = Element(name, attributes, parent, self.filename, self.saveMark(), caseInsensitive=self.caseInsensitive, preserveCase=self.preserveCase, namespace=namespaces.get(''))
        revspaces = _reverseDict(newspaces)
        el.addPrefixes(revspaces)
        if newspaces:
            rscopy = self.nsstack[-1][2].copy()
            rscopy.update(revspaces)
            self.nsstack.append((namespaces, el, rscopy))
        self.elementstack.append(el)
        if parent:
            parent.appendChild(el)
        if self.beExtremelyLenient and el.tagName in self.soonClosers:
            self.gotTagEnd(name)

    def _gotStandalone(self, factory, data):
        if False:
            for i in range(10):
                print('nop')
        parent = self._getparent()
        te = factory(data, parent)
        if parent:
            parent.appendChild(te)
        elif self.beExtremelyLenient:
            self.documents.append(te)

    def gotText(self, data):
        if False:
            i = 10
            return i + 15
        if data.strip() or self.shouldPreserveSpace():
            self._gotStandalone(Text, data)

    def gotComment(self, data):
        if False:
            i = 10
            return i + 15
        self._gotStandalone(Comment, data)

    def gotEntityReference(self, entityRef):
        if False:
            while True:
                i = 10
        self._gotStandalone(EntityReference, entityRef)

    def gotCData(self, cdata):
        if False:
            i = 10
            return i + 15
        self._gotStandalone(CDATASection, cdata)

    def gotTagEnd(self, name):
        if False:
            for i in range(10):
                print('nop')
        if not self.elementstack:
            if self.beExtremelyLenient:
                return
            raise MismatchedTags(*(self.filename, 'NOTHING', name) + self.saveMark() + (0, 0))
        el = self.elementstack.pop()
        pfxdix = self.nsstack[-1][2]
        if self.nsstack[-1][1] is el:
            nstuple = self.nsstack.pop()
        else:
            nstuple = None
        if self.caseInsensitive:
            tn = el.tagName.lower()
            cname = name.lower()
        else:
            tn = el.tagName
            cname = name
        nsplit = name.split(':', 1)
        if len(nsplit) == 2:
            (pfx, newname) = nsplit
            ns = pfxdix.get(pfx, None)
            if ns is not None:
                if el.namespace != ns:
                    if not self.beExtremelyLenient:
                        raise MismatchedTags(*(self.filename, el.tagName, name) + self.saveMark() + el._markpos)
        if not tn == cname:
            if self.beExtremelyLenient:
                if self.elementstack:
                    lastEl = self.elementstack[0]
                    for idx in range(len(self.elementstack)):
                        if self.elementstack[-(idx + 1)].tagName == cname:
                            self.elementstack[-(idx + 1)].endTag(name)
                            break
                    else:
                        self.elementstack.append(el)
                        if nstuple is not None:
                            self.nsstack.append(nstuple)
                        return
                    del self.elementstack[-(idx + 1):]
                    if not self.elementstack:
                        self.documents.append(lastEl)
                        return
            else:
                raise MismatchedTags(*(self.filename, el.tagName, name) + self.saveMark() + el._markpos)
        el.endTag(name)
        if not self.elementstack:
            self.documents.append(el)
        if self.beExtremelyLenient and el.tagName == 'script':
            self._fixScriptElement(el)

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        XMLParser.connectionLost(self, reason)
        if self.elementstack:
            if self.beExtremelyLenient:
                self.documents.append(self.elementstack[0])
            else:
                raise MismatchedTags(*(self.filename, self.elementstack[-1], 'END_OF_FILE') + self.saveMark() + self.elementstack[-1]._markpos)

def parse(readable, *args, **kwargs):
    if False:
        return 10
    '\n    Parse HTML or XML readable.\n    '
    if not hasattr(readable, 'read'):
        readable = open(readable, 'rb')
    mdp = MicroDOMParser(*args, **kwargs)
    mdp.filename = getattr(readable, 'name', '<xmlfile />')
    mdp.makeConnection(None)
    if hasattr(readable, 'getvalue'):
        mdp.dataReceived(readable.getvalue())
    else:
        r = readable.read(1024)
        while r:
            mdp.dataReceived(r)
            r = readable.read(1024)
    mdp.connectionLost(None)
    if not mdp.documents:
        raise ParseError(mdp.filename, 0, 0, 'No top-level Nodes in document')
    if mdp.beExtremelyLenient:
        if len(mdp.documents) == 1:
            d = mdp.documents[0]
            if not isinstance(d, Element):
                el = Element('html')
                el.appendChild(d)
                d = el
        else:
            d = Element('html')
            for child in mdp.documents:
                d.appendChild(child)
    else:
        d = mdp.documents[0]
    doc = Document(d)
    doc.doctype = mdp._mddoctype
    return doc

def parseString(st, *args, **kw):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(st, str):
        return parse(BytesIO(st.encode('UTF-16')), *args, **kw)
    return parse(BytesIO(st), *args, **kw)

def parseXML(readable):
    if False:
        print('Hello World!')
    '\n    Parse an XML readable object.\n    '
    return parse(readable, caseInsensitive=0, preserveCase=1)

def parseXMLString(st):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse an XML readable object.\n    '
    return parseString(st, caseInsensitive=0, preserveCase=1)

class lmx:
    """
    Easy creation of XML.
    """

    def __init__(self, node='div'):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node, str):
            node = Element(node)
        self.node = node

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name[0] == '_':
            raise AttributeError('no private attrs')
        return lambda **kw: self.add(name, **kw)

    def __setitem__(self, key, val):
        if False:
            print('Hello World!')
        self.node.setAttribute(key, val)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self.node.getAttribute(key)

    def text(self, txt, raw=0):
        if False:
            return 10
        nn = Text(txt, raw=raw)
        self.node.appendChild(nn)
        return self

    def add(self, tagName, **kw):
        if False:
            return 10
        newNode = Element(tagName, caseInsensitive=0, preserveCase=0)
        self.node.appendChild(newNode)
        xf = lmx(newNode)
        for (k, v) in kw.items():
            if k[0] == '_':
                k = k[1:]
            xf[k] = v
        return xf