"""
DOM-like XML processing support.

This module provides support for parsing XML into DOM-like object structures
and serializing such structures to an XML string representation, optimized
for use in streaming XML applications.
"""
from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux

def _splitPrefix(name):
    if False:
        while True:
            i = 10
    'Internal method for splitting a prefixed Element name into its\n    respective parts'
    ntok = name.split(':', 1)
    if len(ntok) == 2:
        return ntok
    else:
        return (None, ntok[0])
G_PREFIXES = {'http://www.w3.org/XML/1998/namespace': 'xml'}

class _ListSerializer:
    """Internal class which serializes an Element tree into a buffer"""

    def __init__(self, prefixes=None, prefixesInScope=None):
        if False:
            i = 10
            return i + 15
        self.writelist = []
        self.prefixes = {}
        if prefixes:
            self.prefixes.update(prefixes)
        self.prefixes.update(G_PREFIXES)
        self.prefixStack = [G_PREFIXES.values()] + (prefixesInScope or [])
        self.prefixCounter = 0

    def getValue(self):
        if False:
            for i in range(10):
                print('nop')
        return ''.join(self.writelist)

    def getPrefix(self, uri):
        if False:
            while True:
                i = 10
        if uri not in self.prefixes:
            self.prefixes[uri] = 'xn%d' % self.prefixCounter
            self.prefixCounter = self.prefixCounter + 1
        return self.prefixes[uri]

    def prefixInScope(self, prefix):
        if False:
            return 10
        stack = self.prefixStack
        for i in range(-1, (len(self.prefixStack) + 1) * -1, -1):
            if prefix in stack[i]:
                return True
        return False

    def serialize(self, elem, closeElement=1, defaultUri=''):
        if False:
            for i in range(10):
                print('nop')
        write = self.writelist.append
        if isinstance(elem, SerializedXML):
            write(elem)
            return
        if isinstance(elem, str):
            write(escapeToXml(elem))
            return
        name = elem.name
        uri = elem.uri
        (defaultUri, currentDefaultUri) = (elem.defaultUri, defaultUri)
        for (p, u) in elem.localPrefixes.items():
            self.prefixes[u] = p
        self.prefixStack.append(list(elem.localPrefixes.keys()))
        if defaultUri is None:
            defaultUri = currentDefaultUri
        if uri is None:
            uri = defaultUri
        prefix = None
        if uri != defaultUri or uri in self.prefixes:
            prefix = self.getPrefix(uri)
            inScope = self.prefixInScope(prefix)
        if not prefix:
            write('<%s' % name)
        else:
            write(f'<{prefix}:{name}')
            if not inScope:
                write(f" xmlns:{prefix}='{uri}'")
                self.prefixStack[-1].append(prefix)
                inScope = True
        if defaultUri != currentDefaultUri and (uri != defaultUri or not prefix or (not inScope)):
            write(" xmlns='%s'" % defaultUri)
        for (p, u) in elem.localPrefixes.items():
            write(f" xmlns:{p}='{u}'")
        for (k, v) in elem.attributes.items():
            if isinstance(k, tuple):
                (attr_uri, attr_name) = k
                attr_prefix = self.getPrefix(attr_uri)
                if not self.prefixInScope(attr_prefix):
                    write(f" xmlns:{attr_prefix}='{attr_uri}'")
                    self.prefixStack[-1].append(attr_prefix)
                write(f" {attr_prefix}:{attr_name}='{escapeToXml(v, 1)}'")
            else:
                write(f" {k}='{escapeToXml(v, 1)}'")
        if closeElement == 0:
            write('>')
            return
        if len(elem.children) > 0:
            write('>')
            for c in elem.children:
                self.serialize(c, defaultUri=defaultUri)
            if not prefix:
                write('</%s>' % name)
            else:
                write(f'</{prefix}:{name}>')
        else:
            write('/>')
        self.prefixStack.pop()
SerializerClass = _ListSerializer

def escapeToXml(text, isattrib=0):
    if False:
        return 10
    'Escape text to proper XML form, per section 2.3 in the XML specification.\n\n    @type text: C{str}\n    @param text: Text to escape\n\n    @type isattrib: C{bool}\n    @param isattrib: Triggers escaping of characters necessary for use as\n                     attribute values\n    '
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    if isattrib == 1:
        text = text.replace("'", '&apos;')
        text = text.replace('"', '&quot;')
    return text

def unescapeFromXml(text):
    if False:
        print('Hello World!')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&apos;', "'")
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    return text

def generateOnlyInterface(list, int):
    if False:
        i = 10
        return i + 15
    'Filters items in a list by class'
    for n in list:
        if int.providedBy(n):
            yield n

def generateElementsQNamed(list, name, uri):
    if False:
        return 10
    'Filters Element items in a list with matching name and URI.'
    for n in list:
        if IElement.providedBy(n) and n.name == name and (n.uri == uri):
            yield n

def generateElementsNamed(list, name):
    if False:
        return 10
    'Filters Element items in a list with matching name, regardless of URI.'
    for n in list:
        if IElement.providedBy(n) and n.name == name:
            yield n

class SerializedXML(str):
    """Marker class for pre-serialized XML in the DOM."""
    pass

class Namespace:
    """Convenience object for tracking namespace declarations."""

    def __init__(self, uri):
        if False:
            print('Hello World!')
        self._uri = uri

    def __getattr__(self, n):
        if False:
            print('Hello World!')
        return (self._uri, n)

    def __getitem__(self, n):
        if False:
            while True:
                i = 10
        return (self._uri, n)

class IElement(Interface):
    """
    Interface to XML element nodes.

    See L{Element} for a detailed example of its general use.

    Warning: this Interface is not yet complete!
    """
    uri = Attribute(" Element's namespace URI ")
    name = Attribute(" Element's local name ")
    defaultUri = Attribute(' Default namespace URI of child elements ')
    attributes = Attribute(' Dictionary of element attributes ')
    children = Attribute(' List of child nodes ')
    parent = Attribute(" Reference to element's parent element ")
    localPrefixes = Attribute(' Dictionary of local prefixes ')

    def toXml(prefixes=None, closeElement=1, defaultUri='', prefixesInScope=None):
        if False:
            return 10
        "Serializes object to a (partial) XML document\n\n        @param prefixes: dictionary that maps namespace URIs to suggested\n                         prefix names.\n        @type prefixes: L{dict}\n\n        @param closeElement: flag that determines whether to include the\n            closing tag of the element in the serialized string. A value of\n            C{0} only generates the element's start tag. A value of C{1} yields\n            a complete serialization.\n        @type closeElement: L{int}\n\n        @param defaultUri: Initial default namespace URI. This is most useful\n            for partial rendering, where the logical parent element (of which\n            the starttag was already serialized) declares a default namespace\n            that should be inherited.\n        @type defaultUri: L{str}\n\n        @param prefixesInScope: list of prefixes that are assumed to be\n            declared by ancestors.\n        @type prefixesInScope: L{list}\n\n        @return: (partial) serialized XML\n        @rtype: L{str}\n        "

    def addElement(name, defaultUri=None, content=None):
        if False:
            i = 10
            return i + 15
        '\n        Create an element and add as child.\n\n        The new element is added to this element as a child, and will have\n        this element as its parent.\n\n        @param name: element name. This can be either a L{str} object that\n            contains the local name, or a tuple of (uri, local_name) for a\n            fully qualified name. In the former case, the namespace URI is\n            inherited from this element.\n        @type name: L{str} or L{tuple} of (L{str}, L{str})\n\n        @param defaultUri: default namespace URI for child elements. If\n            L{None}, this is inherited from this element.\n        @type defaultUri: L{str}\n\n        @param content: text contained by the new element.\n        @type content: L{str}\n\n        @return: the created element\n        @rtype: object providing L{IElement}\n        '

    def addChild(node):
        if False:
            while True:
                i = 10
        '\n        Adds a node as child of this element.\n\n        The C{node} will be added to the list of childs of this element, and\n        will have this element set as its parent when C{node} provides\n        L{IElement}. If C{node} is a L{str} and the current last child is\n        character data (L{str}), the text from C{node} is appended to the\n        existing last child.\n\n        @param node: the child node.\n        @type node: L{str} or object implementing L{IElement}\n        '

    def addContent(text):
        if False:
            return 10
        '\n        Adds character data to this element.\n\n        If the current last child of this element is a string, the text will\n        be appended to that string. Otherwise, the text will be added as a new\n        child.\n\n        @param text: The character data to be added to this element.\n        @type text: L{str}\n        '

@implementer(IElement)
class Element:
    """Represents an XML element node.

    An Element contains a series of attributes (name/value pairs), content
    (character data), and other child Element objects. When building a document
    with markup (such as HTML or XML), use this object as the starting point.

    Element objects fully support XML Namespaces. The fully qualified name of
    the XML Element it represents is stored in the C{uri} and C{name}
    attributes, where C{uri} holds the namespace URI. There is also a default
    namespace, for child elements. This is stored in the C{defaultUri}
    attribute. Note that C{''} means the empty namespace.

    Serialization of Elements through C{toXml()} will use these attributes
    for generating proper serialized XML. When both C{uri} and C{defaultUri}
    are not None in the Element and all of its descendents, serialization
    proceeds as expected:

      >>> from twisted.words.xish import domish
      >>> root = domish.Element(('myns', 'root'))
      >>> root.addElement('child', content='test')
      <twisted.words.xish.domish.Element object at 0x83002ac>
      >>> root.toXml()
      u"<root xmlns='myns'><child>test</child></root>"

    For partial serialization, needed for streaming XML, a special value for
    namespace URIs can be used: L{None}.

    Using L{None} as the value for C{uri} means: this element is in whatever
    namespace inherited by the closest logical ancestor when the complete XML
    document has been serialized. The serialized start tag will have a
    non-prefixed name, and no xmlns declaration will be generated.

    Similarly, L{None} for C{defaultUri} means: the default namespace for my
    child elements is inherited from the logical ancestors of this element,
    when the complete XML document has been serialized.

    To illustrate, an example from a Jabber stream. Assume the start tag of the
    root element of the stream has already been serialized, along with several
    complete child elements, and sent off, looking like this::

      <stream:stream xmlns:stream='http://etherx.jabber.org/streams'
                     xmlns='jabber:client' to='example.com'>
        ...

    Now suppose we want to send a complete element represented by an
    object C{message} created like:

      >>> message = domish.Element((None, 'message'))
      >>> message['to'] = 'user@example.com'
      >>> message.addElement('body', content='Hi!')
      <twisted.words.xish.domish.Element object at 0x8276e8c>
      >>> message.toXml()
      u"<message to='user@example.com'><body>Hi!</body></message>"

    As, you can see, this XML snippet has no xmlns declaration. When sent
    off, it inherits the C{jabber:client} namespace from the root element.
    Note that this renders the same as using C{''} instead of L{None}:

      >>> presence = domish.Element(('', 'presence'))
      >>> presence.toXml()
      u"<presence/>"

    However, if this object has a parent defined, the difference becomes
    clear:

      >>> child = message.addElement(('http://example.com/', 'envelope'))
      >>> child.addChild(presence)
      <twisted.words.xish.domish.Element object at 0x8276fac>
      >>> message.toXml()
      u"<message to='user@example.com'><body>Hi!</body><envelope xmlns='http://example.com/'><presence xmlns=''/></envelope></message>"

    As, you can see, the <presence/> element is now in the empty namespace, not
    in the default namespace of the parent or the streams'.

    @type uri: L{str} or None
    @ivar uri: URI of this Element's name

    @type name: L{str}
    @ivar name: Name of this Element

    @type defaultUri: L{str} or None
    @ivar defaultUri: URI this Element exists within

    @type children: L{list}
    @ivar children: List of child Elements and content

    @type parent: L{Element}
    @ivar parent: Reference to the parent Element, if any.

    @type attributes: L{dict}
    @ivar attributes: Dictionary of attributes associated with this Element.

    @type localPrefixes: L{dict}
    @ivar localPrefixes: Dictionary of namespace declarations on this
                         element. The key is the prefix to bind the
                         namespace uri to.
    """
    _idCounter = 0

    def __init__(self, qname, defaultUri=None, attribs=None, localPrefixes=None):
        if False:
            print('Hello World!')
        '\n        @param qname: Tuple of (uri, name)\n        @param defaultUri: The default URI of the element; defaults to the URI\n                           specified in C{qname}\n        @param attribs: Dictionary of attributes\n        @param localPrefixes: Dictionary of namespace declarations on this\n                              element. The key is the prefix to bind the\n                              namespace uri to.\n        '
        self.localPrefixes = localPrefixes or {}
        (self.uri, self.name) = qname
        if defaultUri is None and self.uri not in self.localPrefixes.values():
            self.defaultUri = self.uri
        else:
            self.defaultUri = defaultUri
        self.attributes = attribs or {}
        self.children = []
        self.parent = None

    def __getattr__(self, key):
        if False:
            for i in range(10):
                print('nop')
        for n in self.children:
            if IElement.providedBy(n) and n.name == key:
                return n
        if key.startswith('_'):
            raise AttributeError(key)
        else:
            return None

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self.attributes[self._dqa(key)]

    def __delitem__(self, key):
        if False:
            return 10
        del self.attributes[self._dqa(key)]

    def __setitem__(self, key, value):
        if False:
            return 10
        self.attributes[self._dqa(key)] = value

    def __unicode__(self):
        if False:
            return 10
        '\n        Retrieve the first CData (content) node\n        '
        for n in self.children:
            if isinstance(n, str):
                return n
        return ''

    def __bytes__(self):
        if False:
            return 10
        '\n        Retrieve the first character data node as UTF-8 bytes.\n        '
        return str(self).encode('utf-8')
    __str__ = __unicode__

    def _dqa(self, attr):
        if False:
            for i in range(10):
                print('nop')
        'Dequalify an attribute key as needed'
        if isinstance(attr, tuple) and (not attr[0]):
            return attr[1]
        else:
            return attr

    def getAttribute(self, attribname, default=None):
        if False:
            return 10
        'Retrieve the value of attribname, if it exists'
        return self.attributes.get(attribname, default)

    def hasAttribute(self, attrib):
        if False:
            print('Hello World!')
        'Determine if the specified attribute exists'
        return self._dqa(attrib) in self.attributes

    def compareAttribute(self, attrib, value):
        if False:
            while True:
                i = 10
        'Safely compare the value of an attribute against a provided value.\n\n        L{None}-safe.\n        '
        return self.attributes.get(self._dqa(attrib), None) == value

    def swapAttributeValues(self, left, right):
        if False:
            print('Hello World!')
        'Swap the values of two attribute.'
        d = self.attributes
        l = d[left]
        d[left] = d[right]
        d[right] = l

    def addChild(self, node):
        if False:
            while True:
                i = 10
        'Add a child to this Element.'
        if IElement.providedBy(node):
            node.parent = self
        self.children.append(node)
        return node

    def addContent(self, text: str) -> str:
        if False:
            print('Hello World!')
        'Add some text data to this Element.'
        if not isinstance(text, str):
            raise TypeError(f'Expected str not {text!r} ({type(text).__name__})')
        c = self.children
        if len(c) > 0 and isinstance(c[-1], str):
            c[-1] = c[-1] + text
        else:
            c.append(text)
        return cast(str, c[-1])

    def addElement(self, name, defaultUri=None, content=None):
        if False:
            i = 10
            return i + 15
        if isinstance(name, tuple):
            if defaultUri is None:
                defaultUri = name[0]
            child = Element(name, defaultUri)
        else:
            if defaultUri is None:
                defaultUri = self.defaultUri
            child = Element((defaultUri, name), defaultUri)
        self.addChild(child)
        if content:
            child.addContent(content)
        return child

    def addRawXml(self, rawxmlstring):
        if False:
            return 10
        "Add a pre-serialized chunk o' XML as a child of this Element."
        self.children.append(SerializedXML(rawxmlstring))

    def addUniqueId(self):
        if False:
            return 10
        'Add a unique (across a given Python session) id attribute to this\n        Element.\n        '
        self.attributes['id'] = 'H_%d' % Element._idCounter
        Element._idCounter = Element._idCounter + 1

    def elements(self, uri=None, name=None):
        if False:
            i = 10
            return i + 15
        '\n        Iterate across all children of this Element that are Elements.\n\n        Returns a generator over the child elements. If both the C{uri} and\n        C{name} parameters are set, the returned generator will only yield\n        on elements matching the qualified name.\n\n        @param uri: Optional element URI.\n        @type uri: L{str}\n        @param name: Optional element name.\n        @type name: L{str}\n        @return: Iterator that yields objects implementing L{IElement}.\n        '
        if name is None:
            return generateOnlyInterface(self.children, IElement)
        else:
            return generateElementsQNamed(self.children, name, uri)

    def toXml(self, prefixes=None, closeElement=1, defaultUri='', prefixesInScope=None):
        if False:
            for i in range(10):
                print('nop')
        'Serialize this Element and all children to a string.'
        s = SerializerClass(prefixes=prefixes, prefixesInScope=prefixesInScope)
        s.serialize(self, closeElement=closeElement, defaultUri=defaultUri)
        return s.getValue()

    def firstChildElement(self):
        if False:
            return 10
        for c in self.children:
            if IElement.providedBy(c):
                return c
        return None

class ParserError(Exception):
    """Exception thrown when a parsing error occurs"""
    pass

def elementStream():
    if False:
        return 10
    'Preferred method to construct an ElementStream\n\n    Uses Expat-based stream if available, and falls back to Sux if necessary.\n    '
    try:
        es = ExpatElementStream()
        return es
    except ImportError:
        if SuxElementStream is None:
            raise Exception('No parsers available :(')
        es = SuxElementStream()
        return es

class SuxElementStream(sux.XMLParser):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.connectionMade()
        self.DocumentStartEvent = None
        self.ElementEvent = None
        self.DocumentEndEvent = None
        self.currElem = None
        self.rootElem = None
        self.documentStarted = False
        self.defaultNsStack = []
        self.prefixStack = []

    def parse(self, buffer):
        if False:
            i = 10
            return i + 15
        try:
            self.dataReceived(buffer)
        except sux.ParseError as e:
            raise ParserError(str(e))

    def findUri(self, prefix):
        if False:
            i = 10
            return i + 15
        stack = self.prefixStack
        for i in range(-1, (len(self.prefixStack) + 1) * -1, -1):
            if prefix in stack[i]:
                return stack[i][prefix]
        return None

    def gotTagStart(self, name, attributes):
        if False:
            for i in range(10):
                print('nop')
        defaultUri = None
        localPrefixes = {}
        attribs = {}
        uri = None
        for (k, v) in list(attributes.items()):
            if k.startswith('xmlns'):
                (x, p) = _splitPrefix(k)
                if x is None:
                    defaultUri = v
                else:
                    localPrefixes[p] = v
                del attributes[k]
        self.prefixStack.append(localPrefixes)
        if defaultUri is None:
            if len(self.defaultNsStack) > 0:
                defaultUri = self.defaultNsStack[-1]
            else:
                defaultUri = ''
        (prefix, name) = _splitPrefix(name)
        if prefix is None:
            uri = defaultUri
        else:
            uri = self.findUri(prefix)
        for (k, v) in attributes.items():
            (p, n) = _splitPrefix(k)
            if p is None:
                attribs[n] = v
            else:
                attribs[self.findUri(p), n] = unescapeFromXml(v)
        e = Element((uri, name), defaultUri, attribs, localPrefixes)
        self.defaultNsStack.append(defaultUri)
        if self.documentStarted:
            if self.currElem is None:
                self.currElem = e
            else:
                self.currElem = self.currElem.addChild(e)
        else:
            self.rootElem = e
            self.documentStarted = True
            self.DocumentStartEvent(e)

    def gotText(self, data):
        if False:
            print('Hello World!')
        if self.currElem is not None:
            if isinstance(data, bytes):
                data = data.decode('ascii')
            self.currElem.addContent(data)

    def gotCData(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.currElem is not None:
            if isinstance(data, bytes):
                data = data.decode('ascii')
            self.currElem.addContent(data)

    def gotComment(self, data):
        if False:
            return 10
        pass
    entities = {'amp': '&', 'lt': '<', 'gt': '>', 'apos': "'", 'quot': '"'}

    def gotEntityReference(self, entityRef):
        if False:
            while True:
                i = 10
        if entityRef in SuxElementStream.entities:
            data = SuxElementStream.entities[entityRef]
            if isinstance(data, bytes):
                data = data.decode('ascii')
            self.currElem.addContent(data)

    def gotTagEnd(self, name):
        if False:
            print('Hello World!')
        if self.rootElem is None:
            raise ParserError('Element closed after end of document.')
        (prefix, name) = _splitPrefix(name)
        if prefix is None:
            uri = self.defaultNsStack[-1]
        else:
            uri = self.findUri(prefix)
        if self.currElem is None:
            if self.rootElem.name != name or self.rootElem.uri != uri:
                raise ParserError('Mismatched root elements')
            self.DocumentEndEvent()
            self.rootElem = None
        else:
            if self.currElem.name != name or self.currElem.uri != uri:
                raise ParserError('Malformed element close')
            self.prefixStack.pop()
            self.defaultNsStack.pop()
            if self.currElem.parent is None:
                self.currElem.parent = self.rootElem
                self.ElementEvent(self.currElem)
                self.currElem = None
            else:
                self.currElem = self.currElem.parent

class ExpatElementStream:

    def __init__(self):
        if False:
            return 10
        import pyexpat
        self.DocumentStartEvent = None
        self.ElementEvent = None
        self.DocumentEndEvent = None
        self.error = pyexpat.error
        self.parser = pyexpat.ParserCreate('UTF-8', ' ')
        self.parser.StartElementHandler = self._onStartElement
        self.parser.EndElementHandler = self._onEndElement
        self.parser.CharacterDataHandler = self._onCdata
        self.parser.StartNamespaceDeclHandler = self._onStartNamespace
        self.parser.EndNamespaceDeclHandler = self._onEndNamespace
        self.currElem = None
        self.defaultNsStack = ['']
        self.documentStarted = 0
        self.localPrefixes = {}

    def parse(self, buffer):
        if False:
            print('Hello World!')
        try:
            self.parser.Parse(buffer)
        except self.error as e:
            raise ParserError(str(e))

    def _onStartElement(self, name, attrs):
        if False:
            for i in range(10):
                print('nop')
        qname = name.rsplit(' ', 1)
        if len(qname) == 1:
            qname = ('', name)
        newAttrs = {}
        toDelete = []
        for (k, v) in attrs.items():
            if ' ' in k:
                aqname = k.rsplit(' ', 1)
                newAttrs[aqname[0], aqname[1]] = v
                toDelete.append(k)
        attrs.update(newAttrs)
        for k in toDelete:
            del attrs[k]
        e = Element(qname, self.defaultNsStack[-1], attrs, self.localPrefixes)
        self.localPrefixes = {}
        if self.documentStarted == 1:
            if self.currElem != None:
                self.currElem.children.append(e)
                e.parent = self.currElem
            self.currElem = e
        else:
            self.documentStarted = 1
            self.DocumentStartEvent(e)

    def _onEndElement(self, _):
        if False:
            for i in range(10):
                print('nop')
        if self.currElem is None:
            self.DocumentEndEvent()
        elif self.currElem.parent is None:
            self.ElementEvent(self.currElem)
            self.currElem = None
        else:
            self.currElem = self.currElem.parent

    def _onCdata(self, data):
        if False:
            i = 10
            return i + 15
        if self.currElem != None:
            self.currElem.addContent(data)

    def _onStartNamespace(self, prefix, uri):
        if False:
            i = 10
            return i + 15
        if prefix is None:
            self.defaultNsStack.append(uri)
        else:
            self.localPrefixes[prefix] = uri

    def _onEndNamespace(self, prefix):
        if False:
            return 10
        if prefix is None:
            self.defaultNsStack.pop()