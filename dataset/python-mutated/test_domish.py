"""
Tests for L{twisted.words.xish.domish}, a DOM-like library for XMPP.
"""
from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish

class ElementTests(unittest.TestCase):
    """
    Tests for L{domish.Element}.
    """

    def test_interface(self):
        if False:
            print('Hello World!')
        '\n        L{domish.Element} implements L{domish.IElement}.\n        '
        verifyObject(domish.IElement, domish.Element((None, 'foo')))

    def test_escaping(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The built-in entity references are properly encoded.\n        '
        s = '&<>\'"'
        self.assertEqual(domish.escapeToXml(s), '&amp;&lt;&gt;\'"')
        self.assertEqual(domish.escapeToXml(s, 1), '&amp;&lt;&gt;&apos;&quot;')

    def test_namespace(self):
        if False:
            print('Hello World!')
        '\n        An attribute on L{domish.Namespace} yields a qualified name.\n        '
        ns = domish.Namespace('testns')
        self.assertEqual(ns.foo, ('testns', 'foo'))

    def test_elementInit(self):
        if False:
            while True:
                i = 10
        '\n        Basic L{domish.Element} initialization tests.\n        '
        e = domish.Element((None, 'foo'))
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, None)
        self.assertEqual(e.defaultUri, None)
        self.assertEqual(e.parent, None)
        e = domish.Element(('', 'foo'))
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, '')
        self.assertEqual(e.defaultUri, '')
        self.assertEqual(e.parent, None)
        e = domish.Element(('testns', 'foo'))
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, 'testns')
        self.assertEqual(e.defaultUri, 'testns')
        self.assertEqual(e.parent, None)
        e = domish.Element(('testns', 'foo'), 'test2ns')
        self.assertEqual(e.name, 'foo')
        self.assertEqual(e.uri, 'testns')
        self.assertEqual(e.defaultUri, 'test2ns')

    def test_childOps(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Basic L{domish.Element} child tests.\n        '
        e = domish.Element(('testns', 'foo'))
        e.addContent('somecontent')
        b2 = e.addElement(('testns2', 'bar2'))
        e['attrib1'] = 'value1'
        e['testns2', 'attrib2'] = 'value2'
        e.addElement('bar')
        e.addElement('bar')
        e.addContent('abc')
        e.addContent('123')
        self.assertEqual(e.children[-1], 'abc123')
        self.assertEqual(e.bar2, b2)
        e.bar2.addContent('subcontent')
        e.bar2['bar2value'] = 'somevalue'
        self.assertEqual(e.children[1], e.bar2)
        self.assertEqual(e.children[2], e.bar)
        self.assertEqual(e['attrib1'], 'value1')
        del e['attrib1']
        self.assertEqual(e.hasAttribute('attrib1'), 0)
        self.assertEqual(e.hasAttribute('attrib2'), 0)
        self.assertEqual(e['testns2', 'attrib2'], 'value2')

    def test_characterData(self):
        if False:
            while True:
                i = 10
        '\n        Extract character data using L{str}.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addContent('somecontent')
        text = str(element)
        self.assertEqual('somecontent', text)
        self.assertIsInstance(text, str)

    def test_characterDataNativeString(self):
        if False:
            while True:
                i = 10
        '\n        Extract ascii character data using L{str}.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addContent('somecontent')
        text = str(element)
        self.assertEqual('somecontent', text)
        self.assertIsInstance(text, str)

    def test_characterDataUnicode(self):
        if False:
            return 10
        '\n        Extract character data using L{str}.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addContent('☃')
        text = str(element)
        self.assertEqual('☃', text)
        self.assertIsInstance(text, str)

    def test_characterDataBytes(self):
        if False:
            i = 10
            return i + 15
        '\n        Extract character data as UTF-8 using L{bytes}.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addContent('☃')
        text = bytes(element)
        self.assertEqual('☃'.encode(), text)
        self.assertIsInstance(text, bytes)

    def test_characterDataMixed(self):
        if False:
            return 10
        '\n        Mixing addChild with cdata and element, the first cdata is returned.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addChild('abc')
        element.addElement('bar')
        element.addChild('def')
        self.assertEqual('abc', str(element))

    def test_addContent(self):
        if False:
            return 10
        '\n        Unicode strings passed to C{addContent} become the character data.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addContent('unicode')
        self.assertEqual('unicode', str(element))

    def test_addContentNativeStringASCII(self):
        if False:
            while True:
                i = 10
        '\n        ASCII native strings passed to C{addContent} become the character data.\n        '
        element = domish.Element(('testns', 'foo'))
        element.addContent('native')
        self.assertEqual('native', str(element))

    def test_addContentBytes(self):
        if False:
            i = 10
            return i + 15
        '\n        Byte strings passed to C{addContent} are not acceptable on Python 3.\n        '
        element = domish.Element(('testns', 'foo'))
        self.assertRaises(TypeError, element.addContent, b'bytes')

    def test_addElementContent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Content passed to addElement becomes character data on the new child.\n        '
        element = domish.Element(('testns', 'foo'))
        child = element.addElement('bar', content='abc')
        self.assertEqual('abc', str(child))

    def test_elements(self):
        if False:
            print('Hello World!')
        '\n        Calling C{elements} without arguments on a L{domish.Element} returns\n        all child elements, whatever the qualified name.\n        '
        e = domish.Element(('testns', 'foo'))
        c1 = e.addElement('name')
        c2 = e.addElement(('testns2', 'baz'))
        c3 = e.addElement('quux')
        c4 = e.addElement(('testns', 'name'))
        elts = list(e.elements())
        self.assertIn(c1, elts)
        self.assertIn(c2, elts)
        self.assertIn(c3, elts)
        self.assertIn(c4, elts)

    def test_elementsWithQN(self):
        if False:
            return 10
        '\n        Calling C{elements} with a namespace and local name on a\n        L{domish.Element} returns all child elements with that qualified name.\n        '
        e = domish.Element(('testns', 'foo'))
        c1 = e.addElement('name')
        c2 = e.addElement(('testns2', 'baz'))
        c3 = e.addElement('quux')
        c4 = e.addElement(('testns', 'name'))
        elts = list(e.elements('testns', 'name'))
        self.assertIn(c1, elts)
        self.assertNotIn(c2, elts)
        self.assertNotIn(c3, elts)
        self.assertIn(c4, elts)

class DomishStreamTestsMixin:
    """
    Mixin defining tests for different stream implementations.

    @ivar streamClass: A no-argument callable which will be used to create an
        XML parser which can produce a stream of elements from incremental
        input.
    """

    def setUp(self):
        if False:
            return 10
        self.doc_started = False
        self.doc_ended = False
        self.root = None
        self.elements = []
        self.stream = self.streamClass()
        self.stream.DocumentStartEvent = self._docStarted
        self.stream.ElementEvent = self.elements.append
        self.stream.DocumentEndEvent = self._docEnded

    def _docStarted(self, root):
        if False:
            print('Hello World!')
        self.root = root
        self.doc_started = True

    def _docEnded(self):
        if False:
            print('Hello World!')
        self.doc_ended = True

    def doTest(self, xml):
        if False:
            print('Hello World!')
        self.stream.parse(xml)

    def testHarness(self):
        if False:
            return 10
        xml = b'<root><child/><child2/></root>'
        self.stream.parse(xml)
        self.assertEqual(self.doc_started, True)
        self.assertEqual(self.root.name, 'root')
        self.assertEqual(self.elements[0].name, 'child')
        self.assertEqual(self.elements[1].name, 'child2')
        self.assertEqual(self.doc_ended, True)

    def testBasic(self):
        if False:
            print('Hello World!')
        xml = b"<stream:stream xmlns:stream='etherx' xmlns='jabber'>\n" + b"  <message to='bar'>" + b"    <x xmlns='xdelay'>some&amp;data&gt;</x>" + b'  </message>' + b'</stream:stream>'
        self.stream.parse(xml)
        self.assertEqual(self.root.name, 'stream')
        self.assertEqual(self.root.uri, 'etherx')
        self.assertEqual(self.elements[0].name, 'message')
        self.assertEqual(self.elements[0].uri, 'jabber')
        self.assertEqual(self.elements[0]['to'], 'bar')
        self.assertEqual(self.elements[0].x.uri, 'xdelay')
        self.assertEqual(str(self.elements[0].x), 'some&data>')

    def testNoRootNS(self):
        if False:
            i = 10
            return i + 15
        xml = b"<stream><error xmlns='etherx'/></stream>"
        self.stream.parse(xml)
        self.assertEqual(self.root.uri, '')
        self.assertEqual(self.elements[0].uri, 'etherx')

    def testNoDefaultNS(self):
        if False:
            print('Hello World!')
        xml = b"<stream:stream xmlns:stream='etherx'><error/></stream:stream>"
        self.stream.parse(xml)
        self.assertEqual(self.root.uri, 'etherx')
        self.assertEqual(self.root.defaultUri, '')
        self.assertEqual(self.elements[0].uri, '')
        self.assertEqual(self.elements[0].defaultUri, '')

    def testChildDefaultNS(self):
        if False:
            return 10
        xml = b"<root xmlns='testns'><child/></root>"
        self.stream.parse(xml)
        self.assertEqual(self.root.uri, 'testns')
        self.assertEqual(self.elements[0].uri, 'testns')

    def testEmptyChildNS(self):
        if False:
            return 10
        xml = b"<root xmlns='testns'>\n                    <child1><child2 xmlns=''/></child1>\n                  </root>"
        self.stream.parse(xml)
        self.assertEqual(self.elements[0].child2.uri, '')

    def test_attributesWithNamespaces(self):
        if False:
            while True:
                i = 10
        '\n        Attributes with namespace are parsed without Exception.\n        (https://twistedmatrix.com/trac/ticket/9730 regression test)\n        '
        xml = b"<root xmlns:test='http://example.org' xml:lang='en'>\n                    <test:test>test</test:test>\n                  </root>"
        self.stream.parse(xml)
        self.assertEqual(self.elements[0].uri, 'http://example.org')

    def testChildPrefix(self):
        if False:
            while True:
                i = 10
        xml = b"<root xmlns='testns' xmlns:foo='testns2'><foo:child/></root>"
        self.stream.parse(xml)
        self.assertEqual(self.root.localPrefixes['foo'], 'testns2')
        self.assertEqual(self.elements[0].uri, 'testns2')

    def testUnclosedElement(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(domish.ParserError, self.stream.parse, b'<root><error></root>')

    def test_namespaceReuse(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that reuse of namespaces does affect an element's serialization.\n\n        When one element uses a prefix for a certain namespace, this is\n        stored in the C{localPrefixes} attribute of the element. We want\n        to make sure that elements created after such use, won't have this\n        prefix end up in their C{localPrefixes} attribute, too.\n        "
        xml = b"<root>\n                    <foo:child1 xmlns:foo='testns'/>\n                    <child2 xmlns='testns'/>\n                  </root>"
        self.stream.parse(xml)
        self.assertEqual('child1', self.elements[0].name)
        self.assertEqual('testns', self.elements[0].uri)
        self.assertEqual('', self.elements[0].defaultUri)
        self.assertEqual({'foo': 'testns'}, self.elements[0].localPrefixes)
        self.assertEqual('child2', self.elements[1].name)
        self.assertEqual('testns', self.elements[1].uri)
        self.assertEqual('testns', self.elements[1].defaultUri)
        self.assertEqual({}, self.elements[1].localPrefixes)

class DomishExpatStreamTests(DomishStreamTestsMixin, unittest.TestCase):
    """
    Tests for L{domish.ExpatElementStream}, the expat-based element stream
    implementation.
    """
    streamClass = domish.ExpatElementStream
    if requireModule('pyexpat', default=None) is None:
        skip = 'pyexpat is required for ExpatElementStream tests.'

class DomishSuxStreamTests(DomishStreamTestsMixin, unittest.TestCase):
    """
    Tests for L{domish.SuxElementStream}, the L{twisted.web.sux}-based element
    stream implementation.
    """
    streamClass = domish.SuxElementStream

class SerializerTests(unittest.TestCase):

    def testNoNamespace(self):
        if False:
            print('Hello World!')
        e = domish.Element((None, 'foo'))
        self.assertEqual(e.toXml(), '<foo/>')
        self.assertEqual(e.toXml(closeElement=0), '<foo>')

    def testDefaultNamespace(self):
        if False:
            print('Hello World!')
        e = domish.Element(('testns', 'foo'))
        self.assertEqual(e.toXml(), "<foo xmlns='testns'/>")

    def testOtherNamespace(self):
        if False:
            print('Hello World!')
        e = domish.Element(('testns', 'foo'), 'testns2')
        self.assertEqual(e.toXml({'testns': 'bar'}), "<bar:foo xmlns:bar='testns' xmlns='testns2'/>")

    def testChildDefaultNamespace(self):
        if False:
            while True:
                i = 10
        e = domish.Element(('testns', 'foo'))
        e.addElement('bar')
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")

    def testChildSameNamespace(self):
        if False:
            return 10
        e = domish.Element(('testns', 'foo'))
        e.addElement(('testns', 'bar'))
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")

    def testChildSameDefaultNamespace(self):
        if False:
            while True:
                i = 10
        e = domish.Element(('testns', 'foo'))
        e.addElement('bar', 'testns')
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")

    def testChildOtherDefaultNamespace(self):
        if False:
            return 10
        e = domish.Element(('testns', 'foo'))
        e.addElement(('testns2', 'bar'), 'testns2')
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar xmlns='testns2'/></foo>")

    def testOnlyChildDefaultNamespace(self):
        if False:
            return 10
        e = domish.Element((None, 'foo'))
        e.addElement(('ns2', 'bar'), 'ns2')
        self.assertEqual(e.toXml(), "<foo><bar xmlns='ns2'/></foo>")

    def testOnlyChildDefaultNamespace2(self):
        if False:
            i = 10
            return i + 15
        e = domish.Element((None, 'foo'))
        e.addElement('bar')
        self.assertEqual(e.toXml(), '<foo><bar/></foo>')

    def testChildInDefaultNamespace(self):
        if False:
            for i in range(10):
                print('nop')
        e = domish.Element(('testns', 'foo'), 'testns2')
        e.addElement(('testns2', 'bar'))
        self.assertEqual(e.toXml(), "<xn0:foo xmlns:xn0='testns' xmlns='testns2'><bar/></xn0:foo>")

    def testQualifiedAttribute(self):
        if False:
            while True:
                i = 10
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'/>")

    def testQualifiedAttributeDefaultNS(self):
        if False:
            while True:
                i = 10
        e = domish.Element(('testns', 'foo'), attribs={('testns', 'bar'): 'baz'})
        self.assertEqual(e.toXml(), "<foo xmlns='testns' xmlns:xn0='testns' xn0:bar='baz'/>")

    def testTwoChilds(self):
        if False:
            while True:
                i = 10
        e = domish.Element(('', 'foo'))
        child1 = e.addElement(('testns', 'bar'), 'testns2')
        child1.addElement(('testns2', 'quux'))
        child2 = e.addElement(('testns3', 'baz'), 'testns4')
        child2.addElement(('testns', 'quux'))
        self.assertEqual(e.toXml(), "<foo><xn0:bar xmlns:xn0='testns' xmlns='testns2'><quux/></xn0:bar><xn1:baz xmlns:xn1='testns3' xmlns='testns4'><xn0:quux xmlns:xn0='testns'/></xn1:baz></foo>")

    def testXMLNamespace(self):
        if False:
            while True:
                i = 10
        e = domish.Element((None, 'foo'), attribs={('http://www.w3.org/XML/1998/namespace', 'lang'): 'en_US'})
        self.assertEqual(e.toXml(), "<foo xml:lang='en_US'/>")

    def testQualifiedAttributeGivenListOfPrefixes(self):
        if False:
            print('Hello World!')
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        self.assertEqual(e.toXml({'testns2': 'qux'}), "<foo xmlns:qux='testns2' qux:bar='baz'/>")

    def testNSPrefix(self):
        if False:
            while True:
                i = 10
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        c = e.addElement(('testns2', 'qux'))
        c['testns2', 'bar'] = 'quux'
        self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'><xn0:qux xn0:bar='quux'/></foo>")

    def testDefaultNSPrefix(self):
        if False:
            i = 10
            return i + 15
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        c = e.addElement(('testns2', 'qux'))
        c['testns2', 'bar'] = 'quux'
        c.addElement('foo')
        self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'><xn0:qux xn0:bar='quux'><xn0:foo/></xn0:qux></foo>")

    def testPrefixScope(self):
        if False:
            i = 10
            return i + 15
        e = domish.Element(('testns', 'foo'))
        self.assertEqual(e.toXml(prefixes={'testns': 'bar'}, prefixesInScope=['bar']), '<bar:foo/>')

    def testLocalPrefixes(self):
        if False:
            print('Hello World!')
        e = domish.Element(('testns', 'foo'), localPrefixes={'bar': 'testns'})
        self.assertEqual(e.toXml(), "<bar:foo xmlns:bar='testns'/>")

    def testLocalPrefixesWithChild(self):
        if False:
            i = 10
            return i + 15
        e = domish.Element(('testns', 'foo'), localPrefixes={'bar': 'testns'})
        e.addElement('baz')
        self.assertIdentical(e.baz.defaultUri, None)
        self.assertEqual(e.toXml(), "<bar:foo xmlns:bar='testns'><baz/></bar:foo>")

    def test_prefixesReuse(self):
        if False:
            return 10
        '\n        Test that prefixes passed to serialization are not modified.\n\n        This test makes sure that passing a dictionary of prefixes repeatedly\n        to C{toXml} of elements does not cause serialization errors. A\n        previous implementation changed the passed in dictionary internally,\n        causing havoc later on.\n        '
        prefixes = {'testns': 'foo'}
        s = domish.SerializerClass(prefixes=prefixes)
        self.assertNotIdentical(prefixes, s.prefixes)
        e = domish.Element(('testns2', 'foo'), localPrefixes={'quux': 'testns2'})
        self.assertEqual("<quux:foo xmlns:quux='testns2'/>", e.toXml(prefixes=prefixes))
        e = domish.Element(('testns2', 'foo'))
        self.assertEqual("<foo xmlns='testns2'/>", e.toXml(prefixes=prefixes))

    def testRawXMLSerialization(self):
        if False:
            print('Hello World!')
        e = domish.Element((None, 'foo'))
        e.addRawXml('<abc123>')
        self.assertEqual(e.toXml(), '<foo><abc123></foo>')

    def testRawXMLWithUnicodeSerialization(self):
        if False:
            i = 10
            return i + 15
        e = domish.Element((None, 'foo'))
        e.addRawXml('<degree>°</degree>')
        self.assertEqual(e.toXml(), '<foo><degree>°</degree></foo>')

    def testUnicodeSerialization(self):
        if False:
            i = 10
            return i + 15
        e = domish.Element((None, 'foo'))
        e['test'] = 'my valueȡe'
        e.addContent('A degree symbol...°')
        self.assertEqual(e.toXml(), "<foo test='my valueȡe'>A degree symbol...°</foo>")