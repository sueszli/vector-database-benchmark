from xml.sax import make_parser, ContentHandler, SAXException, SAXReaderNotAvailable, SAXParseException
import unittest
from unittest import mock
try:
    make_parser()
except SAXReaderNotAvailable:
    raise unittest.SkipTest('no XML parsers available')
from xml.sax.saxutils import XMLGenerator, escape, unescape, quoteattr, XMLFilterBase, prepare_input_source
from xml.sax.expatreader import create_parser
from xml.sax.handler import feature_namespaces, feature_external_ges, LexicalHandler
from xml.sax.xmlreader import InputSource, AttributesImpl, AttributesNSImpl
from io import BytesIO, StringIO
import codecs
import os.path
import shutil
import sys
from urllib.error import URLError
import urllib.request
from test.support import os_helper
from test.support import findfile
from test.support.os_helper import FakePath, TESTFN
TEST_XMLFILE = findfile('test.xml', subdir='xmltestdata')
TEST_XMLFILE_OUT = findfile('test.xml.out', subdir='xmltestdata')
try:
    TEST_XMLFILE.encode('utf-8')
    TEST_XMLFILE_OUT.encode('utf-8')
except UnicodeEncodeError:
    raise unittest.SkipTest('filename is not encodable to utf8')
supports_nonascii_filenames = True
if not os.path.supports_unicode_filenames:
    try:
        os_helper.TESTFN_UNICODE.encode(sys.getfilesystemencoding())
    except (UnicodeError, TypeError):
        supports_nonascii_filenames = False
requires_nonascii_filenames = unittest.skipUnless(supports_nonascii_filenames, 'Requires non-ascii filenames support')
ns_uri = 'http://www.python.org/xml-ns/saxtest/'

class XmlTestBase(unittest.TestCase):

    def verify_empty_attrs(self, attrs):
        if False:
            while True:
                i = 10
        self.assertRaises(KeyError, attrs.getValue, 'attr')
        self.assertRaises(KeyError, attrs.getValueByQName, 'attr')
        self.assertRaises(KeyError, attrs.getNameByQName, 'attr')
        self.assertRaises(KeyError, attrs.getQNameByName, 'attr')
        self.assertRaises(KeyError, attrs.__getitem__, 'attr')
        self.assertEqual(attrs.getLength(), 0)
        self.assertEqual(attrs.getNames(), [])
        self.assertEqual(attrs.getQNames(), [])
        self.assertEqual(len(attrs), 0)
        self.assertNotIn('attr', attrs)
        self.assertEqual(list(attrs.keys()), [])
        self.assertEqual(attrs.get('attrs'), None)
        self.assertEqual(attrs.get('attrs', 25), 25)
        self.assertEqual(list(attrs.items()), [])
        self.assertEqual(list(attrs.values()), [])

    def verify_empty_nsattrs(self, attrs):
        if False:
            return 10
        self.assertRaises(KeyError, attrs.getValue, (ns_uri, 'attr'))
        self.assertRaises(KeyError, attrs.getValueByQName, 'ns:attr')
        self.assertRaises(KeyError, attrs.getNameByQName, 'ns:attr')
        self.assertRaises(KeyError, attrs.getQNameByName, (ns_uri, 'attr'))
        self.assertRaises(KeyError, attrs.__getitem__, (ns_uri, 'attr'))
        self.assertEqual(attrs.getLength(), 0)
        self.assertEqual(attrs.getNames(), [])
        self.assertEqual(attrs.getQNames(), [])
        self.assertEqual(len(attrs), 0)
        self.assertNotIn((ns_uri, 'attr'), attrs)
        self.assertEqual(list(attrs.keys()), [])
        self.assertEqual(attrs.get((ns_uri, 'attr')), None)
        self.assertEqual(attrs.get((ns_uri, 'attr'), 25), 25)
        self.assertEqual(list(attrs.items()), [])
        self.assertEqual(list(attrs.values()), [])

    def verify_attrs_wattr(self, attrs):
        if False:
            while True:
                i = 10
        self.assertEqual(attrs.getLength(), 1)
        self.assertEqual(attrs.getNames(), ['attr'])
        self.assertEqual(attrs.getQNames(), ['attr'])
        self.assertEqual(len(attrs), 1)
        self.assertIn('attr', attrs)
        self.assertEqual(list(attrs.keys()), ['attr'])
        self.assertEqual(attrs.get('attr'), 'val')
        self.assertEqual(attrs.get('attr', 25), 'val')
        self.assertEqual(list(attrs.items()), [('attr', 'val')])
        self.assertEqual(list(attrs.values()), ['val'])
        self.assertEqual(attrs.getValue('attr'), 'val')
        self.assertEqual(attrs.getValueByQName('attr'), 'val')
        self.assertEqual(attrs.getNameByQName('attr'), 'attr')
        self.assertEqual(attrs['attr'], 'val')
        self.assertEqual(attrs.getQNameByName('attr'), 'attr')

def xml_str(doc, encoding=None):
    if False:
        return 10
    if encoding is None:
        return doc
    return '<?xml version="1.0" encoding="%s"?>\n%s' % (encoding, doc)

def xml_bytes(doc, encoding, decl_encoding=...):
    if False:
        return 10
    if decl_encoding is ...:
        decl_encoding = encoding
    return xml_str(doc, decl_encoding).encode(encoding, 'xmlcharrefreplace')

def make_xml_file(doc, encoding, decl_encoding=...):
    if False:
        i = 10
        return i + 15
    if decl_encoding is ...:
        decl_encoding = encoding
    with open(TESTFN, 'w', encoding=encoding, errors='xmlcharrefreplace') as f:
        f.write(xml_str(doc, decl_encoding))

class ParseTest(unittest.TestCase):
    data = '<money value="$£€𐅻">$£€𐅻</money>'

    def tearDown(self):
        if False:
            return 10
        os_helper.unlink(TESTFN)

    def check_parse(self, f):
        if False:
            print('Hello World!')
        from xml.sax import parse
        result = StringIO()
        parse(f, XMLGenerator(result, 'utf-8'))
        self.assertEqual(result.getvalue(), xml_str(self.data, 'utf-8'))

    def test_parse_text(self):
        if False:
            for i in range(10):
                print('nop')
        encodings = ('us-ascii', 'iso-8859-1', 'utf-8', 'utf-16', 'utf-16le', 'utf-16be')
        for encoding in encodings:
            self.check_parse(StringIO(xml_str(self.data, encoding)))
            make_xml_file(self.data, encoding)
            with open(TESTFN, 'r', encoding=encoding) as f:
                self.check_parse(f)
            self.check_parse(StringIO(self.data))
            make_xml_file(self.data, encoding, None)
            with open(TESTFN, 'r', encoding=encoding) as f:
                self.check_parse(f)

    def test_parse_bytes(self):
        if False:
            print('Hello World!')
        encodings = ('us-ascii', 'utf-8', 'utf-16', 'utf-16le', 'utf-16be')
        for encoding in encodings:
            self.check_parse(BytesIO(xml_bytes(self.data, encoding)))
            make_xml_file(self.data, encoding)
            self.check_parse(TESTFN)
            with open(TESTFN, 'rb') as f:
                self.check_parse(f)
            self.check_parse(BytesIO(xml_bytes(self.data, encoding, None)))
            make_xml_file(self.data, encoding, None)
            self.check_parse(TESTFN)
            with open(TESTFN, 'rb') as f:
                self.check_parse(f)
        self.check_parse(BytesIO(xml_bytes(self.data, 'utf-8-sig', 'utf-8')))
        make_xml_file(self.data, 'utf-8-sig', 'utf-8')
        self.check_parse(TESTFN)
        with open(TESTFN, 'rb') as f:
            self.check_parse(f)
        self.check_parse(BytesIO(xml_bytes(self.data, 'utf-8-sig', None)))
        make_xml_file(self.data, 'utf-8-sig', None)
        self.check_parse(TESTFN)
        with open(TESTFN, 'rb') as f:
            self.check_parse(f)
        self.check_parse(BytesIO(xml_bytes(self.data, 'iso-8859-1')))
        make_xml_file(self.data, 'iso-8859-1')
        self.check_parse(TESTFN)
        with open(TESTFN, 'rb') as f:
            self.check_parse(f)
        with self.assertRaises(SAXException):
            self.check_parse(BytesIO(xml_bytes(self.data, 'iso-8859-1', None)))
        make_xml_file(self.data, 'iso-8859-1', None)
        with self.assertRaises(SAXException):
            self.check_parse(TESTFN)
        with open(TESTFN, 'rb') as f:
            with self.assertRaises(SAXException):
                self.check_parse(f)

    def test_parse_path_object(self):
        if False:
            for i in range(10):
                print('nop')
        make_xml_file(self.data, 'utf-8', None)
        self.check_parse(FakePath(TESTFN))

    def test_parse_InputSource(self):
        if False:
            print('Hello World!')
        make_xml_file(self.data, 'iso-8859-1', None)
        with open(TESTFN, 'rb') as f:
            input = InputSource()
            input.setByteStream(f)
            input.setEncoding('iso-8859-1')
            self.check_parse(input)

    def test_parse_close_source(self):
        if False:
            print('Hello World!')
        builtin_open = open
        fileobj = None

        def mock_open(*args):
            if False:
                i = 10
                return i + 15
            nonlocal fileobj
            fileobj = builtin_open(*args)
            return fileobj
        with mock.patch('xml.sax.saxutils.open', side_effect=mock_open):
            make_xml_file(self.data, 'iso-8859-1', None)
            with self.assertRaises(SAXException):
                self.check_parse(TESTFN)
            self.assertTrue(fileobj.closed)

    def check_parseString(self, s):
        if False:
            print('Hello World!')
        from xml.sax import parseString
        result = StringIO()
        parseString(s, XMLGenerator(result, 'utf-8'))
        self.assertEqual(result.getvalue(), xml_str(self.data, 'utf-8'))

    def test_parseString_text(self):
        if False:
            return 10
        encodings = ('us-ascii', 'iso-8859-1', 'utf-8', 'utf-16', 'utf-16le', 'utf-16be')
        for encoding in encodings:
            self.check_parseString(xml_str(self.data, encoding))
        self.check_parseString(self.data)

    def test_parseString_bytes(self):
        if False:
            i = 10
            return i + 15
        encodings = ('us-ascii', 'utf-8', 'utf-16', 'utf-16le', 'utf-16be')
        for encoding in encodings:
            self.check_parseString(xml_bytes(self.data, encoding))
            self.check_parseString(xml_bytes(self.data, encoding, None))
        self.check_parseString(xml_bytes(self.data, 'utf-8-sig', 'utf-8'))
        self.check_parseString(xml_bytes(self.data, 'utf-8-sig', None))
        self.check_parseString(xml_bytes(self.data, 'iso-8859-1'))
        with self.assertRaises(SAXException):
            self.check_parseString(xml_bytes(self.data, 'iso-8859-1', None))

class MakeParserTest(unittest.TestCase):

    def test_make_parser2(self):
        if False:
            print('Hello World!')
        from xml.sax import make_parser
        p = make_parser()
        from xml.sax import make_parser
        p = make_parser()
        from xml.sax import make_parser
        p = make_parser()
        from xml.sax import make_parser
        p = make_parser()
        from xml.sax import make_parser
        p = make_parser()
        from xml.sax import make_parser
        p = make_parser()

    def test_make_parser3(self):
        if False:
            for i in range(10):
                print('nop')
        make_parser(['module'])
        make_parser(('module',))
        make_parser({'module'})
        make_parser(frozenset({'module'}))
        make_parser({'module': None})
        make_parser(iter(['module']))

    def test_make_parser4(self):
        if False:
            for i in range(10):
                print('nop')
        make_parser([])
        make_parser(tuple())
        make_parser(set())
        make_parser(frozenset())
        make_parser({})
        make_parser(iter([]))

    def test_make_parser5(self):
        if False:
            while True:
                i = 10
        make_parser(['module1', 'module2'])
        make_parser(('module1', 'module2'))
        make_parser({'module1', 'module2'})
        make_parser(frozenset({'module1', 'module2'}))
        make_parser({'module1': None, 'module2': None})
        make_parser(iter(['module1', 'module2']))

class SaxutilsTest(unittest.TestCase):

    def test_escape_basic(self):
        if False:
            print('Hello World!')
        self.assertEqual(escape('Donald Duck & Co'), 'Donald Duck &amp; Co')

    def test_escape_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(escape('<Donald Duck & Co>'), '&lt;Donald Duck &amp; Co&gt;')

    def test_escape_extra(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(escape('Hei på deg', {'å': '&aring;'}), 'Hei p&aring; deg')

    def test_unescape_basic(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(unescape('Donald Duck &amp; Co'), 'Donald Duck & Co')

    def test_unescape_all(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(unescape('&lt;Donald Duck &amp; Co&gt;'), '<Donald Duck & Co>')

    def test_unescape_extra(self):
        if False:
            print('Hello World!')
        self.assertEqual(unescape('Hei på deg', {'å': '&aring;'}), 'Hei p&aring; deg')

    def test_unescape_amp_extra(self):
        if False:
            while True:
                i = 10
        self.assertEqual(unescape('&amp;foo;', {'&foo;': 'splat'}), '&foo;')

    def test_quoteattr_basic(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(quoteattr('Donald Duck & Co'), '"Donald Duck &amp; Co"')

    def test_single_quoteattr(self):
        if False:
            while True:
                i = 10
        self.assertEqual(quoteattr('Includes "double" quotes'), '\'Includes "double" quotes\'')

    def test_double_quoteattr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(quoteattr("Includes 'single' quotes"), '"Includes \'single\' quotes"')

    def test_single_double_quoteattr(self):
        if False:
            return 10
        self.assertEqual(quoteattr('Includes \'single\' and "double" quotes'), '"Includes \'single\' and &quot;double&quot; quotes"')

    def test_make_parser(self):
        if False:
            return 10
        p = make_parser(['xml.parsers.no_such_parser'])

class PrepareInputSourceTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.file = os_helper.TESTFN
        with open(self.file, 'w') as tmp:
            tmp.write('This was read from a file.')

    def tearDown(self):
        if False:
            return 10
        os_helper.unlink(self.file)

    def make_byte_stream(self):
        if False:
            i = 10
            return i + 15
        return BytesIO(b'This is a byte stream.')

    def make_character_stream(self):
        if False:
            i = 10
            return i + 15
        return StringIO('This is a character stream.')

    def checkContent(self, stream, content):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNotNone(stream)
        self.assertEqual(stream.read(), content)
        stream.close()

    def test_character_stream(self):
        if False:
            print('Hello World!')
        src = InputSource(self.file)
        src.setCharacterStream(self.make_character_stream())
        prep = prepare_input_source(src)
        self.assertIsNone(prep.getByteStream())
        self.checkContent(prep.getCharacterStream(), 'This is a character stream.')

    def test_byte_stream(self):
        if False:
            for i in range(10):
                print('nop')
        src = InputSource(self.file)
        src.setByteStream(self.make_byte_stream())
        prep = prepare_input_source(src)
        self.assertIsNone(prep.getCharacterStream())
        self.checkContent(prep.getByteStream(), b'This is a byte stream.')

    def test_system_id(self):
        if False:
            return 10
        src = InputSource(self.file)
        prep = prepare_input_source(src)
        self.assertIsNone(prep.getCharacterStream())
        self.checkContent(prep.getByteStream(), b'This was read from a file.')

    def test_string(self):
        if False:
            return 10
        prep = prepare_input_source(self.file)
        self.assertIsNone(prep.getCharacterStream())
        self.checkContent(prep.getByteStream(), b'This was read from a file.')

    def test_path_objects(self):
        if False:
            while True:
                i = 10
        prep = prepare_input_source(FakePath(self.file))
        self.assertIsNone(prep.getCharacterStream())
        self.checkContent(prep.getByteStream(), b'This was read from a file.')

    def test_binary_file(self):
        if False:
            print('Hello World!')
        prep = prepare_input_source(self.make_byte_stream())
        self.assertIsNone(prep.getCharacterStream())
        self.checkContent(prep.getByteStream(), b'This is a byte stream.')

    def test_text_file(self):
        if False:
            while True:
                i = 10
        prep = prepare_input_source(self.make_character_stream())
        self.assertIsNone(prep.getByteStream())
        self.checkContent(prep.getCharacterStream(), 'This is a character stream.')

class XmlgenTest:

    def test_xmlgen_basic(self):
        if False:
            return 10
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc></doc>'))

    def test_xmlgen_basic_empty(self):
        if False:
            i = 10
            return i + 15
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc/>'))

    def test_xmlgen_content(self):
        if False:
            while True:
                i = 10
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.characters('huhei')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc>huhei</doc>'))

    def test_xmlgen_content_empty(self):
        if False:
            return 10
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.characters('huhei')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc>huhei</doc>'))

    def test_xmlgen_pi(self):
        if False:
            print('Hello World!')
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.processingInstruction('test', 'data')
        gen.startElement('doc', {})
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<?test data?><doc></doc>'))

    def test_xmlgen_content_escape(self):
        if False:
            while True:
                i = 10
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.characters('<huhei&')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc>&lt;huhei&amp;</doc>'))

    def test_xmlgen_attr_escape(self):
        if False:
            i = 10
            return i + 15
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startElement('doc', {'a': '"'})
        gen.startElement('e', {'a': "'"})
        gen.endElement('e')
        gen.startElement('e', {'a': '\'"'})
        gen.endElement('e')
        gen.startElement('e', {'a': '\n\r\t'})
        gen.endElement('e')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc a=\'"\'><e a="\'"></e><e a="\'&quot;"></e><e a="&#10;&#13;&#9;"></e></doc>'))

    def test_xmlgen_encoding(self):
        if False:
            print('Hello World!')
        encodings = ('iso-8859-15', 'utf-8', 'utf-8-sig', 'utf-16', 'utf-16be', 'utf-16le', 'utf-32', 'utf-32be', 'utf-32le')
        for encoding in encodings:
            result = self.ioclass()
            gen = XMLGenerator(result, encoding=encoding)
            gen.startDocument()
            gen.startElement('doc', {'a': '€'})
            gen.characters('€')
            gen.endElement('doc')
            gen.endDocument()
            self.assertEqual(result.getvalue(), self.xml('<doc a="€">€</doc>', encoding=encoding))

    def test_xmlgen_unencodable(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.ioclass()
        gen = XMLGenerator(result, encoding='ascii')
        gen.startDocument()
        gen.startElement('doc', {'a': '€'})
        gen.characters('€')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc a="&#8364;">&#8364;</doc>', encoding='ascii'))

    def test_xmlgen_ignorable(self):
        if False:
            i = 10
            return i + 15
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.ignorableWhitespace(' ')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc> </doc>'))

    def test_xmlgen_ignorable_empty(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startElement('doc', {})
        gen.ignorableWhitespace(' ')
        gen.endElement('doc')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<doc> </doc>'))

    def test_xmlgen_encoding_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        encodings = ('iso-8859-15', 'utf-8', 'utf-8-sig', 'utf-16', 'utf-16be', 'utf-16le', 'utf-32', 'utf-32be', 'utf-32le')
        for encoding in encodings:
            result = self.ioclass()
            gen = XMLGenerator(result, encoding=encoding)
            gen.startDocument()
            gen.startElement('doc', {'a': '€'})
            gen.characters('€'.encode(encoding))
            gen.ignorableWhitespace(' '.encode(encoding))
            gen.endElement('doc')
            gen.endDocument()
            self.assertEqual(result.getvalue(), self.xml('<doc a="€">€ </doc>', encoding=encoding))

    def test_xmlgen_ns(self):
        if False:
            print('Hello World!')
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startPrefixMapping('ns1', ns_uri)
        gen.startElementNS((ns_uri, 'doc'), 'ns1:doc', {})
        gen.startElementNS((None, 'udoc'), None, {})
        gen.endElementNS((None, 'udoc'), None)
        gen.endElementNS((ns_uri, 'doc'), 'ns1:doc')
        gen.endPrefixMapping('ns1')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<ns1:doc xmlns:ns1="%s"><udoc></udoc></ns1:doc>' % ns_uri))

    def test_xmlgen_ns_empty(self):
        if False:
            print('Hello World!')
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startPrefixMapping('ns1', ns_uri)
        gen.startElementNS((ns_uri, 'doc'), 'ns1:doc', {})
        gen.startElementNS((None, 'udoc'), None, {})
        gen.endElementNS((None, 'udoc'), None)
        gen.endElementNS((ns_uri, 'doc'), 'ns1:doc')
        gen.endPrefixMapping('ns1')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<ns1:doc xmlns:ns1="%s"><udoc/></ns1:doc>' % ns_uri))

    def test_1463026_1(self):
        if False:
            i = 10
            return i + 15
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startElementNS((None, 'a'), 'a', {(None, 'b'): 'c'})
        gen.endElementNS((None, 'a'), 'a')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<a b="c"></a>'))

    def test_1463026_1_empty(self):
        if False:
            return 10
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startElementNS((None, 'a'), 'a', {(None, 'b'): 'c'})
        gen.endElementNS((None, 'a'), 'a')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<a b="c"/>'))

    def test_1463026_2(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startPrefixMapping(None, 'qux')
        gen.startElementNS(('qux', 'a'), 'a', {})
        gen.endElementNS(('qux', 'a'), 'a')
        gen.endPrefixMapping(None)
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<a xmlns="qux"></a>'))

    def test_1463026_2_empty(self):
        if False:
            return 10
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startPrefixMapping(None, 'qux')
        gen.startElementNS(('qux', 'a'), 'a', {})
        gen.endElementNS(('qux', 'a'), 'a')
        gen.endPrefixMapping(None)
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<a xmlns="qux"/>'))

    def test_1463026_3(self):
        if False:
            while True:
                i = 10
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startPrefixMapping('my', 'qux')
        gen.startElementNS(('qux', 'a'), 'a', {(None, 'b'): 'c'})
        gen.endElementNS(('qux', 'a'), 'a')
        gen.endPrefixMapping('my')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<my:a xmlns:my="qux" b="c"></my:a>'))

    def test_1463026_3_empty(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.ioclass()
        gen = XMLGenerator(result, short_empty_elements=True)
        gen.startDocument()
        gen.startPrefixMapping('my', 'qux')
        gen.startElementNS(('qux', 'a'), 'a', {(None, 'b'): 'c'})
        gen.endElementNS(('qux', 'a'), 'a')
        gen.endPrefixMapping('my')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<my:a xmlns:my="qux" b="c"/>'))

    def test_5027_1(self):
        if False:
            while True:
                i = 10
        test_xml = StringIO('<?xml version="1.0"?><a:g1 xmlns:a="http://example.com/ns"><a:g2 xml:lang="en">Hello</a:g2></a:g1>')
        parser = make_parser()
        parser.setFeature(feature_namespaces, True)
        result = self.ioclass()
        gen = XMLGenerator(result)
        parser.setContentHandler(gen)
        parser.parse(test_xml)
        self.assertEqual(result.getvalue(), self.xml('<a:g1 xmlns:a="http://example.com/ns"><a:g2 xml:lang="en">Hello</a:g2></a:g1>'))

    def test_5027_2(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startDocument()
        gen.startPrefixMapping('a', 'http://example.com/ns')
        gen.startElementNS(('http://example.com/ns', 'g1'), 'g1', {})
        lang_attr = {('http://www.w3.org/XML/1998/namespace', 'lang'): 'en'}
        gen.startElementNS(('http://example.com/ns', 'g2'), 'g2', lang_attr)
        gen.characters('Hello')
        gen.endElementNS(('http://example.com/ns', 'g2'), 'g2')
        gen.endElementNS(('http://example.com/ns', 'g1'), 'g1')
        gen.endPrefixMapping('a')
        gen.endDocument()
        self.assertEqual(result.getvalue(), self.xml('<a:g1 xmlns:a="http://example.com/ns"><a:g2 xml:lang="en">Hello</a:g2></a:g1>'))

    def test_no_close_file(self):
        if False:
            i = 10
            return i + 15
        result = self.ioclass()

        def func(out):
            if False:
                i = 10
                return i + 15
            gen = XMLGenerator(out)
            gen.startDocument()
            gen.startElement('doc', {})
        func(result)
        self.assertFalse(result.closed)

    def test_xmlgen_fragment(self):
        if False:
            while True:
                i = 10
        result = self.ioclass()
        gen = XMLGenerator(result)
        gen.startElement('foo', {'a': '1.0'})
        gen.characters('Hello')
        gen.endElement('foo')
        gen.startElement('bar', {'b': '2.0'})
        gen.endElement('bar')
        self.assertEqual(result.getvalue(), self.xml('<foo a="1.0">Hello</foo><bar b="2.0"></bar>')[len(self.xml('')):])

class StringXmlgenTest(XmlgenTest, unittest.TestCase):
    ioclass = StringIO

    def xml(self, doc, encoding='iso-8859-1'):
        if False:
            i = 10
            return i + 15
        return '<?xml version="1.0" encoding="%s"?>\n%s' % (encoding, doc)
    test_xmlgen_unencodable = None

class BytesXmlgenTest(XmlgenTest, unittest.TestCase):
    ioclass = BytesIO

    def xml(self, doc, encoding='iso-8859-1'):
        if False:
            print('Hello World!')
        return ('<?xml version="1.0" encoding="%s"?>\n%s' % (encoding, doc)).encode(encoding, 'xmlcharrefreplace')

class WriterXmlgenTest(BytesXmlgenTest):

    class ioclass(list):
        write = list.append
        closed = False

        def seekable(self):
            if False:
                return 10
            return True

        def tell(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self)

        def getvalue(self):
            if False:
                for i in range(10):
                    print('nop')
            return b''.join(self)

class StreamWriterXmlgenTest(XmlgenTest, unittest.TestCase):

    def ioclass(self):
        if False:
            while True:
                i = 10
        raw = BytesIO()
        writer = codecs.getwriter('ascii')(raw, 'xmlcharrefreplace')
        writer.getvalue = raw.getvalue
        return writer

    def xml(self, doc, encoding='iso-8859-1'):
        if False:
            while True:
                i = 10
        return ('<?xml version="1.0" encoding="%s"?>\n%s' % (encoding, doc)).encode('ascii', 'xmlcharrefreplace')

class StreamReaderWriterXmlgenTest(XmlgenTest, unittest.TestCase):
    fname = os_helper.TESTFN + '-codecs'

    def ioclass(self):
        if False:
            return 10
        writer = codecs.open(self.fname, 'w', encoding='ascii', errors='xmlcharrefreplace', buffering=0)

        def cleanup():
            if False:
                for i in range(10):
                    print('nop')
            writer.close()
            os_helper.unlink(self.fname)
        self.addCleanup(cleanup)

        def getvalue():
            if False:
                for i in range(10):
                    print('nop')
            writer.close()
            with open(writer.name, 'rb') as f:
                return f.read()
        writer.getvalue = getvalue
        return writer

    def xml(self, doc, encoding='iso-8859-1'):
        if False:
            i = 10
            return i + 15
        return ('<?xml version="1.0" encoding="%s"?>\n%s' % (encoding, doc)).encode('ascii', 'xmlcharrefreplace')
start = b'<?xml version="1.0" encoding="iso-8859-1"?>\n'

class XMLFilterBaseTest(unittest.TestCase):

    def test_filter_basic(self):
        if False:
            return 10
        result = BytesIO()
        gen = XMLGenerator(result)
        filter = XMLFilterBase()
        filter.setContentHandler(gen)
        filter.startDocument()
        filter.startElement('doc', {})
        filter.characters('content')
        filter.ignorableWhitespace(' ')
        filter.endElement('doc')
        filter.endDocument()
        self.assertEqual(result.getvalue(), start + b'<doc>content </doc>')
with open(TEST_XMLFILE_OUT, 'rb') as f:
    xml_test_out = f.read()

class ExpatReaderTest(XmlTestBase):

    def test_expat_binary_file(self):
        if False:
            while True:
                i = 10
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        with open(TEST_XMLFILE, 'rb') as f:
            parser.parse(f)
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_text_file(self):
        if False:
            print('Hello World!')
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        with open(TEST_XMLFILE, 'rt', encoding='iso-8859-1') as f:
            parser.parse(f)
        self.assertEqual(result.getvalue(), xml_test_out)

    @requires_nonascii_filenames
    def test_expat_binary_file_nonascii(self):
        if False:
            print('Hello World!')
        fname = os_helper.TESTFN_UNICODE
        shutil.copyfile(TEST_XMLFILE, fname)
        self.addCleanup(os_helper.unlink, fname)
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        parser.parse(open(fname, 'rb'))
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_binary_file_bytes_name(self):
        if False:
            return 10
        fname = os.fsencode(TEST_XMLFILE)
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        with open(fname, 'rb') as f:
            parser.parse(f)
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_binary_file_int_name(self):
        if False:
            print('Hello World!')
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        with open(TEST_XMLFILE, 'rb') as f:
            with open(f.fileno(), 'rb', closefd=False) as f2:
                parser.parse(f2)
        self.assertEqual(result.getvalue(), xml_test_out)

    class TestDTDHandler:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self._notations = []
            self._entities = []

        def notationDecl(self, name, publicId, systemId):
            if False:
                i = 10
                return i + 15
            self._notations.append((name, publicId, systemId))

        def unparsedEntityDecl(self, name, publicId, systemId, ndata):
            if False:
                return 10
            self._entities.append((name, publicId, systemId, ndata))

    class TestEntityRecorder:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.entities = []

        def resolveEntity(self, publicId, systemId):
            if False:
                return 10
            self.entities.append((publicId, systemId))
            source = InputSource()
            source.setPublicId(publicId)
            source.setSystemId(systemId)
            return source

    def test_expat_dtdhandler(self):
        if False:
            print('Hello World!')
        parser = create_parser()
        handler = self.TestDTDHandler()
        parser.setDTDHandler(handler)
        parser.feed('<!DOCTYPE doc [\n')
        parser.feed('  <!ENTITY img SYSTEM "expat.gif" NDATA GIF>\n')
        parser.feed('  <!NOTATION GIF PUBLIC "-//CompuServe//NOTATION Graphics Interchange Format 89a//EN">\n')
        parser.feed(']>\n')
        parser.feed('<doc></doc>')
        parser.close()
        self.assertEqual(handler._notations, [('GIF', '-//CompuServe//NOTATION Graphics Interchange Format 89a//EN', None)])
        self.assertEqual(handler._entities, [('img', None, 'expat.gif', 'GIF')])

    def test_expat_external_dtd_enabled(self):
        if False:
            print('Hello World!')
        self.addCleanup(urllib.request.urlcleanup)
        parser = create_parser()
        parser.setFeature(feature_external_ges, True)
        resolver = self.TestEntityRecorder()
        parser.setEntityResolver(resolver)
        with self.assertRaises(URLError):
            parser.feed('<!DOCTYPE external SYSTEM "unsupported://non-existing">\n')
        self.assertEqual(resolver.entities, [(None, 'unsupported://non-existing')])

    def test_expat_external_dtd_default(self):
        if False:
            i = 10
            return i + 15
        parser = create_parser()
        resolver = self.TestEntityRecorder()
        parser.setEntityResolver(resolver)
        parser.feed('<!DOCTYPE external SYSTEM "unsupported://non-existing">\n')
        parser.feed('<doc />')
        parser.close()
        self.assertEqual(resolver.entities, [])

    class TestEntityResolver:

        def resolveEntity(self, publicId, systemId):
            if False:
                i = 10
                return i + 15
            inpsrc = InputSource()
            inpsrc.setByteStream(BytesIO(b'<entity/>'))
            return inpsrc

    def test_expat_entityresolver_enabled(self):
        if False:
            i = 10
            return i + 15
        parser = create_parser()
        parser.setFeature(feature_external_ges, True)
        parser.setEntityResolver(self.TestEntityResolver())
        result = BytesIO()
        parser.setContentHandler(XMLGenerator(result))
        parser.feed('<!DOCTYPE doc [\n')
        parser.feed('  <!ENTITY test SYSTEM "whatever">\n')
        parser.feed(']>\n')
        parser.feed('<doc>&test;</doc>')
        parser.close()
        self.assertEqual(result.getvalue(), start + b'<doc><entity></entity></doc>')

    def test_expat_entityresolver_default(self):
        if False:
            print('Hello World!')
        parser = create_parser()
        self.assertEqual(parser.getFeature(feature_external_ges), False)
        parser.setEntityResolver(self.TestEntityResolver())
        result = BytesIO()
        parser.setContentHandler(XMLGenerator(result))
        parser.feed('<!DOCTYPE doc [\n')
        parser.feed('  <!ENTITY test SYSTEM "whatever">\n')
        parser.feed(']>\n')
        parser.feed('<doc>&test;</doc>')
        parser.close()
        self.assertEqual(result.getvalue(), start + b'<doc></doc>')

    class AttrGatherer(ContentHandler):

        def startElement(self, name, attrs):
            if False:
                print('Hello World!')
            self._attrs = attrs

        def startElementNS(self, name, qname, attrs):
            if False:
                i = 10
                return i + 15
            self._attrs = attrs

    def test_expat_attrs_empty(self):
        if False:
            while True:
                i = 10
        parser = create_parser()
        gather = self.AttrGatherer()
        parser.setContentHandler(gather)
        parser.feed('<doc/>')
        parser.close()
        self.verify_empty_attrs(gather._attrs)

    def test_expat_attrs_wattr(self):
        if False:
            print('Hello World!')
        parser = create_parser()
        gather = self.AttrGatherer()
        parser.setContentHandler(gather)
        parser.feed("<doc attr='val'/>")
        parser.close()
        self.verify_attrs_wattr(gather._attrs)

    def test_expat_nsattrs_empty(self):
        if False:
            i = 10
            return i + 15
        parser = create_parser(1)
        gather = self.AttrGatherer()
        parser.setContentHandler(gather)
        parser.feed('<doc/>')
        parser.close()
        self.verify_empty_nsattrs(gather._attrs)

    def test_expat_nsattrs_wattr(self):
        if False:
            while True:
                i = 10
        parser = create_parser(1)
        gather = self.AttrGatherer()
        parser.setContentHandler(gather)
        parser.feed("<doc xmlns:ns='%s' ns:attr='val'/>" % ns_uri)
        parser.close()
        attrs = gather._attrs
        self.assertEqual(attrs.getLength(), 1)
        self.assertEqual(attrs.getNames(), [(ns_uri, 'attr')])
        self.assertTrue(attrs.getQNames() == [] or attrs.getQNames() == ['ns:attr'])
        self.assertEqual(len(attrs), 1)
        self.assertIn((ns_uri, 'attr'), attrs)
        self.assertEqual(attrs.get((ns_uri, 'attr')), 'val')
        self.assertEqual(attrs.get((ns_uri, 'attr'), 25), 'val')
        self.assertEqual(list(attrs.items()), [((ns_uri, 'attr'), 'val')])
        self.assertEqual(list(attrs.values()), ['val'])
        self.assertEqual(attrs.getValue((ns_uri, 'attr')), 'val')
        self.assertEqual(attrs[ns_uri, 'attr'], 'val')

    def test_expat_inpsource_filename(self):
        if False:
            while True:
                i = 10
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        parser.parse(TEST_XMLFILE)
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_inpsource_sysid(self):
        if False:
            for i in range(10):
                print('nop')
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        parser.parse(InputSource(TEST_XMLFILE))
        self.assertEqual(result.getvalue(), xml_test_out)

    @requires_nonascii_filenames
    def test_expat_inpsource_sysid_nonascii(self):
        if False:
            for i in range(10):
                print('nop')
        fname = os_helper.TESTFN_UNICODE
        shutil.copyfile(TEST_XMLFILE, fname)
        self.addCleanup(os_helper.unlink, fname)
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        parser.parse(InputSource(fname))
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_inpsource_byte_stream(self):
        if False:
            while True:
                i = 10
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        inpsrc = InputSource()
        with open(TEST_XMLFILE, 'rb') as f:
            inpsrc.setByteStream(f)
            parser.parse(inpsrc)
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_inpsource_character_stream(self):
        if False:
            for i in range(10):
                print('nop')
        parser = create_parser()
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        inpsrc = InputSource()
        with open(TEST_XMLFILE, 'rt', encoding='iso-8859-1') as f:
            inpsrc.setCharacterStream(f)
            parser.parse(inpsrc)
        self.assertEqual(result.getvalue(), xml_test_out)

    def test_expat_incremental(self):
        if False:
            for i in range(10):
                print('nop')
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser = create_parser()
        parser.setContentHandler(xmlgen)
        parser.feed('<doc>')
        parser.feed('</doc>')
        parser.close()
        self.assertEqual(result.getvalue(), start + b'<doc></doc>')

    def test_expat_incremental_reset(self):
        if False:
            print('Hello World!')
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser = create_parser()
        parser.setContentHandler(xmlgen)
        parser.feed('<doc>')
        parser.feed('text')
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser.setContentHandler(xmlgen)
        parser.reset()
        parser.feed('<doc>')
        parser.feed('text')
        parser.feed('</doc>')
        parser.close()
        self.assertEqual(result.getvalue(), start + b'<doc>text</doc>')

    def test_expat_locator_noinfo(self):
        if False:
            print('Hello World!')
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser = create_parser()
        parser.setContentHandler(xmlgen)
        parser.feed('<doc>')
        parser.feed('</doc>')
        parser.close()
        self.assertEqual(parser.getSystemId(), None)
        self.assertEqual(parser.getPublicId(), None)
        self.assertEqual(parser.getLineNumber(), 1)

    def test_expat_locator_withinfo(self):
        if False:
            return 10
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser = create_parser()
        parser.setContentHandler(xmlgen)
        parser.parse(TEST_XMLFILE)
        self.assertEqual(parser.getSystemId(), TEST_XMLFILE)
        self.assertEqual(parser.getPublicId(), None)

    @requires_nonascii_filenames
    def test_expat_locator_withinfo_nonascii(self):
        if False:
            i = 10
            return i + 15
        fname = os_helper.TESTFN_UNICODE
        shutil.copyfile(TEST_XMLFILE, fname)
        self.addCleanup(os_helper.unlink, fname)
        result = BytesIO()
        xmlgen = XMLGenerator(result)
        parser = create_parser()
        parser.setContentHandler(xmlgen)
        parser.parse(fname)
        self.assertEqual(parser.getSystemId(), fname)
        self.assertEqual(parser.getPublicId(), None)

class ErrorReportingTest(unittest.TestCase):

    def test_expat_inpsource_location(self):
        if False:
            while True:
                i = 10
        parser = create_parser()
        parser.setContentHandler(ContentHandler())
        source = InputSource()
        source.setByteStream(BytesIO(b'<foo bar foobar>'))
        name = 'a file name'
        source.setSystemId(name)
        try:
            parser.parse(source)
            self.fail()
        except SAXException as e:
            self.assertEqual(e.getSystemId(), name)

    def test_expat_incomplete(self):
        if False:
            print('Hello World!')
        parser = create_parser()
        parser.setContentHandler(ContentHandler())
        self.assertRaises(SAXParseException, parser.parse, StringIO('<foo>'))
        self.assertEqual(parser.getColumnNumber(), 5)
        self.assertEqual(parser.getLineNumber(), 1)

    def test_sax_parse_exception_str(self):
        if False:
            print('Hello World!')
        str(SAXParseException('message', None, self.DummyLocator(1, 1)))
        str(SAXParseException('message', None, self.DummyLocator(None, 1)))
        str(SAXParseException('message', None, self.DummyLocator(1, None)))
        str(SAXParseException('message', None, self.DummyLocator(None, None)))

    class DummyLocator:

        def __init__(self, lineno, colno):
            if False:
                i = 10
                return i + 15
            self._lineno = lineno
            self._colno = colno

        def getPublicId(self):
            if False:
                while True:
                    i = 10
            return 'pubid'

        def getSystemId(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'sysid'

        def getLineNumber(self):
            if False:
                i = 10
                return i + 15
            return self._lineno

        def getColumnNumber(self):
            if False:
                while True:
                    i = 10
            return self._colno

class XmlReaderTest(XmlTestBase):

    def test_attrs_empty(self):
        if False:
            print('Hello World!')
        self.verify_empty_attrs(AttributesImpl({}))

    def test_attrs_wattr(self):
        if False:
            return 10
        self.verify_attrs_wattr(AttributesImpl({'attr': 'val'}))

    def test_nsattrs_empty(self):
        if False:
            return 10
        self.verify_empty_nsattrs(AttributesNSImpl({}, {}))

    def test_nsattrs_wattr(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = AttributesNSImpl({(ns_uri, 'attr'): 'val'}, {(ns_uri, 'attr'): 'ns:attr'})
        self.assertEqual(attrs.getLength(), 1)
        self.assertEqual(attrs.getNames(), [(ns_uri, 'attr')])
        self.assertEqual(attrs.getQNames(), ['ns:attr'])
        self.assertEqual(len(attrs), 1)
        self.assertIn((ns_uri, 'attr'), attrs)
        self.assertEqual(list(attrs.keys()), [(ns_uri, 'attr')])
        self.assertEqual(attrs.get((ns_uri, 'attr')), 'val')
        self.assertEqual(attrs.get((ns_uri, 'attr'), 25), 'val')
        self.assertEqual(list(attrs.items()), [((ns_uri, 'attr'), 'val')])
        self.assertEqual(list(attrs.values()), ['val'])
        self.assertEqual(attrs.getValue((ns_uri, 'attr')), 'val')
        self.assertEqual(attrs.getValueByQName('ns:attr'), 'val')
        self.assertEqual(attrs.getNameByQName('ns:attr'), (ns_uri, 'attr'))
        self.assertEqual(attrs[ns_uri, 'attr'], 'val')
        self.assertEqual(attrs.getQNameByName((ns_uri, 'attr')), 'ns:attr')

class LexicalHandlerTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = None
        self.specified_version = '1.0'
        self.specified_encoding = 'UTF-8'
        self.specified_doctype = 'wish'
        self.specified_entity_names = ('nbsp', 'source', 'target')
        self.specified_comment = ('Comment in a DTD', 'Really! You think so?')
        self.test_data = StringIO()
        self.test_data.write('<?xml version="{}" encoding="{}"?>\n'.format(self.specified_version, self.specified_encoding))
        self.test_data.write('<!DOCTYPE {} [\n'.format(self.specified_doctype))
        self.test_data.write('<!-- {} -->\n'.format(self.specified_comment[0]))
        self.test_data.write('<!ELEMENT {} (to,from,heading,body,footer)>\n'.format(self.specified_doctype))
        self.test_data.write('<!ELEMENT to (#PCDATA)>\n')
        self.test_data.write('<!ELEMENT from (#PCDATA)>\n')
        self.test_data.write('<!ELEMENT heading (#PCDATA)>\n')
        self.test_data.write('<!ELEMENT body (#PCDATA)>\n')
        self.test_data.write('<!ELEMENT footer (#PCDATA)>\n')
        self.test_data.write('<!ENTITY {} "&#xA0;">\n'.format(self.specified_entity_names[0]))
        self.test_data.write('<!ENTITY {} "Written by: Alexander.">\n'.format(self.specified_entity_names[1]))
        self.test_data.write('<!ENTITY {} "Hope it gets to: Aristotle.">\n'.format(self.specified_entity_names[2]))
        self.test_data.write(']>\n')
        self.test_data.write('<{}>'.format(self.specified_doctype))
        self.test_data.write('<to>Aristotle</to>\n')
        self.test_data.write('<from>Alexander</from>\n')
        self.test_data.write('<heading>Supplication</heading>\n')
        self.test_data.write('<body>Teach me patience!</body>\n')
        self.test_data.write('<footer>&{};&{};&{};</footer>\n'.format(self.specified_entity_names[1], self.specified_entity_names[0], self.specified_entity_names[2]))
        self.test_data.write('<!-- {} -->\n'.format(self.specified_comment[1]))
        self.test_data.write('</{}>\n'.format(self.specified_doctype))
        self.test_data.seek(0)
        self.version = None
        self.encoding = None
        self.standalone = None
        self.doctype = None
        self.publicID = None
        self.systemID = None
        self.end_of_dtd = False
        self.comments = []

    def test_handlers(self):
        if False:
            print('Hello World!')

        class TestLexicalHandler(LexicalHandler):

            def __init__(self, test_harness, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__(*args, **kwargs)
                self.test_harness = test_harness

            def startDTD(self, doctype, publicID, systemID):
                if False:
                    while True:
                        i = 10
                self.test_harness.doctype = doctype
                self.test_harness.publicID = publicID
                self.test_harness.systemID = systemID

            def endDTD(self):
                if False:
                    while True:
                        i = 10
                self.test_harness.end_of_dtd = True

            def comment(self, text):
                if False:
                    while True:
                        i = 10
                self.test_harness.comments.append(text)
        self.parser = create_parser()
        self.parser.setContentHandler(ContentHandler())
        self.parser.setProperty('http://xml.org/sax/properties/lexical-handler', TestLexicalHandler(self))
        source = InputSource()
        source.setCharacterStream(self.test_data)
        self.parser.parse(source)
        self.assertEqual(self.doctype, self.specified_doctype)
        self.assertIsNone(self.publicID)
        self.assertIsNone(self.systemID)
        self.assertTrue(self.end_of_dtd)
        self.assertEqual(len(self.comments), len(self.specified_comment))
        self.assertEqual(f' {self.specified_comment[0]} ', self.comments[0])

class CDATAHandlerTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.parser = None
        self.specified_chars = []
        self.specified_chars.append(('Parseable character data', False))
        self.specified_chars.append(('<> &% - assorted other XML junk.', True))
        self.char_index = 0
        self.test_data = StringIO()
        self.test_data.write('<root_doc>\n')
        self.test_data.write('<some_pcdata>\n')
        self.test_data.write(f'{self.specified_chars[0][0]}\n')
        self.test_data.write('</some_pcdata>\n')
        self.test_data.write('<some_cdata>\n')
        self.test_data.write(f'<![CDATA[{self.specified_chars[1][0]}]]>\n')
        self.test_data.write('</some_cdata>\n')
        self.test_data.write('</root_doc>\n')
        self.test_data.seek(0)
        self.chardata = []
        self.in_cdata = False

    def test_handlers(self):
        if False:
            for i in range(10):
                print('nop')

        class TestLexicalHandler(LexicalHandler):

            def __init__(self, test_harness, *args, **kwargs):
                if False:
                    return 10
                super().__init__(*args, **kwargs)
                self.test_harness = test_harness

            def startCDATA(self):
                if False:
                    i = 10
                    return i + 15
                self.test_harness.in_cdata = True

            def endCDATA(self):
                if False:
                    return 10
                self.test_harness.in_cdata = False

        class TestCharHandler(ContentHandler):

            def __init__(self, test_harness, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__(*args, **kwargs)
                self.test_harness = test_harness

            def characters(self, content):
                if False:
                    return 10
                if content != '\n':
                    h = self.test_harness
                    t = h.specified_chars[h.char_index]
                    h.assertEqual(t[0], content)
                    h.assertEqual(t[1], h.in_cdata)
                    h.char_index += 1
        self.parser = create_parser()
        self.parser.setContentHandler(TestCharHandler(self))
        self.parser.setProperty('http://xml.org/sax/properties/lexical-handler', TestLexicalHandler(self))
        source = InputSource()
        source.setCharacterStream(self.test_data)
        self.parser.parse(source)
        self.assertFalse(self.in_cdata)
        self.assertEqual(self.char_index, 2)
if __name__ == '__main__':
    unittest.main()