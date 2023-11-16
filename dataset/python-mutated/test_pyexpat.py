from io import BytesIO
import os
import platform
import sys
import sysconfig
import unittest
import traceback
from xml.parsers import expat
from xml.parsers.expat import errors
from test.support import sortdict

class SetAttributeTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.parser = expat.ParserCreate(namespace_separator='!')

    def test_buffer_text(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIs(self.parser.buffer_text, False)
        for x in (0, 1, 2, 0):
            self.parser.buffer_text = x
            self.assertIs(self.parser.buffer_text, bool(x))

    def test_namespace_prefixes(self):
        if False:
            return 10
        self.assertIs(self.parser.namespace_prefixes, False)
        for x in (0, 1, 2, 0):
            self.parser.namespace_prefixes = x
            self.assertIs(self.parser.namespace_prefixes, bool(x))

    def test_ordered_attributes(self):
        if False:
            while True:
                i = 10
        self.assertIs(self.parser.ordered_attributes, False)
        for x in (0, 1, 2, 0):
            self.parser.ordered_attributes = x
            self.assertIs(self.parser.ordered_attributes, bool(x))

    def test_specified_attributes(self):
        if False:
            while True:
                i = 10
        self.assertIs(self.parser.specified_attributes, False)
        for x in (0, 1, 2, 0):
            self.parser.specified_attributes = x
            self.assertIs(self.parser.specified_attributes, bool(x))

    def test_invalid_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AttributeError):
            self.parser.returns_unicode = 1
        with self.assertRaises(AttributeError):
            self.parser.returns_unicode
        self.assertRaises(TypeError, setattr, self.parser, range(15), 0)
        self.assertRaises(TypeError, self.parser.__setattr__, range(15), 0)
        self.assertRaises(TypeError, getattr, self.parser, range(15))
data = b'<?xml version="1.0" encoding="iso-8859-1" standalone="no"?>\n<?xml-stylesheet href="stylesheet.css"?>\n<!-- comment data -->\n<!DOCTYPE quotations SYSTEM "quotations.dtd" [\n<!ELEMENT root ANY>\n<!ATTLIST root attr1 CDATA #REQUIRED attr2 CDATA #IMPLIED>\n<!NOTATION notation SYSTEM "notation.jpeg">\n<!ENTITY acirc "&#226;">\n<!ENTITY external_entity SYSTEM "entity.file">\n<!ENTITY unparsed_entity SYSTEM "entity.file" NDATA notation>\n%unparsed_entity;\n]>\n\n<root attr1="value1" attr2="value2&#8000;">\n<myns:subelement xmlns:myns="http://www.python.org/namespace">\n     Contents of subelements\n</myns:subelement>\n<sub2><![CDATA[contents of CDATA section]]></sub2>\n&external_entity;\n&skipped_entity;\n\xb5\n</root>\n'

class ParseTest(unittest.TestCase):

    class Outputter:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.out = []

        def StartElementHandler(self, name, attrs):
            if False:
                print('Hello World!')
            self.out.append('Start element: ' + repr(name) + ' ' + sortdict(attrs))

        def EndElementHandler(self, name):
            if False:
                for i in range(10):
                    print('nop')
            self.out.append('End element: ' + repr(name))

        def CharacterDataHandler(self, data):
            if False:
                return 10
            data = data.strip()
            if data:
                self.out.append('Character data: ' + repr(data))

        def ProcessingInstructionHandler(self, target, data):
            if False:
                return 10
            self.out.append('PI: ' + repr(target) + ' ' + repr(data))

        def StartNamespaceDeclHandler(self, prefix, uri):
            if False:
                while True:
                    i = 10
            self.out.append('NS decl: ' + repr(prefix) + ' ' + repr(uri))

        def EndNamespaceDeclHandler(self, prefix):
            if False:
                return 10
            self.out.append('End of NS decl: ' + repr(prefix))

        def StartCdataSectionHandler(self):
            if False:
                print('Hello World!')
            self.out.append('Start of CDATA section')

        def EndCdataSectionHandler(self):
            if False:
                print('Hello World!')
            self.out.append('End of CDATA section')

        def CommentHandler(self, text):
            if False:
                for i in range(10):
                    print('nop')
            self.out.append('Comment: ' + repr(text))

        def NotationDeclHandler(self, *args):
            if False:
                return 10
            (name, base, sysid, pubid) = args
            self.out.append('Notation declared: %s' % (args,))

        def UnparsedEntityDeclHandler(self, *args):
            if False:
                print('Hello World!')
            (entityName, base, systemId, publicId, notationName) = args
            self.out.append('Unparsed entity decl: %s' % (args,))

        def NotStandaloneHandler(self):
            if False:
                print('Hello World!')
            self.out.append('Not standalone')
            return 1

        def ExternalEntityRefHandler(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            (context, base, sysId, pubId) = args
            self.out.append('External entity ref: %s' % (args[1:],))
            return 1

        def StartDoctypeDeclHandler(self, *args):
            if False:
                while True:
                    i = 10
            self.out.append(('Start doctype', args))
            return 1

        def EndDoctypeDeclHandler(self):
            if False:
                print('Hello World!')
            self.out.append('End doctype')
            return 1

        def EntityDeclHandler(self, *args):
            if False:
                i = 10
                return i + 15
            self.out.append(('Entity declaration', args))
            return 1

        def XmlDeclHandler(self, *args):
            if False:
                print('Hello World!')
            self.out.append(('XML declaration', args))
            return 1

        def ElementDeclHandler(self, *args):
            if False:
                print('Hello World!')
            self.out.append(('Element declaration', args))
            return 1

        def AttlistDeclHandler(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            self.out.append(('Attribute list declaration', args))
            return 1

        def SkippedEntityHandler(self, *args):
            if False:
                return 10
            self.out.append(('Skipped entity', args))
            return 1

        def DefaultHandler(self, userData):
            if False:
                while True:
                    i = 10
            pass

        def DefaultHandlerExpand(self, userData):
            if False:
                while True:
                    i = 10
            pass
    handler_names = ['StartElementHandler', 'EndElementHandler', 'CharacterDataHandler', 'ProcessingInstructionHandler', 'UnparsedEntityDeclHandler', 'NotationDeclHandler', 'StartNamespaceDeclHandler', 'EndNamespaceDeclHandler', 'CommentHandler', 'StartCdataSectionHandler', 'EndCdataSectionHandler', 'DefaultHandler', 'DefaultHandlerExpand', 'NotStandaloneHandler', 'ExternalEntityRefHandler', 'StartDoctypeDeclHandler', 'EndDoctypeDeclHandler', 'EntityDeclHandler', 'XmlDeclHandler', 'ElementDeclHandler', 'AttlistDeclHandler', 'SkippedEntityHandler']

    def _hookup_callbacks(self, parser, handler):
        if False:
            print('Hello World!')
        '\n        Set each of the callbacks defined on handler and named in\n        self.handler_names on the given parser.\n        '
        for name in self.handler_names:
            setattr(parser, name, getattr(handler, name))

    def _verify_parse_output(self, operations):
        if False:
            while True:
                i = 10
        expected_operations = [('XML declaration', ('1.0', 'iso-8859-1', 0)), 'PI: \'xml-stylesheet\' \'href="stylesheet.css"\'', "Comment: ' comment data '", 'Not standalone', ('Start doctype', ('quotations', 'quotations.dtd', None, 1)), ('Element declaration', ('root', (2, 0, None, ()))), ('Attribute list declaration', ('root', 'attr1', 'CDATA', None, 1)), ('Attribute list declaration', ('root', 'attr2', 'CDATA', None, 0)), "Notation declared: ('notation', None, 'notation.jpeg', None)", ('Entity declaration', ('acirc', 0, 'â', None, None, None, None)), ('Entity declaration', ('external_entity', 0, None, None, 'entity.file', None, None)), "Unparsed entity decl: ('unparsed_entity', None, 'entity.file', None, 'notation')", 'Not standalone', 'End doctype', "Start element: 'root' {'attr1': 'value1', 'attr2': 'value2ὀ'}", "NS decl: 'myns' 'http://www.python.org/namespace'", "Start element: 'http://www.python.org/namespace!subelement' {}", "Character data: 'Contents of subelements'", "End element: 'http://www.python.org/namespace!subelement'", "End of NS decl: 'myns'", "Start element: 'sub2' {}", 'Start of CDATA section', "Character data: 'contents of CDATA section'", 'End of CDATA section', "End element: 'sub2'", "External entity ref: (None, 'entity.file', None)", ('Skipped entity', ('skipped_entity', 0)), "Character data: 'µ'", "End element: 'root'"]
        for (operation, expected_operation) in zip(operations, expected_operations):
            self.assertEqual(operation, expected_operation)

    def test_parse_bytes(self):
        if False:
            while True:
                i = 10
        out = self.Outputter()
        parser = expat.ParserCreate(namespace_separator='!')
        self._hookup_callbacks(parser, out)
        parser.Parse(data, True)
        operations = out.out
        self._verify_parse_output(operations)
        self.assertRaises(AttributeError, getattr, parser, '\ud800')

    def test_parse_str(self):
        if False:
            for i in range(10):
                print('nop')
        out = self.Outputter()
        parser = expat.ParserCreate(namespace_separator='!')
        self._hookup_callbacks(parser, out)
        parser.Parse(data.decode('iso-8859-1'), True)
        operations = out.out
        self._verify_parse_output(operations)

    def test_parse_file(self):
        if False:
            for i in range(10):
                print('nop')
        out = self.Outputter()
        parser = expat.ParserCreate(namespace_separator='!')
        self._hookup_callbacks(parser, out)
        file = BytesIO(data)
        parser.ParseFile(file)
        operations = out.out
        self._verify_parse_output(operations)

    def test_parse_again(self):
        if False:
            while True:
                i = 10
        parser = expat.ParserCreate()
        file = BytesIO(data)
        parser.ParseFile(file)
        with self.assertRaises(expat.error) as cm:
            parser.ParseFile(file)
        self.assertEqual(expat.ErrorString(cm.exception.code), expat.errors.XML_ERROR_FINISHED)

class NamespaceSeparatorTest(unittest.TestCase):

    def test_legal(self):
        if False:
            i = 10
            return i + 15
        expat.ParserCreate()
        expat.ParserCreate(namespace_separator=None)
        expat.ParserCreate(namespace_separator=' ')

    def test_illegal(self):
        if False:
            return 10
        try:
            expat.ParserCreate(namespace_separator=42)
            self.fail()
        except TypeError as e:
            self.assertEqual(str(e), "ParserCreate() argument 'namespace_separator' must be str or None, not int")
        try:
            expat.ParserCreate(namespace_separator='too long')
            self.fail()
        except ValueError as e:
            self.assertEqual(str(e), 'namespace_separator must be at most one character, omitted, or None')

    def test_zero_length(self):
        if False:
            print('Hello World!')
        expat.ParserCreate(namespace_separator='')

class InterningTest(unittest.TestCase):

    def test(self):
        if False:
            return 10
        p = expat.ParserCreate()
        L = []

        def collector(name, *args):
            if False:
                while True:
                    i = 10
            L.append(name)
        p.StartElementHandler = collector
        p.EndElementHandler = collector
        p.Parse(b'<e> <e/> <e></e> </e>', True)
        tag = L[0]
        self.assertEqual(len(L), 6)
        for entry in L:
            self.assertTrue(tag is entry)

    def test_issue9402(self):
        if False:
            i = 10
            return i + 15

        class ExternalOutputter:

            def __init__(self, parser):
                if False:
                    i = 10
                    return i + 15
                self.parser = parser
                self.parser_result = None

            def ExternalEntityRefHandler(self, context, base, sysId, pubId):
                if False:
                    return 10
                external_parser = self.parser.ExternalEntityParserCreate('')
                self.parser_result = external_parser.Parse(b'', True)
                return 1
        parser = expat.ParserCreate(namespace_separator='!')
        parser.buffer_text = 1
        out = ExternalOutputter(parser)
        parser.ExternalEntityRefHandler = out.ExternalEntityRefHandler
        parser.Parse(data, True)
        self.assertEqual(out.parser_result, 1)

class BufferTextTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.stuff = []
        self.parser = expat.ParserCreate()
        self.parser.buffer_text = 1
        self.parser.CharacterDataHandler = self.CharacterDataHandler

    def check(self, expected, label):
        if False:
            while True:
                i = 10
        self.assertEqual(self.stuff, expected, '%s\nstuff    = %r\nexpected = %r' % (label, self.stuff, map(str, expected)))

    def CharacterDataHandler(self, text):
        if False:
            for i in range(10):
                print('nop')
        self.stuff.append(text)

    def StartElementHandler(self, name, attrs):
        if False:
            for i in range(10):
                print('nop')
        self.stuff.append('<%s>' % name)
        bt = attrs.get('buffer-text')
        if bt == 'yes':
            self.parser.buffer_text = 1
        elif bt == 'no':
            self.parser.buffer_text = 0

    def EndElementHandler(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.stuff.append('</%s>' % name)

    def CommentHandler(self, data):
        if False:
            print('Hello World!')
        self.stuff.append('<!--%s-->' % data)

    def setHandlers(self, handlers=[]):
        if False:
            for i in range(10):
                print('nop')
        for name in handlers:
            setattr(self.parser, name, getattr(self, name))

    def test_default_to_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        parser = expat.ParserCreate()
        self.assertFalse(parser.buffer_text)

    def test_buffering_enabled(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.parser.buffer_text)
        self.parser.Parse(b'<a>1<b/>2<c/>3</a>', True)
        self.assertEqual(self.stuff, ['123'], 'buffered text not properly collapsed')

    def test1(self):
        if False:
            for i in range(10):
                print('nop')
        self.setHandlers(['StartElementHandler'])
        self.parser.Parse(b"<a>1<b buffer-text='no'/>2\n3<c buffer-text='yes'/>4\n5</a>", True)
        self.assertEqual(self.stuff, ['<a>', '1', '<b>', '2', '\n', '3', '<c>', '4\n5'], 'buffering control not reacting as expected')

    def test2(self):
        if False:
            return 10
        self.parser.Parse(b'<a>1<b/>&lt;2&gt;<c/>&#32;\n&#x20;3</a>', True)
        self.assertEqual(self.stuff, ['1<2> \n 3'], 'buffered text not properly collapsed')

    def test3(self):
        if False:
            for i in range(10):
                print('nop')
        self.setHandlers(['StartElementHandler'])
        self.parser.Parse(b'<a>1<b/>2<c/>3</a>', True)
        self.assertEqual(self.stuff, ['<a>', '1', '<b>', '2', '<c>', '3'], 'buffered text not properly split')

    def test4(self):
        if False:
            return 10
        self.setHandlers(['StartElementHandler', 'EndElementHandler'])
        self.parser.CharacterDataHandler = None
        self.parser.Parse(b'<a>1<b/>2<c/>3</a>', True)
        self.assertEqual(self.stuff, ['<a>', '<b>', '</b>', '<c>', '</c>', '</a>'])

    def test5(self):
        if False:
            for i in range(10):
                print('nop')
        self.setHandlers(['StartElementHandler', 'EndElementHandler'])
        self.parser.Parse(b'<a>1<b></b>2<c/>3</a>', True)
        self.assertEqual(self.stuff, ['<a>', '1', '<b>', '</b>', '2', '<c>', '</c>', '3', '</a>'])

    def test6(self):
        if False:
            i = 10
            return i + 15
        self.setHandlers(['CommentHandler', 'EndElementHandler', 'StartElementHandler'])
        self.parser.Parse(b'<a>1<b/>2<c></c>345</a> ', True)
        self.assertEqual(self.stuff, ['<a>', '1', '<b>', '</b>', '2', '<c>', '</c>', '345', '</a>'], 'buffered text not properly split')

    def test7(self):
        if False:
            for i in range(10):
                print('nop')
        self.setHandlers(['CommentHandler', 'EndElementHandler', 'StartElementHandler'])
        self.parser.Parse(b'<a>1<b/>2<c></c>3<!--abc-->4<!--def-->5</a> ', True)
        self.assertEqual(self.stuff, ['<a>', '1', '<b>', '</b>', '2', '<c>', '</c>', '3', '<!--abc-->', '4', '<!--def-->', '5', '</a>'], 'buffered text not properly split')

class HandlerExceptionTest(unittest.TestCase):

    def StartElementHandler(self, name, attrs):
        if False:
            print('Hello World!')
        raise RuntimeError(name)

    def check_traceback_entry(self, entry, filename, funcname):
        if False:
            i = 10
            return i + 15
        self.assertEqual(os.path.basename(entry[0]), filename)
        self.assertEqual(entry[2], funcname)

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        parser = expat.ParserCreate()
        parser.StartElementHandler = self.StartElementHandler
        try:
            parser.Parse(b'<a><b><c/></b></a>', True)
            self.fail()
        except RuntimeError as e:
            self.assertEqual(e.args[0], 'a', "Expected RuntimeError for element 'a', but" + ' found %r' % e.args[0])
            entries = traceback.extract_tb(e.__traceback__)
            self.assertEqual(len(entries), 3)
            self.check_traceback_entry(entries[0], 'test_pyexpat.py', 'test_exception')
            self.check_traceback_entry(entries[1], 'pyexpat.c', 'StartElement')
            self.check_traceback_entry(entries[2], 'test_pyexpat.py', 'StartElementHandler')
            if sysconfig.is_python_build() and (not (sys.platform == 'win32' and platform.machine() == 'ARM')):
                self.assertIn('call_with_frame("StartElement"', entries[1][3])

class PositionTest(unittest.TestCase):

    def StartElementHandler(self, name, attrs):
        if False:
            i = 10
            return i + 15
        self.check_pos('s')

    def EndElementHandler(self, name):
        if False:
            print('Hello World!')
        self.check_pos('e')

    def check_pos(self, event):
        if False:
            print('Hello World!')
        pos = (event, self.parser.CurrentByteIndex, self.parser.CurrentLineNumber, self.parser.CurrentColumnNumber)
        self.assertTrue(self.upto < len(self.expected_list), 'too many parser events')
        expected = self.expected_list[self.upto]
        self.assertEqual(pos, expected, 'Expected position %s, got position %s' % (pos, expected))
        self.upto += 1

    def test(self):
        if False:
            i = 10
            return i + 15
        self.parser = expat.ParserCreate()
        self.parser.StartElementHandler = self.StartElementHandler
        self.parser.EndElementHandler = self.EndElementHandler
        self.upto = 0
        self.expected_list = [('s', 0, 1, 0), ('s', 5, 2, 1), ('s', 11, 3, 2), ('e', 15, 3, 6), ('e', 17, 4, 1), ('e', 22, 5, 0)]
        xml = b'<a>\n <b>\n  <c/>\n </b>\n</a>'
        self.parser.Parse(xml, True)

class sf1296433Test(unittest.TestCase):

    def test_parse_only_xml_data(self):
        if False:
            for i in range(10):
                print('nop')
        xml = "<?xml version='1.0' encoding='iso8859'?><s>%s</s>" % ('a' * 1025)

        class SpecificException(Exception):
            pass

        def handler(text):
            if False:
                while True:
                    i = 10
            raise SpecificException
        parser = expat.ParserCreate()
        parser.CharacterDataHandler = handler
        self.assertRaises(Exception, parser.Parse, xml.encode('iso8859'))

class ChardataBufferTest(unittest.TestCase):
    """
    test setting of chardata buffer size
    """

    def test_1025_bytes(self):
        if False:
            return 10
        self.assertEqual(self.small_buffer_test(1025), 2)

    def test_1000_bytes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.small_buffer_test(1000), 1)

    def test_wrong_size(self):
        if False:
            return 10
        parser = expat.ParserCreate()
        parser.buffer_text = 1
        with self.assertRaises(ValueError):
            parser.buffer_size = -1
        with self.assertRaises(ValueError):
            parser.buffer_size = 0
        with self.assertRaises((ValueError, OverflowError)):
            parser.buffer_size = sys.maxsize + 1
        with self.assertRaises(TypeError):
            parser.buffer_size = 512.0

    def test_unchanged_size(self):
        if False:
            print('Hello World!')
        xml1 = b"<?xml version='1.0' encoding='iso8859'?><s>" + b'a' * 512
        xml2 = b'a' * 512 + b'</s>'
        parser = expat.ParserCreate()
        parser.CharacterDataHandler = self.counting_handler
        parser.buffer_size = 512
        parser.buffer_text = 1
        self.n = 0
        parser.Parse(xml1)
        self.assertEqual(self.n, 1)
        parser.buffer_size = parser.buffer_size
        self.assertEqual(self.n, 1)
        parser.Parse(xml2)
        self.assertEqual(self.n, 2)

    def test_disabling_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        xml1 = b"<?xml version='1.0' encoding='iso8859'?><a>" + b'a' * 512
        xml2 = b'b' * 1024
        xml3 = b'c' * 1024 + b'</a>'
        parser = expat.ParserCreate()
        parser.CharacterDataHandler = self.counting_handler
        parser.buffer_text = 1
        parser.buffer_size = 1024
        self.assertEqual(parser.buffer_size, 1024)
        self.n = 0
        parser.Parse(xml1, False)
        self.assertEqual(parser.buffer_size, 1024)
        self.assertEqual(self.n, 1)
        parser.buffer_text = 0
        self.assertFalse(parser.buffer_text)
        self.assertEqual(parser.buffer_size, 1024)
        for i in range(10):
            parser.Parse(xml2, False)
        self.assertEqual(self.n, 11)
        parser.buffer_text = 1
        self.assertTrue(parser.buffer_text)
        self.assertEqual(parser.buffer_size, 1024)
        parser.Parse(xml3, True)
        self.assertEqual(self.n, 12)

    def counting_handler(self, text):
        if False:
            while True:
                i = 10
        self.n += 1

    def small_buffer_test(self, buffer_len):
        if False:
            return 10
        xml = b"<?xml version='1.0' encoding='iso8859'?><s>" + b'a' * buffer_len + b'</s>'
        parser = expat.ParserCreate()
        parser.CharacterDataHandler = self.counting_handler
        parser.buffer_size = 1024
        parser.buffer_text = 1
        self.n = 0
        parser.Parse(xml)
        return self.n

    def test_change_size_1(self):
        if False:
            print('Hello World!')
        xml1 = b"<?xml version='1.0' encoding='iso8859'?><a><s>" + b'a' * 1024
        xml2 = b'aaa</s><s>' + b'a' * 1025 + b'</s></a>'
        parser = expat.ParserCreate()
        parser.CharacterDataHandler = self.counting_handler
        parser.buffer_text = 1
        parser.buffer_size = 1024
        self.assertEqual(parser.buffer_size, 1024)
        self.n = 0
        parser.Parse(xml1, False)
        parser.buffer_size *= 2
        self.assertEqual(parser.buffer_size, 2048)
        parser.Parse(xml2, True)
        self.assertEqual(self.n, 2)

    def test_change_size_2(self):
        if False:
            print('Hello World!')
        xml1 = b"<?xml version='1.0' encoding='iso8859'?><a>a<s>" + b'a' * 1023
        xml2 = b'aaa</s><s>' + b'a' * 1025 + b'</s></a>'
        parser = expat.ParserCreate()
        parser.CharacterDataHandler = self.counting_handler
        parser.buffer_text = 1
        parser.buffer_size = 2048
        self.assertEqual(parser.buffer_size, 2048)
        self.n = 0
        parser.Parse(xml1, False)
        parser.buffer_size = parser.buffer_size // 2
        self.assertEqual(parser.buffer_size, 1024)
        parser.Parse(xml2, True)
        self.assertEqual(self.n, 4)

class MalformedInputTest(unittest.TestCase):

    def test1(self):
        if False:
            for i in range(10):
                print('nop')
        xml = b'\x00\r\n'
        parser = expat.ParserCreate()
        try:
            parser.Parse(xml, True)
            self.fail()
        except expat.ExpatError as e:
            self.assertEqual(str(e), 'unclosed token: line 2, column 0')

    def test2(self):
        if False:
            print('Hello World!')
        xml = b"<?xml version\xc2\x85='1.0'?>\r\n"
        parser = expat.ParserCreate()
        err_pattern = 'XML declaration not well-formed: line 1, column \\d+'
        with self.assertRaisesRegex(expat.ExpatError, err_pattern):
            parser.Parse(xml, True)

class ErrorMessageTest(unittest.TestCase):

    def test_codes(self):
        if False:
            return 10
        self.assertEqual(errors.XML_ERROR_SYNTAX, errors.messages[errors.codes[errors.XML_ERROR_SYNTAX]])

    def test_expaterror(self):
        if False:
            while True:
                i = 10
        xml = b'<'
        parser = expat.ParserCreate()
        try:
            parser.Parse(xml, True)
            self.fail()
        except expat.ExpatError as e:
            self.assertEqual(e.code, errors.codes[errors.XML_ERROR_UNCLOSED_TOKEN])

class ForeignDTDTests(unittest.TestCase):
    """
    Tests for the UseForeignDTD method of expat parser objects.
    """

    def test_use_foreign_dtd(self):
        if False:
            print('Hello World!')
        '\n        If UseForeignDTD is passed True and a document without an external\n        entity reference is parsed, ExternalEntityRefHandler is first called\n        with None for the public and system ids.\n        '
        handler_call_args = []

        def resolve_entity(context, base, system_id, public_id):
            if False:
                print('Hello World!')
            handler_call_args.append((public_id, system_id))
            return 1
        parser = expat.ParserCreate()
        parser.UseForeignDTD(True)
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_ALWAYS)
        parser.ExternalEntityRefHandler = resolve_entity
        parser.Parse(b"<?xml version='1.0'?><element/>")
        self.assertEqual(handler_call_args, [(None, None)])
        handler_call_args[:] = []
        parser = expat.ParserCreate()
        parser.UseForeignDTD()
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_ALWAYS)
        parser.ExternalEntityRefHandler = resolve_entity
        parser.Parse(b"<?xml version='1.0'?><element/>")
        self.assertEqual(handler_call_args, [(None, None)])

    def test_ignore_use_foreign_dtd(self):
        if False:
            print('Hello World!')
        '\n        If UseForeignDTD is passed True and a document with an external\n        entity reference is parsed, ExternalEntityRefHandler is called with\n        the public and system ids from the document.\n        '
        handler_call_args = []

        def resolve_entity(context, base, system_id, public_id):
            if False:
                return 10
            handler_call_args.append((public_id, system_id))
            return 1
        parser = expat.ParserCreate()
        parser.UseForeignDTD(True)
        parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_ALWAYS)
        parser.ExternalEntityRefHandler = resolve_entity
        parser.Parse(b"<?xml version='1.0'?><!DOCTYPE foo PUBLIC 'bar' 'baz'><element/>")
        self.assertEqual(handler_call_args, [('bar', 'baz')])
if __name__ == '__main__':
    unittest.main()