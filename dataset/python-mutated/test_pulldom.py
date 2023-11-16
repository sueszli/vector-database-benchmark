import io
import unittest
import xml.sax
from xml.sax.xmlreader import AttributesImpl
from xml.sax.handler import feature_external_ges
from xml.dom import pulldom
from test.support import findfile
tstfile = findfile('test.xml', subdir='xmltestdata')
SMALL_SAMPLE = '<?xml version="1.0"?>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:xdc="http://www.xml.com/books">\n<!-- A comment -->\n<title>Introduction to XSL</title>\n<hr/>\n<p><xdc:author xdc:attrib="prefixed attribute" attrib="other attrib">A. Namespace</xdc:author></p>\n</html>'

class PullDOMTestCase(unittest.TestCase):

    def test_parse(self):
        if False:
            while True:
                i = 10
        'Minimal test of DOMEventStream.parse()'
        handler = pulldom.parse(tstfile)
        self.addCleanup(handler.stream.close)
        list(handler)
        with open(tstfile, 'rb') as fin:
            list(pulldom.parse(fin))

    def test_parse_semantics(self):
        if False:
            while True:
                i = 10
        'Test DOMEventStream parsing semantics.'
        items = pulldom.parseString(SMALL_SAMPLE)
        (evt, node) = next(items)
        self.assertTrue(hasattr(node, 'createElement'))
        self.assertEqual(pulldom.START_DOCUMENT, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.START_ELEMENT, evt)
        self.assertEqual('html', node.tagName)
        self.assertEqual(2, len(node.attributes))
        self.assertEqual(node.attributes.getNamedItem('xmlns:xdc').value, 'http://www.xml.com/books')
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        (evt, node) = next(items)
        self.assertEqual('title', node.tagName)
        title_node = node
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        self.assertEqual('Introduction to XSL', node.data)
        (evt, node) = next(items)
        self.assertEqual(pulldom.END_ELEMENT, evt)
        self.assertEqual('title', node.tagName)
        self.assertTrue(title_node is node)
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.START_ELEMENT, evt)
        self.assertEqual('hr', node.tagName)
        (evt, node) = next(items)
        self.assertEqual(pulldom.END_ELEMENT, evt)
        self.assertEqual('hr', node.tagName)
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.START_ELEMENT, evt)
        self.assertEqual('p', node.tagName)
        (evt, node) = next(items)
        self.assertEqual(pulldom.START_ELEMENT, evt)
        self.assertEqual('xdc:author', node.tagName)
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.END_ELEMENT, evt)
        self.assertEqual('xdc:author', node.tagName)
        (evt, node) = next(items)
        self.assertEqual(pulldom.END_ELEMENT, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.CHARACTERS, evt)
        (evt, node) = next(items)
        self.assertEqual(pulldom.END_ELEMENT, evt)

    def test_expandItem(self):
        if False:
            print('Hello World!')
        'Ensure expandItem works as expected.'
        items = pulldom.parseString(SMALL_SAMPLE)
        for (evt, item) in items:
            if evt == pulldom.START_ELEMENT and item.tagName == 'title':
                items.expandNode(item)
                self.assertEqual(1, len(item.childNodes))
                break
        else:
            self.fail('No "title" element detected in SMALL_SAMPLE!')
        for (evt, node) in items:
            if evt == pulldom.START_ELEMENT:
                break
        self.assertEqual('hr', node.tagName, 'expandNode did not leave DOMEventStream in the correct state.')
        items.expandNode(node)
        self.assertEqual(next(items)[0], pulldom.CHARACTERS)
        (evt, node) = next(items)
        self.assertEqual(node.tagName, 'p')
        items.expandNode(node)
        next(items)
        (evt, node) = next(items)
        self.assertEqual(node.tagName, 'html')
        with self.assertRaises(StopIteration):
            next(items)
        items.clear()
        self.assertIsNone(items.parser)
        self.assertIsNone(items.stream)

    @unittest.expectedFailure
    def test_comment(self):
        if False:
            while True:
                i = 10
        'PullDOM does not receive "comment" events.'
        items = pulldom.parseString(SMALL_SAMPLE)
        for (evt, _) in items:
            if evt == pulldom.COMMENT:
                break
        else:
            self.fail('No comment was encountered')

    @unittest.expectedFailure
    def test_end_document(self):
        if False:
            for i in range(10):
                print('nop')
        'PullDOM does not receive "end-document" events.'
        items = pulldom.parseString(SMALL_SAMPLE)
        for (evt, node) in items:
            if evt == pulldom.END_ELEMENT and node.tagName == 'html':
                break
        try:
            (evt, node) = next(items)
            self.assertEqual(pulldom.END_DOCUMENT, evt)
        except StopIteration:
            self.fail('Ran out of events, but should have received END_DOCUMENT')

    def test_getitem_deprecation(self):
        if False:
            for i in range(10):
                print('nop')
        parser = pulldom.parseString(SMALL_SAMPLE)
        with self.assertWarnsRegex(DeprecationWarning, 'Use iterator protocol instead'):
            self.assertEqual(parser[-1][0], pulldom.START_DOCUMENT)

    def test_external_ges_default(self):
        if False:
            return 10
        parser = pulldom.parseString(SMALL_SAMPLE)
        saxparser = parser.parser
        ges = saxparser.getFeature(feature_external_ges)
        self.assertEqual(ges, False)

class ThoroughTestCase(unittest.TestCase):
    """Test the hard-to-reach parts of pulldom."""

    def test_thorough_parse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test some of the hard-to-reach parts of PullDOM.'
        self._test_thorough(pulldom.parse(None, parser=SAXExerciser()))

    @unittest.expectedFailure
    def test_sax2dom_fail(self):
        if False:
            i = 10
            return i + 15
        'SAX2DOM can"t handle a PI before the root element.'
        pd = SAX2DOMTestHelper(None, SAXExerciser(), 12)
        self._test_thorough(pd)

    def test_thorough_sax2dom(self):
        if False:
            for i in range(10):
                print('nop')
        'Test some of the hard-to-reach parts of SAX2DOM.'
        pd = SAX2DOMTestHelper(None, SAX2DOMExerciser(), 12)
        self._test_thorough(pd, False)

    def _test_thorough(self, pd, before_root=True):
        if False:
            i = 10
            return i + 15
        'Test some of the hard-to-reach parts of the parser, using a mock\n        parser.'
        (evt, node) = next(pd)
        self.assertEqual(pulldom.START_DOCUMENT, evt)
        self.assertTrue(hasattr(node, 'createElement'))
        if before_root:
            (evt, node) = next(pd)
            self.assertEqual(pulldom.COMMENT, evt)
            self.assertEqual('a comment', node.data)
            (evt, node) = next(pd)
            self.assertEqual(pulldom.PROCESSING_INSTRUCTION, evt)
            self.assertEqual('target', node.target)
            self.assertEqual('data', node.data)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.START_ELEMENT, evt)
        self.assertEqual('html', node.tagName)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.COMMENT, evt)
        self.assertEqual('a comment', node.data)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.PROCESSING_INSTRUCTION, evt)
        self.assertEqual('target', node.target)
        self.assertEqual('data', node.data)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.START_ELEMENT, evt)
        self.assertEqual('p', node.tagName)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.CHARACTERS, evt)
        self.assertEqual('text', node.data)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.END_ELEMENT, evt)
        self.assertEqual('p', node.tagName)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.END_ELEMENT, evt)
        self.assertEqual('html', node.tagName)
        (evt, node) = next(pd)
        self.assertEqual(pulldom.END_DOCUMENT, evt)

class SAXExerciser(object):
    """A fake sax parser that calls some of the harder-to-reach sax methods to
    ensure it emits the correct events"""

    def setContentHandler(self, handler):
        if False:
            while True:
                i = 10
        self._handler = handler

    def parse(self, _):
        if False:
            return 10
        h = self._handler
        h.startDocument()
        h.comment('a comment')
        h.processingInstruction('target', 'data')
        h.startElement('html', AttributesImpl({}))
        h.comment('a comment')
        h.processingInstruction('target', 'data')
        h.startElement('p', AttributesImpl({'class': 'paraclass'}))
        h.characters('text')
        h.endElement('p')
        h.endElement('html')
        h.endDocument()

    def stub(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Stub method. Does nothing.'
        pass
    setProperty = stub
    setFeature = stub

class SAX2DOMExerciser(SAXExerciser):
    """The same as SAXExerciser, but without the processing instruction and
    comment before the root element, because S2D can"t handle it"""

    def parse(self, _):
        if False:
            return 10
        h = self._handler
        h.startDocument()
        h.startElement('html', AttributesImpl({}))
        h.comment('a comment')
        h.processingInstruction('target', 'data')
        h.startElement('p', AttributesImpl({'class': 'paraclass'}))
        h.characters('text')
        h.endElement('p')
        h.endElement('html')
        h.endDocument()

class SAX2DOMTestHelper(pulldom.DOMEventStream):
    """Allows us to drive SAX2DOM from a DOMEventStream."""

    def reset(self):
        if False:
            print('Hello World!')
        self.pulldom = pulldom.SAX2DOM()
        self.parser.setFeature(xml.sax.handler.feature_namespaces, 1)
        self.parser.setContentHandler(self.pulldom)

class SAX2DOMTestCase(unittest.TestCase):

    def confirm(self, test, testname='Test'):
        if False:
            i = 10
            return i + 15
        self.assertTrue(test, testname)

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure SAX2DOM can parse from a stream.'
        with io.StringIO(SMALL_SAMPLE) as fin:
            sd = SAX2DOMTestHelper(fin, xml.sax.make_parser(), len(SMALL_SAMPLE))
            for (evt, node) in sd:
                if evt == pulldom.START_ELEMENT and node.tagName == 'html':
                    break
            self.assertGreater(len(node.childNodes), 0)

    def testSAX2DOM(self):
        if False:
            print('Hello World!')
        'Ensure SAX2DOM expands nodes as expected.'
        sax2dom = pulldom.SAX2DOM()
        sax2dom.startDocument()
        sax2dom.startElement('doc', {})
        sax2dom.characters('text')
        sax2dom.startElement('subelm', {})
        sax2dom.characters('text')
        sax2dom.endElement('subelm')
        sax2dom.characters('text')
        sax2dom.endElement('doc')
        sax2dom.endDocument()
        doc = sax2dom.document
        root = doc.documentElement
        (text1, elm1, text2) = root.childNodes
        text3 = elm1.childNodes[0]
        self.assertIsNone(text1.previousSibling)
        self.assertIs(text1.nextSibling, elm1)
        self.assertIs(elm1.previousSibling, text1)
        self.assertIs(elm1.nextSibling, text2)
        self.assertIs(text2.previousSibling, elm1)
        self.assertIsNone(text2.nextSibling)
        self.assertIsNone(text3.previousSibling)
        self.assertIsNone(text3.nextSibling)
        self.assertIs(root.parentNode, doc)
        self.assertIs(text1.parentNode, root)
        self.assertIs(elm1.parentNode, root)
        self.assertIs(text2.parentNode, root)
        self.assertIs(text3.parentNode, elm1)
        doc.unlink()
if __name__ == '__main__':
    unittest.main()