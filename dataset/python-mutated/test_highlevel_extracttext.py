import unittest
from helpers import absolute_sample_path
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextContainer

def run_with_string(sample_path, laparams=None):
    if False:
        while True:
            i = 10
    if laparams is None:
        laparams = {}
    absolute_path = absolute_sample_path(sample_path)
    s = extract_text(absolute_path, laparams=LAParams(**laparams))
    return s

def run_with_file(sample_path):
    if False:
        print('Hello World!')
    absolute_path = absolute_sample_path(sample_path)
    with open(absolute_path, 'rb') as in_file:
        s = extract_text(in_file)
    return s
test_strings = {'simple1.pdf': 'Hello \n\nWorld\n\nHello \n\nWorld\n\nH e l l o  \n\nW o r l d\n\nH e l l o  \n\nW o r l d\n\n\x0c', 'simple1.pdf_no_boxes_flow': 'Hello \n\nWorld\n\nHello \n\nWorld\n\nH e l l o  \n\nW o r l d\n\nH e l l o  \n\nW o r l d\n\n\x0c', 'simple2.pdf': '\x0c', 'simple3.pdf': 'Hello\n\nHello\nあ\nい\nう\nえ\nお\nあ\nい\nう\nえ\nお\nWorld\n\nWorld\n\n\x0c', 'simple4.pdf': 'Text1\nText2\nText3\n\n\x0c', 'simple5.pdf': 'Heading\n\nLink to heading that is working with vim-pandoc.\n\nLink to heading “that is” not working with vim-pandoc.\n\nSubheading\n\nSome “more text”\n\n1\n\n\x0c', 'zen_of_python_corrupted.pdf': 'Mai 30, 18 13:27\n\nzen_of_python.txt', 'contrib/issue_566_test_1.pdf': 'ISSUE Date：2019-4-25 Buyer：黎荣', 'contrib/issue_566_test_2.pdf': '甲方：中国饮料有限公司（盖章）', 'contrib/issue-625-identity-cmap.pdf': 'Termin płatności: 2021-05-03'}

class TestExtractText(unittest.TestCase):

    def test_simple1_with_string(self):
        if False:
            for i in range(10):
                print('nop')
        test_file = 'simple1.pdf'
        s = run_with_string(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple1_no_boxes_flow(self):
        if False:
            return 10
        test_file = 'simple1.pdf'
        s = run_with_string(test_file, laparams={'boxes_flow': None})
        self.assertEqual(s, test_strings['simple1.pdf_no_boxes_flow'])

    def test_simple2_with_string(self):
        if False:
            i = 10
            return i + 15
        test_file = 'simple2.pdf'
        s = run_with_string(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple3_with_string(self):
        if False:
            for i in range(10):
                print('nop')
        test_file = 'simple3.pdf'
        s = run_with_string(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple4_with_string(self):
        if False:
            i = 10
            return i + 15
        test_file = 'simple4.pdf'
        s = run_with_string(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple5_with_string(self):
        if False:
            while True:
                i = 10
        test_file = 'simple5.pdf'
        s = run_with_string(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple1_with_file(self):
        if False:
            for i in range(10):
                print('nop')
        test_file = 'simple1.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple2_with_file(self):
        if False:
            while True:
                i = 10
        test_file = 'simple2.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple3_with_file(self):
        if False:
            while True:
                i = 10
        test_file = 'simple3.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple4_with_file(self):
        if False:
            while True:
                i = 10
        test_file = 'simple4.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_simple5_with_file(self):
        if False:
            for i in range(10):
                print('nop')
        test_file = 'simple5.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s, test_strings[test_file])

    def test_zlib_corrupted(self):
        if False:
            return 10
        test_file = 'zen_of_python_corrupted.pdf'
        s = run_with_file(test_file)
        expected = test_strings[test_file]
        self.assertEqual(s[:len(expected)], expected)

    def test_issue_566_cmap_bytes(self):
        if False:
            while True:
                i = 10
        test_file = 'contrib/issue_566_test_1.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s.strip(), test_strings[test_file])

    def test_issue_566_cid_range(self):
        if False:
            while True:
                i = 10
        test_file = 'contrib/issue_566_test_2.pdf'
        s = run_with_file(test_file)
        self.assertEqual(s.strip(), test_strings[test_file])

    def test_issue_625_identity_cmap(self):
        if False:
            print('Hello World!')
        test_file = 'contrib/issue-625-identity-cmap.pdf'
        lines = run_with_file(test_file).splitlines()
        self.assertEqual(lines[6], test_strings[test_file])

class TestExtractPages(unittest.TestCase):

    def _get_test_file_path(self):
        if False:
            print('Hello World!')
        test_file = 'simple4.pdf'
        return absolute_sample_path(test_file)

    def test_line_margin(self):
        if False:
            while True:
                i = 10
        pages = list(extract_pages(self._get_test_file_path(), laparams=LAParams(line_margin=0.19)))
        self.assertEqual(len(pages), 1)
        page = pages[0]
        elements = [element for element in page if isinstance(element, LTTextContainer)]
        self.assertEqual(len(elements), 3)
        self.assertEqual(elements[0].get_text(), 'Text1\n')
        self.assertEqual(elements[1].get_text(), 'Text2\n')
        self.assertEqual(elements[2].get_text(), 'Text3\n')
        pages = list(extract_pages(self._get_test_file_path(), laparams=LAParams(line_margin=0.21)))
        self.assertEqual(len(pages), 1)
        page = pages[0]
        elements = [element for element in page if isinstance(element, LTTextContainer)]
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].get_text(), 'Text1\nText2\nText3\n')

    def test_no_boxes_flow(self):
        if False:
            while True:
                i = 10
        pages = list(extract_pages(self._get_test_file_path(), laparams=LAParams(boxes_flow=None)))
        self.assertEqual(len(pages), 1)
        page = pages[0]
        elements = [element for element in page if isinstance(element, LTTextContainer)]
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].get_text(), 'Text1\nText2\nText3\n')