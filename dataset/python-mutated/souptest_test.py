import unittest
from r2.lib.souptest import souptest_fragment, SoupDetectedCrasherError, SoupError, SoupSyntaxError, SoupUnexpectedCDataSectionError, SoupUnexpectedCommentError, SoupUnsupportedAttrError, SoupUnsupportedEntityError, SoupUnsupportedNodeError, SoupUnsupportedSchemeError, SoupUnsupportedTagError

class TestSoupTest(unittest.TestCase):

    def assertFragmentRaises(self, fragment, error):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(error, souptest_fragment, fragment)

    def assertFragmentValid(self, fragment):
        if False:
            for i in range(10):
                print('nop')
        souptest_fragment(fragment)

    def test_benign(self):
        if False:
            print('Hello World!')
        'A typical example of what we might get out of `safemarkdown()`'
        testcase = '\n            <!-- SC_OFF -->\n            <div class="md"><a href="http://zombo.com/">Welcome</a></div>\n            <!-- SC_ON -->\n        '
        self.assertFragmentValid(testcase)

    def test_unbalanced(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFragmentRaises('<div></div></div>', SoupSyntaxError)

    def test_unclosed_comment(self):
        if False:
            print('Hello World!')
        self.assertFragmentRaises('<!--', SoupSyntaxError)

    def test_invalid_comment(self):
        if False:
            print('Hello World!')
        testcase = '<!--[if IE 6]>WHAT YEAR IS IT?<![endif]-->'
        self.assertFragmentRaises(testcase, SoupUnexpectedCommentError)

    def test_quoting(self):
        if False:
            print('Hello World!')
        self.assertFragmentRaises('<div class=`poor IE`></div>', SoupSyntaxError)

    def test_processing_instruction(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFragmentRaises('<?php not even once ?>', SoupUnsupportedNodeError)

    def test_doctype(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFragmentRaises('<!DOCTYPE VRML>', SoupSyntaxError)

    def test_entity_declarations(self):
        if False:
            i = 10
            return i + 15
        testcase = '<!ENTITY lol "bad things">'
        self.assertFragmentRaises(testcase, SoupSyntaxError)
        testcase = '<!DOCTYPE div- [<!ENTITY lol "bad things">]>'
        self.assertFragmentRaises(testcase, SoupSyntaxError)

    def test_cdata_section(self):
        if False:
            print('Hello World!')
        testcase = '<![CDATA[If only XHTML 2 went anywhere]]>'
        self.assertFragmentRaises(testcase, SoupUnexpectedCDataSectionError)

    def test_entities(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFragmentRaises('&xml:what;', SoupError)
        self.assertFragmentRaises('&foo,bar;', SoupError)
        self.assertFragmentRaises('&#999999999999;', SoupUnsupportedEntityError)
        self.assertFragmentRaises('&#00;', SoupUnsupportedEntityError)
        self.assertFragmentRaises('&foo-bar;', SoupUnsupportedEntityError)
        self.assertFragmentRaises('&foobar;', SoupUnsupportedEntityError)
        self.assertFragmentValid('&nbsp;')
        self.assertFragmentValid('&Omicron;')

    def test_tag_whitelist(self):
        if False:
            return 10
        testcase = '<div><a><a><script>alert(1)</script></a></a></div>'
        self.assertFragmentRaises(testcase, SoupUnsupportedTagError)

    def test_attr_whitelist(self):
        if False:
            while True:
                i = 10
        testcase = '<div><a><a><em onclick="alert(1)">FOO!</em></a></a></div>'
        self.assertFragmentRaises(testcase, SoupUnsupportedAttrError)

    def test_tag_xmlns(self):
        if False:
            i = 10
            return i + 15
        self.assertFragmentRaises('<xml:div></xml:div>', SoupUnsupportedTagError)
        self.assertFragmentRaises('<div xmlns="http://zombo.com/foo"></div>', SoupError)

    def test_attr_xmlns(self):
        if False:
            return 10
        self.assertFragmentRaises('<div xml:class="baz"></div>', SoupUnsupportedAttrError)

    def test_schemes(self):
        if False:
            while True:
                i = 10
        self.assertFragmentValid('<a href="http://google.com">a</a>')
        self.assertFragmentValid('<a href="Http://google.com">a</a>')
        self.assertFragmentValid('<a href="/google.com">a</a>')
        self.assertFragmentRaises('<a href="javascript://google.com">a</a>', SoupUnsupportedSchemeError)

    def test_crashers(self):
        if False:
            i = 10
            return i + 15
        self.assertFragmentRaises('<a href="http://example.com/%%30%30">foo</a>', SoupDetectedCrasherError)
        self.assertFragmentRaises('<a href="http://example.com/%0%30">foo</a>', SoupDetectedCrasherError)
        self.assertFragmentRaises('<a href="http://example.com/%%300">foo</a>', SoupDetectedCrasherError)
        self.assertFragmentRaises('<a href="http://%s.com">foo</a>' % ('x' * 300), SoupDetectedCrasherError)