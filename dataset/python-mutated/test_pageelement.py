"""Tests of the bs4.element.PageElement class"""
import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import Comment, ResultSet, SoupStrainer
from . import SoupTest

class TestEncoding(SoupTest):
    """Test the ability to encode objects into strings."""

    def test_unicode_string_can_be_encoded(self):
        if False:
            while True:
                i = 10
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert soup.b.string.encode('utf-8') == '☃'.encode('utf-8')

    def test_tag_containing_unicode_string_can_be_encoded(self):
        if False:
            return 10
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert soup.b.encode('utf-8') == html.encode('utf-8')

    def test_encoding_substitutes_unrecognized_characters_by_default(self):
        if False:
            i = 10
            return i + 15
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert soup.b.encode('ascii') == b'<b>&#9731;</b>'

    def test_encoding_can_be_made_strict(self):
        if False:
            for i in range(10):
                print('nop')
        html = '<b>☃</b>'
        soup = self.soup(html)
        with pytest.raises(UnicodeEncodeError):
            soup.encode('ascii', errors='strict')

    def test_decode_contents(self):
        if False:
            print('Hello World!')
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert '☃' == soup.b.decode_contents()

    def test_encode_contents(self):
        if False:
            i = 10
            return i + 15
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert '☃'.encode('utf8') == soup.b.encode_contents(encoding='utf8')

    def test_encode_deeply_nested_document(self):
        if False:
            for i in range(10):
                print('nop')
        limit = sys.getrecursionlimit() + 1
        markup = '<span>' * limit
        soup = self.soup(markup)
        encoded = soup.encode()
        assert limit == encoded.count(b'<span>')

    def test_deprecated_renderContents(self):
        if False:
            for i in range(10):
                print('nop')
        html = '<b>☃</b>'
        soup = self.soup(html)
        soup.renderContents()
        assert '☃'.encode('utf8') == soup.b.renderContents()

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        html = '<b>☃</b>'
        soup = self.soup(html)
        assert html == repr(soup)

class TestFormatters(SoupTest):
    """Test the formatting feature, used by methods like decode() and
    prettify(), and the formatters themselves.
    """

    def test_default_formatter_is_minimal(self):
        if False:
            for i in range(10):
                print('nop')
        markup = '<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
        soup = self.soup(markup)
        decoded = soup.decode(formatter='minimal')
        assert decoded == self.document_for('<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>')

    def test_formatter_html(self):
        if False:
            for i in range(10):
                print('nop')
        markup = '<br><b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
        soup = self.soup(markup)
        decoded = soup.decode(formatter='html')
        assert decoded == self.document_for('<br/><b>&lt;&lt;Sacr&eacute; bleu!&gt;&gt;</b>')

    def test_formatter_html5(self):
        if False:
            print('Hello World!')
        markup = '<br><b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
        soup = self.soup(markup)
        decoded = soup.decode(formatter='html5')
        assert decoded == self.document_for('<br><b>&lt;&lt;Sacr&eacute; bleu!&gt;&gt;</b>')

    def test_formatter_minimal(self):
        if False:
            i = 10
            return i + 15
        markup = '<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
        soup = self.soup(markup)
        decoded = soup.decode(formatter='minimal')
        assert decoded == self.document_for('<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>')

    def test_formatter_null(self):
        if False:
            i = 10
            return i + 15
        markup = '<b>&lt;&lt;Sacré bleu!&gt;&gt;</b>'
        soup = self.soup(markup)
        decoded = soup.decode(formatter=None)
        assert decoded == self.document_for('<b><<Sacré bleu!>></b>')

    def test_formatter_custom(self):
        if False:
            for i in range(10):
                print('nop')
        markup = '<b>&lt;foo&gt;</b><b>bar</b><br/>'
        soup = self.soup(markup)
        decoded = soup.decode(formatter=lambda x: x.upper())
        assert decoded == self.document_for('<b><FOO></b><b>BAR</b><br/>')

    def test_formatter_is_run_on_attribute_values(self):
        if False:
            i = 10
            return i + 15
        markup = '<a href="http://a.com?a=b&c=é">e</a>'
        soup = self.soup(markup)
        a = soup.a
        expect_minimal = '<a href="http://a.com?a=b&amp;c=é">e</a>'
        assert expect_minimal == a.decode()
        assert expect_minimal == a.decode(formatter='minimal')
        expect_html = '<a href="http://a.com?a=b&amp;c=&eacute;">e</a>'
        assert expect_html == a.decode(formatter='html')
        assert markup == a.decode(formatter=None)
        expect_upper = '<a href="HTTP://A.COM?A=B&C=É">E</a>'
        assert expect_upper == a.decode(formatter=lambda x: x.upper())

    def test_formatter_skips_script_tag_for_html_documents(self):
        if False:
            while True:
                i = 10
        doc = '\n  <script type="text/javascript">\n   console.log("< < hey > > ");\n  </script>\n'
        encoded = BeautifulSoup(doc, 'html.parser').encode()
        assert b'< < hey > >' in encoded

    def test_formatter_skips_style_tag_for_html_documents(self):
        if False:
            for i in range(10):
                print('nop')
        doc = '\n  <style type="text/css">\n   console.log("< < hey > > ");\n  </style>\n'
        encoded = BeautifulSoup(doc, 'html.parser').encode()
        assert b'< < hey > >' in encoded

    def test_prettify_leaves_preformatted_text_alone(self):
        if False:
            print('Hello World!')
        soup = self.soup('<div>  foo  <pre>  \tbar\n  \n  </pre>  baz  <textarea> eee\nfff\t</textarea></div>')
        assert '<div>\n foo\n <pre>  \tbar\n  \n  </pre>\n baz\n <textarea> eee\nfff\t</textarea>\n</div>\n' == soup.div.prettify()

    def test_prettify_handles_nested_string_literal_tags(self):
        if False:
            while True:
                i = 10
        markup = '<div><pre><code>some\n<script><pre>code</pre></script> for you \n</code></pre></div>'
        expect = '<div>\n <pre><code>some\n<script><pre>code</pre></script> for you \n</code></pre>\n</div>\n'
        soup = self.soup(markup)
        assert expect == soup.div.prettify()

    def test_prettify_accepts_formatter_function(self):
        if False:
            print('Hello World!')
        soup = BeautifulSoup('<html><body>foo</body></html>', 'html.parser')
        pretty = soup.prettify(formatter=lambda x: x.upper())
        assert 'FOO' in pretty

    def test_prettify_outputs_unicode_by_default(self):
        if False:
            print('Hello World!')
        soup = self.soup('<a></a>')
        assert str == type(soup.prettify())

    def test_prettify_can_encode_data(self):
        if False:
            while True:
                i = 10
        soup = self.soup('<a></a>')
        assert bytes == type(soup.prettify('utf-8'))

    def test_html_entity_substitution_off_by_default(self):
        if False:
            while True:
                i = 10
        markup = '<b>Sacré bleu!</b>'
        soup = self.soup(markup)
        encoded = soup.b.encode('utf-8')
        assert encoded == markup.encode('utf-8')

    def test_encoding_substitution(self):
        if False:
            while True:
                i = 10
        meta_tag = '<meta content="text/html; charset=x-sjis" http-equiv="Content-type"/>'
        soup = self.soup(meta_tag)
        assert soup.meta['content'] == 'text/html; charset=x-sjis'
        utf_8 = soup.encode('utf-8')
        assert b'charset=utf-8' in utf_8
        euc_jp = soup.encode('euc_jp')
        assert b'charset=euc_jp' in euc_jp
        shift_jis = soup.encode('shift-jis')
        assert b'charset=shift-jis' in shift_jis
        utf_16_u = soup.encode('utf-16').decode('utf-16')
        assert 'charset=utf-16' in utf_16_u

    def test_encoding_substitution_doesnt_happen_if_tag_is_strained(self):
        if False:
            while True:
                i = 10
        markup = '<head><meta content="text/html; charset=x-sjis" http-equiv="Content-type"/></head><pre>foo</pre>'
        strainer = SoupStrainer('pre')
        soup = self.soup(markup, parse_only=strainer)
        assert soup.contents[0].name == 'pre'

class TestPersistence(SoupTest):
    """Testing features like pickle and deepcopy."""

    def setup_method(self):
        if False:
            print('Hello World!')
        self.page = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN"\n"http://www.w3.org/TR/REC-html40/transitional.dtd">\n<html>\n<head>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n<title>Beautiful Soup: We called him Tortoise because he taught us.</title>\n<link rev="made" href="mailto:leonardr@segfault.org">\n<meta name="Description" content="Beautiful Soup: an HTML parser optimized for screen-scraping.">\n<meta name="generator" content="Markov Approximation 1.4 (module: leonardr)">\n<meta name="author" content="Leonard Richardson">\n</head>\n<body>\n<a href="foo">foo</a>\n<a href="foo"><b>bar</b></a>\n</body>\n</html>'
        self.tree = self.soup(self.page)

    def test_pickle_and_unpickle_identity(self):
        if False:
            i = 10
            return i + 15
        dumped = pickle.dumps(self.tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == self.tree.decode()

    def test_deepcopy_identity(self):
        if False:
            while True:
                i = 10
        copied = copy.deepcopy(self.tree)
        assert copied.decode() == self.tree.decode()

    def test_copy_deeply_nested_document(self):
        if False:
            while True:
                i = 10
        limit = sys.getrecursionlimit() + 1
        markup = '<span>' * limit
        soup = self.soup(markup)
        copied = copy.copy(soup)
        copied = copy.deepcopy(soup)

    def test_copy_preserves_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        soup = BeautifulSoup(b'<p>&nbsp;</p>', 'html.parser')
        encoding = soup.original_encoding
        copy = soup.__copy__()
        assert '<p>\xa0</p>' == str(copy)
        assert encoding == copy.original_encoding

    def test_copy_preserves_builder_information(self):
        if False:
            i = 10
            return i + 15
        tag = self.soup('<p></p>').p
        tag.sourceline = 10
        tag.sourcepos = 33
        copied = tag.__copy__()
        assert tag.sourceline == copied.sourceline
        assert tag.sourcepos == copied.sourcepos
        assert tag.can_be_empty_element == copied.can_be_empty_element
        assert tag.cdata_list_attributes == copied.cdata_list_attributes
        assert tag.preserve_whitespace_tags == copied.preserve_whitespace_tags
        assert tag.interesting_string_types == copied.interesting_string_types

    def test_unicode_pickle(self):
        if False:
            while True:
                i = 10
        html = '<b>☃</b>'
        soup = self.soup(html)
        dumped = pickle.dumps(soup, pickle.HIGHEST_PROTOCOL)
        loaded = pickle.loads(dumped)
        assert loaded.decode() == soup.decode()

    def test_copy_navigablestring_is_not_attached_to_tree(self):
        if False:
            print('Hello World!')
        html = '<b>Foo<a></a></b><b>Bar</b>'
        soup = self.soup(html)
        s1 = soup.find(string='Foo')
        s2 = copy.copy(s1)
        assert s1 == s2
        assert None == s2.parent
        assert None == s2.next_element
        assert None != s1.next_sibling
        assert None == s2.next_sibling
        assert None == s2.previous_element

    def test_copy_navigablestring_subclass_has_same_type(self):
        if False:
            i = 10
            return i + 15
        html = '<b><!--Foo--></b>'
        soup = self.soup(html)
        s1 = soup.string
        s2 = copy.copy(s1)
        assert s1 == s2
        assert isinstance(s2, Comment)

    def test_copy_entire_soup(self):
        if False:
            i = 10
            return i + 15
        html = '<div><b>Foo<a></a></b><b>Bar</b></div>end'
        soup = self.soup(html)
        soup_copy = copy.copy(soup)
        assert soup == soup_copy

    def test_copy_tag_copies_contents(self):
        if False:
            while True:
                i = 10
        html = '<div><b>Foo<a></a></b><b>Bar</b></div>end'
        soup = self.soup(html)
        div = soup.div
        div_copy = copy.copy(div)
        assert str(div) == str(div_copy)
        assert div == div_copy
        assert div is not div_copy
        assert None == div_copy.parent
        assert None == div_copy.previous_element
        assert None == div_copy.find(string='Bar').next_element
        assert None != div.find(string='Bar').next_element