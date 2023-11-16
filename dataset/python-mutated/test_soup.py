"""Tests of Beautiful Soup as a whole."""
from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile
from bs4 import BeautifulSoup, BeautifulStoneSoup, GuessedAtParserWarning, MarkupResemblesLocatorWarning, dammit
from bs4.builder import builder_registry, TreeBuilder, ParserRejectedMarkup
from bs4.element import Comment, SoupStrainer, PYTHON_SPECIFIC_ENCODINGS, Tag, NavigableString
from . import default_builder, LXML_PRESENT, SoupTest
import warnings

class TestConstructor(SoupTest):

    def test_short_unicode_input(self):
        if False:
            return 10
        data = '<h1>éé</h1>'
        soup = self.soup(data)
        assert 'éé' == soup.h1.string

    def test_embedded_null(self):
        if False:
            for i in range(10):
                print('nop')
        data = '<h1>foo\x00bar</h1>'
        soup = self.soup(data)
        assert 'foo\x00bar' == soup.h1.string

    def test_exclude_encodings(self):
        if False:
            for i in range(10):
                print('nop')
        utf8_data = 'Räksmörgås'.encode('utf-8')
        soup = self.soup(utf8_data, exclude_encodings=['utf-8'])
        assert 'windows-1252' == soup.original_encoding

    def test_custom_builder_class(self):
        if False:
            i = 10
            return i + 15

        class Mock(object):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                self.called_with = kwargs
                self.is_xml = True
                self.store_line_numbers = False
                self.cdata_list_attributes = []
                self.preserve_whitespace_tags = []
                self.string_containers = {}

            def initialize_soup(self, soup):
                if False:
                    while True:
                        i = 10
                pass

            def feed(self, markup):
                if False:
                    for i in range(10):
                        print('nop')
                self.fed = markup

            def reset(self):
                if False:
                    return 10
                pass

            def ignore(self, ignore):
                if False:
                    for i in range(10):
                        print('nop')
                pass
            set_up_substitutions = can_be_empty_element = ignore

            def prepare_markup(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                yield ('prepared markup', 'original encoding', 'declared encoding', 'contains replacement characters')
        kwargs = dict(var='value', convertEntities=True)
        with warnings.catch_warnings(record=True):
            soup = BeautifulSoup('', builder=Mock, **kwargs)
        assert isinstance(soup.builder, Mock)
        assert dict(var='value') == soup.builder.called_with
        assert 'prepared markup' == soup.builder.fed
        builder = Mock(**kwargs)
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup('', builder=builder, ignored_value=True)
        msg = str(w[0].message)
        assert msg.startswith('Keyword arguments to the BeautifulSoup constructor will be ignored.')
        assert builder == soup.builder
        assert kwargs == builder.called_with

    def test_parser_markup_rejection(self):
        if False:
            i = 10
            return i + 15

        class Mock(TreeBuilder):

            def feed(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                raise ParserRejectedMarkup('Nope.')

        def prepare_markup(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            yield (markup, None, None, False)
            yield (markup, None, None, False)
        import re
        with pytest.raises(ParserRejectedMarkup) as exc_info:
            BeautifulSoup('', builder=Mock)
        assert 'The markup you provided was rejected by the parser. Trying a different parser or a different encoding may help.' in str(exc_info.value)

    def test_cdata_list_attributes(self):
        if False:
            return 10
        markup = '<a id=" an id " class=" a class "></a>'
        soup = self.soup(markup)
        a = soup.a
        assert ' an id ' == a['id']
        assert ['a', 'class'] == a['class']
        soup = self.soup(markup, builder=default_builder, multi_valued_attributes=None)
        assert ' a class ' == soup.a['class']
        for switcheroo in ({'*': 'id'}, {'a': 'id'}):
            with warnings.catch_warnings(record=True) as w:
                soup = self.soup(markup, builder=None, multi_valued_attributes=switcheroo)
            a = soup.a
            assert ['an', 'id'] == a['id']
            assert ' a class ' == a['class']

    def test_replacement_classes(self):
        if False:
            return 10

        class TagPlus(Tag):
            pass

        class StringPlus(NavigableString):
            pass

        class CommentPlus(Comment):
            pass
        soup = self.soup('<a><b>foo</b>bar</a><!--whee-->', element_classes={Tag: TagPlus, NavigableString: StringPlus, Comment: CommentPlus})
        assert all((isinstance(x, (TagPlus, StringPlus, CommentPlus)) for x in soup.recursiveChildGenerator()))

    def test_alternate_string_containers(self):
        if False:
            return 10

        class PString(NavigableString):
            pass

        class BString(NavigableString):
            pass
        soup = self.soup('<div>Hello.<p>Here is <b>some <i>bolded</i></b> text', string_containers={'b': BString, 'p': PString})
        assert isinstance(soup.div.contents[0], NavigableString)
        assert isinstance(soup.p.contents[0], PString)
        for s in soup.b.strings:
            assert isinstance(s, BString)
        assert [] == soup.string_container_stack

class TestOutput(SoupTest):

    @pytest.mark.parametrize('eventual_encoding,actual_encoding', [('utf-8', 'utf-8'), ('utf-16', 'utf-16')])
    def test_decode_xml_declaration(self, eventual_encoding, actual_encoding):
        if False:
            print('Hello World!')
        soup = self.soup('<tag></tag>')
        soup.is_xml = True
        assert f'<?xml version="1.0" encoding="{actual_encoding}"?>\n<tag></tag>' == soup.decode(eventual_encoding=eventual_encoding)

    @pytest.mark.parametrize('eventual_encoding', [x for x in PYTHON_SPECIFIC_ENCODINGS] + [None])
    def test_decode_xml_declaration_with_missing_or_python_internal_eventual_encoding(self, eventual_encoding):
        if False:
            for i in range(10):
                print('nop')
        soup = BeautifulSoup('<tag></tag>', 'html.parser')
        soup.is_xml = True
        assert f'<?xml version="1.0"?>\n<tag></tag>' == soup.decode(eventual_encoding=eventual_encoding)

    def test(self):
        if False:
            i = 10
            return i + 15
        soup = self.soup('<tag></tag>')
        assert b'<tag></tag>' == soup.encode(encoding='utf-8')
        assert b'<tag></tag>' == soup.encode_contents(encoding='utf-8')
        assert '<tag></tag>' == soup.decode_contents()
        assert '<tag>\n</tag>\n' == soup.prettify()

class TestWarnings(SoupTest):

    def _assert_warning(self, warnings, cls):
        if False:
            i = 10
            return i + 15
        for w in warnings:
            if isinstance(w.message, cls):
                assert w.filename == __file__
                return w
        raise Exception('%s warning not found in %r' % (cls, warnings))

    def _assert_no_parser_specified(self, w):
        if False:
            print('Hello World!')
        warning = self._assert_warning(w, GuessedAtParserWarning)
        message = str(warning.message)
        assert message.startswith(BeautifulSoup.NO_PARSER_SPECIFIED_WARNING[:60])

    def test_warning_if_no_parser_specified(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup('<a><b></b></a>')
        self._assert_no_parser_specified(w)

    def test_warning_if_parser_specified_too_vague(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup('<a><b></b></a>', 'html')
        self._assert_no_parser_specified(w)

    def test_no_warning_if_explicit_parser_specified(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup('<a><b></b></a>')
        assert [] == w

    def test_parseOnlyThese_renamed_to_parse_only(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup('<a><b></b></a>', 'html.parser', parseOnlyThese=SoupStrainer('b'))
        warning = self._assert_warning(w, DeprecationWarning)
        msg = str(warning.message)
        assert 'parseOnlyThese' in msg
        assert 'parse_only' in msg
        assert b'<b></b>' == soup.encode()

    def test_fromEncoding_renamed_to_from_encoding(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            utf8 = b'\xc3\xa9'
            soup = BeautifulSoup(utf8, 'html.parser', fromEncoding='utf8')
        warning = self._assert_warning(w, DeprecationWarning)
        msg = str(warning.message)
        assert 'fromEncoding' in msg
        assert 'from_encoding' in msg
        assert 'utf8' == soup.original_encoding

    def test_unrecognized_keyword_argument(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            self.soup('<a>', no_such_argument=True)

    @pytest.mark.parametrize('extension', ['markup.html', 'markup.htm', 'markup.HTML', 'markup.txt', 'markup.xhtml', 'markup.xml', '/home/user/file', 'c:\\user\x0cile'])
    def test_resembles_filename_warning(self, extension):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup('markup' + extension, 'html.parser')
            warning = self._assert_warning(w, MarkupResemblesLocatorWarning)
            assert 'looks more like a filename' in str(warning.message)

    @pytest.mark.parametrize('extension', ['markuphtml', 'markup.com', '', 'markup.js'])
    def test_resembles_filename_no_warning(self, extension):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup('markup' + extension)
        assert [] == w

    def test_url_warning_with_bytes_url(self):
        if False:
            while True:
                i = 10
        url = b'http://www.crummybytes.com/'
        with warnings.catch_warnings(record=True) as warning_list:
            soup = BeautifulSoup(url, 'html.parser')
        warning = self._assert_warning(warning_list, MarkupResemblesLocatorWarning)
        assert 'looks more like a URL' in str(warning.message)
        assert url not in str(warning.message).encode('utf8')

    def test_url_warning_with_unicode_url(self):
        if False:
            return 10
        url = 'http://www.crummyunicode.com/'
        with warnings.catch_warnings(record=True) as warning_list:
            soup = BeautifulSoup(url, 'html.parser')
        warning = self._assert_warning(warning_list, MarkupResemblesLocatorWarning)
        assert 'looks more like a URL' in str(warning.message)
        assert url not in str(warning.message)

    def test_url_warning_with_bytes_and_space(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as warning_list:
            soup = self.soup(b'http://www.crummybytes.com/ is great')
        assert not any(('looks more like a URL' in str(w.message) for w in warning_list))

    def test_url_warning_with_unicode_and_space(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as warning_list:
            soup = self.soup('http://www.crummyunicode.com/ is great')
        assert not any(('looks more like a URL' in str(w.message) for w in warning_list))

class TestSelectiveParsing(SoupTest):

    def test_parse_with_soupstrainer(self):
        if False:
            while True:
                i = 10
        markup = 'No<b>Yes</b><a>No<b>Yes <c>Yes</c></b>'
        strainer = SoupStrainer('b')
        soup = self.soup(markup, parse_only=strainer)
        assert soup.encode() == b'<b>Yes</b><b>Yes <c>Yes</c></b>'

class TestNewTag(SoupTest):
    """Test the BeautifulSoup.new_tag() method."""

    def test_new_tag(self):
        if False:
            return 10
        soup = self.soup('')
        new_tag = soup.new_tag('foo', bar='baz', attrs={'name': 'a name'})
        assert isinstance(new_tag, Tag)
        assert 'foo' == new_tag.name
        assert dict(bar='baz', name='a name') == new_tag.attrs
        assert None == new_tag.parent

    @pytest.mark.skipif(not LXML_PRESENT, reason='lxml not installed, cannot parse XML document')
    def test_xml_tag_inherits_self_closing_rules_from_builder(self):
        if False:
            return 10
        xml_soup = BeautifulSoup('', 'xml')
        xml_br = xml_soup.new_tag('br')
        xml_p = xml_soup.new_tag('p')
        assert b'<br/>' == xml_br.encode()
        assert b'<p/>' == xml_p.encode()

    def test_tag_inherits_self_closing_rules_from_builder(self):
        if False:
            while True:
                i = 10
        html_soup = BeautifulSoup('', 'html.parser')
        html_br = html_soup.new_tag('br')
        html_p = html_soup.new_tag('p')
        assert b'<br/>' == html_br.encode()
        assert b'<p></p>' == html_p.encode()

class TestNewString(SoupTest):
    """Test the BeautifulSoup.new_string() method."""

    def test_new_string_creates_navigablestring(self):
        if False:
            while True:
                i = 10
        soup = self.soup('')
        s = soup.new_string('foo')
        assert 'foo' == s
        assert isinstance(s, NavigableString)

    def test_new_string_can_create_navigablestring_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        soup = self.soup('')
        s = soup.new_string('foo', Comment)
        assert 'foo' == s
        assert isinstance(s, Comment)

class TestPickle(SoupTest):

    def test_normal_pickle(self):
        if False:
            while True:
                i = 10
        soup = self.soup('<a>some markup</a>')
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert 'some markup' == unpickled.a.string

    def test_pickle_with_no_builder(self):
        if False:
            for i in range(10):
                print('nop')
        soup = self.soup('some markup')
        soup.builder = None
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert 'some markup' == unpickled.string

class TestEncodingConversion(SoupTest):

    def setup_method(self):
        if False:
            return 10
        self.unicode_data = '<html><head><meta charset="utf-8"/></head><body><foo>Sacré bleu!</foo></body></html>'
        self.utf8_data = self.unicode_data.encode('utf-8')
        assert self.utf8_data == b'<html><head><meta charset="utf-8"/></head><body><foo>Sacr\xc3\xa9 bleu!</foo></body></html>'

    def test_ascii_in_unicode_out(self):
        if False:
            print('Hello World!')
        chardet = dammit.chardet_dammit
        logging.disable(logging.WARNING)
        try:

            def noop(str):
                if False:
                    for i in range(10):
                        print('nop')
                return None
            dammit.chardet_dammit = noop
            ascii = b'<foo>a</foo>'
            soup_from_ascii = self.soup(ascii)
            unicode_output = soup_from_ascii.decode()
            assert isinstance(unicode_output, str)
            assert unicode_output == self.document_for(ascii.decode())
            assert soup_from_ascii.original_encoding.lower() == 'utf-8'
        finally:
            logging.disable(logging.NOTSET)
            dammit.chardet_dammit = chardet

    def test_unicode_in_unicode_out(self):
        if False:
            print('Hello World!')
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.decode() == self.unicode_data
        assert soup_from_unicode.foo.string == 'Sacré bleu!'
        assert soup_from_unicode.original_encoding == None

    def test_utf8_in_unicode_out(self):
        if False:
            i = 10
            return i + 15
        soup_from_utf8 = self.soup(self.utf8_data)
        assert soup_from_utf8.decode() == self.unicode_data
        assert soup_from_utf8.foo.string == 'Sacré bleu!'

    def test_utf8_out(self):
        if False:
            while True:
                i = 10
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.encode('utf-8') == self.utf8_data