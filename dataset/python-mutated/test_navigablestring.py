import pytest
from bs4.element import CData, Comment, Declaration, Doctype, NavigableString, RubyParenthesisString, RubyTextString, Script, Stylesheet, TemplateString
from . import SoupTest

class TestNavigableString(SoupTest):

    def test_text_acquisition_methods(self):
        if False:
            print('Hello World!')
        s = NavigableString('fee ')
        cdata = CData('fie ')
        comment = Comment('foe ')
        assert 'fee ' == s.get_text()
        assert 'fee' == s.get_text(strip=True)
        assert ['fee '] == list(s.strings)
        assert ['fee'] == list(s.stripped_strings)
        assert ['fee '] == list(s._all_strings())
        assert 'fie ' == cdata.get_text()
        assert 'fie' == cdata.get_text(strip=True)
        assert ['fie '] == list(cdata.strings)
        assert ['fie'] == list(cdata.stripped_strings)
        assert ['fie '] == list(cdata._all_strings())
        assert '' == comment.get_text()
        assert [] == list(comment.strings)
        assert [] == list(comment.stripped_strings)
        assert [] == list(comment._all_strings())
        assert 'foe' == comment.get_text(strip=True, types=Comment)
        assert 'foe ' == comment.get_text(types=(Comment, NavigableString))

    def test_string_has_immutable_name_property(self):
        if False:
            print('Hello World!')
        string = self.soup('s').string
        assert None == string.name
        with pytest.raises(AttributeError):
            string.name = 'foo'

class TestNavigableStringSubclasses(SoupTest):

    def test_cdata(self):
        if False:
            i = 10
            return i + 15
        soup = self.soup('')
        cdata = CData('foo')
        soup.insert(1, cdata)
        assert str(soup) == '<![CDATA[foo]]>'
        assert soup.find(string='foo') == 'foo'
        assert soup.contents[0] == 'foo'

    def test_cdata_is_never_formatted(self):
        if False:
            for i in range(10):
                print('nop')
        'Text inside a CData object is passed into the formatter.\n\n        But the return value is ignored.\n        '
        self.count = 0

        def increment(*args):
            if False:
                for i in range(10):
                    print('nop')
            self.count += 1
            return 'BITTER FAILURE'
        soup = self.soup('')
        cdata = CData('<><><>')
        soup.insert(1, cdata)
        assert b'<![CDATA[<><><>]]>' == soup.encode(formatter=increment)
        assert 1 == self.count

    def test_doctype_ends_in_newline(self):
        if False:
            return 10
        doctype = Doctype('foo')
        soup = self.soup('')
        soup.insert(1, doctype)
        assert soup.encode() == b'<!DOCTYPE foo>\n'

    def test_declaration(self):
        if False:
            while True:
                i = 10
        d = Declaration('foo')
        assert '<?foo?>' == d.output_ready()

    def test_default_string_containers(self):
        if False:
            for i in range(10):
                print('nop')
        soup = self.soup('<div>text</div><script>text</script><style>text</style>')
        assert [NavigableString, Script, Stylesheet] == [x.__class__ for x in soup.find_all(string=True)]
        soup = self.soup('<template>Some text<p>In a tag</p></template>Some text outside')
        assert all((isinstance(x, TemplateString) for x in soup.template._all_strings(types=None)))
        outside = soup.template.next_sibling
        assert isinstance(outside, NavigableString)
        assert not isinstance(outside, TemplateString)
        markup = b'<template>Some text<p>In a tag</p><!--with a comment--></template>'
        soup = self.soup(markup)
        assert markup == soup.template.encode('utf8')

    def test_ruby_strings(self):
        if False:
            return 10
        markup = '<ruby>漢 <rp>(</rp><rt>kan</rt><rp>)</rp> 字 <rp>(</rp><rt>ji</rt><rp>)</rp></ruby>'
        soup = self.soup(markup)
        assert isinstance(soup.rp.string, RubyParenthesisString)
        assert isinstance(soup.rt.string, RubyTextString)
        assert '漢字' == soup.get_text(strip=True)
        assert '漢(kan)字(ji)' == soup.get_text(strip=True, types=(NavigableString, RubyTextString, RubyParenthesisString))