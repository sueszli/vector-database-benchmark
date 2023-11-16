"""
Python Markdown

A Python implementation of John Gruber's Markdown.

Documentation: https://python-markdown.github.io/
GitHub: https://github.com/Python-Markdown/markdown/
PyPI: https://pypi.org/project/Markdown/

Started by Manfred Stienstra (http://www.dwerg.net/).
Maintained for a few years by Yuri Takhteyev (http://www.freewisdom.org).
Currently maintained by Waylan Limberg (https://github.com/waylan),
Dmitry Shachnev (https://github.com/mitya57) and Isaac Muse (https://github.com/facelessuser).

Copyright 2007-2023 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""
from unittest import TestSuite
from markdown.test_tools import TestCase
from ..blocks.test_html_blocks import TestHTMLBlocks
from markdown import Markdown
from xml.etree.ElementTree import Element

class TestMarkdownInHTMLPostProcessor(TestCase):
    """ Ensure any remaining elements in HTML stash are properly serialized. """

    def test_stash_to_string(self):
        if False:
            print('Hello World!')
        element = Element('div')
        element.text = 'Foo bar.'
        md = Markdown(extensions=['md_in_html'])
        result = md.postprocessors['raw_html'].stash_to_string(element)
        self.assertEqual(result, '<div>Foo bar.</div>')

class TestDefaultwMdInHTML(TestHTMLBlocks):
    """ Ensure the md_in_html extension does not break the default behavior. """
    default_kwargs = {'extensions': ['md_in_html']}

class TestMdInHTML(TestCase):
    default_kwargs = {'extensions': ['md_in_html']}

    def test_md1_paragraph(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p markdown="1">*foo*</p>', '<p><em>foo</em></p>')

    def test_md1_p_linebreaks(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <p markdown="1">\n                *foo*\n                </p>\n                '), self.dedent('\n                <p>\n                <em>foo</em>\n                </p>\n                '))

    def test_md1_p_blank_lines(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <p markdown="1">\n\n                *foo*\n\n                </p>\n                '), self.dedent('\n                <p>\n\n                <em>foo</em>\n\n                </p>\n                '))

    def test_md1_div(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<div markdown="1">*foo*</div>', self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                '))

    def test_md1_div_linebreaks(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                *foo*\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                '))

    def test_md1_code_span(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                `<h1>code span</h1>`\n                </div>\n                '), self.dedent('\n                <div>\n                <p><code>&lt;h1&gt;code span&lt;/h1&gt;</code></p>\n                </div>\n                '))

    def test_md1_code_span_oneline(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<div markdown="1">`<h1>code span</h1>`</div>', self.dedent('\n                <div>\n                <p><code>&lt;h1&gt;code span&lt;/h1&gt;</code></p>\n                </div>\n                '))

    def test_md1_code_span_unclosed(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                `<p>`\n                </div>\n                '), self.dedent('\n                <div>\n                <p><code>&lt;p&gt;</code></p>\n                </div>\n                '))

    def test_md1_code_span_script_tag(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                `<script>`\n                </div>\n                '), self.dedent('\n                <div>\n                <p><code>&lt;script&gt;</code></p>\n                </div>\n                '))

    def test_md1_div_blank_lines(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                *foo*\n\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                '))

    def test_md1_div_multi(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                *foo*\n\n                __bar__\n\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                <p><strong>bar</strong></p>\n                </div>\n                '))

    def test_md1_div_nested(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                <div markdown="1">\n                *foo*\n                </div>\n\n                </div>\n                '), self.dedent('\n                <div>\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                </div>\n                '))

    def test_md1_div_multi_nest(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                <div markdown="1">\n                <p markdown="1">*foo*</p>\n                </div>\n\n                </div>\n                '), self.dedent('\n                <div>\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                </div>\n                '))

    def text_md1_details(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <details markdown="1">\n                <summary>Click to expand</summary>\n                *foo*\n                </details>\n                '), self.dedent('\n                <details>\n                <summary>Click to expand</summary>\n                <p><em>foo</em></p>\n                </details>\n                '))

    def test_md1_mix(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                A _Markdown_ paragraph before a raw child.\n\n                <p markdown="1">A *raw* child.</p>\n\n                A _Markdown_ tail to the raw child.\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A <em>Markdown</em> paragraph before a raw child.</p>\n                <p>A <em>raw</em> child.</p>\n                <p>A <em>Markdown</em> tail to the raw child.</p>\n                </div>\n                '))

    def test_md1_deep_mix(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                A _Markdown_ paragraph before a raw child.\n\n                A second Markdown paragraph\n                with two lines.\n\n                <div markdown="1">\n\n                A *raw* child.\n\n                <p markdown="1">*foo*</p>\n\n                Raw child tail.\n\n                </div>\n\n                A _Markdown_ tail to the raw child.\n\n                A second tail item\n                with two lines.\n\n                <p markdown="1">More raw.</p>\n\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A <em>Markdown</em> paragraph before a raw child.</p>\n                <p>A second Markdown paragraph\n                with two lines.</p>\n                <div>\n                <p>A <em>raw</em> child.</p>\n                <p><em>foo</em></p>\n                <p>Raw child tail.</p>\n                </div>\n                <p>A <em>Markdown</em> tail to the raw child.</p>\n                <p>A second tail item\n                with two lines.</p>\n                <p>More raw.</p>\n                </div>\n                '))

    def test_md1_div_raw_inline(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                <em>foo</em>\n\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                '))

    def test_no_md1_paragraph(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p>*foo*</p>', '<p>*foo*</p>')

    def test_no_md1_nest(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                A _Markdown_ paragraph before a raw child.\n\n                <p>A *raw* child.</p>\n\n                A _Markdown_ tail to the raw child.\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A <em>Markdown</em> paragraph before a raw child.</p>\n                <p>A *raw* child.</p>\n                <p>A <em>Markdown</em> tail to the raw child.</p>\n                </div>\n                '))

    def test_md1_nested_empty(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                A _Markdown_ paragraph before a raw empty tag.\n\n                <img src="image.png" alt="An image" />\n\n                A _Markdown_ tail to the raw empty tag.\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A <em>Markdown</em> paragraph before a raw empty tag.</p>\n                <p><img src="image.png" alt="An image" /></p>\n                <p>A <em>Markdown</em> tail to the raw empty tag.</p>\n                </div>\n                '))

    def test_md1_nested_empty_block(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                A _Markdown_ paragraph before a raw empty tag.\n\n                <hr />\n\n                A _Markdown_ tail to the raw empty tag.\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A <em>Markdown</em> paragraph before a raw empty tag.</p>\n                <hr />\n                <p>A <em>Markdown</em> tail to the raw empty tag.</p>\n                </div>\n                '))

    def test_empty_tags(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                <div></div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div></div>\n                </div>\n                '))

    def test_orphan_end_tag_in_raw_html(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                <div>\n                Test\n\n                </pre>\n\n                Test\n                </div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div>\n                Test\n\n                </pre>\n\n                Test\n                </div>\n                </div>\n                '))

    def test_complex_nested_case(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                **test**\n                <div>\n                **test**\n                <img src=""/>\n                <code>Test</code>\n                <span>**test**</span>\n                <p>Test 2</p>\n                </div>\n                </div>\n                '), self.dedent('\n                <div>\n                <p><strong>test</strong></p>\n                <div>\n                **test**\n                <img src=""/>\n                <code>Test</code>\n                <span>**test**</span>\n                <p>Test 2</p>\n                </div>\n                </div>\n                '))

    def test_complex_nested_case_whitespace(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                Text with space\t\n                <div markdown="1">\t\n                \t\n                 <div>\n                **test**\n                <img src=""/>\n                <code>Test</code>\n                <span>**test**</span>\n                  <div>With whitespace</div>\n                <p>Test 2</p>\n                </div>\n                **test**\n                </div>\n                '), self.dedent('\n                <p>Text with space </p>\n                <div>\n                <div>\n                **test**\n                <img src=""/>\n                <code>Test</code>\n                <span>**test**</span>\n                  <div>With whitespace</div>\n                <p>Test 2</p>\n                </div>\n                <p><strong>test</strong></p>\n                </div>\n                '))

    def test_md1_intail_md1(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<div markdown="1">*foo*</div><div markdown="1">*bar*</div>', self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                <div>\n                <p><em>bar</em></p>\n                </div>\n                '))

    def test_md1_no_blank_line_before(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                A _Markdown_ paragraph with no blank line after.\n                <div markdown="1">\n                A _Markdown_ paragraph in an HTML block with no blank line before.\n                </div>\n                '), self.dedent('\n                <p>A <em>Markdown</em> paragraph with no blank line after.</p>\n                <div>\n                <p>A <em>Markdown</em> paragraph in an HTML block with no blank line before.</p>\n                </div>\n                '))

    def test_md1_no_line_break(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('A _Markdown_ paragraph with <div markdown="1">no _line break_.</div>', '<p>A <em>Markdown</em> paragraph with <div markdown="1">no <em>line break</em>.</div></p>')

    def test_md1_in_tail(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div></div><div markdown="1">\n                A _Markdown_ paragraph in an HTML block in tail of previous element.\n                </div>\n                '), self.dedent('\n                <div></div>\n                <div>\n                <p>A <em>Markdown</em> paragraph in an HTML block in tail of previous element.</p>\n                </div>\n                '))

    def test_md1_PI_oneliner(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('<div markdown="1"><?php print("foo"); ?></div>', self.dedent('\n                <div>\n                <?php print("foo"); ?>\n                </div>\n                '))

    def test_md1_PI_multiline(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                <?php print("foo"); ?>\n                </div>\n                '), self.dedent('\n                <div>\n                <?php print("foo"); ?>\n                </div>\n                '))

    def test_md1_PI_blank_lines(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                <?php print("foo"); ?>\n\n                </div>\n                '), self.dedent('\n                <div>\n                <?php print("foo"); ?>\n                </div>\n                '))

    def test_md_span_paragraph(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<p markdown="span">*foo*</p>', '<p><em>foo</em></p>')

    def test_md_block_paragraph(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<p markdown="block">*foo*</p>', self.dedent('\n                <p>\n                <p><em>foo</em></p>\n                </p>\n                '))

    def test_md_span_div(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<div markdown="span">*foo*</div>', '<div><em>foo</em></div>')

    def test_md_block_div(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('<div markdown="block">*foo*</div>', self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </div>\n                '))

    def test_md_span_nested_in_block(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="block">\n                <div markdown="span">*foo*</div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div><em>foo</em></div>\n                </div>\n                '))

    def test_md_block_nested_in_span(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="span">\n                <div markdown="block">*foo*</div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div><em>foo</em></div>\n                </div>\n                '))

    def test_md_block_after_span_nested_in_block(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="block">\n                <div markdown="span">*foo*</div>\n                <div markdown="block">*bar*</div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div><em>foo</em></div>\n                <div>\n                <p><em>bar</em></p>\n                </div>\n                </div>\n                '))

    def test_nomd_nested_in_md1(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                *foo*\n                <div>\n                *foo*\n                <p>*bar*</p>\n                *baz*\n                </div>\n                *bar*\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                <div>\n                *foo*\n                <p>*bar*</p>\n                *baz*\n                </div>\n                <p><em>bar</em></p>\n                </div>\n                '))

    def test_md1_nested_in_nomd(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div>\n                <div markdown="1">*foo*</div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div markdown="1">*foo*</div>\n                </div>\n                '))

    def test_md1_single_quotes(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders("<p markdown='1'>*foo*</p>", '<p><em>foo</em></p>')

    def test_md1_no_quotes(self):
        if False:
            return 10
        self.assertMarkdownRenders('<p markdown=1>*foo*</p>', '<p><em>foo</em></p>')

    def test_md_no_value(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('<p markdown>*foo*</p>', '<p><em>foo</em></p>')

    def test_md1_preserve_attrs(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1" id="parent">\n\n                <div markdown="1" class="foo">\n                <p markdown="1" class="bar baz">*foo*</p>\n                </div>\n\n                </div>\n                '), self.dedent('\n                <div id="parent">\n                <div class="foo">\n                <p class="bar baz"><em>foo</em></p>\n                </div>\n                </div>\n                '))

    def test_md1_unclosed_div(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                _foo_\n\n                <div class="unclosed">\n\n                _bar_\n\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                <div class="unclosed">\n\n                _bar_\n\n                </div>\n                </div>\n                '))

    def test_md1_orphan_endtag(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n\n                _foo_\n\n                </p>\n\n                _bar_\n\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em></p>\n                </p>\n                <p><em>bar</em></p>\n                </div>\n                '))

    def test_md1_unclosed_p(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <p markdown="1">_foo_\n                <p markdown="1">_bar_\n                '), self.dedent('\n                <p><em>foo</em>\n                </p>\n                <p><em>bar</em>\n\n                </p>\n                '))

    def test_md1_nested_unclosed_p(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                <p markdown="1">_foo_\n                <p markdown="1">_bar_\n                </div>\n                '), self.dedent('\n                <div>\n                <p><em>foo</em>\n                </p>\n                <p><em>bar</em>\n                </p>\n                </div>\n                '))

    def test_md1_nested_comment(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                A *Markdown* paragraph.\n                <!-- foobar -->\n                A *Markdown* paragraph.\n                </div>\n                '), self.dedent('\n                <div>\n                <p>A <em>Markdown</em> paragraph.</p>\n                <!-- foobar -->\n                <p>A <em>Markdown</em> paragraph.</p>\n                </div>\n                '))

    def test_md1_nested_link_ref(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                [link]: http://example.com\n                <div markdown="1">\n                [link][link]\n                </div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div>\n                <p><a href="http://example.com">link</a></p>\n                </div>\n                </div>\n                '))

    def test_md1_hr_only_start(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr markdown="1">\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p><em>emphasis2</em></p>\n                '))

    def test_md1_hr_self_close(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr markdown="1" />\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p><em>emphasis2</em></p>\n                '))

    def test_md1_hr_start_and_end(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr markdown="1"></hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p></hr>\n                <em>emphasis2</em></p>\n                '))

    def test_md1_hr_only_end(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                </hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em>\n                </hr>\n                <em>emphasis2</em></p>\n                '))

    def test_md1_hr_with_content(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr markdown="1">\n                **content**\n                </hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p><strong>content</strong>\n                </hr>\n                <em>emphasis2</em></p>\n                '))

    def test_no_md1_hr_with_content(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                *emphasis1*\n                <hr>\n                **content**\n                </hr>\n                *emphasis2*\n                '), self.dedent('\n                <p><em>emphasis1</em></p>\n                <hr>\n                <p><strong>content</strong>\n                </hr>\n                <em>emphasis2</em></p>\n                '))

    def test_md1_nested_abbr_ref(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                *[abbr]: Abbreviation\n                <div markdown="1">\n                abbr\n                </div>\n                </div>\n                '), self.dedent('\n                <div>\n                <div>\n                <p><abbr title="Abbreviation">abbr</abbr></p>\n                </div>\n                </div>\n                '), extensions=['md_in_html', 'abbr'])

    def test_md1_nested_footnote_ref(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <div markdown="1">\n                [^1]: The footnote.\n                <div markdown="1">\n                Paragraph with a footnote.[^1]\n                </div>\n                </div>\n                '), '<div>\n<div>\n<p>Paragraph with a footnote.<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></p>\n</div>\n</div>\n<div class="footnote">\n<hr />\n<ol>\n<li id="fn:1">\n<p>The footnote.&#160;<a class="footnote-backref" href="#fnref:1" title="Jump back to footnote 1 in the text">&#8617;</a></p>\n</li>\n</ol>\n</div>', extensions=['md_in_html', 'footnotes'])

def load_tests(loader, tests, pattern):
    if False:
        print('Hello World!')
    " Ensure `TestHTMLBlocks` doesn't get run twice by excluding it here. "
    suite = TestSuite()
    for test_class in [TestDefaultwMdInHTML, TestMdInHTML, TestMarkdownInHTMLPostProcessor]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite