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
from markdown.test_tools import TestCase

class TestAbbr(TestCase):
    default_kwargs = {'extensions': ['abbr']}

    def test_abbr_upper(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n\n                *[ABBR]: Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr></p>\n                '))

    def test_abbr_lower(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                abbr\n\n                *[abbr]: Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">abbr</abbr></p>\n                '))

    def test_abbr_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                The HTML specification\n                is maintained by the W3C.\n\n                *[HTML]: Hyper Text Markup Language\n                *[W3C]:  World Wide Web Consortium\n                '), self.dedent('\n                <p>The <abbr title="Hyper Text Markup Language">HTML</abbr> specification\n                is maintained by the <abbr title="World Wide Web Consortium">W3C</abbr>.</p>\n                '))

    def test_abbr_override(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n\n                *[ABBR]: Ignored\n                *[ABBR]: The override\n                '), self.dedent('\n                <p><abbr title="The override">ABBR</abbr></p>\n                '))

    def test_abbr_no_blank_Lines(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n                *[ABBR]: Abbreviation\n                ABBR\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr></p>\n                <p><abbr title="Abbreviation">ABBR</abbr></p>\n                '))

    def test_abbr_no_space(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n\n                *[ABBR]:Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr></p>\n                '))

    def test_abbr_extra_space(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n\n                *[ABBR] :      Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr></p>\n                '))

    def test_abbr_line_break(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n\n                *[ABBR]:\n                    Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr></p>\n                '))

    def test_abbr_ignore_unmatched_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ABBR abbr\n\n                *[ABBR]: Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr> abbr</p>\n                '))

    def test_abbr_partial_word(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ABBR ABBREVIATION\n\n                *[ABBR]: Abbreviation\n                '), self.dedent('\n                <p><abbr title="Abbreviation">ABBR</abbr> ABBREVIATION</p>\n                '))

    def test_abbr_unused(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                foo bar\n\n                *[ABBR]: Abbreviation\n                '), self.dedent('\n                <p>foo bar</p>\n                '))

    def test_abbr_double_quoted(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ABBR\n\n                *[ABBR]: "Abbreviation"\n                '), self.dedent('\n                <p><abbr title="&quot;Abbreviation&quot;">ABBR</abbr></p>\n                '))

    def test_abbr_single_quoted(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent("\n                ABBR\n\n                *[ABBR]: 'Abbreviation'\n                "), self.dedent('\n                <p><abbr title="\'Abbreviation\'">ABBR</abbr></p>\n                '))