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

class TestParagraphBlocks(TestCase):

    def test_simple_paragraph(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('A simple paragraph.', '<p>A simple paragraph.</p>')

    def test_blank_line_before_paragraph(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('\nA paragraph preceded by a blank line.', '<p>A paragraph preceded by a blank line.</p>')

    def test_multiline_paragraph(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                This is a paragraph\n                on multiple lines\n                with hard returns.\n                '), self.dedent('\n                <p>This is a paragraph\n                on multiple lines\n                with hard returns.</p>\n                '))

    def test_paragraph_long_line(self):
        if False:
            return 10
        self.assertMarkdownRenders('A very long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long paragraph on 1 line.', '<p>A very long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long paragraph on 1 line.</p>')

    def test_2_paragraphs_long_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('A very long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long paragraph on 1 line.\n\nA new long long long long long long long long long long long long long long long long paragraph on 1 line.', '<p>A very long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long paragraph on 1 line.</p>\n<p>A new long long long long long long long long long long long long long long long long paragraph on 1 line.</p>')

    def test_consecutive_paragraphs(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                Paragraph 1.\n\n                Paragraph 2.\n                '), self.dedent('\n                <p>Paragraph 1.</p>\n                <p>Paragraph 2.</p>\n                '))

    def test_consecutive_paragraphs_tab(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                Paragraph followed by a line with a tab only.\n                \t\n                Paragraph after a line with a tab only.\n                '), self.dedent('\n                <p>Paragraph followed by a line with a tab only.</p>\n                <p>Paragraph after a line with a tab only.</p>\n                '))

    def test_consecutive_paragraphs_space(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                Paragraph followed by a line with a space only.\n\n                Paragraph after a line with a space only.\n                '), self.dedent('\n                <p>Paragraph followed by a line with a space only.</p>\n                <p>Paragraph after a line with a space only.</p>\n                '))

    def test_consecutive_multiline_paragraphs(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                Paragraph 1, line 1.\n                Paragraph 1, line 2.\n\n                Paragraph 2, line 1.\n                Paragraph 2, line 2.\n                '), self.dedent('\n                <p>Paragraph 1, line 1.\n                Paragraph 1, line 2.</p>\n                <p>Paragraph 2, line 1.\n                Paragraph 2, line 2.</p>\n                '))

    def test_paragraph_leading_space(self):
        if False:
            return 10
        self.assertMarkdownRenders(' A paragraph with 1 leading space.', '<p>A paragraph with 1 leading space.</p>')

    def test_paragraph_2_leading_spaces(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('  A paragraph with 2 leading spaces.', '<p>A paragraph with 2 leading spaces.</p>')

    def test_paragraph_3_leading_spaces(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('   A paragraph with 3 leading spaces.', '<p>A paragraph with 3 leading spaces.</p>')

    def test_paragraph_trailing_leading_space(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(' A paragraph with 1 trailing and 1 leading space. ', '<p>A paragraph with 1 trailing and 1 leading space. </p>')

    def test_paragraph_trailing_tab(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('A paragraph with 1 trailing tab.\t', '<p>A paragraph with 1 trailing tab.    </p>')

    def test_paragraphs_CR(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('Paragraph 1, line 1.\rParagraph 1, line 2.\r\rParagraph 2, line 1.\rParagraph 2, line 2.\r', self.dedent('\n                <p>Paragraph 1, line 1.\n                Paragraph 1, line 2.</p>\n                <p>Paragraph 2, line 1.\n                Paragraph 2, line 2.</p>\n                '))

    def test_paragraphs_LF(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('Paragraph 1, line 1.\nParagraph 1, line 2.\n\nParagraph 2, line 1.\nParagraph 2, line 2.\n', self.dedent('\n                <p>Paragraph 1, line 1.\n                Paragraph 1, line 2.</p>\n                <p>Paragraph 2, line 1.\n                Paragraph 2, line 2.</p>\n                '))

    def test_paragraphs_CR_LF(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('Paragraph 1, line 1.\r\nParagraph 1, line 2.\r\n\r\nParagraph 2, line 1.\r\nParagraph 2, line 2.\r\n', self.dedent('\n                <p>Paragraph 1, line 1.\n                Paragraph 1, line 2.</p>\n                <p>Paragraph 2, line 1.\n                Paragraph 2, line 2.</p>\n                '))