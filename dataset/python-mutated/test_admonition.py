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

Copyright 2007-2019 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""
from markdown.test_tools import TestCase

class TestAdmonition(TestCase):

    def test_with_lists(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                - List\n\n                    !!! note "Admontion"\n\n                        - Paragraph\n\n                            Paragraph\n                '), self.dedent('\n                <ul>\n                <li>\n                <p>List</p>\n                <div class="admonition note">\n                <p class="admonition-title">Admontion</p>\n                <ul>\n                <li>\n                <p>Paragraph</p>\n                <p>Paragraph</p>\n                </li>\n                </ul>\n                </div>\n                </li>\n                </ul>\n                '), extensions=['admonition'])

    def test_with_big_lists(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                - List\n\n                    !!! note "Admontion"\n\n                        - Paragraph\n\n                            Paragraph\n\n                        - Paragraph\n\n                            paragraph\n                '), self.dedent('\n                <ul>\n                <li>\n                <p>List</p>\n                <div class="admonition note">\n                <p class="admonition-title">Admontion</p>\n                <ul>\n                <li>\n                <p>Paragraph</p>\n                <p>Paragraph</p>\n                </li>\n                <li>\n                <p>Paragraph</p>\n                <p>paragraph</p>\n                </li>\n                </ul>\n                </div>\n                </li>\n                </ul>\n                '), extensions=['admonition'])

    def test_with_complex_lists(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                - List\n\n                    !!! note "Admontion"\n\n                        - Paragraph\n\n                            !!! note "Admontion"\n\n                                1. Paragraph\n\n                                    Paragraph\n                '), self.dedent('\n                <ul>\n                <li>\n                <p>List</p>\n                <div class="admonition note">\n                <p class="admonition-title">Admontion</p>\n                <ul>\n                <li>\n                <p>Paragraph</p>\n                <div class="admonition note">\n                <p class="admonition-title">Admontion</p>\n                <ol>\n                <li>\n                <p>Paragraph</p>\n                <p>Paragraph</p>\n                </li>\n                </ol>\n                </div>\n                </li>\n                </ul>\n                </div>\n                </li>\n                </ul>\n                '), extensions=['admonition'])

    def test_definition_list(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                - List\n\n                    !!! note "Admontion"\n\n                        Term\n\n                        :   Definition\n\n                            More text\n\n                        :   Another\n                            definition\n\n                            Even more text\n                '), self.dedent('\n                <ul>\n                <li>\n                <p>List</p>\n                <div class="admonition note">\n                <p class="admonition-title">Admontion</p>\n                <dl>\n                <dt>Term</dt>\n                <dd>\n                <p>Definition</p>\n                <p>More text</p>\n                </dd>\n                <dd>\n                <p>Another\n                definition</p>\n                <p>Even more text</p>\n                </dd>\n                </dl>\n                </div>\n                </li>\n                </ul>\n                '), extensions=['admonition', 'def_list'])

    def test_with_preceding_text(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                foo\n                **foo**\n                !!! note "Admonition"\n                '), self.dedent('\n                <p>foo\n                <strong>foo</strong></p>\n                <div class="admonition note">\n                <p class="admonition-title">Admonition</p>\n                </div>\n                '), extensions=['admonition'])

    def test_admontion_detabbing(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                !!! note "Admonition"\n                    - Parent 1\n\n                        - Child 1\n                        - Child 2\n                '), self.dedent('\n                <div class="admonition note">\n                <p class="admonition-title">Admonition</p>\n                <ul>\n                <li>\n                <p>Parent 1</p>\n                <ul>\n                <li>Child 1</li>\n                <li>Child 2</li>\n                </ul>\n                </li>\n                </ul>\n                </div>\n                '), extensions=['admonition'])

    def test_admonition_first_indented(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                !!! danger "This is not"\n                        one long admonition title\n                '), self.dedent('\n                <div class="admonition danger">\n                <p class="admonition-title">This is not</p>\n                <pre><code>one long admonition title\n                </code></pre>\n                </div>\n                '), extensions=['admonition'])