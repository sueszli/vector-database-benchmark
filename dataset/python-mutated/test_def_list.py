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

class TestDefList(TestCase):

    def test_def_list_with_ol(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n\n                term\n\n                :   this is a definition for term. it has\n                    multiple lines in the first paragraph.\n\n                    1.  first thing\n\n                        first thing details in a second paragraph.\n\n                    1.  second thing\n\n                        second thing details in a second paragraph.\n\n                    1.  third thing\n\n                        third thing details in a second paragraph.\n                '), self.dedent('\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>this is a definition for term. it has\n                multiple lines in the first paragraph.</p>\n                <ol>\n                <li>\n                <p>first thing</p>\n                <p>first thing details in a second paragraph.</p>\n                </li>\n                <li>\n                <p>second thing</p>\n                <p>second thing details in a second paragraph.</p>\n                </li>\n                <li>\n                <p>third thing</p>\n                <p>third thing details in a second paragraph.</p>\n                </li>\n                </ol>\n                </dd>\n                </dl>\n                '), extensions=['def_list'])

    def test_def_list_with_ul(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n\n                term\n\n                :   this is a definition for term. it has\n                    multiple lines in the first paragraph.\n\n                    -   first thing\n\n                        first thing details in a second paragraph.\n\n                    -   second thing\n\n                        second thing details in a second paragraph.\n\n                    -   third thing\n\n                        third thing details in a second paragraph.\n                '), self.dedent('\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>this is a definition for term. it has\n                multiple lines in the first paragraph.</p>\n                <ul>\n                <li>\n                <p>first thing</p>\n                <p>first thing details in a second paragraph.</p>\n                </li>\n                <li>\n                <p>second thing</p>\n                <p>second thing details in a second paragraph.</p>\n                </li>\n                <li>\n                <p>third thing</p>\n                <p>third thing details in a second paragraph.</p>\n                </li>\n                </ul>\n                </dd>\n                </dl>\n                '), extensions=['def_list'])

    def test_def_list_with_nesting(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n\n                term\n\n                :   this is a definition for term. it has\n                    multiple lines in the first paragraph.\n\n                    1.  first thing\n\n                        first thing details in a second paragraph.\n\n                        -   first nested thing\n\n                            second nested thing details\n                '), self.dedent('\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>this is a definition for term. it has\n                multiple lines in the first paragraph.</p>\n                <ol>\n                <li>\n                <p>first thing</p>\n                <p>first thing details in a second paragraph.</p>\n                <ul>\n                <li>\n                <p>first nested thing</p>\n                <p>second nested thing details</p>\n                </li>\n                </ul>\n                </li>\n                </ol>\n                </dd>\n                </dl>\n                '), extensions=['def_list'])

    def test_def_list_with_nesting_self(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n\n                term\n\n                :   this is a definition for term. it has\n                    multiple lines in the first paragraph.\n\n                    inception\n\n                    :   this is a definition for term. it has\n                        multiple lines in the first paragraph.\n\n                        - bullet point\n\n                          another paragraph\n                '), self.dedent('\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>this is a definition for term. it has\n                multiple lines in the first paragraph.</p>\n                <dl>\n                <dt>inception</dt>\n                <dd>\n                <p>this is a definition for term. it has\n                multiple lines in the first paragraph.</p>\n                <ul>\n                <li>bullet point</li>\n                </ul>\n                <p>another paragraph</p>\n                </dd>\n                </dl>\n                </dd>\n                </dl>\n                '), extensions=['def_list'])

    def test_def_list_unreasonable_nesting(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n\n                turducken\n\n                :   this is a definition for term. it has\n                    multiple lines in the first paragraph.\n\n                    1.  ordered list\n\n                        - nested list\n\n                            term\n\n                            :   definition\n\n                                -   item 1 paragraph 1\n\n                                    item 1 paragraph 2\n                '), self.dedent('\n                <dl>\n                <dt>turducken</dt>\n                <dd>\n                <p>this is a definition for term. it has\n                multiple lines in the first paragraph.</p>\n                <ol>\n                <li>\n                <p>ordered list</p>\n                <ul>\n                <li>\n                <p>nested list</p>\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>definition</p>\n                <ul>\n                <li>\n                <p>item 1 paragraph 1</p>\n                <p>item 1 paragraph 2</p>\n                </li>\n                </ul>\n                </dd>\n                </dl>\n                </li>\n                </ul>\n                </li>\n                </ol>\n                </dd>\n                </dl>\n                '), extensions=['def_list'])

    def test_def_list_nested_admontions(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                term\n\n                :   definition\n\n                    !!! note "Admontion"\n\n                        term\n\n                        :   definition\n\n                            1.  list\n\n                                continue\n                '), self.dedent('\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>definition</p>\n                <div class="admonition note">\n                <p class="admonition-title">Admontion</p>\n                <dl>\n                <dt>term</dt>\n                <dd>\n                <p>definition</p>\n                <ol>\n                <li>\n                <p>list</p>\n                <p>continue</p>\n                </li>\n                </ol>\n                </dd>\n                </dl>\n                </div>\n                </dd>\n                </dl>\n                '), extensions=['def_list', 'admonition'])