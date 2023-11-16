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

class TestAttrList(TestCase):
    maxDiff = None

    def test_empty_list(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('*foo*{ }', '<p><em>foo</em>{ }</p>', extensions=['attr_list'])

    def test_table_td(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                | A { .foo }  | *B*{ .foo } | C { } | D{ .foo }     | E { .foo } F |\n                |-------------|-------------|-------|---------------|--------------|\n                | a { .foo }  | *b*{ .foo } | c { } | d{ .foo }     | e { .foo } f |\n                | valid on td | inline      | empty | missing space | not at end   |\n                '), self.dedent('\n                <table>\n                <thead>\n                <tr>\n                <th class="foo">A</th>\n                <th><em class="foo">B</em></th>\n                <th>C { }</th>\n                <th>D{ .foo }</th>\n                <th>E { .foo } F</th>\n                </tr>\n                </thead>\n                <tbody>\n                <tr>\n                <td class="foo">a</td>\n                <td><em class="foo">b</em></td>\n                <td>c { }</td>\n                <td>d{ .foo }</td>\n                <td>e { .foo } f</td>\n                </tr>\n                <tr>\n                <td>valid on td</td>\n                <td>inline</td>\n                <td>empty</td>\n                <td>missing space</td>\n                <td>not at end</td>\n                </tr>\n                </tbody>\n                </table>\n                '), extensions=['attr_list', 'tables'])