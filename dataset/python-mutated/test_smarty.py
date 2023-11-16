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

Copyright 2007-2022 The Python Markdown Project (v. 1.7 and later)
Copyright 2004, 2005, 2006 Yuri Takhteyev (v. 0.2-1.6b)
Copyright 2004 Manfred Stienstra (the original version)

License: BSD (see LICENSE.md for details).
"""
from markdown.test_tools import TestCase

class TestSmarty(TestCase):
    default_kwargs = {'extensions': ['smarty']}

    def test_basic(self):
        if False:
            return 10
        self.assertMarkdownRenders("It's fun. What's fun?", '<p>It&rsquo;s fun. What&rsquo;s fun?</p>')
        self.assertMarkdownRenders('"Isn\'t this fun"? --- she said...', '<p>&ldquo;Isn&rsquo;t this fun&rdquo;? &mdash; she said&hellip;</p>')
        self.assertMarkdownRenders('"\'Quoted\' words in a larger quote."', '<p>&ldquo;&lsquo;Quoted&rsquo; words in a larger quote.&rdquo;</p>')
        self.assertMarkdownRenders('\'Quoted "words" in a larger quote.\'', '<p>&lsquo;Quoted &ldquo;words&rdquo; in a larger quote.&rsquo;</p>')
        self.assertMarkdownRenders('"quoted" text and **bold "quoted" text**', '<p>&ldquo;quoted&rdquo; text and <strong>bold &ldquo;quoted&rdquo; text</strong></p>')
        self.assertMarkdownRenders("'quoted' text and **bold 'quoted' text**", '<p>&lsquo;quoted&rsquo; text and <strong>bold &lsquo;quoted&rsquo; text</strong></p>')
        self.assertMarkdownRenders('em-dashes (---) and ellipes (...)', '<p>em-dashes (&mdash;) and ellipes (&hellip;)</p>')
        self.assertMarkdownRenders('"[Link](http://example.com)" --- she said.', '<p>&ldquo;<a href="http://example.com">Link</a>&rdquo; &mdash; she said.</p>')
        self.assertMarkdownRenders('"Ellipsis within quotes..."', '<p>&ldquo;Ellipsis within quotes&hellip;&rdquo;</p>')
        self.assertMarkdownRenders("*Custer*'s Last Stand", '<p><em>Custer</em>&rsquo;s Last Stand</p>')

    def test_years(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders("1440--80's", '<p>1440&ndash;80&rsquo;s</p>')
        self.assertMarkdownRenders("1440--'80s", '<p>1440&ndash;&rsquo;80s</p>')
        self.assertMarkdownRenders("1440---'80s", '<p>1440&mdash;&rsquo;80s</p>')
        self.assertMarkdownRenders("1960's", '<p>1960&rsquo;s</p>')
        self.assertMarkdownRenders("one two '60s", '<p>one two &rsquo;60s</p>')
        self.assertMarkdownRenders("'60s", '<p>&rsquo;60s</p>')

    def test_wrapping_line(self):
        if False:
            for i in range(10):
                print('nop')
        text = "A line that 'wraps' with\n*emphasis* at the beginning of the next line."
        html = '<p>A line that &lsquo;wraps&rsquo; with\n<em>emphasis</em> at the beginning of the next line.</p>'
        self.assertMarkdownRenders(text, html)

    def test_escaped(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('Escaped \\-- ndash', '<p>Escaped -- ndash</p>')
        self.assertMarkdownRenders('\\\'Escaped\\\' \\"quotes\\"', '<p>\'Escaped\' "quotes"</p>')
        self.assertMarkdownRenders('Escaped ellipsis\\...', '<p>Escaped ellipsis...</p>')
        self.assertMarkdownRenders('\'Escaped \\"quotes\\" in real ones\'', '<p>&lsquo;Escaped "quotes" in real ones&rsquo;</p>')
        self.assertMarkdownRenders('\\\'"Real" quotes in escaped ones\\\'', "<p>'&ldquo;Real&rdquo; quotes in escaped ones'</p>")

    def test_escaped_attr(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('![x"x](x)', '<p><img alt="x&quot;x" src="x" /></p>')

    def test_code_spans(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders('Skip `"code" -- --- \'spans\' ...`.', '<p>Skip <code>"code" -- --- \'spans\' ...</code>.</p>')

    def test_code_blocks(self):
        if False:
            return 10
        text = '    Also skip "code" \'blocks\'\n    foo -- bar --- baz ...'
        html = '<pre><code>Also skip "code" \'blocks\'\nfoo -- bar --- baz ...\n</code></pre>'
        self.assertMarkdownRenders(text, html)

    def test_horizontal_rule(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders('--- -- ---', '<hr />')

class TestSmartyAngledQuotes(TestCase):
    default_kwargs = {'extensions': ['smarty'], 'extension_configs': {'smarty': {'smart_angled_quotes': True}}}

    def test_angled_quotes(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders('<<hello>>', '<p>&laquo;hello&raquo;</p>')
        self.assertMarkdownRenders('Кавычки-<<ёлочки>>', '<p>Кавычки-&laquo;ёлочки&raquo;</p>')
        self.assertMarkdownRenders('Anführungszeichen->>Chevrons<<', '<p>Anführungszeichen-&raquo;Chevrons&laquo;</p>')

class TestSmartyCustomSubstitutions(TestCase):
    default_kwargs = {'extensions': ['smarty'], 'extension_configs': {'smarty': {'smart_angled_quotes': True, 'substitutions': {'ndash': '–', 'mdash': '—', 'ellipsis': '…', 'left-single-quote': '&sbquo;', 'right-single-quote': '&lsquo;', 'left-double-quote': '&bdquo;', 'right-double-quote': '&ldquo;', 'left-angle-quote': '[', 'right-angle-quote': ']'}}}}

    def test_custom_substitutions(self):
        if False:
            i = 10
            return i + 15
        text = '<< The "Unicode char of the year 2014"\nis the \'mdash\': ---\nMust not be confused with \'ndash\'  (--) ... >>'
        html = '<p>[ The &bdquo;Unicode char of the year 2014&ldquo;\nis the &sbquo;mdash&lsquo;: —\nMust not be confused with &sbquo;ndash&lsquo;  (–) … ]</p>'
        self.assertMarkdownRenders(text, html)