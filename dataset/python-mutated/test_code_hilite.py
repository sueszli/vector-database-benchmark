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
from markdown.extensions.codehilite import CodeHiliteExtension, CodeHilite
from markdown import extensions, treeprocessors
import os
import xml.etree.ElementTree as etree
try:
    import pygments
    has_pygments = True
except ImportError:
    has_pygments = False
required_pygments_version = os.environ.get('PYGMENTS_VERSION', '')

class TestCodeHiliteClass(TestCase):
    """ Test the markdown.extensions.codehilite.CodeHilite class. """

    def setUp(self):
        if False:
            while True:
                i = 10
        if has_pygments and pygments.__version__ != required_pygments_version:
            self.skipTest(f'Pygments=={required_pygments_version} is required')
    maxDiff = None

    def assertOutputEquals(self, source, expected, **options):
        if False:
            while True:
                i = 10
        '\n        Test that source code block results in the expected output with given options.\n        '
        output = CodeHilite(source, **options).hilite()
        self.assertMultiLineEqual(output.strip(), expected)

    def test_codehilite_defaults(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code># A Code Comment\n</code></pre>'
        self.assertOutputEquals('# A Code Comment', expected)

    def test_codehilite_guess_lang(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="cp">&lt;?php</span> <span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span><span class="p">);</span> <span class="cp">?&gt;</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code>&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, guess_lang=True)

    def test_codehilite_guess_lang_plain_text(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="err">plain text</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code>plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, guess_lang=True)

    def test_codehilite_set_lang(self):
        if False:
            return 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="cp">&lt;?php</span> <span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span><span class="p">);</span> <span class="cp">?&gt;</span><span class="x"></span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='php')

    def test_codehilite_bad_lang(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="cp">&lt;?php</span> <span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span><span class="p">);</span> <span class="cp">?&gt;</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-unkown">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='unkown')

    def test_codehilite_use_pygments_false(self):
        if False:
            while True:
                i = 10
        expected = '<pre class="codehilite"><code class="language-php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='php', use_pygments=False)

    def test_codehilite_lang_prefix_empty(self):
        if False:
            for i in range(10):
                print('nop')
        expected = '<pre class="codehilite"><code class="php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='php', use_pygments=False, lang_prefix='')

    def test_codehilite_lang_prefix(self):
        if False:
            i = 10
            return i + 15
        expected = '<pre class="codehilite"><code class="lang-php">&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertOutputEquals('<?php print("Hello World"); ?>', expected, lang='php', use_pygments=False, lang_prefix='lang-')

    def test_codehilite_linenos_true(self):
        if False:
            return 10
        if has_pygments:
            expected = '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div></td><td class="code"><div class="codehilite"><pre><span></span><code>plain text\n</code></pre></div>\n</td></tr></table>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenos=True)

    def test_codehilite_linenos_false(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code>plain text\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenos=False)

    def test_codehilite_linenos_none(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code>plain text\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenos=None)

    def test_codehilite_linenos_table(self):
        if False:
            return 10
        if has_pygments:
            expected = '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div></td><td class="code"><div class="codehilite"><pre><span></span><code>plain text\n</code></pre></div>\n</td></tr></table>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenos='table')

    def test_codehilite_linenos_inline(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="linenos">1</span>plain text\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenos='inline')

    def test_codehilite_linenums_true(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div></td><td class="code"><div class="codehilite"><pre><span></span><code>plain text\n</code></pre></div>\n</td></tr></table>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenums=True)

    def test_codehilite_set_cssclass(self):
        if False:
            return 10
        if has_pygments:
            expected = '<div class="override"><pre><span></span><code>plain text\n</code></pre></div>'
        else:
            expected = '<pre class="override"><code class="language-text">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', cssclass='override')

    def test_codehilite_set_css_class(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="override"><pre><span></span><code>plain text\n</code></pre></div>'
        else:
            expected = '<pre class="override"><code class="language-text">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', css_class='override')

    def test_codehilite_linenostart(self):
        if False:
            return 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="linenos">42</span>plain text\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">plain text\n</code></pre>'
        self.assertOutputEquals('plain text', expected, lang='text', linenos='inline', linenostart=42)

    def test_codehilite_linenos_hl_lines(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="linenos">1</span><span class="hll">line 1\n</span><span class="linenos">2</span>line 2\n<span class="linenos">3</span><span class="hll">line 3\n</span></code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">line 1\nline 2\nline 3\n</code></pre>'
        self.assertOutputEquals('line 1\nline 2\nline 3', expected, lang='text', linenos='inline', hl_lines=[1, 3])

    def test_codehilite_linenos_linenostep(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="linenos"> </span>line 1\n<span class="linenos">2</span>line 2\n<span class="linenos"> </span>line 3\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">line 1\nline 2\nline 3\n</code></pre>'
        self.assertOutputEquals('line 1\nline 2\nline 3', expected, lang='text', linenos='inline', linenostep=2)

    def test_codehilite_linenos_linenospecial(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="linenos">1</span>line 1\n<span class="linenos special">2</span>line 2\n<span class="linenos">3</span>line 3\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text linenums">line 1\nline 2\nline 3\n</code></pre>'
        self.assertOutputEquals('line 1\nline 2\nline 3', expected, lang='text', linenos='inline', linenospecial=2)

    def test_codehilite_startinline(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span><span class="p">);</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-php">print(&quot;Hello World&quot;);\n</code></pre>'
        self.assertOutputEquals('print("Hello World");', expected, lang='php', startinline=True)

class TestCodeHiliteExtension(TestCase):
    """ Test codehilite extension. """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if has_pygments and pygments.__version__ != required_pygments_version:
            self.skipTest(f'Pygments=={required_pygments_version} is required')
        if has_pygments:

            class CustomAddLangHtmlFormatter(pygments.formatters.HtmlFormatter):

                def __init__(self, lang_str='', **options):
                    if False:
                        print('Hello World!')
                    super().__init__(**options)
                    self.lang_str = lang_str

                def _wrap_code(self, source):
                    if False:
                        return 10
                    yield (0, f'<code class="{self.lang_str}">')
                    yield from source
                    yield (0, '</code>')
        else:
            CustomAddLangHtmlFormatter = None
        self.custom_pygments_formatter = CustomAddLangHtmlFormatter
    maxDiff = None

    def testBasicCodeHilite(self):
        if False:
            return 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t# A Code Comment', expected, extensions=['codehilite'])

    def testLinenumsTrue(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div></td><td class="code"><div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n</code></pre></div>\n</td></tr></table>'
        else:
            expected = '<pre class="codehilite"><code class="linenums"># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t# A Code Comment', expected, extensions=[CodeHiliteExtension(linenums=True)])

    def testLinenumsFalse(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="c1"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-python"># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t#!Python\n\t# A Code Comment', expected, extensions=[CodeHiliteExtension(linenums=False)])

    def testLinenumsNone(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t# A Code Comment', expected, extensions=[CodeHiliteExtension(linenums=None)])

    def testLinenumsNoneWithShebang(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div></td><td class="code"><div class="codehilite"><pre><span></span><code><span class="c1"># A Code Comment</span>\n</code></pre></div>\n</td></tr></table>'
        else:
            expected = '<pre class="codehilite"><code class="language-python linenums"># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t#!Python\n\t# A Code Comment', expected, extensions=[CodeHiliteExtension(linenums=None)])

    def testLinenumsNoneWithColon(self):
        if False:
            return 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="c1"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-python"># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t:::Python\n\t# A Code Comment', expected, extensions=[CodeHiliteExtension(linenums=None)])

    def testHighlightLinesWithColon(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="hll"><span class="c1">#line 1</span>\n</span><span class="c1">#line 2</span>\n<span class="c1">#line 3</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-python">#line 1\n#line 2\n#line 3\n</code></pre>'
        self.assertMarkdownRenders('\t:::Python hl_lines="1"\n\t#line 1\n\t#line 2\n\t#line 3', expected, extensions=['codehilite'])
        self.assertMarkdownRenders("\t:::Python hl_lines='1'\n\t#line 1\n\t#line 2\n\t#line 3", expected, extensions=['codehilite'])

    def testUsePygmentsFalse(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders('\t:::Python\n\t# A Code Comment', '<pre class="codehilite"><code class="language-python"># A Code Comment\n</code></pre>', extensions=[CodeHiliteExtension(use_pygments=False)])

    def testLangPrefixEmpty(self):
        if False:
            return 10
        self.assertMarkdownRenders('\t:::Python\n\t# A Code Comment', '<pre class="codehilite"><code class="python"># A Code Comment\n</code></pre>', extensions=[CodeHiliteExtension(use_pygments=False, lang_prefix='')])

    def testLangPrefix(self):
        if False:
            return 10
        self.assertMarkdownRenders('\t:::Python\n\t# A Code Comment', '<pre class="codehilite"><code class="lang-python"># A Code Comment\n</code></pre>', extensions=[CodeHiliteExtension(use_pygments=False, lang_prefix='lang-')])

    def testDoubleEscape(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="p">&lt;</span><span class="nt">span</span><span class="p">&gt;</span>This<span class="ni">&amp;amp;</span>That<span class="p">&lt;/</span><span class="nt">span</span><span class="p">&gt;</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-html">&lt;span&gt;This&amp;amp;That&lt;/span&gt;\n</code></pre>'
        self.assertMarkdownRenders('\t:::html\n\t<span>This&amp;That</span>', expected, extensions=['codehilite'])

    def testEntitiesIntact(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code>&lt; &amp;lt; and &gt; &amp;gt;\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text">&lt; &amp;lt; and &gt; &amp;gt;\n</code></pre>'
        self.assertMarkdownRenders('\t:::text\n\t< &lt; and > &gt;', expected, extensions=['codehilite'])

    def testHighlightAmps(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code>&amp;\n&amp;amp;\n&amp;amp;amp;\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-text">&amp;\n&amp;amp;\n&amp;amp;amp;\n</code></pre>'
        self.assertMarkdownRenders('\t:::text\n\t&\n\t&amp;\n\t&amp;amp;', expected, extensions=['codehilite'])

    def testUnknownOption(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="err"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t# A Code Comment', expected, extensions=[CodeHiliteExtension(unknown='some value')])

    def testMultipleBlocksSameStyle(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite" style="background: #202020"><pre style="line-height: 125%; margin: 0;"><span></span><code><span style="color: #999999; font-style: italic"># First Code Block</span>\n</code></pre></div>\n\n<p>Normal paragraph</p>\n<div class="codehilite" style="background: #202020"><pre style="line-height: 125%; margin: 0;"><span></span><code><span style="color: #999999; font-style: italic"># Second Code Block</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-python"># First Code Block\n</code></pre>\n\n<p>Normal paragraph</p>\n<pre class="codehilite"><code class="language-python"># Second Code Block\n</code></pre>'
        self.assertMarkdownRenders('\t:::Python\n\t# First Code Block\n\nNormal paragraph\n\n\t:::Python\n\t# Second Code Block', expected, extensions=[CodeHiliteExtension(pygments_style='native', noclasses=True)])

    def testFormatterLangStr(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code class="language-python"><span class="c1"># A Code Comment</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-python"># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t:::Python\n\t# A Code Comment', expected, extensions=[CodeHiliteExtension(guess_lang=False, pygments_formatter=self.custom_pygments_formatter)])

    def testFormatterLangStrGuessLang(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code class="language-js+php"><span class="cp">&lt;?php</span> <span class="k">print</span><span class="p">(</span><span class="s2">&quot;Hello World&quot;</span><span class="p">);</span> <span class="cp">?&gt;</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code>&lt;?php print(&quot;Hello World&quot;); ?&gt;\n</code></pre>'
        self.assertMarkdownRenders('\t<?php print("Hello World"); ?>', expected, extensions=[CodeHiliteExtension(pygments_formatter=self.custom_pygments_formatter)])

    def testFormatterLangStrEmptyLang(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code class="language-text"># A Code Comment\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code># A Code Comment\n</code></pre>'
        self.assertMarkdownRenders('\t# A Code Comment', expected, extensions=[CodeHiliteExtension(guess_lang=False, pygments_formatter=self.custom_pygments_formatter)])

    def testDoesntCrashWithEmptyCodeTag(self):
        if False:
            print('Hello World!')
        expected = '<h1>Hello</h1>\n<pre><code></code></pre>'
        self.assertMarkdownRenders('# Hello', expected, extensions=[CodeHiliteExtension(), _ExtensionThatAddsAnEmptyCodeTag()])

class _ExtensionThatAddsAnEmptyCodeTag(extensions.Extension):

    def extendMarkdown(self, md):
        if False:
            for i in range(10):
                print('nop')
        md.treeprocessors.register(_AddCodeTagTreeprocessor(), 'add-code-tag', 40)

class _AddCodeTagTreeprocessor(treeprocessors.Treeprocessor):

    def run(self, root: etree.Element):
        if False:
            while True:
                i = 10
        pre = etree.SubElement(root, 'pre')
        etree.SubElement(pre, 'code')