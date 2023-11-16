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
import markdown
import markdown.extensions.codehilite
import os
try:
    import pygments
    import pygments.formatters
    has_pygments = True
except ImportError:
    has_pygments = False
required_pygments_version = os.environ.get('PYGMENTS_VERSION', '')

class TestFencedCode(TestCase):

    def testBasicFence(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                A paragraph before a fenced code block:\n\n                ```\n                Fenced code block\n                ```\n                '), self.dedent('\n                <p>A paragraph before a fenced code block:</p>\n                <pre><code>Fenced code block\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testNestedFence(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ````\n\n                ```\n                ````\n                '), self.dedent('\n                <pre><code>\n                ```\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedTildes(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                ~~~\n                # Arbitrary code\n                ``` # these backticks will not close the block\n                ~~~\n                '), self.dedent('\n                <pre><code># Arbitrary code\n                ``` # these backticks will not close the block\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedLanguageNoDot(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ``` python\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedLanguageWithDot(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ``` .python\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def test_fenced_code_in_raw_html(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                <details>\n                ```\n                Begone placeholders!\n                ```\n                </details>\n                '), self.dedent('\n                <details>\n\n                <pre><code>Begone placeholders!\n                </code></pre>\n\n                </details>\n                '), extensions=['fenced_code'])

    def testFencedLanguageInAttr(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ``` {.python}\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedMultipleClassesInAttr(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                ``` {.python .foo .bar}\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre class="foo bar"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedIdInAttr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ``` { #foo }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo"><code># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedIdAndLangInAttr(self):
        if False:
            print('Hello World!')
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python #foo }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedIdAndLangAndClassInAttr(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python #foo .bar }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo" class="bar"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedLanguageIdAndPygmentsDisabledInAttrNoCodehilite(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python #foo use_pygments=False }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedLanguageIdAndPygmentsEnabledInAttrNoCodehilite(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python #foo use_pygments=True }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code'])

    def testFencedLanguageNoCodehiliteWithAttrList(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python foo=bar }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python" foo="bar"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code', 'attr_list'])

    def testFencedLanguagePygmentsDisabledInAttrNoCodehiliteWithAttrList(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python foo=bar use_pygments=False }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python" foo="bar"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code', 'attr_list'])

    def testFencedLanguagePygmentsEnabledInAttrNoCodehiliteWithAttrList(self):
        if False:
            i = 10
            return i + 15
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python foo=bar use_pygments=True }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code', 'attr_list'])

    def testFencedLanguageNoPrefix(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ``` python\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="python"># Some python code\n                </code></pre>\n                '), extensions=[markdown.extensions.fenced_code.FencedCodeExtension(lang_prefix='')])

    def testFencedLanguageAltPrefix(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ``` python\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="lang-python"># Some python code\n                </code></pre>\n                '), extensions=[markdown.extensions.fenced_code.FencedCodeExtension(lang_prefix='lang-')])

    def testFencedCodeEscapedAttrs(self):
        if False:
            while True:
                i = 10
        self.assertMarkdownRenders(self.dedent('\n                ``` { ."weird #"foo bar=">baz }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="&quot;foo"><code class="language-&quot;weird" bar="&quot;&gt;baz"># Some python code\n                </code></pre>\n                '), extensions=['fenced_code', 'attr_list'])

class TestFencedCodeWithCodehilite(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if has_pygments and pygments.__version__ != required_pygments_version:
            self.skipTest(f'Pygments=={required_pygments_version} is required')

    def test_shebang(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '\n            <div class="codehilite"><pre><span></span><code>#!test\n            </code></pre></div>\n            '
        else:
            expected = '\n            <pre class="codehilite"><code>#!test\n            </code></pre>\n            '
        self.assertMarkdownRenders(self.dedent('\n                ```\n                #!test\n                ```\n                '), self.dedent(expected), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False), 'fenced_code'])

    def testFencedCodeWithHighlightLines(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = self.dedent('\n                <div class="codehilite"><pre><span></span><code><span class="hll">line 1\n                </span>line 2\n                <span class="hll">line 3\n                </span></code></pre></div>\n                ')
        else:
            expected = self.dedent('\n                    <pre class="codehilite"><code>line 1\n                    line 2\n                    line 3\n                    </code></pre>\n                    ')
        self.assertMarkdownRenders(self.dedent('\n                ```hl_lines="1 3"\n                line 1\n                line 2\n                line 3\n                ```\n                '), expected, extensions=[markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False), 'fenced_code'])

    def testFencedLanguageAndHighlightLines(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="hll"><span class="n">line</span> <span class="mi">1</span>\n</span><span class="n">line</span> <span class="mi">2</span>\n<span class="hll"><span class="n">line</span> <span class="mi">3</span>\n</span></code></pre></div>'
        else:
            expected = self.dedent('\n                    <pre class="codehilite"><code class="language-python">line 1\n                    line 2\n                    line 3\n                    </code></pre>\n                    ')
        self.assertMarkdownRenders(self.dedent('\n                ``` .python hl_lines="1 3"\n                line 1\n                line 2\n                line 3\n                ```\n                '), expected, extensions=[markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False), 'fenced_code'])

    def testFencedLanguageAndPygmentsDisabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ``` .python\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(use_pygments=False), 'fenced_code'])

    def testFencedLanguageDoubleEscape(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="p">&lt;</span><span class="nt">span</span><span class="p">&gt;</span>This<span class="ni">&amp;amp;</span>That<span class="p">&lt;/</span><span class="nt">span</span><span class="p">&gt;</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-html">&lt;span&gt;This&amp;amp;That&lt;/span&gt;\n</code></pre>'
        self.assertMarkdownRenders(self.dedent('\n                ```html\n                <span>This&amp;That</span>\n                ```\n                '), expected, extensions=[markdown.extensions.codehilite.CodeHiliteExtension(), 'fenced_code'])

    def testFencedAmps(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = self.dedent('\n                <div class="codehilite"><pre><span></span><code>&amp;\n                &amp;amp;\n                &amp;amp;amp;\n                </code></pre></div>\n                ')
        else:
            expected = self.dedent('\n                <pre class="codehilite"><code class="language-text">&amp;\n                &amp;amp;\n                &amp;amp;amp;\n                </code></pre>\n                ')
        self.assertMarkdownRenders(self.dedent('\n                ```text\n                &\n                &amp;\n                &amp;amp;\n                ```\n                '), expected, extensions=[markdown.extensions.codehilite.CodeHiliteExtension(), 'fenced_code'])

    def testFencedCodeWithHighlightLinesInAttr(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = self.dedent('\n                <div class="codehilite"><pre><span></span><code><span class="hll">line 1\n                </span>line 2\n                <span class="hll">line 3\n                </span></code></pre></div>\n                ')
        else:
            expected = self.dedent('\n                    <pre class="codehilite"><code>line 1\n                    line 2\n                    line 3\n                    </code></pre>\n                    ')
        self.assertMarkdownRenders(self.dedent('\n                ```{ hl_lines="1 3" }\n                line 1\n                line 2\n                line 3\n                ```\n                '), expected, extensions=[markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False), 'fenced_code'])

    def testFencedLanguageAndHighlightLinesInAttr(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<div class="codehilite"><pre><span></span><code><span class="hll"><span class="n">line</span> <span class="mi">1</span>\n</span><span class="n">line</span> <span class="mi">2</span>\n<span class="hll"><span class="n">line</span> <span class="mi">3</span>\n</span></code></pre></div>'
        else:
            expected = self.dedent('\n                    <pre class="codehilite"><code class="language-python">line 1\n                    line 2\n                    line 3\n                    </code></pre>\n                    ')
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python hl_lines="1 3" }\n                line 1\n                line 2\n                line 3\n                ```\n                '), expected, extensions=[markdown.extensions.codehilite.CodeHiliteExtension(linenums=None, guess_lang=False), 'fenced_code'])

    def testFencedLanguageIdInAttrAndPygmentsDisabled(self):
        if False:
            return 10
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python #foo }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(use_pygments=False), 'fenced_code'])

    def testFencedLanguageIdAndPygmentsDisabledInAttr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python #foo use_pygments=False }\n                # Some python code\n                ```\n                '), self.dedent('\n                <pre id="foo"><code class="language-python"># Some python code\n                </code></pre>\n                '), extensions=['codehilite', 'fenced_code'])

    def testFencedLanguageAttrCssclass(self):
        if False:
            i = 10
            return i + 15
        if has_pygments:
            expected = self.dedent('\n                <div class="pygments"><pre><span></span><code><span class="c1"># Some python code</span>\n                </code></pre></div>\n                ')
        else:
            expected = '<pre class="pygments"><code class="language-python"># Some python code\n</code></pre>'
        self.assertMarkdownRenders(self.dedent("\n                ``` { .python css_class='pygments' }\n                # Some python code\n                ```\n                "), expected, extensions=['codehilite', 'fenced_code'])

    def testFencedLanguageAttrLinenums(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<table class="codehilitetable"><tr><td class="linenos"><div class="linenodiv"><pre>1</pre></div></td><td class="code"><div class="codehilite"><pre><span></span><code><span class="c1"># Some python code</span>\n</code></pre></div>\n</td></tr></table>'
        else:
            expected = '<pre class="codehilite"><code class="language-python linenums"># Some python code\n</code></pre>'
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python linenums=True }\n                # Some python code\n                ```\n                '), expected, extensions=['codehilite', 'fenced_code'])

    def testFencedLanguageAttrGuesslang(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = self.dedent('\n                <div class="codehilite"><pre><span></span><code># Some python code\n                </code></pre></div>\n                ')
        else:
            expected = '<pre class="codehilite"><code># Some python code\n</code></pre>'
        self.assertMarkdownRenders(self.dedent('\n                ``` { guess_lang=False }\n                # Some python code\n                ```\n                '), expected, extensions=['codehilite', 'fenced_code'])

    def testFencedLanguageAttrNoclasses(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<div class="codehilite" style="background: #f8f8f8"><pre style="line-height: 125%; margin: 0;"><span></span><code><span style="color: #408080; font-style: italic"># Some python code</span>\n</code></pre></div>'
        else:
            expected = '<pre class="codehilite"><code class="language-python"># Some python code\n</code></pre>'
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python noclasses=True }\n                # Some python code\n                ```\n                '), expected, extensions=['codehilite', 'fenced_code'])

    def testFencedMultipleBlocksSameStyle(self):
        if False:
            print('Hello World!')
        if has_pygments:
            expected = '<div class="codehilite" style="background: #202020"><pre style="line-height: 125%; margin: 0;"><span></span><code><span style="color: #999999; font-style: italic"># First Code Block</span>\n</code></pre></div>\n\n<p>Normal paragraph</p>\n<div class="codehilite" style="background: #202020"><pre style="line-height: 125%; margin: 0;"><span></span><code><span style="color: #999999; font-style: italic"># Second Code Block</span>\n</code></pre></div>'
        else:
            expected = '\n            <pre class="codehilite"><code class="language-python"># First Code Block\n            </code></pre>\n\n            <p>Normal paragraph</p>\n            <pre class="codehilite"><code class="language-python"># Second Code Block\n            </code></pre>\n            '
        self.assertMarkdownRenders(self.dedent('\n                ``` { .python }\n                # First Code Block\n                ```\n\n                Normal paragraph\n\n                ``` { .python }\n                # Second Code Block\n                ```\n                '), self.dedent(expected), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(pygments_style='native', noclasses=True), 'fenced_code'])

    def testCustomPygmentsFormatter(self):
        if False:
            return 10
        if has_pygments:

            class CustomFormatter(pygments.formatters.HtmlFormatter):

                def wrap(self, source, outfile):
                    if False:
                        while True:
                            i = 10
                    return self._wrap_div(self._wrap_code(source))

                def _wrap_code(self, source):
                    if False:
                        print('Hello World!')
                    yield (0, '<code>')
                    for (i, t) in source:
                        if i == 1:
                            t += '<br>'
                        yield (i, t)
                    yield (0, '</code>')
            expected = '\n            <div class="codehilite"><code>hello world\n            <br>hello another world\n            <br></code></div>\n            '
        else:
            CustomFormatter = None
            expected = '\n            <pre class="codehilite"><code>hello world\n            hello another world\n            </code></pre>\n            '
        self.assertMarkdownRenders(self.dedent('\n                ```\n                hello world\n                hello another world\n                ```\n                '), self.dedent(expected), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(pygments_formatter=CustomFormatter, guess_lang=False), 'fenced_code'])

    def testPygmentsAddLangClassFormatter(self):
        if False:
            while True:
                i = 10
        if has_pygments:

            class CustomAddLangHtmlFormatter(pygments.formatters.HtmlFormatter):

                def __init__(self, lang_str='', **options):
                    if False:
                        print('Hello World!')
                    super().__init__(**options)
                    self.lang_str = lang_str

                def _wrap_code(self, source):
                    if False:
                        print('Hello World!')
                    yield (0, f'<code class="{self.lang_str}">')
                    yield from source
                    yield (0, '</code>')
            expected = '\n                <div class="codehilite"><pre><span></span><code class="language-text">hello world\n                hello another world\n                </code></pre></div>\n                '
        else:
            CustomAddLangHtmlFormatter = None
            expected = '\n                <pre class="codehilite"><code class="language-text">hello world\n                hello another world\n                </code></pre>\n                '
        self.assertMarkdownRenders(self.dedent('\n                ```text\n                hello world\n                hello another world\n                ```\n                '), self.dedent(expected), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(guess_lang=False, pygments_formatter=CustomAddLangHtmlFormatter), 'fenced_code'])

    def testSvgCustomPygmentsFormatter(self):
        if False:
            for i in range(10):
                print('nop')
        if has_pygments:
            expected = '\n            <?xml version="1.0"?>\n            <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.0//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">\n            <svg xmlns="http://www.w3.org/2000/svg">\n            <g font-family="monospace" font-size="14px">\n            <text x="0" y="14" xml:space="preserve">hello&#160;world</text>\n            <text x="0" y="33" xml:space="preserve">hello&#160;another&#160;world</text>\n            <text x="0" y="52" xml:space="preserve"></text></g></svg>\n            '
        else:
            expected = '\n            <pre class="codehilite"><code>hello world\n            hello another world\n            </code></pre>\n            '
        self.assertMarkdownRenders(self.dedent('\n                ```\n                hello world\n                hello another world\n                ```\n                '), self.dedent(expected), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(pygments_formatter='svg', linenos=False, guess_lang=False), 'fenced_code'])

    def testInvalidCustomPygmentsFormatter(self):
        if False:
            while True:
                i = 10
        if has_pygments:
            expected = '\n            <div class="codehilite"><pre><span></span><code>hello world\n            hello another world\n            </code></pre></div>\n            '
        else:
            expected = '\n            <pre class="codehilite"><code>hello world\n            hello another world\n            </code></pre>\n            '
        self.assertMarkdownRenders(self.dedent('\n                ```\n                hello world\n                hello another world\n                ```\n                '), self.dedent(expected), extensions=[markdown.extensions.codehilite.CodeHiliteExtension(pygments_formatter='invalid', guess_lang=False), 'fenced_code'])