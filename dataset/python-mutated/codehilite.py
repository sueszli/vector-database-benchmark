"""
Adds code/syntax highlighting to standard Python-Markdown code blocks.

See the [documentation](https://Python-Markdown.github.io/extensions/code_hilite)
for details.
"""
from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any
if TYPE_CHECKING:
    import xml.etree.ElementTree as etree
try:
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import get_formatter_by_name
    from pygments.util import ClassNotFound
    pygments = True
except ImportError:
    pygments = False

def parse_hl_lines(expr: str) -> list[int]:
    if False:
        while True:
            i = 10
    "Support our syntax for emphasizing certain lines of code.\n\n    `expr` should be like '1 2' to emphasize lines 1 and 2 of a code block.\n    Returns a list of integers, the line numbers to emphasize.\n    "
    if not expr:
        return []
    try:
        return list(map(int, expr.split()))
    except ValueError:
        return []

class CodeHilite:
    """
    Determine language of source code, and pass it on to the Pygments highlighter.

    Usage:

    ```python
    code = CodeHilite(src=some_code, lang='python')
    html = code.hilite()
    ```

    Arguments:
        src: Source string or any object with a `.readline` attribute.

    Keyword arguments:
        lang (str): String name of Pygments lexer to use for highlighting. Default: `None`.
        guess_lang (bool): Auto-detect which lexer to use.
            Ignored if `lang` is set to a valid value. Default: `True`.
        use_pygments (bool): Pass code to Pygments for code highlighting. If `False`, the code is
            instead wrapped for highlighting by a JavaScript library. Default: `True`.
        pygments_formatter (str): The name of a Pygments formatter or a formatter class used for
            highlighting the code blocks. Default: `html`.
        linenums (bool): An alias to Pygments `linenos` formatter option. Default: `None`.
        css_class (str): An alias to Pygments `cssclass` formatter option. Default: 'codehilite'.
        lang_prefix (str): Prefix prepended to the language. Default: "language-".

    Other Options:

    Any other options are accepted and passed on to the lexer and formatter. Therefore,
    valid options include any options which are accepted by the `html` formatter or
    whichever lexer the code's language uses. Note that most lexers do not have any
    options. However, a few have very useful options, such as PHP's `startinline` option.
    Any invalid options are ignored without error.

    * **Formatter options**: <https://pygments.org/docs/formatters/#HtmlFormatter>
    * **Lexer Options**: <https://pygments.org/docs/lexers/>

    Additionally, when Pygments is enabled, the code's language is passed to the
    formatter as an extra option `lang_str`, whose value being `{lang_prefix}{lang}`.
    This option has no effect to the Pygments' builtin formatters.

    Advanced Usage:

    ```python
    code = CodeHilite(
        src = some_code,
        lang = 'php',
        startinline = True,      # Lexer option. Snippet does not start with `<?php`.
        linenostart = 42,        # Formatter option. Snippet starts on line 42.
        hl_lines = [45, 49, 50], # Formatter option. Highlight lines 45, 49, and 50.
        linenos = 'inline'       # Formatter option. Avoid alignment problems.
    )
    html = code.hilite()
    ```

    """

    def __init__(self, src: str, **options):
        if False:
            i = 10
            return i + 15
        self.src = src
        self.lang: str | None = options.pop('lang', None)
        self.guess_lang: bool = options.pop('guess_lang', True)
        self.use_pygments: bool = options.pop('use_pygments', True)
        self.lang_prefix: str = options.pop('lang_prefix', 'language-')
        self.pygments_formatter: str | Callable = options.pop('pygments_formatter', 'html')
        if 'linenos' not in options:
            options['linenos'] = options.pop('linenums', None)
        if 'cssclass' not in options:
            options['cssclass'] = options.pop('css_class', 'codehilite')
        if 'wrapcode' not in options:
            options['wrapcode'] = True
        options['full'] = False
        self.options = options

    def hilite(self, shebang: bool=True) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pass code to the [Pygments](https://pygments.org/) highlighter with\n        optional line numbers. The output should then be styled with CSS to\n        your liking. No styles are applied by default - only styling hooks\n        (i.e.: `<span class="k">`).\n\n        returns : A string of html.\n\n        '
        self.src = self.src.strip('\n')
        if self.lang is None and shebang:
            self._parseHeader()
        if pygments and self.use_pygments:
            try:
                lexer = get_lexer_by_name(self.lang, **self.options)
            except ValueError:
                try:
                    if self.guess_lang:
                        lexer = guess_lexer(self.src, **self.options)
                    else:
                        lexer = get_lexer_by_name('text', **self.options)
                except ValueError:
                    lexer = get_lexer_by_name('text', **self.options)
            if not self.lang:
                self.lang = lexer.aliases[0]
            lang_str = f'{self.lang_prefix}{self.lang}'
            if isinstance(self.pygments_formatter, str):
                try:
                    formatter = get_formatter_by_name(self.pygments_formatter, **self.options)
                except ClassNotFound:
                    formatter = get_formatter_by_name('html', **self.options)
            else:
                formatter = self.pygments_formatter(lang_str=lang_str, **self.options)
            return highlight(self.src, lexer, formatter)
        else:
            txt = self.src.replace('&', '&amp;')
            txt = txt.replace('<', '&lt;')
            txt = txt.replace('>', '&gt;')
            txt = txt.replace('"', '&quot;')
            classes = []
            if self.lang:
                classes.append('{}{}'.format(self.lang_prefix, self.lang))
            if self.options['linenos']:
                classes.append('linenums')
            class_str = ''
            if classes:
                class_str = ' class="{}"'.format(' '.join(classes))
            return '<pre class="{}"><code{}>{}\n</code></pre>\n'.format(self.options['cssclass'], class_str, txt)

    def _parseHeader(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines language of a code block from shebang line and whether the\n        said line should be removed or left in place. If the shebang line\n        contains a path (even a single /) then it is assumed to be a real\n        shebang line and left alone. However, if no path is given\n        (e.i.: `#!python` or `:::python`) then it is assumed to be a mock shebang\n        for language identification of a code fragment and removed from the\n        code block prior to processing for code highlighting. When a mock\n        shebang (e.i: `#!python`) is found, line numbering is turned on. When\n        colons are found in place of a shebang (e.i.: `:::python`), line\n        numbering is left in the current state - off by default.\n\n        Also parses optional list of highlight lines, like:\n\n            :::python hl_lines="1 3"\n        '
        import re
        lines = self.src.split('\n')
        fl = lines.pop(0)
        c = re.compile('\n            (?:(?:^::+)|(?P<shebang>^[#]!)) # Shebang or 2 or more colons\n            (?P<path>(?:/\\w+)*[/ ])?        # Zero or 1 path\n            (?P<lang>[\\w#.+-]*)             # The language\n            \\s*                             # Arbitrary whitespace\n            # Optional highlight lines, single- or double-quote-delimited\n            (hl_lines=(?P<quot>"|\')(?P<hl_lines>.*?)(?P=quot))?\n            ', re.VERBOSE)
        m = c.search(fl)
        if m:
            try:
                self.lang = m.group('lang').lower()
            except IndexError:
                self.lang = None
            if m.group('path'):
                lines.insert(0, fl)
            if self.options['linenos'] is None and m.group('shebang'):
                self.options['linenos'] = True
            self.options['hl_lines'] = parse_hl_lines(m.group('hl_lines'))
        else:
            lines.insert(0, fl)
        self.src = '\n'.join(lines).strip('\n')

class HiliteTreeprocessor(Treeprocessor):
    """ Highlight source code in code blocks. """
    config: dict[str, Any]

    def code_unescape(self, text: str) -> str:
        if False:
            i = 10
            return i + 15
        'Unescape code.'
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        return text

    def run(self, root: etree.Element) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Find code blocks and store in `htmlStash`. '
        blocks = root.iter('pre')
        for block in blocks:
            if len(block) == 1 and block[0].tag == 'code':
                local_config = self.config.copy()
                text = block[0].text
                if text is None:
                    continue
                code = CodeHilite(self.code_unescape(text), tab_length=self.md.tab_length, style=local_config.pop('pygments_style', 'default'), **local_config)
                placeholder = self.md.htmlStash.store(code.hilite())
                block.clear()
                block.tag = 'p'
                block.text = placeholder

class CodeHiliteExtension(Extension):
    """ Add source code highlighting to markdown code blocks. """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.config = {'linenums': [None, 'Use lines numbers. True|table|inline=yes, False=no, None=auto. Default: `None`.'], 'guess_lang': [True, 'Automatic language detection - Default: `True`.'], 'css_class': ['codehilite', 'Set class name for wrapper <div> - Default: `codehilite`.'], 'pygments_style': ['default', 'Pygments HTML Formatter Style (Colorscheme). Default: `default`.'], 'noclasses': [False, 'Use inline styles instead of CSS classes - Default `False`.'], 'use_pygments': [True, 'Highlight code blocks with pygments. Disable if using a JavaScript library. Default: `True`.'], 'lang_prefix': ['language-', 'Prefix prepended to the language when `use_pygments` is false. Default: `language-`.'], 'pygments_formatter': ['html', 'Use a specific formatter for Pygments highlighting. Default: `html`.']}
        ' Default configuration options. '
        for (key, value) in kwargs.items():
            if key in self.config:
                self.setConfig(key, value)
            else:
                if isinstance(value, str):
                    try:
                        value = parseBoolValue(value, preserve_none=True)
                    except ValueError:
                        pass
                self.config[key] = [value, '']

    def extendMarkdown(self, md):
        if False:
            for i in range(10):
                print('nop')
        ' Add `HilitePostprocessor` to Markdown instance. '
        hiliter = HiliteTreeprocessor(md)
        hiliter.config = self.getConfigs()
        md.treeprocessors.register(hiliter, 'hilite', 30)
        md.registerExtension(self)

def makeExtension(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    return CodeHiliteExtension(**kwargs)