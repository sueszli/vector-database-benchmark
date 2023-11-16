"""
Fenced Code Extension for Python Markdown
=========================================

This extension adds Fenced Code Blocks to Python-Markdown.

    >>> import markdown
    >>> text = '''
    ... A paragraph before a fenced code block:
    ...
    ... ~~~
    ... Fenced code block
    ... ~~~
    ... '''
    >>> html = markdown.markdown(text, extensions=['fenced_code'])
    >>> print html
    <p>A paragraph before a fenced code block:</p>
    <pre><code>Fenced code block
    </code></pre>

Works with safe_mode also (we check this because we are using the HtmlStash):

    >>> print markdown.markdown(text, extensions=['fenced_code'], safe_mode='replace')
    <p>A paragraph before a fenced code block:</p>
    <pre><code>Fenced code block
    </code></pre>

Include tilde's in a code block and wrap with blank lines:

    >>> text = '''
    ... ~~~~~~~~
    ...
    ... ~~~~
    ... ~~~~~~~~'''
    >>> print markdown.markdown(text, extensions=['fenced_code'])
    <pre><code>
    ~~~~
    </code></pre>

Removes trailing whitespace from code blocks that cause horizontal scrolling
    >>> import markdown
    >>> text = '''
    ... A paragraph before a fenced code block:
    ...
    ... ~~~
    ... Fenced code block    							
    ... ~~~
    ... '''
    >>> html = markdown.markdown(text, extensions=['fenced_code'])
    >>> print html
    <p>A paragraph before a fenced code block:</p>
    <pre><code>Fenced code block
    </code></pre>

Language tags:

    >>> text = '''
    ... ~~~~{.python}
    ... # Some python code
    ... ~~~~'''
    >>> print markdown.markdown(text, extensions=['fenced_code'])
    <pre><code class="python"># Some python code
    </code></pre>

Copyright 2007-2008 [Waylan Limberg](http://achinghead.com/).

Project website: <http://packages.python.org/Markdown/extensions/fenced_code_blocks.html>
Contact: markdown@freewisdom.org

License: BSD (see ../docs/LICENSE for details)

Dependencies:
* [Python 2.4+](http://python.org)
* [Markdown 2.0+](http://packages.python.org/Markdown/)
* [Pygments (optional)](http://pygments.org)

"""
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence
import lxml.html
from django.utils.html import escape
from markdown import Markdown
from markdown.extensions import Extension, codehilite
from markdown.extensions.codehilite import CodeHiliteExtension, parse_hl_lines
from markdown.preprocessors import Preprocessor
from pygments.lexers import find_lexer_class_by_name
from pygments.util import ClassNotFound
from typing_extensions import override
from zerver.lib.exceptions import MarkdownRenderingError
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITES
from zerver.lib.tex import render_tex
FENCE_RE = re.compile('\n    # ~~~ or ```\n    (?P<fence>\n        ^(?:~{3,}|`{3,})\n    )\n\n    [ ]* # spaces\n\n    (?:\n        # language, like ".py" or "{javascript}"\n        \\{?\\.?\n        (?P<lang>\n            [a-zA-Z0-9_+-./#]+\n        ) # "py" or "javascript"\n\n        [ ]* # spaces\n\n        # header for features that use fenced block header syntax (like spoilers)\n        (?P<header>\n            [^ ~`][^~`]*\n        )?\n        \\}?\n    )?\n    $\n    ', re.VERBOSE)
CODE_WRAP = '<pre><code{}>{}\n</code></pre>'
LANG_TAG = ' class="{}"'

def validate_curl_content(lines: List[str]) -> None:
    if False:
        return 10
    error_msg = '\nMissing required -X argument in curl command:\n\n{command}\n'.strip()
    for line in lines:
        regex = 'curl [-](sS)?X "?(GET|DELETE|PATCH|POST)"?'
        if line.startswith('curl') and re.search(regex, line) is None:
            raise MarkdownRenderingError(error_msg.format(command=line.strip()))
CODE_VALIDATORS: Dict[Optional[str], Callable[[List[str]], None]] = {'curl': validate_curl_content}

class FencedCodeExtension(Extension):

    def __init__(self, config: Mapping[str, Any]={}) -> None:
        if False:
            return 10
        self.config = {'run_content_validators': [config.get('run_content_validators', False), 'Boolean specifying whether to run content validation code in CodeHandler']}
        for (key, value) in config.items():
            self.setConfig(key, value)

    @override
    def extendMarkdown(self, md: Markdown) -> None:
        if False:
            return 10
        'Add FencedBlockPreprocessor to the Markdown instance.'
        md.registerExtension(self)
        processor = FencedBlockPreprocessor(md, run_content_validators=self.config['run_content_validators'][0])
        md.preprocessors.register(processor, 'fenced_code_block', PREPROCESSOR_PRIORITES['fenced_code_block'])

class ZulipBaseHandler:

    def __init__(self, processor: 'FencedBlockPreprocessor', output: MutableSequence[str], fence: Optional[str]=None, process_contents: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.processor = processor
        self.output = output
        self.fence = fence
        self.process_contents = process_contents
        self.lines: List[str] = []

    def handle_line(self, line: str) -> None:
        if False:
            while True:
                i = 10
        if line.rstrip() == self.fence:
            self.done()
        else:
            self.lines.append(line.rstrip())

    def done(self) -> None:
        if False:
            print('Hello World!')
        if self.lines:
            text = '\n'.join(self.lines)
            text = self.format_text(text)
            if not self.process_contents:
                text = self.processor.placeholder(text)
            processed_lines = text.split('\n')
            self.output.append('')
            self.output.extend(processed_lines)
            self.output.append('')
        self.processor.pop()

    def format_text(self, text: str) -> str:
        if False:
            return 10
        'Returns a formatted text.\n        Subclasses should override this method.\n        '
        raise NotImplementedError

def generic_handler(processor: 'FencedBlockPreprocessor', output: MutableSequence[str], fence: str, lang: Optional[str], header: Optional[str], run_content_validators: bool=False, default_language: Optional[str]=None) -> ZulipBaseHandler:
    if False:
        for i in range(10):
            print('nop')
    if lang is not None:
        lang = lang.lower()
    if lang in ('quote', 'quoted'):
        return QuoteHandler(processor, output, fence, default_language)
    elif lang == 'math':
        return TexHandler(processor, output, fence)
    elif lang == 'spoiler':
        return SpoilerHandler(processor, output, fence, header)
    else:
        return CodeHandler(processor, output, fence, lang, run_content_validators)

def check_for_new_fence(processor: 'FencedBlockPreprocessor', output: MutableSequence[str], line: str, run_content_validators: bool=False, default_language: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    m = FENCE_RE.match(line)
    if m:
        fence = m.group('fence')
        lang: Optional[str] = m.group('lang')
        header: Optional[str] = m.group('header')
        if not lang and default_language:
            lang = default_language
        handler = generic_handler(processor, output, fence, lang, header, run_content_validators, default_language)
        processor.push(handler)
    else:
        output.append(line)

class OuterHandler(ZulipBaseHandler):

    def __init__(self, processor: 'FencedBlockPreprocessor', output: MutableSequence[str], run_content_validators: bool=False, default_language: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        self.run_content_validators = run_content_validators
        self.default_language = default_language
        super().__init__(processor, output)

    @override
    def handle_line(self, line: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        check_for_new_fence(self.processor, self.output, line, self.run_content_validators, self.default_language)

class CodeHandler(ZulipBaseHandler):

    def __init__(self, processor: 'FencedBlockPreprocessor', output: MutableSequence[str], fence: str, lang: Optional[str], run_content_validators: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self.lang = lang
        self.run_content_validators = run_content_validators
        super().__init__(processor, output, fence)

    @override
    def done(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.run_content_validators:
            validator = CODE_VALIDATORS.get(self.lang, lambda text: None)
            validator(self.lines)
        super().done()

    @override
    def format_text(self, text: str) -> str:
        if False:
            while True:
                i = 10
        return self.processor.format_code(self.lang, text)

class QuoteHandler(ZulipBaseHandler):

    def __init__(self, processor: 'FencedBlockPreprocessor', output: MutableSequence[str], fence: str, default_language: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.default_language = default_language
        super().__init__(processor, output, fence, process_contents=True)

    @override
    def handle_line(self, line: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if line.rstrip() == self.fence:
            self.done()
        else:
            check_for_new_fence(self.processor, self.lines, line, default_language=self.default_language)

    @override
    def format_text(self, text: str) -> str:
        if False:
            while True:
                i = 10
        return self.processor.format_quote(text)

class SpoilerHandler(ZulipBaseHandler):

    def __init__(self, processor: 'FencedBlockPreprocessor', output: MutableSequence[str], fence: str, spoiler_header: Optional[str]) -> None:
        if False:
            print('Hello World!')
        self.spoiler_header = spoiler_header
        super().__init__(processor, output, fence, process_contents=True)

    @override
    def handle_line(self, line: str) -> None:
        if False:
            i = 10
            return i + 15
        if line.rstrip() == self.fence:
            self.done()
        else:
            check_for_new_fence(self.processor, self.lines, line)

    @override
    def format_text(self, text: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.processor.format_spoiler(self.spoiler_header, text)

class TexHandler(ZulipBaseHandler):

    @override
    def format_text(self, text: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.processor.format_tex(text)

class CodeHilite(codehilite.CodeHilite):

    def _parseHeader(self) -> None:
        if False:
            i = 10
            return i + 15
        lines = self.src.split('\n')
        fl = lines[0]
        c = re.compile('\n            (?:(?:^::+)|(?P<shebang>^[#]!)) # Shebang or 2 or more colons\n            (?P<path>(?:/\\w+)*[/ ])?        # Zero or 1 path\n            (?P<lang>[\\w#.+-]*)             # The language\n            \\s*                             # Arbitrary whitespace\n            # Optional highlight lines, single- or double-quote-delimited\n            (hl_lines=(?P<quot>"|\')(?P<hl_lines>.*?)(?P=quot))?\n            ', re.VERBOSE)
        m = c.search(fl)
        if m:
            try:
                self.lang = m.group('lang').lower()
            except IndexError:
                self.lang = None
            if self.options['linenos'] is None and m.group('shebang'):
                self.options['linenos'] = True
            self.options['hl_lines'] = parse_hl_lines(m.group('hl_lines'))
        self.src = '\n'.join(lines).strip('\n')

class FencedBlockPreprocessor(Preprocessor):

    def __init__(self, md: Markdown, run_content_validators: bool=False) -> None:
        if False:
            return 10
        super().__init__(md)
        self.checked_for_codehilite = False
        self.run_content_validators = run_content_validators
        self.codehilite_conf: Mapping[str, Sequence[Any]] = {}

    def push(self, handler: ZulipBaseHandler) -> None:
        if False:
            print('Hello World!')
        self.handlers.append(handler)

    def pop(self) -> None:
        if False:
            print('Hello World!')
        self.handlers.pop()

    @override
    def run(self, lines: Iterable[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Match and store Fenced Code Blocks in the HtmlStash.'
        from zerver.lib.markdown import ZulipMarkdown
        output: List[str] = []
        processor = self
        self.handlers: List[ZulipBaseHandler] = []
        default_language = None
        if isinstance(self.md, ZulipMarkdown) and self.md.zulip_realm is not None:
            default_language = self.md.zulip_realm.default_code_block_language
        handler = OuterHandler(processor, output, self.run_content_validators, default_language)
        self.push(handler)
        for line in lines:
            self.handlers[-1].handle_line(line)
        while self.handlers:
            self.handlers[-1].done()
        if len(output) > 2 and output[-2] != '':
            output.append('')
        return output

    def format_code(self, lang: Optional[str], text: str) -> str:
        if False:
            print('Hello World!')
        if lang:
            langclass = LANG_TAG.format(lang)
        else:
            langclass = ''
        if not self.checked_for_codehilite:
            for ext in self.md.registeredExtensions:
                if isinstance(ext, CodeHiliteExtension):
                    self.codehilite_conf = ext.config
                    break
            self.checked_for_codehilite = True
        if self.codehilite_conf:
            highliter = CodeHilite(text, linenums=self.codehilite_conf['linenums'][0], guess_lang=self.codehilite_conf['guess_lang'][0], css_class=self.codehilite_conf['css_class'][0], style=self.codehilite_conf['pygments_style'][0], use_pygments=self.codehilite_conf['use_pygments'][0], lang=lang or None, noclasses=self.codehilite_conf['noclasses'][0], startinline=True)
            code = highliter.hilite().rstrip('\n')
        else:
            code = CODE_WRAP.format(langclass, self._escape(text))
        if lang:
            div_tag = lxml.html.fromstring(code)
            try:
                code_language = find_lexer_class_by_name(lang).name
            except ClassNotFound:
                code_language = lang
            div_tag.attrib['data-code-language'] = code_language
            code = lxml.html.tostring(div_tag, encoding='unicode')
        return code

    def format_quote(self, text: str) -> str:
        if False:
            while True:
                i = 10
        paragraphs = text.split('\n')
        quoted_paragraphs = []
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            quoted_paragraphs.append('\n'.join(('> ' + line for line in lines)))
        return '\n'.join(quoted_paragraphs)

    def format_spoiler(self, header: Optional[str], text: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        output = []
        header_div_open_html = '<div class="spoiler-block"><div class="spoiler-header">'
        end_header_start_content_html = '</div><div class="spoiler-content" aria-hidden="true">'
        footer_html = '</div></div>'
        output.append(self.placeholder(header_div_open_html))
        if header is not None:
            output.append(header)
        output.append(self.placeholder(end_header_start_content_html))
        output.append(text)
        output.append(self.placeholder(footer_html))
        return '\n\n'.join(output)

    def format_tex(self, text: str) -> str:
        if False:
            return 10
        paragraphs = text.split('\n\n')
        tex_paragraphs = []
        for paragraph in paragraphs:
            html = render_tex(paragraph, is_inline=False)
            if html is not None:
                tex_paragraphs.append(html)
            else:
                tex_paragraphs.append('<span class="tex-error">' + escape(paragraph) + '</span>')
        return '\n\n'.join(tex_paragraphs)

    def placeholder(self, code: str) -> str:
        if False:
            print('Hello World!')
        return self.md.htmlStash.store(code)

    def _escape(self, txt: str) -> str:
        if False:
            while True:
                i = 10
        'basic html escaping'
        txt = txt.replace('&', '&amp;')
        txt = txt.replace('<', '&lt;')
        txt = txt.replace('>', '&gt;')
        txt = txt.replace('"', '&quot;')
        return txt

def makeExtension(*args: Any, **kwargs: Any) -> FencedCodeExtension:
    if False:
        i = 10
        return i + 15
    return FencedCodeExtension(kwargs)
if __name__ == '__main__':
    import doctest
    doctest.testmod()