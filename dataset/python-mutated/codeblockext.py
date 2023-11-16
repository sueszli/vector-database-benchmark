from __future__ import annotations
__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'The MIT License <http://opensource.org/licenses/MIT>'
__copyright__ = 'Copyright (C) 2015 Gina Häußge - Released under terms of the MIT License'
from typing import Any
import sphinx.highlighting
from docutils import nodes
from docutils.parsers.rst import directives
from pygments import highlight
from pygments.filters import ErrorToken, VisibleWhitespaceFilter
from pygments.lexers.python import PythonConsoleLexer
from pygments.util import ClassNotFound
from six import text_type
from sphinx.directives.code import CodeBlock
from sphinx.ext import doctest

def _merge_dict(a, b):
    if False:
        return 10
    '\n    Little helper to merge two dicts a and b on the fly.\n    '
    result = dict(a)
    result.update(b)
    return result

class literal_block_ext(nodes.General, nodes.FixedTextElement):
    """
    Custom node which is basically the same as a :class:`literal_block`, just with whitespace support and introduced
    in order to be able to have a custom visitor.
    """

    @classmethod
    def from_literal_block(cls, block):
        if False:
            i = 10
            return i + 15
        '\n        Factory method constructing an instance exactly copying all attributes over from ``block`` and settings a\n        custom ``tagname``.\n        '
        new = literal_block_ext()
        for a in ('attributes', 'basic_attributes', 'child_text_separator', 'children', 'document', 'known_attributes', 'line', 'list_attributes', 'local_attributes', 'parent', 'rawsource', 'source'):
            setattr(new, a, getattr(block, a))
        new.tagname = 'literal_block_ext'
        return new

class CodeBlockExt(CodeBlock):
    """
    This is basically an extension of a regular :class:`CodeBlock` directive which just supports an additional option
    ``whitespace`` which if present will enable (together with everything else in here) to render whitespace in
    code blocks.
    """
    option_spec = _merge_dict(CodeBlock.option_spec, {'whitespace': directives.flag})

    def run(self) -> list[nodes.Node]:
        if False:
            print('Hello World!')
        code_block = CodeBlock.run(self)

        def find_and_wrap_literal_block(node):
            if False:
                while True:
                    i = 10
            '\n            Recursive method to turn all literal blocks located within a node into :class:`literal_block_ext`.\n            '
            if isinstance(node, nodes.container):
                children = []
                for child in node.children:
                    children.append(find_and_wrap_literal_block(child))
                node.children = children
                return node
            elif isinstance(node, nodes.literal_block):
                return self._wrap_literal_block(node)
            else:
                return node
        return list(map(find_and_wrap_literal_block, code_block))

    def _wrap_literal_block(self, node):
        if False:
            i = 10
            return i + 15
        literal = literal_block_ext.from_literal_block(node)
        literal['whitespace'] = 'whitespace' in self.options
        return literal

class PygmentsBridgeExt:
    """
    Wrapper for :class:`PygmentsBridge`, delegates everything to the wrapped ``bridge`` but :method:`highlight_block`,
    which calls the parent implementation for lexer selection, then
    """

    def __init__(self, bridge, whitespace):
        if False:
            return 10
        self._bridge = bridge
        self._whitespace = whitespace

    def __getattr__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._bridge, item)

    def highlight_block(self, source, lang, opts=None, warn=None, force=False, **kwargs):
        if False:
            return 10
        if not self._whitespace:
            return self._bridge.highlight_block(source, lang, opts=opts, warn=warn, force=force, **kwargs)

        class whitespace:

            def __init__(self, lexer):
                if False:
                    print('Hello World!')
                self._lexer = lexer
                self._orig_filters = lexer.filters
                self._orig_tabsize = lexer.tabsize

            def __enter__(self):
                if False:
                    print('Hello World!')
                new_filters = list(self._orig_filters)
                new_filters.append(VisibleWhitespaceFilter(spaces=True, tabs=True, tabsize=self._lexer.tabsize))
                self._lexer.filters = new_filters
                self._lexer.tabsize = 0
                return self._lexer

            def __exit__(self, type, value, traceback):
                if False:
                    while True:
                        i = 10
                self._lexer.filters = self._orig_filters
                self._lexer.tabsize = self._orig_tabsize
        if not isinstance(source, str):
            source = source.decode()
        if lang in ('py', 'python'):
            if source.startswith('>>>'):
                lexer = sphinx.highlighting.lexers['pycon']
            elif not force:
                if self.try_parse(source):
                    lexer = sphinx.highlighting.lexers['python']
                else:
                    lexer = sphinx.highlighting.lexers['none']
            else:
                lexer = sphinx.highlighting.lexers['python']
        elif lang in ('python3', 'py3') and source.startswith('>>>'):
            lexer = sphinx.highlighting.lexers['pycon3']
        elif lang == 'guess':
            lexer = sphinx.highlighting.guess_lexer(source)
        elif lang in sphinx.highlighting.lexers:
            lexer = sphinx.highlighting.lexers[lang]
        else:
            try:
                lexer = sphinx.highlighting.lexers[lang] = sphinx.highlighting.get_lexer_by_name(lang, **opts or {})
            except ClassNotFound:
                if warn:
                    warn('Pygments lexer name %r is not known' % lang)
                    lexer = sphinx.highlighting.lexers['none']
                else:
                    raise
            else:
                lexer.add_filter('raiseonerror')
        if not isinstance(source, str):
            source = source.decode()
        if isinstance(lexer, PythonConsoleLexer) and self._bridge.trim_doctest_flags:
            source = doctest.blankline_re.sub('', source)
            source = doctest.doctestopt_re.sub('', source)
        formatter = self._bridge.get_formatter(**kwargs)
        try:
            with whitespace(lexer) as l:
                hlsource = highlight(source, l, formatter)
        except ErrorToken:
            with whitespace(sphinx.highlighting.lexers['none']) as l:
                hlsource = highlight(source, l, formatter)
        return hlsource

class whitespace_highlighter:
    """
    Context manager for adapting the used highlighter on a translator for a given node's whitespace properties.
    """

    def __init__(self, translator, node):
        if False:
            print('Hello World!')
        self.translator = translator
        self.node = node
        self._orig_highlighter = self.translator.highlighter

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        whitespace = self.node['whitespace'] if 'whitespace' in self.node else False
        if whitespace:
            self.translator.highlighter = PygmentsBridgeExt(self._orig_highlighter, whitespace)
        return self.translator

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        self.translator.highlighter = self._orig_highlighter

def visit_literal_block_ext(translator, node):
    if False:
        i = 10
        return i + 15
    '\n    When our custom code block is visited, we temporarily exchange the highlighter used in the translator, call the\n    visitor for regular literal blocks, then switch back again.\n    '
    with whitespace_highlighter(translator, node):
        translator.visit_literal_block(node)

def depart_literal_block_ext(translator, node):
    if False:
        print('Hello World!')
    '\n    Just call the depart function for regular literal blocks.\n    '
    with whitespace_highlighter(translator, node):
        translator.depart_literal_block(node)

def setup(app):
    if False:
        return 10
    app.add_directive('code-block-ext', CodeBlockExt)
    handler = (visit_literal_block_ext, depart_literal_block_ext)
    app.add_node(literal_block_ext, html=handler, latex=handler, text=handler)
    return {'version': '0.1'}