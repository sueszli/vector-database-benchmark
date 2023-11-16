"""Highlight code blocks using Pygments."""
from __future__ import annotations
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any
from pygments import highlight
from pygments.filters import ErrorToken
from pygments.formatters import HtmlFormatter, LatexFormatter
from pygments.lexers import CLexer, PythonConsoleLexer, PythonLexer, RstLexer, TextLexer, get_lexer_by_name, guess_lexer
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound
from sphinx.locale import __
from sphinx.pygments_styles import NoneStyle, SphinxStyle
from sphinx.util import logging, texescape
if TYPE_CHECKING:
    from pygments.formatter import Formatter
    from pygments.lexer import Lexer
    from pygments.style import Style
logger = logging.getLogger(__name__)
lexers: dict[str, Lexer] = {}
lexer_classes: dict[str, type[Lexer] | partial[Lexer]] = {'none': partial(TextLexer, stripnl=False), 'python': partial(PythonLexer, stripnl=False), 'pycon': partial(PythonConsoleLexer, stripnl=False), 'rest': partial(RstLexer, stripnl=False), 'c': partial(CLexer, stripnl=False)}
escape_hl_chars = {ord('\\'): '\\PYGZbs{}', ord('{'): '\\PYGZob{}', ord('}'): '\\PYGZcb{}'}
_LATEX_ADD_STYLES = '\n% Sphinx redefinitions\n% Originally to obtain a straight single quote via package textcomp, then\n% to fix problems for the 5.0.0 inline code highlighting (captions!).\n% The \\text is from amstext, a dependency of sphinx.sty.  It is here only\n% to avoid build errors if for some reason expansion is in math mode.\n\\def\\PYGZbs{\\text\\textbackslash}\n\\def\\PYGZus{\\_}\n\\def\\PYGZob{\\{}\n\\def\\PYGZcb{\\}}\n\\def\\PYGZca{\\text\\textasciicircum}\n\\def\\PYGZam{\\&}\n\\def\\PYGZlt{\\text\\textless}\n\\def\\PYGZgt{\\text\\textgreater}\n\\def\\PYGZsh{\\#}\n\\def\\PYGZpc{\\%}\n\\def\\PYGZdl{\\$}\n\\def\\PYGZhy{\\sphinxhyphen}% defined in sphinxlatexstyletext.sty\n\\def\\PYGZsq{\\text\\textquotesingle}\n\\def\\PYGZdq{"}\n\\def\\PYGZti{\\text\\textasciitilde}\n\\makeatletter\n% use \\protected to allow syntax highlighting in captions\n\\protected\\def\\PYG#1#2{\\PYG@reset\\PYG@toks#1+\\relax+{\\PYG@do{#2}}}\n\\makeatother\n'

class PygmentsBridge:
    html_formatter = HtmlFormatter
    latex_formatter = LatexFormatter

    def __init__(self, dest: str='html', stylename: str='sphinx', latex_engine: str | None=None) -> None:
        if False:
            return 10
        self.dest = dest
        self.latex_engine = latex_engine
        style = self.get_style(stylename)
        self.formatter_args: dict[str, Any] = {'style': style}
        if dest == 'html':
            self.formatter = self.html_formatter
        else:
            self.formatter = self.latex_formatter
            self.formatter_args['commandprefix'] = 'PYG'

    def get_style(self, stylename: str) -> Style:
        if False:
            i = 10
            return i + 15
        if stylename is None or stylename == 'sphinx':
            return SphinxStyle
        elif stylename == 'none':
            return NoneStyle
        elif '.' in stylename:
            (module, stylename) = stylename.rsplit('.', 1)
            return getattr(import_module(module), stylename)
        else:
            return get_style_by_name(stylename)

    def get_formatter(self, **kwargs: Any) -> Formatter:
        if False:
            while True:
                i = 10
        kwargs.update(self.formatter_args)
        return self.formatter(**kwargs)

    def get_lexer(self, source: str, lang: str, opts: dict | None=None, force: bool=False, location: Any=None) -> Lexer:
        if False:
            return 10
        if not opts:
            opts = {}
        if lang in {'py', 'python', 'py3', 'python3', 'default'}:
            if source.startswith('>>>'):
                lang = 'pycon'
            else:
                lang = 'python'
        if lang == 'pycon3':
            lang = 'pycon'
        if lang in lexers:
            return lexers[lang]
        elif lang in lexer_classes:
            lexer = lexer_classes[lang](**opts)
        else:
            try:
                if lang == 'guess':
                    lexer = guess_lexer(source, **opts)
                else:
                    lexer = get_lexer_by_name(lang, **opts)
            except ClassNotFound:
                logger.warning(__('Pygments lexer name %r is not known'), lang, location=location)
                lexer = lexer_classes['none'](**opts)
        if not force:
            lexer.add_filter('raiseonerror')
        return lexer

    def highlight_block(self, source: str, lang: str, opts: dict | None=None, force: bool=False, location: Any=None, **kwargs: Any) -> str:
        if False:
            print('Hello World!')
        if not isinstance(source, str):
            source = source.decode()
        lexer = self.get_lexer(source, lang, opts, force, location)
        formatter = self.get_formatter(**kwargs)
        try:
            hlsource = highlight(source, lexer, formatter)
        except ErrorToken as err:
            if lang == 'default':
                lang = 'none'
            else:
                logger.warning(__('Lexing literal_block %r as "%s" resulted in an error at token: %r. Retrying in relaxed mode.'), source, lang, str(err), type='misc', subtype='highlighting_failure', location=location)
                if force:
                    lang = 'none'
                else:
                    force = True
            lexer = self.get_lexer(source, lang, opts, force, location)
            hlsource = highlight(source, lexer, formatter)
        if self.dest == 'html':
            return hlsource
        else:
            return texescape.hlescape(hlsource, self.latex_engine)

    def get_stylesheet(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        formatter = self.get_formatter()
        if self.dest == 'html':
            return formatter.get_style_defs('.highlight')
        else:
            return formatter.get_style_defs() + _LATEX_ADD_STYLES