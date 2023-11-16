from __future__ import annotations
import os.path
import re
from typing import Any
import pygments
import pygments.formatters
import pygments.lexers
from pwnlib.lexer import PwntoolsLexer
import pwndbg.gdblib.config
from pwndbg.color import disable_colors
from pwndbg.color import message
from pwndbg.color import theme
pwndbg.gdblib.config.add_param('syntax-highlight', True, 'Source code / assembly syntax highlight')
style = theme.add_param('syntax-highlight-style', 'monokai', 'Source code / assembly syntax highlight stylename of pygments module')
formatter = pygments.formatters.Terminal256Formatter(style=str(style))
pwntools_lexer = PwntoolsLexer()
lexer_cache: dict[str, Any] = {}

@pwndbg.gdblib.config.trigger(style)
def check_style() -> None:
    if False:
        return 10
    global formatter
    try:
        formatter = pygments.formatters.Terminal256Formatter(style=str(style))
        from pwndbg.commands.context import get_highlight_source
        get_highlight_source.cache.clear()
    except pygments.util.ClassNotFound:
        print(message.warn(f"The pygment formatter style '{style}' is not found, restore to default"))
        style.revert_default()

def syntax_highlight(code, filename='.asm'):
    if False:
        return 10
    if disable_colors:
        return code
    filename = os.path.basename(filename)
    lexer = lexer_cache.get(filename, None)
    if not lexer:
        for glob_pat in PwntoolsLexer.filenames:
            pat = '^' + glob_pat.replace('.', '\\.').replace('*', '.*') + '$'
            if re.match(pat, filename):
                lexer = pwntools_lexer
                break
    if not lexer:
        try:
            lexer = pygments.lexers.guess_lexer_for_filename(filename, code, stripnl=False)
        except pygments.util.ClassNotFound:
            pass
    if lexer:
        lexer_cache[filename] = lexer
        code = pygments.highlight(code, lexer, formatter).rstrip()
    return code