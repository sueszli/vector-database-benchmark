import io
import sys
from . import is_terminal

class TextPecker:

    def __init__(self, s):
        if False:
            print('Hello World!')
        self.str = s
        self.i = 0

    def read(self, n):
        if False:
            while True:
                i = 10
        self.i += n
        return self.str[self.i - n:self.i]

    def peek(self, n):
        if False:
            return 10
        if n >= 0:
            return self.str[self.i:self.i + n]
        else:
            return self.str[self.i + n - 1:self.i - 1]

    def peekline(self):
        if False:
            return 10
        out = ''
        i = self.i
        while i < len(self.str) and self.str[i] != '\n':
            out += self.str[i]
            i += 1
        return out

    def readline(self):
        if False:
            while True:
                i = 10
        out = self.peekline()
        self.i += len(out)
        return out

def process_directive(directive, arguments, out, state_hook):
    if False:
        return 10
    if directive == 'container' and arguments == 'experimental':
        state_hook('text', '**', out)
        out.write('++ Experimental ++')
        state_hook('**', 'text', out)
    else:
        state_hook('text', '**', out)
        out.write(directive.title())
        out.write(':\n')
        state_hook('**', 'text', out)
        if arguments:
            out.write(arguments)
            out.write('\n')

def rst_to_text(text, state_hook=None, references=None):
    if False:
        return 10
    '\n    Convert rST to a more human text form.\n\n    This is a very loose conversion. No advanced rST features are supported.\n    The generated output directly depends on the input (e.g. indentation of\n    admonitions).\n    '
    state_hook = state_hook or (lambda old_state, new_state, out: None)
    references = references or {}
    state = 'text'
    inline_mode = 'replace'
    text = TextPecker(text)
    out = io.StringIO()
    inline_single = ('*', '`')
    while True:
        char = text.read(1)
        if not char:
            break
        next = text.peek(1)
        if state == 'text':
            if char == '\\' and text.peek(1) in inline_single:
                continue
            if text.peek(-1) != '\\':
                if char in inline_single and next != char:
                    state_hook(state, char, out)
                    state = char
                    continue
                if char == next == '*':
                    state_hook(state, '**', out)
                    state = '**'
                    text.read(1)
                    continue
                if char == next == '`':
                    state_hook(state, '``', out)
                    state = '``'
                    text.read(1)
                    continue
                if text.peek(-1).isspace() and char == ':' and (text.peek(5) == 'ref:`'):
                    text.read(5)
                    ref = ''
                    while True:
                        char = text.peek(1)
                        if char == '`':
                            text.read(1)
                            break
                        if char == '\n':
                            text.read(1)
                            continue
                        ref += text.read(1)
                    try:
                        out.write(references[ref])
                    except KeyError:
                        raise ValueError("Undefined reference in Archiver help: %r — please add reference substitution to 'rst_plain_text_references'" % ref)
                    continue
                if char == ':' and text.peek(2) == ':\n':
                    text.read(2)
                    state_hook(state, 'code-block', out)
                    state = 'code-block'
                    out.write(':\n')
                    continue
            if text.peek(-2) in ('\n\n', '') and char == next == '.':
                text.read(2)
                (directive, is_directive, arguments) = text.readline().partition('::')
                text.read(1)
                if not is_directive:
                    if directive == 'nanorst: inline-fill':
                        inline_mode = 'fill'
                    elif directive == 'nanorst: inline-replace':
                        inline_mode = 'replace'
                    continue
                process_directive(directive, arguments.strip(), out, state_hook)
                continue
        if state in inline_single and char == state:
            state_hook(state, 'text', out)
            state = 'text'
            if inline_mode == 'fill':
                out.write(2 * ' ')
            continue
        if state == '``' and char == next == '`':
            state_hook(state, 'text', out)
            state = 'text'
            text.read(1)
            if inline_mode == 'fill':
                out.write(4 * ' ')
            continue
        if state == '**' and char == next == '*':
            state_hook(state, 'text', out)
            state = 'text'
            text.read(1)
            continue
        if state == 'code-block' and char == next == '\n' and (text.peek(5)[1:] != '    '):
            state_hook(state, 'text', out)
            state = 'text'
        out.write(char)
    assert state == 'text', 'Invalid final state %r (This usually indicates unmatched */**)' % state
    return out.getvalue()

class RstToTextLazy:

    def __init__(self, str, state_hook=None, references=None):
        if False:
            for i in range(10):
                print('nop')
        self.str = str
        self.state_hook = state_hook
        self.references = references
        self._rst = None

    @property
    def rst(self):
        if False:
            while True:
                i = 10
        if self._rst is None:
            self._rst = rst_to_text(self.str, self.state_hook, self.references)
        return self._rst

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        return getattr(self.rst, item)

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.rst

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        return self.rst + other

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.rst)

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return item in self.rst

def ansi_escapes(old_state, new_state, out):
    if False:
        i = 10
        return i + 15
    if old_state == 'text' and new_state in ('*', '`', '``'):
        out.write('\x1b[4m')
    if old_state == 'text' and new_state == '**':
        out.write('\x1b[1m')
    if old_state in ('*', '`', '``', '**') and new_state == 'text':
        out.write('\x1b[0m')

def rst_to_terminal(rst, references=None, destination=sys.stdout):
    if False:
        i = 10
        return i + 15
    '\n    Convert *rst* to a lazy string.\n\n    If *destination* is a file-like object connected to a terminal,\n    enrich text with suitable ANSI escapes. Otherwise return plain text.\n    '
    if is_terminal(destination):
        rst_state_hook = ansi_escapes
    else:
        rst_state_hook = None
    return RstToTextLazy(rst, rst_state_hook, references)