"""utilities for generating and formatting literal Python code."""
import re
from mako import exceptions

class PythonPrinter:

    def __init__(self, stream):
        if False:
            i = 10
            return i + 15
        self.indent = 0
        self.indent_detail = []
        self.indentstring = '    '
        self.stream = stream
        self.lineno = 1
        self.line_buffer = []
        self.in_indent_lines = False
        self._reset_multi_line_flags()
        self.source_map = {}
        self._re_space_comment = re.compile('^\\s*#')
        self._re_space = re.compile('^\\s*$')
        self._re_indent = re.compile(':[ \\t]*(?:#.*)?$')
        self._re_compound = re.compile('^\\s*(if|try|elif|while|for|with)')
        self._re_indent_keyword = re.compile('^\\s*(def|class|else|elif|except|finally)')
        self._re_unindentor = re.compile('^\\s*(else|elif|except|finally).*\\:')

    def _update_lineno(self, num):
        if False:
            i = 10
            return i + 15
        self.lineno += num

    def start_source(self, lineno):
        if False:
            i = 10
            return i + 15
        if self.lineno not in self.source_map:
            self.source_map[self.lineno] = lineno

    def write_blanks(self, num):
        if False:
            for i in range(10):
                print('nop')
        self.stream.write('\n' * num)
        self._update_lineno(num)

    def write_indented_block(self, block, starting_lineno=None):
        if False:
            while True:
                i = 10
        'print a line or lines of python which already contain indentation.\n\n        The indentation of the total block of lines will be adjusted to that of\n        the current indent level.'
        self.in_indent_lines = False
        for (i, l) in enumerate(re.split('\\r?\\n', block)):
            self.line_buffer.append(l)
            if starting_lineno is not None:
                self.start_source(starting_lineno + i)
            self._update_lineno(1)

    def writelines(self, *lines):
        if False:
            return 10
        'print a series of lines of python.'
        for line in lines:
            self.writeline(line)

    def writeline(self, line):
        if False:
            for i in range(10):
                print('nop')
        'print a line of python, indenting it according to the current\n        indent level.\n\n        this also adjusts the indentation counter according to the\n        content of the line.\n\n        '
        if not self.in_indent_lines:
            self._flush_adjusted_lines()
            self.in_indent_lines = True
        if line is None or self._re_space_comment.match(line) or self._re_space.match(line):
            hastext = False
        else:
            hastext = True
        is_comment = line and len(line) and (line[0] == '#')
        if not is_comment and (not hastext or self._is_unindentor(line)) and (self.indent > 0):
            self.indent -= 1
            if len(self.indent_detail) == 0:
                raise exceptions.MakoException('Too many whitespace closures')
            self.indent_detail.pop()
        if line is None:
            return
        self.stream.write(self._indent_line(line) + '\n')
        self._update_lineno(len(line.split('\n')))
        if self._re_indent.search(line):
            match = self._re_compound.match(line)
            if match:
                indentor = match.group(1)
                self.indent += 1
                self.indent_detail.append(indentor)
            else:
                indentor = None
                m2 = self._re_indent_keyword.match(line)
                if m2:
                    self.indent += 1
                    self.indent_detail.append(indentor)

    def close(self):
        if False:
            while True:
                i = 10
        'close this printer, flushing any remaining lines.'
        self._flush_adjusted_lines()

    def _is_unindentor(self, line):
        if False:
            while True:
                i = 10
        "return true if the given line is an 'unindentor',\n        relative to the last 'indent' event received.\n\n        "
        if len(self.indent_detail) == 0:
            return False
        indentor = self.indent_detail[-1]
        if indentor is None:
            return False
        match = self._re_unindentor.match(line)
        return bool(match)

    def _indent_line(self, line, stripspace=''):
        if False:
            print('Hello World!')
        'indent the given line according to the current indent level.\n\n        stripspace is a string of space that will be truncated from the\n        start of the line before indenting.'
        if stripspace == '':
            return self.indentstring * self.indent + line
        return re.sub('^%s' % stripspace, self.indentstring * self.indent, line)

    def _reset_multi_line_flags(self):
        if False:
            return 10
        'reset the flags which would indicate we are in a backslashed\n        or triple-quoted section.'
        (self.backslashed, self.triplequoted) = (False, False)

    def _in_multi_line(self, line):
        if False:
            i = 10
            return i + 15
        'return true if the given line is part of a multi-line block,\n        via backslash or triple-quote.'
        current_state = self.backslashed or self.triplequoted
        self.backslashed = bool(re.search('\\\\$', line))
        triples = len(re.findall('\\"\\"\\"|\\\'\\\'\\\'', line))
        if triples == 1 or triples % 2 != 0:
            self.triplequoted = not self.triplequoted
        return current_state

    def _flush_adjusted_lines(self):
        if False:
            i = 10
            return i + 15
        stripspace = None
        self._reset_multi_line_flags()
        for entry in self.line_buffer:
            if self._in_multi_line(entry):
                self.stream.write(entry + '\n')
            else:
                entry = entry.expandtabs()
                if stripspace is None and re.search('^[ \\t]*[^# \\t]', entry):
                    stripspace = re.match('^([ \\t]*)', entry).group(1)
                self.stream.write(self._indent_line(entry, stripspace) + '\n')
        self.line_buffer = []
        self._reset_multi_line_flags()

def adjust_whitespace(text):
    if False:
        while True:
            i = 10
    'remove the left-whitespace margin of a block of Python code.'
    state = [False, False]
    (backslashed, triplequoted) = (0, 1)

    def in_multi_line(line):
        if False:
            return 10
        start_state = state[backslashed] or state[triplequoted]
        if re.search('\\\\$', line):
            state[backslashed] = True
        else:
            state[backslashed] = False

        def match(reg, t):
            if False:
                while True:
                    i = 10
            m = re.match(reg, t)
            if m:
                return (m, t[len(m.group(0)):])
            else:
                return (None, t)
        while line:
            if state[triplequoted]:
                (m, line) = match('%s' % state[triplequoted], line)
                if m:
                    state[triplequoted] = False
                else:
                    (m, line) = match('.*?(?=%s|$)' % state[triplequoted], line)
            else:
                (m, line) = match('#', line)
                if m:
                    return start_state
                (m, line) = match('\\"\\"\\"|\\\'\\\'\\\'', line)
                if m:
                    state[triplequoted] = m.group(0)
                    continue
                (m, line) = match('.*?(?=\\"\\"\\"|\\\'\\\'\\\'|#|$)', line)
        return start_state

    def _indent_line(line, stripspace=''):
        if False:
            i = 10
            return i + 15
        return re.sub('^%s' % stripspace, '', line)
    lines = []
    stripspace = None
    for line in re.split('\\r?\\n', text):
        if in_multi_line(line):
            lines.append(line)
        else:
            line = line.expandtabs()
            if stripspace is None and re.search('^[ \\t]*[^# \\t]', line):
                stripspace = re.match('^([ \\t]*)', line).group(1)
            lines.append(_indent_line(line, stripspace))
    return '\n'.join(lines)