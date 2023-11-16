import tokenize
from pywin import default_scintilla_encoding
from . import PyParse

class AutoIndent:
    menudefs = [('edit', [None, ('_Indent region', '<<indent-region>>'), ('_Dedent region', '<<dedent-region>>'), ('Comment _out region', '<<comment-region>>'), ('U_ncomment region', '<<uncomment-region>>'), ('Tabify region', '<<tabify-region>>'), ('Untabify region', '<<untabify-region>>'), ('Toggle tabs', '<<toggle-tabs>>'), ('New indent width', '<<change-indentwidth>>')])]
    keydefs = {'<<smart-backspace>>': ['<Key-BackSpace>'], '<<newline-and-indent>>': ['<Key-Return>', '<KP_Enter>'], '<<smart-indent>>': ['<Key-Tab>']}
    windows_keydefs = {'<<indent-region>>': ['<Control-bracketright>'], '<<dedent-region>>': ['<Control-bracketleft>'], '<<comment-region>>': ['<Alt-Key-3>'], '<<uncomment-region>>': ['<Alt-Key-4>'], '<<tabify-region>>': ['<Alt-Key-5>'], '<<untabify-region>>': ['<Alt-Key-6>'], '<<toggle-tabs>>': ['<Alt-Key-t>'], '<<change-indentwidth>>': ['<Alt-Key-u>']}
    unix_keydefs = {'<<indent-region>>': ['<Alt-bracketright>', '<Meta-bracketright>', '<Control-bracketright>'], '<<dedent-region>>': ['<Alt-bracketleft>', '<Meta-bracketleft>', '<Control-bracketleft>'], '<<comment-region>>': ['<Alt-Key-3>', '<Meta-Key-3>'], '<<uncomment-region>>': ['<Alt-Key-4>', '<Meta-Key-4>'], '<<tabify-region>>': ['<Alt-Key-5>', '<Meta-Key-5>'], '<<untabify-region>>': ['<Alt-Key-6>', '<Meta-Key-6>'], '<<toggle-tabs>>': ['<Alt-Key-t>'], '<<change-indentwidth>>': ['<Alt-Key-u>']}
    usetabs = 1
    indentwidth = 4
    tabwidth = 8
    context_use_ps1 = 0
    num_context_lines = (50, 500, 5000000)

    def __init__(self, editwin):
        if False:
            return 10
        self.editwin = editwin
        self.text = editwin.text

    def config(self, **options):
        if False:
            print('Hello World!')
        for (key, value) in options.items():
            if key == 'usetabs':
                self.usetabs = value
            elif key == 'indentwidth':
                self.indentwidth = value
            elif key == 'tabwidth':
                self.tabwidth = value
            elif key == 'context_use_ps1':
                self.context_use_ps1 = value
            else:
                raise KeyError('bad option name: %s' % repr(key))

    def set_indentation_params(self, ispythonsource, guess=1):
        if False:
            for i in range(10):
                print('nop')
        if guess and ispythonsource:
            i = self.guess_indent()
            if 2 <= i <= 8:
                self.indentwidth = i
            if self.indentwidth != self.tabwidth:
                self.usetabs = 0
        self.editwin.set_tabwidth(self.tabwidth)

    def smart_backspace_event(self, event):
        if False:
            while True:
                i = 10
        text = self.text
        (first, last) = self.editwin.get_selection_indices()
        if first and last:
            text.delete(first, last)
            text.mark_set('insert', first)
            return 'break'
        chars = text.get('insert linestart', 'insert')
        if chars == '':
            if text.compare('insert', '>', '1.0'):
                text.delete('insert-1c')
            else:
                text.bell()
            return 'break'
        if chars[-1] not in ' \t':
            text.delete('insert-1c')
            return 'break'
        have = len(chars.expandtabs(self.tabwidth))
        assert have > 0
        want = int((have - 1) / self.indentwidth) * self.indentwidth
        ncharsdeleted = 0
        while 1:
            chars = chars[:-1]
            ncharsdeleted = ncharsdeleted + 1
            have = len(chars.expandtabs(self.tabwidth))
            if have <= want or chars[-1] not in ' \t':
                break
        text.undo_block_start()
        text.delete('insert-%dc' % ncharsdeleted, 'insert')
        if have < want:
            text.insert('insert', ' ' * (want - have))
        text.undo_block_stop()
        return 'break'

    def smart_indent_event(self, event):
        if False:
            print('Hello World!')
        text = self.text
        (first, last) = self.editwin.get_selection_indices()
        text.undo_block_start()
        try:
            if first and last:
                if index2line(first) != index2line(last):
                    return self.indent_region_event(event)
                text.delete(first, last)
                text.mark_set('insert', first)
            prefix = text.get('insert linestart', 'insert')
            (raw, effective) = classifyws(prefix, self.tabwidth)
            if raw == len(prefix):
                self.reindent_to(effective + self.indentwidth)
            else:
                if self.usetabs:
                    pad = '\t'
                else:
                    effective = len(prefix.expandtabs(self.tabwidth))
                    n = self.indentwidth
                    pad = ' ' * (n - effective % n)
                text.insert('insert', pad)
            text.see('insert')
            return 'break'
        finally:
            text.undo_block_stop()

    def newline_and_indent_event(self, event):
        if False:
            while True:
                i = 10
        text = self.text
        (first, last) = self.editwin.get_selection_indices()
        text.undo_block_start()
        try:
            if first and last:
                text.delete(first, last)
                text.mark_set('insert', first)
            line = text.get('insert linestart', 'insert')
            (i, n) = (0, len(line))
            while i < n and line[i] in ' \t':
                i = i + 1
            if i == n:
                text.delete('insert - %d chars' % i, 'insert')
                text.insert('insert linestart', '\n')
                return 'break'
            indent = line[:i]
            i = 0
            while line and line[-1] in ' \t':
                line = line[:-1]
                i = i + 1
            if i:
                text.delete('insert - %d chars' % i, 'insert')
            while text.get('insert') in ' \t':
                text.delete('insert')
            text.insert('insert', '\n')
            lno = index2line(text.index('insert'))
            y = PyParse.Parser(self.indentwidth, self.tabwidth)
            for context in self.num_context_lines:
                startat = max(lno - context, 1)
                startatindex = repr(startat) + '.0'
                rawtext = text.get(startatindex, 'insert')
                y.set_str(rawtext)
                bod = y.find_good_parse_start(self.context_use_ps1, self._build_char_in_string_func(startatindex))
                if bod is not None or startat == 1:
                    break
            y.set_lo(bod or 0)
            c = y.get_continuation_type()
            if c != PyParse.C_NONE:
                if c == PyParse.C_STRING:
                    text.insert('insert', indent)
                elif c == PyParse.C_BRACKET:
                    self.reindent_to(y.compute_bracket_indent())
                elif c == PyParse.C_BACKSLASH:
                    if y.get_num_lines_in_stmt() > 1:
                        text.insert('insert', indent)
                    else:
                        self.reindent_to(y.compute_backslash_indent())
                else:
                    assert 0, 'bogus continuation type ' + repr(c)
                return 'break'
            indent = y.get_base_indent_string()
            text.insert('insert', indent)
            if y.is_block_opener():
                self.smart_indent_event(event)
            elif indent and y.is_block_closer():
                self.smart_backspace_event(event)
            return 'break'
        finally:
            text.see('insert')
            text.undo_block_stop()
    auto_indent = newline_and_indent_event

    def _build_char_in_string_func(self, startindex):
        if False:
            while True:
                i = 10

        def inner(offset, _startindex=startindex, _icis=self.editwin.is_char_in_string):
            if False:
                for i in range(10):
                    print('nop')
            return _icis(_startindex + '+%dc' % offset)
        return inner

    def indent_region_event(self, event):
        if False:
            while True:
                i = 10
        (head, tail, chars, lines) = self.get_region()
        for pos in range(len(lines)):
            line = lines[pos]
            if line:
                (raw, effective) = classifyws(line, self.tabwidth)
                effective = effective + self.indentwidth
                lines[pos] = self._make_blanks(effective) + line[raw:]
        self.set_region(head, tail, chars, lines)
        return 'break'

    def dedent_region_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        (head, tail, chars, lines) = self.get_region()
        for pos in range(len(lines)):
            line = lines[pos]
            if line:
                (raw, effective) = classifyws(line, self.tabwidth)
                effective = max(effective - self.indentwidth, 0)
                lines[pos] = self._make_blanks(effective) + line[raw:]
        self.set_region(head, tail, chars, lines)
        return 'break'

    def comment_region_event(self, event):
        if False:
            i = 10
            return i + 15
        (head, tail, chars, lines) = self.get_region()
        for pos in range(len(lines) - 1):
            line = lines[pos]
            lines[pos] = '##' + line
        self.set_region(head, tail, chars, lines)

    def uncomment_region_event(self, event):
        if False:
            while True:
                i = 10
        (head, tail, chars, lines) = self.get_region()
        for pos in range(len(lines)):
            line = lines[pos]
            if not line:
                continue
            if line[:2] == '##':
                line = line[2:]
            elif line[:1] == '#':
                line = line[1:]
            lines[pos] = line
        self.set_region(head, tail, chars, lines)

    def tabify_region_event(self, event):
        if False:
            i = 10
            return i + 15
        (head, tail, chars, lines) = self.get_region()
        tabwidth = self._asktabwidth()
        for pos in range(len(lines)):
            line = lines[pos]
            if line:
                (raw, effective) = classifyws(line, tabwidth)
                (ntabs, nspaces) = divmod(effective, tabwidth)
                lines[pos] = '\t' * ntabs + ' ' * nspaces + line[raw:]
        self.set_region(head, tail, chars, lines)

    def untabify_region_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        (head, tail, chars, lines) = self.get_region()
        tabwidth = self._asktabwidth()
        for pos in range(len(lines)):
            lines[pos] = lines[pos].expandtabs(tabwidth)
        self.set_region(head, tail, chars, lines)

    def toggle_tabs_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self.editwin.askyesno('Toggle tabs', 'Turn tabs ' + ('on', 'off')[self.usetabs] + '?', parent=self.text):
            self.usetabs = not self.usetabs
        return 'break'

    def change_tabwidth_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        new = self._asktabwidth()
        if new != self.tabwidth:
            self.tabwidth = new
            self.set_indentation_params(0, guess=0)
        return 'break'

    def change_indentwidth_event(self, event):
        if False:
            print('Hello World!')
        new = self.editwin.askinteger('Indent width', 'New indent width (1-16)', parent=self.text, initialvalue=self.indentwidth, minvalue=1, maxvalue=16)
        if new and new != self.indentwidth:
            self.indentwidth = new
        return 'break'

    def get_region(self):
        if False:
            while True:
                i = 10
        text = self.text
        (first, last) = self.editwin.get_selection_indices()
        if first and last:
            head = text.index(first + ' linestart')
            tail = text.index(last + '-1c lineend +1c')
        else:
            head = text.index('insert linestart')
            tail = text.index('insert lineend +1c')
        chars = text.get(head, tail)
        lines = chars.split('\n')
        return (head, tail, chars, lines)

    def set_region(self, head, tail, chars, lines):
        if False:
            i = 10
            return i + 15
        text = self.text
        newchars = '\n'.join(lines)
        if newchars == chars:
            text.bell()
            return
        text.tag_remove('sel', '1.0', 'end')
        text.mark_set('insert', head)
        text.undo_block_start()
        text.delete(head, tail)
        text.insert(head, newchars)
        text.undo_block_stop()
        text.tag_add('sel', head, 'insert')

    def _make_blanks(self, n):
        if False:
            return 10
        if self.usetabs:
            (ntabs, nspaces) = divmod(n, self.tabwidth)
            return '\t' * ntabs + ' ' * nspaces
        else:
            return ' ' * n

    def reindent_to(self, column):
        if False:
            return 10
        text = self.text
        text.undo_block_start()
        if text.compare('insert linestart', '!=', 'insert'):
            text.delete('insert linestart', 'insert')
        if column:
            text.insert('insert', self._make_blanks(column))
        text.undo_block_stop()

    def _asktabwidth(self):
        if False:
            print('Hello World!')
        return self.editwin.askinteger('Tab width', 'Spaces per tab?', parent=self.text, initialvalue=self.tabwidth, minvalue=1, maxvalue=16) or self.tabwidth

    def guess_indent(self):
        if False:
            return 10
        (opener, indented) = IndentSearcher(self.text, self.tabwidth).run()
        if opener and indented:
            (raw, indentsmall) = classifyws(opener, self.tabwidth)
            (raw, indentlarge) = classifyws(indented, self.tabwidth)
        else:
            indentsmall = indentlarge = 0
        return indentlarge - indentsmall

def index2line(index):
    if False:
        i = 10
        return i + 15
    return int(float(index))

def classifyws(s, tabwidth):
    if False:
        print('Hello World!')
    raw = effective = 0
    for ch in s:
        if ch == ' ':
            raw = raw + 1
            effective = effective + 1
        elif ch == '\t':
            raw = raw + 1
            effective = (effective // tabwidth + 1) * tabwidth
        else:
            break
    return (raw, effective)

class IndentSearcher:

    def __init__(self, text, tabwidth):
        if False:
            return 10
        self.text = text
        self.tabwidth = tabwidth
        self.i = self.finished = 0
        self.blkopenline = self.indentedline = None

    def readline(self):
        if False:
            for i in range(10):
                print('nop')
        if self.finished:
            val = ''
        else:
            i = self.i = self.i + 1
            mark = repr(i) + '.0'
            if self.text.compare(mark, '>=', 'end'):
                val = ''
            else:
                val = self.text.get(mark, mark + ' lineend+1c')
        return val.encode(default_scintilla_encoding)

    def run(self):
        if False:
            while True:
                i = 10
        OPENERS = ('class', 'def', 'for', 'if', 'try', 'while')
        INDENT = tokenize.INDENT
        NAME = tokenize.NAME
        save_tabsize = tokenize.tabsize
        tokenize.tabsize = self.tabwidth
        try:
            try:
                for (typ, token, start, end, line) in tokenize.tokenize(self.readline):
                    if typ == NAME and token in OPENERS:
                        self.blkopenline = line
                    elif typ == INDENT and self.blkopenline:
                        self.indentedline = line
                        break
            except (tokenize.TokenError, IndentationError):
                pass
        finally:
            tokenize.tabsize = save_tabsize
        return (self.blkopenline, self.indentedline)