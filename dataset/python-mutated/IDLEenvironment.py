import string
import sys
import win32api
import win32con
import win32ui
from pywin import default_scintilla_encoding
from pywin.mfc.dialog import GetSimpleInput
wordchars = string.ascii_uppercase + string.ascii_lowercase + string.digits

class TextError(Exception):
    pass

class EmptyRange(Exception):
    pass

def GetIDLEModule(module):
    if False:
        print('Hello World!')
    try:
        modname = 'pywin.idle.' + module
        __import__(modname)
    except ImportError as details:
        msg = f"The IDLE extension '{module}' can not be located.\r\n\r\nPlease correct the installation and restart the application.\r\n\r\n{details}"
        win32ui.MessageBox(msg)
        return None
    mod = sys.modules[modname]
    mod.TclError = TextError
    return mod

def fast_readline(self):
    if False:
        for i in range(10):
            print('nop')
    if self.finished:
        val = ''
    else:
        if '_scint_lines' not in self.__dict__:
            self._scint_lines = self.text.edit.GetTextRange().split('\n')
        sl = self._scint_lines
        i = self.i = self.i + 1
        if i >= len(sl):
            val = ''
        else:
            val = sl[i] + '\n'
    return val.encode(default_scintilla_encoding)
try:
    GetIDLEModule('AutoIndent').IndentSearcher.readline = fast_readline
except AttributeError:
    pass

class IDLEEditorWindow:

    def __init__(self, edit):
        if False:
            return 10
        self.edit = edit
        self.text = TkText(edit)
        self.extensions = {}
        self.extension_menus = {}

    def close(self):
        if False:
            while True:
                i = 10
        self.edit = self.text = None
        self.extension_menus = None
        try:
            for ext in self.extensions.values():
                closer = getattr(ext, 'close', None)
                if closer is not None:
                    closer()
        finally:
            self.extensions = {}

    def IDLEExtension(self, extension):
        if False:
            print('Hello World!')
        ext = self.extensions.get(extension)
        if ext is not None:
            return ext
        mod = GetIDLEModule(extension)
        if mod is None:
            return None
        klass = getattr(mod, extension)
        ext = self.extensions[extension] = klass(self)
        events = [item for item in dir(klass) if item[-6:] == '_event']
        for event in events:
            name = '<<{}>>'.format(event[:-6].replace('_', '-'))
            self.edit.bindings.bind(name, getattr(ext, event))
        return ext

    def GetMenuItems(self, menu_name):
        if False:
            i = 10
            return i + 15
        bindings = self.edit.bindings
        ret = []
        for ext in self.extensions.values():
            menudefs = getattr(ext, 'menudefs', [])
            for (name, items) in menudefs:
                if name == menu_name:
                    for (text, event) in [item for item in items if item is not None]:
                        text = text.replace('&', '&&')
                        text = text.replace('_', '&')
                        ret.append((text, event))
        return ret

    def askinteger(self, caption, prompt, parent=None, initialvalue=0, minvalue=None, maxvalue=None):
        if False:
            print('Hello World!')
        while 1:
            rc = GetSimpleInput(prompt, str(initialvalue), caption)
            if rc is None:
                return 0
            err = None
            try:
                rc = int(rc)
            except ValueError:
                err = 'Please enter an integer'
            if not err and minvalue is not None and (rc < minvalue):
                err = f'Please enter an integer greater then or equal to {minvalue}'
            if not err and maxvalue is not None and (rc > maxvalue):
                err = f'Please enter an integer less then or equal to {maxvalue}'
            if err:
                win32ui.MessageBox(err, caption, win32con.MB_OK)
                continue
            return rc

    def askyesno(self, caption, prompt, parent=None):
        if False:
            print('Hello World!')
        return win32ui.MessageBox(prompt, caption, win32con.MB_YESNO) == win32con.IDYES

    def is_char_in_string(self, text_index):
        if False:
            for i in range(10):
                print('nop')
        text_index = self.text._getoffset(text_index)
        c = self.text.edit._GetColorizer()
        if c and c.GetStringStyle(text_index) is None:
            return 0
        return 1

    def get_selection_indices(self):
        if False:
            print('Hello World!')
        try:
            first = self.text.index('sel.first')
            last = self.text.index('sel.last')
            return (first, last)
        except TextError:
            return (None, None)

    def set_tabwidth(self, width):
        if False:
            while True:
                i = 10
        self.edit.SCISetTabWidth(width)

    def get_tabwidth(self):
        if False:
            return 10
        return self.edit.GetTabWidth()

class CallTips:

    def __init__(self, edit):
        if False:
            return 10
        self.edit = edit

    def showtip(self, tip_text):
        if False:
            i = 10
            return i + 15
        self.edit.SCICallTipShow(tip_text)

    def hidetip(self):
        if False:
            i = 10
            return i + 15
        self.edit.SCICallTipCancel()

def TkOffsetToIndex(offset, edit):
    if False:
        print('Hello World!')
    lineoff = 0
    offset = min(offset, edit.GetTextLength())
    line = edit.LineFromChar(offset)
    lineIndex = edit.LineIndex(line)
    return '%d.%d' % (line + 1, offset - lineIndex)

def _NextTok(str, pos):
    if False:
        for i in range(10):
            print('nop')
    end = len(str)
    if pos >= end:
        return (None, 0)
    while pos < end and str[pos] in string.whitespace:
        pos = pos + 1
    if str[pos] in '+-':
        return (str[pos], pos + 1)
    endPos = pos
    while endPos < end and str[endPos] in string.digits + '.':
        endPos = endPos + 1
    if pos != endPos:
        return (str[pos:endPos], endPos)
    endPos = pos
    while endPos < end and str[endPos] not in string.whitespace + string.digits + '+-':
        endPos = endPos + 1
    if pos != endPos:
        return (str[pos:endPos], endPos)
    return (None, 0)

def TkIndexToOffset(bm, edit, marks):
    if False:
        for i in range(10):
            print('nop')
    (base, nextTokPos) = _NextTok(bm, 0)
    if base is None:
        raise ValueError('Empty bookmark ID!')
    if base.find('.') > 0:
        try:
            (line, col) = base.split('.', 2)
            if col == 'first' or col == 'last':
                if line != 'sel':
                    raise ValueError('Tags arent here!')
                sel = edit.GetSel()
                if sel[0] == sel[1]:
                    raise EmptyRange
                if col == 'first':
                    pos = sel[0]
                else:
                    pos = sel[1]
            else:
                line = int(line) - 1
                if line > edit.GetLineCount():
                    pos = edit.GetTextLength() + 1
                else:
                    pos = edit.LineIndex(line)
                    if pos == -1:
                        pos = edit.GetTextLength()
                    pos = pos + int(col)
        except (ValueError, IndexError):
            raise ValueError("Unexpected literal in '%s'" % base)
    elif base == 'insert':
        pos = edit.GetSel()[0]
    elif base == 'end':
        pos = edit.GetTextLength()
        if pos and edit.SCIGetCharAt(pos - 1) != '\n':
            pos = pos + 1
    else:
        try:
            pos = marks[base]
        except KeyError:
            raise ValueError("Unsupported base offset or undefined mark '%s'" % base)
    while 1:
        (word, nextTokPos) = _NextTok(bm, nextTokPos)
        if word is None:
            break
        if word in ('+', '-'):
            (num, nextTokPos) = _NextTok(bm, nextTokPos)
            if num is None:
                raise ValueError('+/- operator needs 2 args')
            (what, nextTokPos) = _NextTok(bm, nextTokPos)
            if what is None:
                raise ValueError('+/- operator needs 2 args')
            if what[0] != 'c':
                raise ValueError('+/- only supports chars')
            if word == '+':
                pos = pos + int(num)
            else:
                pos = pos - int(num)
        elif word == 'wordstart':
            while pos > 0 and edit.SCIGetCharAt(pos - 1) in wordchars:
                pos = pos - 1
        elif word == 'wordend':
            end = edit.GetTextLength()
            while pos < end and edit.SCIGetCharAt(pos) in wordchars:
                pos = pos + 1
        elif word == 'linestart':
            while pos > 0 and edit.SCIGetCharAt(pos - 1) not in '\n\r':
                pos = pos - 1
        elif word == 'lineend':
            end = edit.GetTextLength()
            while pos < end and edit.SCIGetCharAt(pos) not in '\n\r':
                pos = pos + 1
        else:
            raise ValueError("Unsupported relative offset '%s'" % word)
    return max(pos, 0)

class TkText:

    def __init__(self, edit):
        if False:
            print('Hello World!')
        self.calltips = None
        self.edit = edit
        self.marks = {}

    def make_calltip_window(self):
        if False:
            print('Hello World!')
        if self.calltips is None:
            self.calltips = CallTips(self.edit)
        return self.calltips

    def _getoffset(self, index):
        if False:
            while True:
                i = 10
        return TkIndexToOffset(index, self.edit, self.marks)

    def _getindex(self, off):
        if False:
            print('Hello World!')
        return TkOffsetToIndex(off, self.edit)

    def _fix_indexes(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        while start > 0 and ord(self.edit.SCIGetCharAt(start)) & 192 == 128:
            start -= 1
        while end < self.edit.GetTextLength() and ord(self.edit.SCIGetCharAt(end)) & 192 == 128:
            end += 1
        if start > 0 and self.edit.SCIGetCharAt(start) == '\n' and (self.edit.SCIGetCharAt(start - 1) == '\r'):
            start = start - 1
        if end < self.edit.GetTextLength() and self.edit.SCIGetCharAt(end - 1) == '\r' and (self.edit.SCIGetCharAt(end) == '\n'):
            end = end + 1
        return (start, end)

    def bind(self, binding, handler):
        if False:
            print('Hello World!')
        self.edit.bindings.bind(binding, handler)

    def get(self, start, end=None):
        if False:
            i = 10
            return i + 15
        try:
            start = self._getoffset(start)
            if end is None:
                end = start + 1
            else:
                end = self._getoffset(end)
        except EmptyRange:
            return ''
        if end <= start:
            return ''
        max = self.edit.GetTextLength()
        checkEnd = 0
        if end > max:
            end = max
            checkEnd = 1
        (start, end) = self._fix_indexes(start, end)
        ret = self.edit.GetTextRange(start, end)
        if checkEnd and (not ret or ret[-1] != '\n'):
            ret = ret + '\n'
        return ret.replace('\r', '')

    def index(self, spec):
        if False:
            i = 10
            return i + 15
        try:
            return self._getindex(self._getoffset(spec))
        except EmptyRange:
            return ''

    def insert(self, pos, text):
        if False:
            while True:
                i = 10
        try:
            pos = self._getoffset(pos)
        except EmptyRange:
            raise TextError('Empty range')
        self.edit.SetSel((pos, pos))
        bits = text.split('\n')
        self.edit.SCIAddText(bits[0])
        for bit in bits[1:]:
            self.edit.SCINewline()
            self.edit.SCIAddText(bit)

    def delete(self, start, end=None):
        if False:
            return 10
        try:
            start = self._getoffset(start)
            if end is not None:
                end = self._getoffset(end)
        except EmptyRange:
            raise TextError('Empty range')
        if start == end:
            return
        if end is None:
            end = start + 1
        elif end < start:
            return
        if start == self.edit.GetTextLength():
            return
        old = self.edit.GetSel()[0]
        (start, end) = self._fix_indexes(start, end)
        self.edit.SetSel((start, end))
        self.edit.Clear()
        if old >= start and old < end:
            old = start
        elif old >= end:
            old = old - (end - start)
        self.edit.SetSel(old)

    def bell(self):
        if False:
            i = 10
            return i + 15
        win32api.MessageBeep()

    def see(self, pos):
        if False:
            for i in range(10):
                print('nop')
        pass

    def mark_set(self, name, pos):
        if False:
            while True:
                i = 10
        try:
            pos = self._getoffset(pos)
        except EmptyRange:
            raise TextError("Empty range '%s'" % pos)
        if name == 'insert':
            self.edit.SetSel(pos)
        else:
            self.marks[name] = pos

    def tag_add(self, name, start, end):
        if False:
            return 10
        if name != 'sel':
            raise ValueError('Only sel tag is supported')
        try:
            start = self._getoffset(start)
            end = self._getoffset(end)
        except EmptyRange:
            raise TextError('Empty range')
        self.edit.SetSel(start, end)

    def tag_remove(self, name, start, end):
        if False:
            while True:
                i = 10
        if name != 'sel' or start != '1.0' or end != 'end':
            raise ValueError('Cant remove this tag')
        self.edit.SetSel(self.edit.GetSel()[0])

    def compare(self, i1, op, i2):
        if False:
            i = 10
            return i + 15
        try:
            i1 = self._getoffset(i1)
        except EmptyRange:
            i1 = ''
        try:
            i2 = self._getoffset(i2)
        except EmptyRange:
            i2 = ''
        return eval('%d%s%d' % (i1, op, i2))

    def undo_block_start(self):
        if False:
            for i in range(10):
                print('nop')
        self.edit.SCIBeginUndoAction()

    def undo_block_stop(self):
        if False:
            print('Hello World!')
        self.edit.SCIEndUndoAction()

def TestCheck(index, edit, expected=None):
    if False:
        for i in range(10):
            print('nop')
    rc = TkIndexToOffset(index, edit, {})
    if rc != expected:
        print('ERROR: Index', index, ', expected', expected, 'but got', rc)

def TestGet(fr, to, t, expected):
    if False:
        return 10
    got = t.get(fr, to)
    if got != expected:
        print('ERROR: get({}, {}) expected {}, but got {}'.format(repr(fr), repr(to), repr(expected), repr(got)))

def test():
    if False:
        for i in range(10):
            print('nop')
    import pywin.framework.editor
    d = pywin.framework.editor.editorTemplate.OpenDocumentFile(None)
    e = d.GetFirstView()
    t = TkText(e)
    e.SCIAddText('hi there how\nare you today\r\nI hope you are well')
    e.SetSel((4, 4))
    skip = '\n\tTestCheck("insert", e, 4)\n\tTestCheck("insert wordstart", e, 3)\n\tTestCheck("insert wordend", e, 8)\n\tTestCheck("insert linestart", e, 0)\n\tTestCheck("insert lineend", e, 12)\n\tTestCheck("insert + 4 chars", e, 8)\n\tTestCheck("insert +4c", e, 8)\n\tTestCheck("insert - 2 chars", e, 2)\n\tTestCheck("insert -2c", e, 2)\n\tTestCheck("insert-2c", e, 2)\n\tTestCheck("insert-2 c", e, 2)\n\tTestCheck("insert- 2c", e, 2)\n\tTestCheck("1.1", e, 1)\n\tTestCheck("1.0", e, 0)\n\tTestCheck("2.0", e, 13)\n\ttry:\n\t\tTestCheck("sel.first", e, 0)\n\t\tprint("*** sel.first worked with an empty selection")\n\texcept TextError:\n\t\tpass\n\te.SetSel((4,5))\n\tTestCheck("sel.first- 2c", e, 2)\n\tTestCheck("sel.last- 2c", e, 3)\n\t'
    e.SetSel((4, 4))
    TestGet('insert lineend', 'insert lineend +1c', t, '\n')
    e.SetSel((20, 20))
    TestGet('insert lineend', 'insert lineend +1c', t, '\n')
    e.SetSel((35, 35))
    TestGet('insert lineend', 'insert lineend +1c', t, '\n')

class IDLEWrapper:

    def __init__(self, control):
        if False:
            i = 10
            return i + 15
        self.text = control

def IDLETest(extension):
    if False:
        print('Hello World!')
    import os
    import sys
    modname = 'pywin.idle.' + extension
    __import__(modname)
    mod = sys.modules[modname]
    mod.TclError = TextError
    klass = getattr(mod, extension)
    import pywin.framework.editor
    d = pywin.framework.editor.editorTemplate.OpenDocumentFile(None)
    v = d.GetFirstView()
    fname = os.path.splitext(__file__)[0] + '.py'
    v.SCIAddText(open(fname).read())
    d.SetModifiedFlag(0)
    r = klass(IDLEWrapper(TkText(v)))
    return r
if __name__ == '__main__':
    test()