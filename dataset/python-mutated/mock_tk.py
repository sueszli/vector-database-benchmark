"""Classes that replace tkinter gui objects used by an object being tested.

A gui object is anything with a master or parent parameter, which is
typically required in spite of what the doc strings say.
"""
import re
from _tkinter import TclError

class Event:
    """Minimal mock with attributes for testing event handlers.

    This is not a gui object, but is used as an argument for callbacks
    that access attributes of the event passed. If a callback ignores
    the event, other than the fact that is happened, pass 'event'.

    Keyboard, mouse, window, and other sources generate Event instances.
    Event instances have the following attributes: serial (number of
    event), time (of event), type (of event as number), widget (in which
    event occurred), and x,y (position of mouse). There are other
    attributes for specific events, such as keycode for key events.
    tkinter.Event.__doc__ has more but is still not complete.
    """

    def __init__(self, **kwds):
        if False:
            return 10
        'Create event with attributes needed for test'
        self.__dict__.update(kwds)

class Var:
    """Use for String/Int/BooleanVar: incomplete"""

    def __init__(self, master=None, value=None, name=None):
        if False:
            return 10
        self.master = master
        self.value = value
        self.name = name

    def set(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def get(self):
        if False:
            while True:
                i = 10
        return self.value

class Mbox_func:
    """Generic mock for messagebox functions, which all have the same signature.

    Instead of displaying a message box, the mock's call method saves the
    arguments as instance attributes, which test functions can then examine.
    The test can set the result returned to ask function
    """

    def __init__(self, result=None):
        if False:
            i = 10
            return i + 15
        self.result = result

    def __call__(self, title, message, *args, **kwds):
        if False:
            return 10
        self.title = title
        self.message = message
        self.args = args
        self.kwds = kwds
        return self.result

class Mbox:
    """Mock for tkinter.messagebox with an Mbox_func for each function.

    Example usage in test_module.py for testing functions in module.py:
    ---
from idlelib.idle_test.mock_tk import Mbox
import module

orig_mbox = module.messagebox
showerror = Mbox.showerror  # example, for attribute access in test methods

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        module.messagebox = Mbox

    @classmethod
    def tearDownClass(cls):
        module.messagebox = orig_mbox
    ---
    For 'ask' functions, set func.result return value before calling the method
    that uses the message function. When messagebox functions are the
    only GUI calls in a method, this replacement makes the method GUI-free,
    """
    askokcancel = Mbox_func()
    askquestion = Mbox_func()
    askretrycancel = Mbox_func()
    askyesno = Mbox_func()
    askyesnocancel = Mbox_func()
    showerror = Mbox_func()
    showinfo = Mbox_func()
    showwarning = Mbox_func()

class Text:
    """A semi-functional non-gui replacement for tkinter.Text text editors.

    The mock's data model is that a text is a list of 
-terminated lines.
    The mock adds an empty string at  the beginning of the list so that the
    index of actual lines start at 1, as with Tk. The methods never see this.
    Tk initializes files with a terminal 
 that cannot be deleted. It is
    invisible in the sense that one cannot move the cursor beyond it.

    This class is only tested (and valid) with strings of ascii chars.
    For testing, we are not concerned with Tk Text's treatment of,
    for instance, 0-width characters or character + accent.
   """

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            return 10
        'Initialize mock, non-gui, text-only Text widget.\n\n        At present, all args are ignored. Almost all affect visual behavior.\n        There are just a few Text-only options that affect text behavior.\n        '
        self.data = ['', '\n']

    def index(self, index):
        if False:
            print('Hello World!')
        'Return string version of index decoded according to current text.'
        return '%s.%s' % self._decode(index, endflag=1)

    def _decode(self, index, endflag=0):
        if False:
            for i in range(10):
                print('nop')
        "Return a (line, char) tuple of int indexes into self.data.\n\n        This implements .index without converting the result back to a string.\n        The result is constrained by the number of lines and linelengths of\n        self.data. For many indexes, the result is initially (1, 0).\n\n        The input index may have any of several possible forms:\n        * line.char float: converted to 'line.char' string;\n        * 'line.char' string, where line and char are decimal integers;\n        * 'line.char lineend', where lineend='lineend' (and char is ignored);\n        * 'line.end', where end='end' (same as above);\n        * 'insert', the positions before terminal \n;\n        * 'end', whose meaning depends on the endflag passed to ._endex.\n        * 'sel.first' or 'sel.last', where sel is a tag -- not implemented.\n        "
        if isinstance(index, (float, bytes)):
            index = str(index)
        try:
            index = index.lower()
        except AttributeError:
            raise TclError('bad text index "%s"' % index) from None
        lastline = len(self.data) - 1
        if index == 'insert':
            return (lastline, len(self.data[lastline]) - 1)
        elif index == 'end':
            return self._endex(endflag)
        (line, char) = index.split('.')
        line = int(line)
        if line < 1:
            return (1, 0)
        elif line > lastline:
            return self._endex(endflag)
        linelength = len(self.data[line]) - 1
        if char.endswith(' lineend') or char == 'end':
            return (line, linelength)
        if (m := re.fullmatch('end-(\\d*)c', char, re.A)):
            return (line, linelength - int(m.group(1)))
        char = int(char)
        if char < 0:
            char = 0
        elif char > linelength:
            char = linelength
        return (line, char)

    def _endex(self, endflag):
        if False:
            print('Hello World!')
        "Return position for 'end' or line overflow corresponding to endflag.\n\n       -1: position before terminal \n; for .insert(), .delete\n       0: position after terminal \n; for .get, .delete index 1\n       1: same viewed as beginning of non-existent next line (for .index)\n       "
        n = len(self.data)
        if endflag == 1:
            return (n, 0)
        else:
            n -= 1
            return (n, len(self.data[n]) + endflag)

    def insert(self, index, chars):
        if False:
            print('Hello World!')
        'Insert chars before the character at index.'
        if not chars:
            return
        chars = chars.splitlines(True)
        if chars[-1][-1] == '\n':
            chars.append('')
        (line, char) = self._decode(index, -1)
        before = self.data[line][:char]
        after = self.data[line][char:]
        self.data[line] = before + chars[0]
        self.data[line + 1:line + 1] = chars[1:]
        self.data[line + len(chars) - 1] += after

    def get(self, index1, index2=None):
        if False:
            for i in range(10):
                print('nop')
        "Return slice from index1 to index2 (default is 'index1+1')."
        (startline, startchar) = self._decode(index1)
        if index2 is None:
            (endline, endchar) = (startline, startchar + 1)
        else:
            (endline, endchar) = self._decode(index2)
        if startline == endline:
            return self.data[startline][startchar:endchar]
        else:
            lines = [self.data[startline][startchar:]]
            for i in range(startline + 1, endline):
                lines.append(self.data[i])
            lines.append(self.data[endline][:endchar])
            return ''.join(lines)

    def delete(self, index1, index2=None):
        if False:
            return 10
        "Delete slice from index1 to index2 (default is 'index1+1').\n\n        Adjust default index2 ('index+1) for line ends.\n        Do not delete the terminal \n at the very end of self.data ([-1][-1]).\n        "
        (startline, startchar) = self._decode(index1, -1)
        if index2 is None:
            if startchar < len(self.data[startline]) - 1:
                (endline, endchar) = (startline, startchar + 1)
            elif startline < len(self.data) - 1:
                (endline, endchar) = (startline + 1, 0)
            else:
                return
        else:
            (endline, endchar) = self._decode(index2, -1)
        if startline == endline and startchar < endchar:
            self.data[startline] = self.data[startline][:startchar] + self.data[startline][endchar:]
        elif startline < endline:
            self.data[startline] = self.data[startline][:startchar] + self.data[endline][endchar:]
            startline += 1
            for i in range(startline, endline + 1):
                del self.data[startline]

    def compare(self, index1, op, index2):
        if False:
            print('Hello World!')
        (line1, char1) = self._decode(index1)
        (line2, char2) = self._decode(index2)
        if op == '<':
            return line1 < line2 or (line1 == line2 and char1 < char2)
        elif op == '<=':
            return line1 < line2 or (line1 == line2 and char1 <= char2)
        elif op == '>':
            return line1 > line2 or (line1 == line2 and char1 > char2)
        elif op == '>=':
            return line1 > line2 or (line1 == line2 and char1 >= char2)
        elif op == '==':
            return line1 == line2 and char1 == char2
        elif op == '!=':
            return line1 != line2 or char1 != char2
        else:
            raise TclError('bad comparison operator "%s": must be <, <=, ==, >=, >, or !=' % op)

    def mark_set(self, name, index):
        if False:
            print('Hello World!')
        'Set mark *name* before the character at index.'
        pass

    def mark_unset(self, *markNames):
        if False:
            print('Hello World!')
        'Delete all marks in markNames.'

    def tag_remove(self, tagName, index1, index2=None):
        if False:
            while True:
                i = 10
        'Remove tag tagName from all characters between index1 and index2.'
        pass

    def scan_dragto(self, x, y):
        if False:
            while True:
                i = 10
        'Adjust the view of the text according to scan_mark'

    def scan_mark(self, x, y):
        if False:
            return 10
        'Remember the current X, Y coordinates.'

    def see(self, index):
        if False:
            return 10
        'Scroll screen to make the character at INDEX is visible.'
        pass

    def bind(sequence=None, func=None, add=None):
        if False:
            print('Hello World!')
        'Bind to this widget at event sequence a call to function func.'
        pass

class Entry:
    """Mock for tkinter.Entry."""

    def focus_set(self):
        if False:
            print('Hello World!')
        pass