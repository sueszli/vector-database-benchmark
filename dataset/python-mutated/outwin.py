"""Editor window that can serve as an output file.
"""
import re
from tkinter import messagebox
from idlelib.editor import EditorWindow
file_line_pats = ['file "([^"]*)", line (\\d+)', '([^\\s]+)\\((\\d+)\\)', '^(\\s*\\S.*?):\\s*(\\d+):', '([^\\s]+):\\s*(\\d+):', '^\\s*(\\S.*?):\\s*(\\d+):']
file_line_progs = None

def compile_progs():
    if False:
        print('Hello World!')
    'Compile the patterns for matching to file name and line number.'
    global file_line_progs
    file_line_progs = [re.compile(pat, re.IGNORECASE) for pat in file_line_pats]

def file_line_helper(line):
    if False:
        return 10
    "Extract file name and line number from line of text.\n\n    Check if line of text contains one of the file/line patterns.\n    If it does and if the file and line are valid, return\n    a tuple of the file name and line number.  If it doesn't match\n    or if the file or line is invalid, return None.\n    "
    if not file_line_progs:
        compile_progs()
    for prog in file_line_progs:
        match = prog.search(line)
        if match:
            (filename, lineno) = match.group(1, 2)
            try:
                f = open(filename, 'r')
                f.close()
                break
            except OSError:
                continue
    else:
        return None
    try:
        return (filename, int(lineno))
    except TypeError:
        return None

class OutputWindow(EditorWindow):
    """An editor window that can serve as an output file.

    Also the future base class for the Python shell window.
    This class has no input facilities.

    Adds binding to open a file at a line to the text widget.
    """
    rmenu_specs = [('Cut', '<<cut>>', 'rmenu_check_cut'), ('Copy', '<<copy>>', 'rmenu_check_copy'), ('Paste', '<<paste>>', 'rmenu_check_paste'), (None, None, None), ('Go to file/line', '<<goto-file-line>>', None)]
    allow_code_context = False

    def __init__(self, *args):
        if False:
            print('Hello World!')
        EditorWindow.__init__(self, *args)
        self.text.bind('<<goto-file-line>>', self.goto_file_line)

    def ispythonsource(self, filename):
        if False:
            return 10
        'Python source is only part of output: do not colorize.'
        return False

    def short_title(self):
        if False:
            return 10
        'Customize EditorWindow title.'
        return 'Output'

    def maybesave(self):
        if False:
            for i in range(10):
                print('nop')
        'Customize EditorWindow to not display save file messagebox.'
        return 'yes' if self.get_saved() else 'no'

    def write(self, s, tags=(), mark='insert'):
        if False:
            i = 10
            return i + 15
        'Write text to text widget.\n\n        The text is inserted at the given index with the provided\n        tags.  The text widget is then scrolled to make it visible\n        and updated to display it, giving the effect of seeing each\n        line as it is added.\n\n        Args:\n            s: Text to insert into text widget.\n            tags: Tuple of tag strings to apply on the insert.\n            mark: Index for the insert.\n\n        Return:\n            Length of text inserted.\n        '
        assert isinstance(s, str)
        self.text.insert(mark, s, tags)
        self.text.see(mark)
        self.text.update()
        return len(s)

    def writelines(self, lines):
        if False:
            for i in range(10):
                print('nop')
        'Write each item in lines iterable.'
        for line in lines:
            self.write(line)

    def flush(self):
        if False:
            return 10
        'No flushing needed as write() directly writes to widget.'
        pass

    def showerror(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        messagebox.showerror(*args, **kwargs)

    def goto_file_line(self, event=None):
        if False:
            i = 10
            return i + 15
        'Handle request to open file/line.\n\n        If the selected or previous line in the output window\n        contains a file name and line number, then open that file\n        name in a new window and position on the line number.\n\n        Otherwise, display an error messagebox.\n        '
        line = self.text.get('insert linestart', 'insert lineend')
        result = file_line_helper(line)
        if not result:
            line = self.text.get('insert -1line linestart', 'insert -1line lineend')
            result = file_line_helper(line)
            if not result:
                self.showerror('No special line', "The line you point at doesn't look like a valid file name followed by a line number.", parent=self.text)
                return
        (filename, lineno) = result
        self.flist.gotofileline(filename, lineno)

class OnDemandOutputWindow:
    tagdefs = {'stdout': {'foreground': 'blue'}, 'stderr': {'foreground': '#007700'}}

    def __init__(self, flist):
        if False:
            while True:
                i = 10
        self.flist = flist
        self.owin = None

    def write(self, s, tags, mark):
        if False:
            i = 10
            return i + 15
        if not self.owin:
            self.setup()
        self.owin.write(s, tags, mark)

    def setup(self):
        if False:
            while True:
                i = 10
        self.owin = owin = OutputWindow(self.flist)
        text = owin.text
        for (tag, cnf) in self.tagdefs.items():
            if cnf:
                text.tag_configure(tag, **cnf)
        text.tag_raise('sel')
        self.write = self.owin.write
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_outwin', verbosity=2, exit=False)