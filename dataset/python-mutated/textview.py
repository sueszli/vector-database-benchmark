"""Simple text browser for IDLE

"""
from tkinter import Toplevel, Text, TclError, HORIZONTAL, VERTICAL, NS, EW, NSEW, NONE, WORD, SUNKEN
from tkinter.ttk import Frame, Scrollbar, Button
from tkinter.messagebox import showerror
from idlelib.colorizer import color_config

class AutoHideScrollbar(Scrollbar):
    """A scrollbar that is automatically hidden when not needed.

    Only the grid geometry manager is supported.
    """

    def set(self, lo, hi):
        if False:
            i = 10
            return i + 15
        if float(lo) > 0.0 or float(hi) < 1.0:
            self.grid()
        else:
            self.grid_remove()
        super().set(lo, hi)

    def pack(self, **kwargs):
        if False:
            i = 10
            return i + 15
        raise TclError(f'{self.__class__.__name__} does not support "pack"')

    def place(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise TclError(f'{self.__class__.__name__} does not support "place"')

class ScrollableTextFrame(Frame):
    """Display text with scrollbar(s)."""

    def __init__(self, master, wrap=NONE, **kwargs):
        if False:
            while True:
                i = 10
        "Create a frame for Textview.\n\n        master - master widget for this frame\n        wrap - type of text wrapping to use ('word', 'char' or 'none')\n\n        All parameters except for 'wrap' are passed to Frame.__init__().\n\n        The Text widget is accessible via the 'text' attribute.\n\n        Note: Changing the wrapping mode of the text widget after\n        instantiation is not supported.\n        "
        super().__init__(master, **kwargs)
        text = self.text = Text(self, wrap=wrap)
        text.grid(row=0, column=0, sticky=NSEW)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.yscroll = AutoHideScrollbar(self, orient=VERTICAL, takefocus=False, command=text.yview)
        self.yscroll.grid(row=0, column=1, sticky=NS)
        text['yscrollcommand'] = self.yscroll.set
        if wrap == NONE:
            self.xscroll = AutoHideScrollbar(self, orient=HORIZONTAL, takefocus=False, command=text.xview)
            self.xscroll.grid(row=1, column=0, sticky=EW)
            text['xscrollcommand'] = self.xscroll.set
        else:
            self.xscroll = None

class ViewFrame(Frame):
    """Display TextFrame and Close button."""

    def __init__(self, parent, contents, wrap='word'):
        if False:
            print('Hello World!')
        'Create a frame for viewing text with a "Close" button.\n\n        parent - parent widget for this frame\n        contents - text to display\n        wrap - type of text wrapping to use (\'word\', \'char\' or \'none\')\n\n        The Text widget is accessible via the \'text\' attribute.\n        '
        super().__init__(parent)
        self.parent = parent
        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.ok)
        self.textframe = ScrollableTextFrame(self, relief=SUNKEN, height=700)
        text = self.text = self.textframe.text
        text.insert('1.0', contents)
        text.configure(wrap=wrap, highlightthickness=0, state='disabled')
        color_config(text)
        text.focus_set()
        self.button_ok = button_ok = Button(self, text='Close', command=self.ok, takefocus=False)
        self.textframe.pack(side='top', expand=True, fill='both')
        button_ok.pack(side='bottom')

    def ok(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        'Dismiss text viewer dialog.'
        self.parent.destroy()

class ViewWindow(Toplevel):
    """A simple text viewer dialog for IDLE."""

    def __init__(self, parent, title, contents, modal=True, wrap=WORD, *, _htest=False, _utest=False):
        if False:
            while True:
                i = 10
        "Show the given text in a scrollable window with a 'close' button.\n\n        If modal is left True, users cannot interact with other windows\n        until the textview window is closed.\n\n        parent - parent of this dialog\n        title - string which is title of popup dialog\n        contents - text to display in dialog\n        wrap - type of text wrapping to use ('word', 'char' or 'none')\n        _htest - bool; change box location when running htest.\n        _utest - bool; don't wait_window when running unittest.\n        "
        super().__init__(parent)
        self['borderwidth'] = 5
        x = parent.winfo_rootx() + 10
        y = parent.winfo_rooty() + (10 if not _htest else 100)
        self.geometry(f'=750x500+{x}+{y}')
        self.title(title)
        self.viewframe = ViewFrame(self, contents, wrap=wrap)
        self.protocol('WM_DELETE_WINDOW', self.ok)
        self.button_ok = button_ok = Button(self, text='Close', command=self.ok, takefocus=False)
        self.viewframe.pack(side='top', expand=True, fill='both')
        self.is_modal = modal
        if self.is_modal:
            self.transient(parent)
            self.grab_set()
            if not _utest:
                self.wait_window()

    def ok(self, event=None):
        if False:
            print('Hello World!')
        'Dismiss text viewer dialog.'
        if self.is_modal:
            self.grab_release()
        self.destroy()

def view_text(parent, title, contents, modal=True, wrap='word', _utest=False):
    if False:
        while True:
            i = 10
    "Create text viewer for given text.\n\n    parent - parent of this dialog\n    title - string which is the title of popup dialog\n    contents - text to display in this dialog\n    wrap - type of text wrapping to use ('word', 'char' or 'none')\n    modal - controls if users can interact with other windows while this\n            dialog is displayed\n    _utest - bool; controls wait_window on unittest\n    "
    return ViewWindow(parent, title, contents, modal, wrap=wrap, _utest=_utest)

def view_file(parent, title, filename, encoding, modal=True, wrap='word', _utest=False):
    if False:
        i = 10
        return i + 15
    'Create text viewer for text in filename.\n\n    Return error message if file cannot be read.  Otherwise calls view_text\n    with contents of the file.\n    '
    try:
        with open(filename, 'r', encoding=encoding) as file:
            contents = file.read()
    except OSError:
        showerror(title='File Load Error', message=f'Unable to load file {filename!r} .', parent=parent)
    except UnicodeDecodeError as err:
        showerror(title='Unicode Decode Error', message=str(err), parent=parent)
    else:
        return view_text(parent, title, contents, modal, wrap=wrap, _utest=_utest)
    return None
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_textview', verbosity=2, exit=False)
    from idlelib.idle_test.htest import run
    run(ViewWindow)