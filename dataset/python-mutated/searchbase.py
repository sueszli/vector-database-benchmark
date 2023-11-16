"""Define SearchDialogBase used by Search, Replace, and Grep dialogs."""
from tkinter import Toplevel
from tkinter.ttk import Frame, Entry, Label, Button, Checkbutton, Radiobutton
from tkinter.simpledialog import _setup_dialog

class SearchDialogBase:
    """Create most of a 3 or 4 row, 3 column search dialog.

    The left and wide middle column contain:
    1 or 2 labeled text entry lines (make_entry, create_entries);
    a row of standard Checkbuttons (make_frame, create_option_buttons),
    each of which corresponds to a search engine Variable;
    a row of dialog-specific Check/Radiobuttons (create_other_buttons).

    The narrow right column contains command buttons
    (make_button, create_command_buttons).
    These are bound to functions that execute the command.

    Except for command buttons, this base class is not limited to items
    common to all three subclasses.  Rather, it is the Find dialog minus
    the "Find Next" command, its execution function, and the
    default_command attribute needed in create_widgets. The other
    dialogs override attributes and methods, the latter to replace and
    add widgets.
    """
    title = 'Search Dialog'
    icon = 'Search'
    needwrapbutton = 1

    def __init__(self, root, engine):
        if False:
            for i in range(10):
                print('nop')
        'Initialize root, engine, and top attributes.\n\n        top (level widget): set in create_widgets() called from open().\n        frame: container for all widgets in dialog.\n        text (Text searched): set in open(), only used in subclasses().\n        ent (ry): created in make_entry() called from create_entry().\n        row (of grid): 0 in create_widgets(), +1 in make_entry/frame().\n        default_command: set in subclasses, used in create_widgets().\n\n        title (of dialog): class attribute, override in subclasses.\n        icon (of dialog): ditto, use unclear if cannot minimize dialog.\n        '
        self.root = root
        self.bell = root.bell
        self.engine = engine
        self.top = None

    def open(self, text, searchphrase=None):
        if False:
            i = 10
            return i + 15
        'Make dialog visible on top of others and ready to use.'
        self.text = text
        if not self.top:
            self.create_widgets()
        else:
            self.top.deiconify()
            self.top.tkraise()
        self.top.transient(text.winfo_toplevel())
        if searchphrase:
            self.ent.delete(0, 'end')
            self.ent.insert('end', searchphrase)
        self.ent.focus_set()
        self.ent.selection_range(0, 'end')
        self.ent.icursor(0)
        self.top.grab_set()

    def close(self, event=None):
        if False:
            while True:
                i = 10
        'Put dialog away for later use.'
        if self.top:
            self.top.grab_release()
            self.top.transient('')
            self.top.withdraw()

    def create_widgets(self):
        if False:
            while True:
                i = 10
        'Create basic 3 row x 3 col search (find) dialog.\n\n        Other dialogs override subsidiary create_x methods as needed.\n        Replace and Find-in-Files add another entry row.\n        '
        top = Toplevel(self.root)
        top.bind('<Return>', self.default_command)
        top.bind('<Escape>', self.close)
        top.protocol('WM_DELETE_WINDOW', self.close)
        top.wm_title(self.title)
        top.wm_iconname(self.icon)
        _setup_dialog(top)
        self.top = top
        self.frame = Frame(top, padding='5px')
        self.frame.grid(sticky='nwes')
        top.grid_columnconfigure(0, weight=100)
        top.grid_rowconfigure(0, weight=100)
        self.row = 0
        self.frame.grid_columnconfigure(0, pad=2, weight=0)
        self.frame.grid_columnconfigure(1, pad=2, minsize=100, weight=100)
        self.create_entries()
        self.create_option_buttons()
        self.create_other_buttons()
        self.create_command_buttons()

    def make_entry(self, label_text, var):
        if False:
            for i in range(10):
                print('nop')
        'Return (entry, label), .\n\n        entry - gridded labeled Entry for text entry.\n        label - Label widget, returned for testing.\n        '
        label = Label(self.frame, text=label_text)
        label.grid(row=self.row, column=0, sticky='nw')
        entry = Entry(self.frame, textvariable=var, exportselection=0)
        entry.grid(row=self.row, column=1, sticky='nwe')
        self.row = self.row + 1
        return (entry, label)

    def create_entries(self):
        if False:
            print('Hello World!')
        'Create one or more entry lines with make_entry.'
        self.ent = self.make_entry('Find:', self.engine.patvar)[0]

    def make_frame(self, labeltext=None):
        if False:
            return 10
        'Return (frame, label).\n\n        frame - gridded labeled Frame for option or other buttons.\n        label - Label widget, returned for testing.\n        '
        if labeltext:
            label = Label(self.frame, text=labeltext)
            label.grid(row=self.row, column=0, sticky='nw')
        else:
            label = ''
        frame = Frame(self.frame)
        frame.grid(row=self.row, column=1, columnspan=1, sticky='nwe')
        self.row = self.row + 1
        return (frame, label)

    def create_option_buttons(self):
        if False:
            while True:
                i = 10
        'Return (filled frame, options) for testing.\n\n        Options is a list of searchengine booleanvar, label pairs.\n        A gridded frame from make_frame is filled with a Checkbutton\n        for each pair, bound to the var, with the corresponding label.\n        '
        frame = self.make_frame('Options')[0]
        engine = self.engine
        options = [(engine.revar, 'Regular expression'), (engine.casevar, 'Match case'), (engine.wordvar, 'Whole word')]
        if self.needwrapbutton:
            options.append((engine.wrapvar, 'Wrap around'))
        for (var, label) in options:
            btn = Checkbutton(frame, variable=var, text=label)
            btn.pack(side='left', fill='both')
        return (frame, options)

    def create_other_buttons(self):
        if False:
            i = 10
            return i + 15
        'Return (frame, others) for testing.\n\n        Others is a list of value, label pairs.\n        A gridded frame from make_frame is filled with radio buttons.\n        '
        frame = self.make_frame('Direction')[0]
        var = self.engine.backvar
        others = [(1, 'Up'), (0, 'Down')]
        for (val, label) in others:
            btn = Radiobutton(frame, variable=var, value=val, text=label)
            btn.pack(side='left', fill='both')
        return (frame, others)

    def make_button(self, label, command, isdef=0):
        if False:
            i = 10
            return i + 15
        'Return command button gridded in command frame.'
        b = Button(self.buttonframe, text=label, command=command, default=isdef and 'active' or 'normal')
        (cols, rows) = self.buttonframe.grid_size()
        b.grid(pady=1, row=rows, column=0, sticky='ew')
        self.buttonframe.grid(rowspan=rows + 1)
        return b

    def create_command_buttons(self):
        if False:
            print('Hello World!')
        'Place buttons in vertical command frame gridded on right.'
        f = self.buttonframe = Frame(self.frame)
        f.grid(row=0, column=2, padx=2, pady=2, ipadx=2, ipady=2)
        b = self.make_button('Close', self.close)
        b.lower()

class _searchbase(SearchDialogBase):
    """Create auto-opening dialog with no text connection."""

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        import re
        from idlelib import searchengine
        self.root = parent
        self.engine = searchengine.get(parent)
        self.create_widgets()
        print(parent.geometry())
        (width, height, x, y) = list(map(int, re.split('[x+]', parent.geometry())))
        self.top.geometry('+%d+%d' % (x + 40, y + 175))

    def default_command(self, dummy):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_searchbase', verbosity=2, exit=False)
    from idlelib.idle_test.htest import run
    run(_searchbase)