"""This modules handles dialog boxes.

It contains the following public symbols:

SimpleDialog -- A simple but flexible modal dialog box

Dialog -- a base class for dialogs

askinteger -- get an integer from the user

askfloat -- get a float from the user

askstring -- get a string from the user
"""
from tkinter import *
from tkinter import _get_temp_root, _destroy_temp_root
from tkinter import messagebox

class SimpleDialog:

    def __init__(self, master, text='', buttons=[], default=None, cancel=None, title=None, class_=None):
        if False:
            while True:
                i = 10
        if class_:
            self.root = Toplevel(master, class_=class_)
        else:
            self.root = Toplevel(master)
        if title:
            self.root.title(title)
            self.root.iconname(title)
        _setup_dialog(self.root)
        self.message = Message(self.root, text=text, aspect=400)
        self.message.pack(expand=1, fill=BOTH)
        self.frame = Frame(self.root)
        self.frame.pack()
        self.num = default
        self.cancel = cancel
        self.default = default
        self.root.bind('<Return>', self.return_event)
        for num in range(len(buttons)):
            s = buttons[num]
            b = Button(self.frame, text=s, command=lambda self=self, num=num: self.done(num))
            if num == default:
                b.config(relief=RIDGE, borderwidth=8)
            b.pack(side=LEFT, fill=BOTH, expand=1)
        self.root.protocol('WM_DELETE_WINDOW', self.wm_delete_window)
        self.root.transient(master)
        _place_window(self.root, master)

    def go(self):
        if False:
            print('Hello World!')
        self.root.wait_visibility()
        self.root.grab_set()
        self.root.mainloop()
        self.root.destroy()
        return self.num

    def return_event(self, event):
        if False:
            return 10
        if self.default is None:
            self.root.bell()
        else:
            self.done(self.default)

    def wm_delete_window(self):
        if False:
            return 10
        if self.cancel is None:
            self.root.bell()
        else:
            self.done(self.cancel)

    def done(self, num):
        if False:
            while True:
                i = 10
        self.num = num
        self.root.quit()

class Dialog(Toplevel):
    """Class to open dialogs.

    This class is intended as a base class for custom dialogs
    """

    def __init__(self, parent, title=None):
        if False:
            i = 10
            return i + 15
        'Initialize a dialog.\n\n        Arguments:\n\n            parent -- a parent window (the application window)\n\n            title -- the dialog title\n        '
        master = parent
        if master is None:
            master = _get_temp_root()
        Toplevel.__init__(self, master)
        self.withdraw()
        if parent is not None and parent.winfo_viewable():
            self.transient(parent)
        if title:
            self.title(title)
        _setup_dialog(self)
        self.parent = parent
        self.result = None
        body = Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        if self.initial_focus is None:
            self.initial_focus = self
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        _place_window(self, parent)
        self.initial_focus.focus_set()
        self.wait_visibility()
        self.grab_set()
        self.wait_window(self)

    def destroy(self):
        if False:
            i = 10
            return i + 15
        'Destroy the window'
        self.initial_focus = None
        Toplevel.destroy(self)
        _destroy_temp_root(self.master)

    def body(self, master):
        if False:
            while True:
                i = 10
        'create dialog body.\n\n        return widget that should have initial focus.\n        This method should be overridden, and is called\n        by the __init__ method.\n        '
        pass

    def buttonbox(self):
        if False:
            while True:
                i = 10
        'add standard button box.\n\n        override if you do not want the standard buttons\n        '
        box = Frame(self)
        w = Button(box, text='OK', width=10, command=self.ok, default=ACTIVE)
        w.pack(side=LEFT, padx=5, pady=5)
        w = Button(box, text='Cancel', width=10, command=self.cancel)
        w.pack(side=LEFT, padx=5, pady=5)
        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.cancel)
        box.pack()

    def ok(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.validate():
            self.initial_focus.focus_set()
            return
        self.withdraw()
        self.update_idletasks()
        try:
            self.apply()
        finally:
            self.cancel()

    def cancel(self, event=None):
        if False:
            i = 10
            return i + 15
        if self.parent is not None:
            self.parent.focus_set()
        self.destroy()

    def validate(self):
        if False:
            print('Hello World!')
        'validate the data\n\n        This method is called automatically to validate the data before the\n        dialog is destroyed. By default, it always validates OK.\n        '
        return 1

    def apply(self):
        if False:
            for i in range(10):
                print('nop')
        'process the data\n\n        This method is called automatically to process the data, *after*\n        the dialog is destroyed. By default, it does nothing.\n        '
        pass

def _place_window(w, parent=None):
    if False:
        while True:
            i = 10
    w.wm_withdraw()
    w.update_idletasks()
    minwidth = w.winfo_reqwidth()
    minheight = w.winfo_reqheight()
    maxwidth = w.winfo_vrootwidth()
    maxheight = w.winfo_vrootheight()
    if parent is not None and parent.winfo_ismapped():
        x = parent.winfo_rootx() + (parent.winfo_width() - minwidth) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - minheight) // 2
        vrootx = w.winfo_vrootx()
        vrooty = w.winfo_vrooty()
        x = min(x, vrootx + maxwidth - minwidth)
        x = max(x, vrootx)
        y = min(y, vrooty + maxheight - minheight)
        y = max(y, vrooty)
        if w._windowingsystem == 'aqua':
            y = max(y, 22)
    else:
        x = (w.winfo_screenwidth() - minwidth) // 2
        y = (w.winfo_screenheight() - minheight) // 2
    w.wm_maxsize(maxwidth, maxheight)
    w.wm_geometry('+%d+%d' % (x, y))
    w.wm_deiconify()

def _setup_dialog(w):
    if False:
        return 10
    if w._windowingsystem == 'aqua':
        w.tk.call('::tk::unsupported::MacWindowStyle', 'style', w, 'moveableModal', '')
    elif w._windowingsystem == 'x11':
        w.wm_attributes('-type', 'dialog')

class _QueryDialog(Dialog):

    def __init__(self, title, prompt, initialvalue=None, minvalue=None, maxvalue=None, parent=None):
        if False:
            for i in range(10):
                print('nop')
        self.prompt = prompt
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.initialvalue = initialvalue
        Dialog.__init__(self, parent, title)

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.entry = None
        Dialog.destroy(self)

    def body(self, master):
        if False:
            print('Hello World!')
        w = Label(master, text=self.prompt, justify=LEFT)
        w.grid(row=0, padx=5, sticky=W)
        self.entry = Entry(master, name='entry')
        self.entry.grid(row=1, padx=5, sticky=W + E)
        if self.initialvalue is not None:
            self.entry.insert(0, self.initialvalue)
            self.entry.select_range(0, END)
        return self.entry

    def validate(self):
        if False:
            print('Hello World!')
        try:
            result = self.getresult()
        except ValueError:
            messagebox.showwarning('Illegal value', self.errormessage + '\nPlease try again', parent=self)
            return 0
        if self.minvalue is not None and result < self.minvalue:
            messagebox.showwarning('Too small', 'The allowed minimum value is %s. Please try again.' % self.minvalue, parent=self)
            return 0
        if self.maxvalue is not None and result > self.maxvalue:
            messagebox.showwarning('Too large', 'The allowed maximum value is %s. Please try again.' % self.maxvalue, parent=self)
            return 0
        self.result = result
        return 1

class _QueryInteger(_QueryDialog):
    errormessage = 'Not an integer.'

    def getresult(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getint(self.entry.get())

def askinteger(title, prompt, **kw):
    if False:
        for i in range(10):
            print('nop')
    'get an integer from the user\n\n    Arguments:\n\n        title -- the dialog title\n        prompt -- the label text\n        **kw -- see SimpleDialog class\n\n    Return value is an integer\n    '
    d = _QueryInteger(title, prompt, **kw)
    return d.result

class _QueryFloat(_QueryDialog):
    errormessage = 'Not a floating point value.'

    def getresult(self):
        if False:
            for i in range(10):
                print('nop')
        return self.getdouble(self.entry.get())

def askfloat(title, prompt, **kw):
    if False:
        for i in range(10):
            print('nop')
    'get a float from the user\n\n    Arguments:\n\n        title -- the dialog title\n        prompt -- the label text\n        **kw -- see SimpleDialog class\n\n    Return value is a float\n    '
    d = _QueryFloat(title, prompt, **kw)
    return d.result

class _QueryString(_QueryDialog):

    def __init__(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        if 'show' in kw:
            self.__show = kw['show']
            del kw['show']
        else:
            self.__show = None
        _QueryDialog.__init__(self, *args, **kw)

    def body(self, master):
        if False:
            for i in range(10):
                print('nop')
        entry = _QueryDialog.body(self, master)
        if self.__show is not None:
            entry.configure(show=self.__show)
        return entry

    def getresult(self):
        if False:
            return 10
        return self.entry.get()

def askstring(title, prompt, **kw):
    if False:
        while True:
            i = 10
    'get a string from the user\n\n    Arguments:\n\n        title -- the dialog title\n        prompt -- the label text\n        **kw -- see SimpleDialog class\n\n    Return value is a string\n    '
    d = _QueryString(title, prompt, **kw)
    return d.result
if __name__ == '__main__':

    def test():
        if False:
            while True:
                i = 10
        root = Tk()

        def doit(root=root):
            if False:
                for i in range(10):
                    print('nop')
            d = SimpleDialog(root, text='This is a test dialog.  Would this have been an actual dialog, the buttons below would have been glowing in soft pink light.\nDo you believe this?', buttons=['Yes', 'No', 'Cancel'], default=0, cancel=2, title='Test Dialog')
            print(d.go())
            print(askinteger('Spam', 'Egg count', initialvalue=12 * 12))
            print(askfloat('Spam', 'Egg weight\n(in tons)', minvalue=1, maxvalue=100))
            print(askstring('Spam', 'Egg label'))
        t = Button(root, text='Test', command=doit)
        t.pack()
        q = Button(root, text='Quit', command=t.quit)
        q.pack()
        t.mainloop()
    test()