"""Contains the WidgetPropertiesDialog class."""
__all__ = ['WidgetPropertiesDialog']
import Pmw
import tkinter as tk

class WidgetPropertiesDialog(tk.Toplevel):
    """Class to open dialogs to adjust widget properties."""

    def __init__(self, propertyDict, propertyList=None, parent=None, title='Widget Properties'):
        if False:
            return 10
        'Initialize a dialog.\n        Arguments:\n            propertyDict -- a dictionary of properties to be edited\n            parent -- a parent window (the application window)\n            title -- the dialog title\n        '
        self.propertyDict = propertyDict
        self.propertyList = propertyList
        if self.propertyList is None:
            self.propertyList = sorted(self.propertyDict)
        if not parent:
            parent = tk._default_root
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        if title:
            self.title(title)
        self.parent = parent
        self.modifiedDict = {}
        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        self.grab_set()
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.geometry('+%d+%d' % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self.initial_focus.focus_set()
        self.wait_window(self)

    def destroy(self):
        if False:
            print('Hello World!')
        'Destroy the window'
        self.propertyDict = {}
        self.initial_focus = None
        for balloon in self.balloonList:
            balloon.withdraw()
        tk.Toplevel.destroy(self)

    def body(self, master):
        if False:
            return 10
        'create dialog body.\n        return entry that should have initial focus.\n        This method should be overridden, and is called\n        by the __init__ method.\n        '
        count = 0
        entryList = []
        self.balloonList = []
        for property in self.propertyList:
            propertySet = self.propertyDict[property]
            widget = propertySet.get('widget', None)
            initialvalue = widget[property]
            entryType = propertySet.get('type', 'real')
            fAllowNone = propertySet.get('fNone', 0)
            helpString = propertySet.get('help', None)
            label = tk.Label(master, text=property, justify=tk.LEFT)
            label.grid(row=count, column=0, padx=5, sticky=tk.W)
            entry = Pmw.EntryField(master, entry_justify=tk.RIGHT)
            entry.grid(row=count, column=1, padx=5, sticky=tk.W + tk.E)
            if initialvalue is None:
                entry.insert(0, 'None')
            else:
                entry.insert(0, initialvalue)
            balloon = Pmw.Balloon(state='balloon')
            self.balloonList.append(balloon)
            if helpString is None:
                if fAllowNone:
                    extra = ' or None'
                else:
                    extra = ''
            if entryType == 'real':
                if fAllowNone:
                    entry['validate'] = {'validator': self.realOrNone}
                else:
                    entry['validate'] = {'validator': 'real'}
                if helpString is None:
                    helpString = 'Enter a floating point number' + extra + '.'
            elif entryType == 'integer':
                if fAllowNone:
                    entry['validate'] = {'validator': self.intOrNone}
                else:
                    entry['validate'] = {'validator': 'integer'}
                if helpString is None:
                    helpString = f'Enter an integer{extra}.'
            elif helpString is None:
                helpString = f'Enter a string{extra}.'
            balloon.bind(entry, helpString)
            modifiedCallback = lambda f=self.modified, w=widget, e=entry, p=property, t=entryType, fn=fAllowNone: f(w, e, p, t, fn)
            entry['modifiedcommand'] = modifiedCallback
            entryList.append(entry)
            count += 1
        if len(entryList) > 0:
            entry = entryList[0]
            entry.select_range(0, tk.END)
            return entryList[0]
        else:
            return self

    def modified(self, widget, entry, property, type, fNone):
        if False:
            for i in range(10):
                print('nop')
        self.modifiedDict[property] = (widget, entry, type, fNone)

    def buttonbox(self):
        if False:
            i = 10
            return i + 15
        'add standard button box buttons.\n        '
        box = tk.Frame(self)
        w = tk.Button(box, text='OK', width=10, command=self.ok)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text='Cancel', width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.cancel)
        box.pack()

    def realOrNone(self, val):
        if False:
            i = 10
            return i + 15
        val = val.lower()
        if 'none'.find(val) != -1:
            if val == 'none':
                return Pmw.OK
            else:
                return Pmw.PARTIAL
        return Pmw.realvalidator(val)

    def intOrNone(self, val):
        if False:
            print('Hello World!')
        val = val.lower()
        if 'none'.find(val) != -1:
            if val == 'none':
                return Pmw.OK
            else:
                return Pmw.PARTIAL
        return Pmw.integervalidator(val)

    def ok(self, event=None):
        if False:
            i = 10
            return i + 15
        self.withdraw()
        self.update_idletasks()
        self.validateChanges()
        self.apply()
        self.cancel()

    def cancel(self, event=None):
        if False:
            return 10
        self.parent.focus_set()
        self.destroy()

    def validateChanges(self):
        if False:
            i = 10
            return i + 15
        for property in self.modifiedDict:
            tuple = self.modifiedDict[property]
            widget = tuple[0]
            entry = tuple[1]
            type = tuple[2]
            fNone = tuple[3]
            value = entry.get()
            lValue = value.lower()
            if 'none'.find(lValue) != -1:
                if fNone and lValue == 'none':
                    widget[property] = None
            else:
                if type == 'real':
                    value = float(value)
                elif type == 'integer':
                    value = int(value)
                widget[property] = value

    def apply(self):
        if False:
            print('Hello World!')
        'process the data\n\n        This method is called automatically to process the data, *after*\n        the dialog is destroyed. By default, it does nothing.\n        '