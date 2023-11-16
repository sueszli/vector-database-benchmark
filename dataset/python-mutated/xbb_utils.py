"""Utility code for graphical Xbbtools tool."""
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

class NotePad(tk.Toplevel):
    """Top level window for results (translations, BLAST searches...)."""

    def __init__(self, master=None):
        if False:
            i = 10
            return i + 15
        'Set up notepad window.'
        tk.Toplevel.__init__(self, master)
        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar)
        self.filemenu.add_command(label='Save', command=self.save)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Dismiss', command=self.destroy)
        self.menubar.add_cascade(label='File', menu=self.filemenu)
        self.configure(menu=self.menubar)
        self.yscroll = ttk.Scrollbar(self, orient='vertical')
        self.tid = tk.Text(self, width=88, yscrollcommand=self.yscroll.set)
        self.yscroll.configure(command=self.tid.yview)
        self.tid.pack(side='left', fill='both', expand=1)
        self.yscroll.pack(side='right', fill='y')

    def text_id(self):
        if False:
            print('Hello World!')
        'Get reference to notepad window.'
        return self.tid

    def insert(self, start, txt):
        if False:
            return 10
        'Add text to notepad window.'
        self.tid.insert(start, txt)

    def save(self):
        if False:
            for i in range(10):
                print('nop')
        'Save text from notepad to file.'
        filename = filedialog.asksaveasfilename()
        if filename:
            with open(filename, 'w') as fid:
                fid.write(self.tid.get(0.0, 'end'))