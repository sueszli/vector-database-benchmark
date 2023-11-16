"""Help code for graphical Xbbtools tool."""
import tkinter as tk
from tkinter import scrolledtext

class xbbtools_help(tk.Toplevel):
    """Help window."""

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        'Make toplevel help window.'
        tk.Toplevel.__init__(self)
        self.tid = scrolledtext.ScrolledText(self)
        self.tid.pack(fill=tk.BOTH, expand=1)
        self.Styles()
        self.Show()

    def Styles(self):
        if False:
            return 10
        'Define text styles.'
        for c in ['red', 'blue', 'magenta', 'yellow', 'green', 'red4', 'green4', 'blue4']:
            self.tid.tag_configure(c, foreground=c)
        self.tid.tag_config('underline', underline=1)
        self.tid.tag_config('italic', font=('Courier', 6, 'italic'))
        self.tid.tag_config('bold', font=('Courier', 8, 'bold'))
        self.tid.tag_config('title', font=('Courier', 12, 'bold'))
        self.tid.tag_config('small', font=('Courier', 6, ''))
        self.tid.tag_config('highlight', background='gray')

    def Show(self):
        if False:
            for i in range(10):
                print('nop')
        'Display help text.'
        t = self.tid
        t.insert(tk.END, 'XBBtools Help\n', 'title')
        t.insert(tk.END, '\nCopyright 2001 by Thomas Sicheritz-Ponten.\nModified 2016 by Markus Piotrowski.\nAll rights reserved.\nThis code is part of the Biopython distribution and governed by its\nlicense.  Please see the LICENSE file that should have been included\nas part of this package.\n\n', 'italic')
        t.insert(tk.END, 'thomas@biopython.org\n\n', 'blue')
        t.insert(tk.END, '* Goto Field\n', 'bold')
        t.insert(tk.END, '\tinserting one position moves cursor to position\n')
        t.insert(tk.END, "\tinserting two positions, separated by ':' ")
        t.insert(tk.END, 'highlights', 'highlight')
        t.insert(tk.END, ' selected range\n')
        t.insert(tk.END, '\n')
        t.insert(tk.END, '* Search\n', 'bold')
        t.insert(tk.END, '\tambiguous dna values are\n')
        t.insert(tk.END, '\n                A: A\n                C: C\n                G: G\n                T: T\n                M: AC\n                R: AG\n                W: AT\n                S: CG\n                Y: CT\n                K: GT\n                V: ACG\n                H: ACT\n                D: AGT\n                B: CGT\n                X: GATC\n                N: GATC\n                ', 'small')
        t.insert(tk.END, '\n')
        t.insert(tk.END, '* BLAST\n', 'bold')
        t.insert(tk.END, "\tto use the 'Blast' option, you need to have the BLAST binaries installed\n\tand at least one database in BLAST format")