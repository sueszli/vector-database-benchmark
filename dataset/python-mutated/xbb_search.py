"""Search code for graphical Xbbtools tool."""
import re
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import colorchooser
from Bio.Data.IUPACData import ambiguous_dna_values
from Bio.Seq import reverse_complement
import xbb_widget

class DNAsearch:
    """Class to search a DNA sequence."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Set up the alphabet.'
        self.init_alphabet()
        self.sequence = ''

    def init_alphabet(self):
        if False:
            print('Hello World!')
        'Expand alphabet values for ambiguous codes.'
        self.alphabet = ambiguous_dna_values
        other = ''.join(self.alphabet)
        self.alphabet['N'] = self.alphabet['N'] + other
        for key in self.alphabet:
            if key == 'N':
                continue
            if key in self.alphabet[key]:
                continue
            self.alphabet[key] = self.alphabet[key] + key

    def SetSeq(self, seq):
        if False:
            return 10
        'Set sequence.'
        self.sequence = seq

    def SetPattern(self, pattern):
        if False:
            return 10
        'Convert search pattern to regular expression.'
        self.pattern = pattern
        self.rx_pattern = self.IUPAC2regex(pattern)
        self.rx = re.compile(self.rx_pattern)

    def IUPAC2regex(self, s):
        if False:
            return 10
        'Translate search text into pattern.'
        rx = ''
        for i in s:
            r = self.alphabet.get(i, i)
            if len(r) > 1:
                rx = f'{rx}[{r}]'
            else:
                rx += r
        return rx

    def _Search(self, start=0):
        if False:
            for i in range(10):
                print('nop')
        'Search and return MatchObject (PRIVAT).'
        pos = self.rx.search(self.sequence, start)
        return pos

    def Search(self, start=0):
        if False:
            while True:
                i = 10
        'Search for query sequence and return position.'
        pos = self.rx.search(self.sequence, start)
        if pos:
            return pos.start()
        else:
            return -1

    def SearchAll(self):
        if False:
            return 10
        'Search the whole sequence.'
        pos = -1
        positions = []
        while True:
            m = self._Search(pos + 1)
            if not m:
                break
            pos = m.start()
            if pos == -1:
                break
            positions.append(pos)
        return positions

class XDNAsearch(tk.Toplevel, DNAsearch):
    """Graphical tools to perform the DNA search."""

    def __init__(self, seq='', master=None, highlight=0):
        if False:
            while True:
                i = 10
        'Initialize the search GUI.'
        DNAsearch.__init__(self)
        self.master = master
        self.highlight = highlight
        self.colors = []
        self.init_graphics()
        self.sequence = seq
        self.cur_pos = 0

    def init_graphics(self):
        if False:
            for i in range(10):
                print('nop')
        'Build the search window.'
        tk.Toplevel.__init__(self, self.master)
        self.frame = ttk.Frame(self)
        self.frame.pack(fill=tk.BOTH, expand=1)
        self.search_entry = ttk.Entry(self.frame)
        self.search_entry.pack(fill=tk.BOTH, expand=1)
        f2 = ttk.Frame(self.frame)
        f2.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        f = f2
        self.forward = ttk.Button(f, text='Search +', command=self.do_search)
        self.forward.pack(side=tk.LEFT)
        self.forward = ttk.Button(f, text='Search -', command=lambda x=self.do_search: x(other_strand=1))
        self.forward.pack(side=tk.LEFT)
        self.cancel = ttk.Button(f, text='Cancel', command=self.exit)
        self.cancel.pack(side=tk.LEFT)
        self.current_color = 'cyan'
        self.colorb = ttk.Button(f, text='Color', command=self.change_color)
        self.colorb.pack(side=tk.LEFT)
        self.config_color(self.current_color)

    def config_color(self, color=None):
        if False:
            for i in range(10):
                print('nop')
        'Set color for found sequence tag.'
        if not self.highlight:
            return
        if not color:
            color = colorchooser.askcolor()[1]
            if not color:
                color = 'cyan'
        self.current_color = color
        self.current_tag = f'searched_{self.current_color}'
        self.master.tag_config(self.current_tag, background=self.current_color)
        self.master.tag_config(self.current_tag + 'R', background=self.current_color, underline=1)
        self.colors.append(color)

    def change_color(self):
        if False:
            for i in range(10):
                print('nop')
        'Call back for color button.'
        self.config_color()

    def get_pattern(self):
        if False:
            while True:
                i = 10
        'Retrieve query sequence.'
        pattern = self.search_entry.get()
        return pattern

    def do_search(self, other_strand=0):
        if False:
            return 10
        'Start the search.'
        pattern = self.get_pattern()
        if other_strand:
            pattern = reverse_complement(pattern)
        self.SetPattern(pattern)
        pos = self.Search(self.cur_pos)
        self.cur_pos = pos + 1
        w = self.master
        if pos != -1:
            if self.highlight:
                (start, stop) = (pos, pos + len(self.pattern))
                if other_strand:
                    w.tag_add(self.current_tag + 'R', f'1.{start:d}', f'1.{stop}')
                else:
                    w.tag_add(self.current_tag, f'1.{start:d}', f'1.{stop}')
                w.see(f'1.{start:d}')

    def exit(self):
        if False:
            i = 10
            return i + 15
        'Clean up on exit.'
        for c in self.colors:
            self.master.tag_remove(f'searched_{c}', 1.0, tk.END)
            self.master.tag_remove(f'searched_{c}R', 1.0, tk.END)
        self.destroy()
        del self
if __name__ == '__main__':
    win = tk.Tk()
    xbbtools = xbb_widget.xbb_widget()
    seq = 'ATGGTGTGTGTGTACGATCGCCCCCCCCAGTCGATCGATGCATCGTA'
    xbbtools.insert_sequence(('Test_seq', seq))
    xbbtools.search()
    win.mainloop()