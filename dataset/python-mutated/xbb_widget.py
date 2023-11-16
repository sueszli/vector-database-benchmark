"""Widget code for graphical Xbbtools tool."""
import re
import sys
import time
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from Bio.Data import CodonTable
from Bio.SeqIO.FastaIO import SimpleFastaParser
from xbb_utils import NotePad
from xbb_translations import xbb_translations
from xbb_blast import BlastIt
from xbb_search import XDNAsearch
from xbb_help import xbbtools_help

class xbb_widget:
    """Main XBBtools window."""

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        'Set up main window.'
        self.is_a_master = parent is None
        self.parent = parent
        self.init_variables()
        self.main_frame = ttk.Frame(parent)
        if not parent:
            self.parent = self.main_frame.master
            self.parent.option_add('*tearOff', 0)
        self.main_frame.pack(fill='both', expand=1)
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(fill='both', expand=1)
        self.create_menu(self.info_frame)
        self.create_seqinfo(self.info_frame)
        self.seq_frame = ttk.Frame(self.main_frame)
        self.seq_frame.pack(fill='both', expand=1)
        self.create_buttons(self.seq_frame)
        self.create_seqfield(self.seq_frame)
        self.create_bindings()
        self.blastit = 'xbb_blast.py'

    def init_variables(self):
        if False:
            i = 10
            return i + 15
        'Set up some standard values.'
        self.seqwidth = 60
        self.translation_tables = {}
        for (i, table) in CodonTable.unambiguous_dna_by_id.items():
            self.translation_tables[table.names[0]] = i
        self.translator = xbb_translations()

    def create_menu(self, parent):
        if False:
            i = 10
            return i + 15
        'Create the main menu bar.'
        self.menubar = tk.Menu(self.main_frame)
        self.file_menu = tk.Menu(self.menubar)
        menu = self.file_menu
        menu.add_command(label='Open', command=self.open)
        menu.add_command(label='Exit', command=self.exit)
        self.menubar.add_cascade(label='File', menu=self.file_menu)
        self.edit_menu = tk.Menu(self.menubar)
        menu = self.edit_menu
        menu.add_command(label='Complement', command=self.complement)
        menu.add_command(label='Antiparallel', command=self.antiparallel)
        menu.add_command(label='Reverse', command=self.reverse)
        menu.add_command(label='Fix sequence', command=self.fix_sequence)
        menu.add_command(label='Search', command=self.search)
        self.menubar.add_cascade(label='Edit', menu=self.edit_menu)
        self.translation_menu = tk.Menu(self.menubar)
        self.gencode_menu = tk.Menu(self.translation_menu)
        self.frame_menu = tk.Menu(self.translation_menu)
        menu = self.translation_menu
        menu.add_cascade(label='Genetic Codes', menu=self.gencode_menu)
        menu.add_cascade(label='Frame', menu=self.frame_menu)
        menu.add_separator()
        menu.add_command(label='Single frame translation', command=self.translate)
        menu.add_command(label='Three frame translation (+)', command=lambda : self.gcframe(direction='plus'))
        menu.add_command(label='Three frame translation (-)', command=lambda : self.gcframe(direction='minus'))
        menu.add_command(label='Six frame translation', command=self.gcframe)
        menu.add_command(label='Extract to FASTA', command=self.extract)
        self.frame_int = tk.IntVar()
        menu = self.frame_menu
        menu.add_radiobutton(label='+1', variable=self.frame_int, value=1)
        menu.add_radiobutton(label='+2', variable=self.frame_int, value=2)
        menu.add_radiobutton(label='+3', variable=self.frame_int, value=3)
        menu.add_radiobutton(label='-1', variable=self.frame_int, value=-1)
        menu.add_radiobutton(label='-2', variable=self.frame_int, value=-2)
        menu.add_radiobutton(label='-3', variable=self.frame_int, value=-3)
        self.frame_int.set(1)
        self.current_codon_table = tk.StringVar()
        self.current_codon_table.set('Standard')
        self.current_codon_table_id = 1
        keys = list(self.translation_tables.keys())
        keys.remove('Standard')
        keys.sort()
        keys = ['Standard'] + keys
        menu = self.gencode_menu
        for table in keys:
            menu.add_radiobutton(label=table, command=self.set_codon_table, variable=self.current_codon_table)
        self.menubar.add_cascade(label='Translations', menu=self.translation_menu)
        self.tools_menu = tk.Menu(self.menubar)
        menu = self.tools_menu
        menu.add_command(label='Blast', command=self.blast)
        menu.add_command(label='Stats', command=self.statistics)
        self.menubar.add_cascade(label='Tools', menu=self.tools_menu)
        self.help_menu = tk.Menu(self.menubar)
        menu = self.help_menu
        menu.add_command(label='Help', command=lambda : xbbtools_help())
        self.menubar.add_cascade(label='Help', menu=self.help_menu)
        self.parent.config(menu=self.menubar)

    def set_codon_table(self):
        if False:
            while True:
                i = 10
        'Set codon table to selection in Translations menu.'
        self.current_codon_table_id = self.translation_tables[self.current_codon_table.get()]

    def exit(self, *args):
        if False:
            return 10
        'Close the program.'
        if self.is_a_master:
            sys.exit()
        else:
            self.main_frame.destroy()

    def create_seqinfo(self, parent):
        if False:
            print('Hello World!')
        'Set up two info lines at top of main window.'
        self.seq_info1 = ttk.Frame(parent, relief='ridge', borderwidth=5, height=30)
        self.seq_info1.pack(fill='both', expand=1, side='top')
        self.position_ids = {}
        d = self.position_ids
        d['id'] = ttk.Label(self.seq_info1, width=10)
        d['from_id'] = ttk.Label(self.seq_info1, width=10)
        d['to_id'] = ttk.Label(self.seq_info1, width=10)
        d['length_id'] = ttk.Label(self.seq_info1, width=10)
        d['label'] = ttk.Label(self.seq_info1, width=10)
        for i in ['id', 'from_id', 'to_id', 'length_id', 'label']:
            d[i].pack(side='left', fill='both', expand=1)
        self.seq_info2 = ttk.Frame(parent, relief='ridge', borderwidth=5, height=30)
        self.seq_info2.pack(fill='both', expand=1, side='top')
        self.statistics_ids = {}
        d = self.statistics_ids
        d['length_id'] = ttk.Label(self.seq_info2, width=10)
        d['length_id'].pack(side='left', fill='both', expand=1)
        for nt in ['A', 'C', 'G', 'T']:
            d[nt] = ttk.Label(self.seq_info2, width=10)
            d[nt].pack(side='left', fill='both', expand=1)

    def create_buttons(self, parent):
        if False:
            i = 10
            return i + 15
        'Set up buttons.'
        self.button_frame = ttk.Frame(parent)
        self.button_frame.pack(fill='y', side='left')
        self.buttons = {}
        for (text, func) in [('Open', self.open), ('Export', self.export), ('GC Frame', self.gcframe), ('Blast', self.blast), ('Exit', self.exit)]:
            b_id = ttk.Button(self.button_frame, text=text, command=func, width=7)
            b_id.pack(side='top', pady=5, padx=10)
            self.buttons[text] = b_id
        frame = ttk.Frame(self.button_frame)
        label = ttk.Label(frame, text='Goto:')
        label.pack(side='left')
        label.bind('<Button-1>', self.goto)
        self.goto_entry = ttk.Entry(frame, width=5)
        self.goto_entry.pack(side='right', pady=5, padx=4)
        self.goto_entry.bind('<Return>', self.goto)
        frame.pack(side='bottom')

    def create_seqfield(self, parent):
        if False:
            return 10
        'Set up the main text field.'
        self.sequence_id = tk.Text(parent, wrap='char', width=self.seqwidth)
        self.sequence_id.pack(fill='both', expand=1, side='right')

    def create_bindings(self):
        if False:
            for i in range(10):
                print('nop')
        'Bind events to commands.'
        self.sequence_id.bind('<Motion>', self.position)
        self.sequence_id.bind('<Leave>', lambda x, s=self: s.position_ids['id'].configure(text=''))
        self.sequence_id.bind('<1>', self.zero)
        self.sequence_id.bind('<B1-Motion>', self.count_selection)
        self.sequence_id.bind('<Double-Button-1>', self.select_all)

    def zero(self, event):
        if False:
            print('Hello World!')
        'Remove selection.'
        for i in ['from_id', 'to_id', 'length_id']:
            self.position_ids[i].configure(text='')

    def get_length(self):
        if False:
            print('Hello World!')
        'Return length of sequence.'
        self.sequence_length = len(self.sequence_id.get(1.0, 'end'))
        return self.sequence_length

    def select_all(self, event):
        if False:
            i = 10
            return i + 15
        'Select the whole sequence.'
        self.select(1, self.get_length())
        self.count_selection(None)

    def select(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Select subsequence from a to b.'
        w = self.sequence_id
        w.selection_own()
        w.tag_add('sel', '1.%d' % (a - 1), f'1.{b:d}')
        self.count_selection(None)

    def get_selection_or_sequence(self):
        if False:
            return 10
        'Return selected sequence or whole sequence (if nothing selected).\n\n        Whitespaces, digits etc. are removed.\n        '
        seq = self.get_selection()
        if not len(seq):
            seq = self.sequence_id.get(1.0, 'end')
        seq = re.sub('[^A-Z]', '', seq)
        return str(seq)

    def get_selection(self):
        if False:
            for i in range(10):
                print('nop')
        'Return selected sequence or empty text (if nothing selected).'
        w = self.sequence_id
        try:
            return w.selection_get()
        except tk.TclError:
            return ''

    def get_self_selection(self):
        if False:
            for i in range(10):
                print('nop')
        'Return selected sequence or empty text (if no selection).'
        w = self.sequence_id
        try:
            return w.selection_get()
        except tk.TclError:
            return ''

    def count_selection(self, event):
        if False:
            while True:
                i = 10
        'Calculate some data from selected sequence and display it.'
        w = self.sequence_id
        w.selection_own()
        try:
            a = int(w.index('sel.first').split('.')[1]) + 1
            b = int(w.index('sel.last').split('.')[1])
            length = b - a + 1
            self.position_ids['from_id'].configure(text=f'Start:{a:d}')
            self.position_ids['to_id'].configure(text=f'Stop:{b:d}')
            self.position_ids['length_id'].configure(text=f'{length:d} nt')
            self.statistics_ids['length_id'].configure(text=f'Length={length:d}')
            seq = self.get_self_selection()
            for nt in ['A', 'C', 'G', 'T']:
                n = seq.count(nt)
                self.statistics_ids[nt].configure(text=f'{nt}={n:d}')
        except tk.TclError:
            pass

    def position(self, event):
        if False:
            i = 10
            return i + 15
        'Get position of cursor and display it.'
        x = event.x
        y = event.y
        pos = self.sequence_id.index(f'@{x:d},{y:d}').split('.')
        pos = int(pos[1]) + 1
        self.position_ids['id'].configure(text=str(pos))

    def open(self, filename=None):
        if False:
            while True:
                i = 10
        'Open a file.'
        if not filename:
            filename = filedialog.askopenfilename()
        if not filename:
            return
        with open(filename) as handle:
            self.insert_sequence(next(SimpleFastaParser(handle)))

    def insert_sequence(self, name_sequence):
        if False:
            for i in range(10):
                print('nop')
        'Load new sequence in sequence window.'
        (name, sequence) = name_sequence
        self.sequence_id.delete(0.0, 'end')
        self.sequence_id.insert('end', sequence.upper())
        self.fix_sequence()
        self.update_label(name)

    def fix_sequence(self):
        if False:
            print('Hello World!')
        'Do basic formatting of sequence in sequence window.'
        seq = str(self.sequence_id.get(1.0, 'end'))
        seq = seq.upper()
        seq = re.sub('[^A-Z]', '', seq)
        self.sequence_id.delete(0.0, 'end')
        self.sequence_id.insert('end', seq)

    def update_label(self, header):
        if False:
            while True:
                i = 10
        'Update name label.'
        name = header.split(' ')[0]
        name = name.split(',')[0]
        self.position_ids['label'].configure(text=name)

    def export(self):
        if False:
            return 10
        'Export selected text to new text window.'
        seq = self.get_selection_or_sequence()
        if not seq:
            return
        np = NotePad()
        tid = np.text_id()
        tid.insert('end', seq)

    def gcframe(self, direction='both'):
        if False:
            while True:
                i = 10
        'Run pretty print multiple frame translations.'
        seq = self.get_selection_or_sequence()
        if not seq:
            return
        np = NotePad()
        tid = np.text_id()
        tid.insert('end', self.translator.gcframe(seq, self.current_codon_table_id, direction))

    def translate(self):
        if False:
            print('Hello World!')
        'Run pretty print single frame translation.'
        seq = self.get_selection_or_sequence()
        frame = self.frame_int.get()
        if not seq:
            return
        np = NotePad()
        tid = np.text_id()
        tid.insert('end', self.translator.frame_nice(seq, frame, self.current_codon_table_id))

    def extract(self):
        if False:
            for i in range(10):
                print('nop')
        'Make single frame translation and display aa sequence as fasta.'
        seq = self.get_selection_or_sequence()
        frame = self.frame_int.get()
        if not seq:
            return
        aa_seq = self.translator.frame(seq, frame, self.current_codon_table_id)
        aa_seq = re.sub('(.{50})', '\\1\n', str(aa_seq))
        np = NotePad()
        tid = np.text_id()
        tid.insert('end', f'>frame{frame:d}\n{aa_seq}')

    def statistics(self):
        if False:
            i = 10
            return i + 15
        'Calculate statistics of sequence and display in new window.'
        seq = self.get_selection_or_sequence()
        if not seq:
            return
        seq = seq.upper()
        aa = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0}
        for nt in seq:
            if nt not in aa:
                nt = 'N'
            aa[nt] = aa[nt] + 1
        GC = 100.0 * (aa['G'] + aa['C']) / len(seq)
        np = NotePad()
        tid = np.text_id()
        tid.insert('end', '%s\n\n' % time.strftime('%y %b %d, %X\n', time.localtime(time.time())) + 'Length = %d\nA=%d C=%d G=%d T=%d other=%d\nGC=%f\n\n' % (len(seq), aa['A'], aa['C'], aa['G'], aa['T'], aa['N'], GC))

    def blast(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize and start BLASTing.'
        seq = self.get_selection_or_sequence()
        self.blaster = BlastIt(seq, self.parent)

    def reverse(self):
        if False:
            while True:
                i = 10
        'Display reversed sequence in sequence window.'
        w = self.sequence_id
        w.selection_own()
        try:
            (start, stop) = w.tag_ranges('sel')
        except ValueError:
            (start, stop) = (1.0, self.sequence_id.index('end'))
        seq = w.get(start, stop)
        seq = list(re.sub('[^A-Z]', '', seq))
        seq.reverse()
        seq = ''.join(seq)
        w.delete(start, stop)
        w.insert(start, seq)
        w.tag_remove('sel', 1.0, start)
        w.tag_add('sel', start, stop)
        w.tag_remove('sel', stop, 'end')

    def complement(self):
        if False:
            i = 10
            return i + 15
        'Display complement of sequence in sequence window.'
        w = self.sequence_id
        w.selection_own()
        try:
            (start, stop) = w.tag_ranges('sel')
        except ValueError:
            (start, stop) = (1.0, self.sequence_id.index('end'))
        seq = str(w.get(start, stop))
        seq = re.sub('[^A-Z]', '', seq)
        complementary = self.translator.complement(seq)
        w.delete(start, stop)
        w.insert(start, complementary)
        w.tag_remove('sel', 1.0, start)
        w.tag_add('sel', start, stop)
        w.tag_remove('sel', stop, 'end')

    def antiparallel(self):
        if False:
            while True:
                i = 10
        'Display reverse-complemented sequence in sequence window.'
        w = self.sequence_id
        w.selection_own()
        try:
            (start, stop) = w.tag_ranges('sel')
        except tk.TclError:
            (start, stop) = (1.0, self.sequence_id.index('end'))
        seq = str(w.get(start, stop))
        seq = re.sub('[^A-Z]', '', seq)
        antip = self.translator.antiparallel(seq)
        w.delete(start, stop)
        w.insert(start, antip)
        w.tag_remove('sel', 1.0, start)
        w.tag_add('sel', start, stop)
        w.tag_remove('sel', stop, 'end')

    def search(self):
        if False:
            return 10
        'Initialize and start search process.'
        seq = self.get_selection_or_sequence()
        XDNAsearch(seq, master=self.sequence_id, highlight=1)

    def goto(self, *args):
        if False:
            print('Hello World!')
        'Place cursor at chosen position.\n\n        You can also select and mark a range by typing e.g. 50:55\n        into the Goto entry field.\n        '
        pos = self.goto_entry.get()
        try:
            pos = int(pos) - 1
        except ValueError:
            try:
                (start, stop) = pos.split(':')
                start = int(start) - 1
                stop = int(stop)
                self.mark(start, stop)
                return
            except ValueError:
                import traceback
                traceback.print_exc()
                self.goto_entry.delete(0, 'end')
                return
        self.sequence_id.focus()
        self.sequence_id.mark_set('insert', f'1.{pos:d}')

    def mark(self, start, stop):
        if False:
            return 10
        'Mark and put a tag on chosen subsequence from start to stop.'
        self.sequence_id.focus()
        self.sequence_id.mark_set('insert', f'1.{start:d}')
        self.sequence_id.tag_add('sel', f'1.{start:d}', f'1.{stop:d}')
if __name__ == '__main__':
    win = tk.Tk()
    xbbtools = xbb_widget()
    xbbtools.open('test.fas')
    win.mainloop()