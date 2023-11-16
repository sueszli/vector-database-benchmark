"""A small GUI tool to demonstrate some basic sequence operations.

SeqGui (sequence graphical user interface) is a little tool that allows
transcription, translation and back translation of a sequence that the
user can type or copy into a text field. For translation the user can select
from several codon tables which are implemented in Biopython.
It runs as a standalone application.

"""
import tkinter as tk
import tkinter.ttk as ttk
from Bio.Seq import translate, transcribe, back_transcribe
from Bio.Data import CodonTable
main_window = tk.Tk()
main_window.title('Greetings from Biopython')
main_menu = tk.Menu(main_window)
menue_single = tk.Menu(main_menu, tearoff=0)
main_menu.add_cascade(menu=menue_single, label='File')
menue_single.add_command(label='About')
menue_single.add_separator()
menue_single.add_command(label='Exit', command=main_window.destroy)
main_window.config(menu=main_menu)
param_panel = ttk.Frame(main_window, relief=tk.GROOVE, padding=5)
codon_panel = ttk.LabelFrame(param_panel, text='Codon Tables')
codon_scroller = ttk.Scrollbar(codon_panel, orient=tk.VERTICAL)
codon_list = tk.Listbox(codon_panel, height=5, width=25, yscrollcommand=codon_scroller.set)
codon_table_list = sorted((table.names[0] for (n, table) in CodonTable.generic_by_id.items()))
del codon_table_list[codon_table_list.index('Standard')]
codon_table_list.insert(0, 'Standard')
for codon_table in codon_table_list:
    codon_list.insert(tk.END, codon_table)
codon_list.selection_set(0)
codon_list.configure(exportselection=False)
codon_scroller.config(command=codon_list.yview)
transform_panel = ttk.LabelFrame(param_panel, text='Transformation')
transform_var = tk.StringVar()
transform_transcribe = ttk.Radiobutton(transform_panel, text='Transcribe', variable=transform_var, value='transcribe')
transform_translate = ttk.Radiobutton(transform_panel, text='Translate', variable=transform_var, value='translate')
transform_backtranscribe = ttk.Radiobutton(transform_panel, text='Back transcribe', variable=transform_var, value='back transcribe')
transform_translate.invoke()
seq_panel = ttk.Frame(main_window, relief=tk.GROOVE, padding=5)
input_panel = ttk.LabelFrame(seq_panel, text='Original Sequence')
input_scroller = ttk.Scrollbar(input_panel, orient=tk.VERTICAL)
input_text = tk.Text(input_panel, width=39, height=5, yscrollcommand=input_scroller.set)
input_scroller.config(command=input_text.yview)
output_panel = ttk.LabelFrame(seq_panel, text='Transformed Sequence')
output_scroller = ttk.Scrollbar(output_panel, orient=tk.VERTICAL)
output_text = tk.Text(output_panel, width=39, height=5, yscrollcommand=output_scroller.set)
output_scroller.config(command=output_text.yview)
apply_button = ttk.Button(seq_panel, text='Apply')
clear_button = ttk.Button(seq_panel, text='Clear')
close_button = ttk.Button(seq_panel, text='Close', command=main_window.destroy)
statustext = tk.StringVar()
statusbar = ttk.Label(main_window, textvariable=statustext, relief=tk.GROOVE, padding=5)
statustext.set('This is the statusbar')
sizegrip = ttk.Sizegrip(statusbar)

def clear_output():
    if False:
        return 10
    'Clear the output window.'
    input_text.delete(1.0, tk.END)
    output_text.delete(1.0, tk.END)

def apply_operation():
    if False:
        while True:
            i = 10
    'Do the selected operation.'
    codon_table = codon_list.get(codon_list.curselection())
    print(f'Code: {codon_table}')
    seq = ''.join(input_text.get(1.0, tk.END).split())
    print(f'Input sequence: {seq}')
    operation = transform_var.get()
    print(f'Operation: {operation}')
    if operation == 'transcribe':
        result = transcribe(seq)
    elif operation == 'translate':
        result = translate(seq, table=codon_table, to_stop=True)
    elif operation == 'back transcribe':
        result = back_transcribe(seq)
    else:
        result = ''
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, result)
    print(f'Result: {result}')

def set_statusbar(event):
    if False:
        return 10
    'Show statusbar comments from menu selection.'
    index = main_window.call(event.widget, 'index', 'active')
    if index == 0:
        statustext.set('More information about this program')
    elif index == 2:
        statustext.set('Terminate the program')
    else:
        statustext.set('This is the statusbar')
menue_single.bind('<<MenuSelect>>', set_statusbar)
apply_button.config(command=apply_operation)
clear_button.config(command=clear_output)
statusbar.pack(side=tk.BOTTOM, padx=1, fill=tk.X)
sizegrip.pack(side=tk.RIGHT, padx=3, pady=4)
param_panel.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=10, fill=tk.Y)
codon_panel.pack(fill=tk.Y, expand=True)
codon_scroller.pack(side=tk.RIGHT, fill=tk.Y)
codon_list.pack(fill=tk.Y, expand=True)
transform_panel.pack(pady=10, fill=tk.X)
transform_transcribe.pack(anchor=tk.W)
transform_translate.pack(anchor=tk.W)
transform_backtranscribe.pack(anchor=tk.W)
seq_panel.pack(anchor=tk.N, padx=5, pady=10, fill=tk.BOTH, expand=True)
input_panel.pack(fill=tk.BOTH, expand=True)
input_scroller.pack(side=tk.RIGHT, fill=tk.Y)
input_text.pack(fill=tk.BOTH, expand=True)
output_panel.pack(pady=10, fill=tk.BOTH, expand=True)
output_scroller.pack(side=tk.RIGHT, fill=tk.Y)
output_text.pack(fill=tk.BOTH, expand=True)
apply_button.pack(side=tk.LEFT)
clear_button.pack(side=tk.LEFT, padx=10)
close_button.pack(side=tk.LEFT)
main_window.mainloop()