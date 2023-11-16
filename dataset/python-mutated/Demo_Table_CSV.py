import PySimpleGUI as sg
import csv
sg.theme('Dark Red')

def table_example():
    if False:
        return 10
    filename = sg.popup_get_file('filename to open', no_window=True, file_types=(('CSV Files', '*.csv'),))
    if filename == '':
        return
    data = []
    header_list = []
    button = sg.popup_yes_no('Does this file have column names already?')
    if filename is not None:
        with open(filename, 'r') as infile:
            reader = csv.reader(infile)
            if button == 'Yes':
                header_list = next(reader)
            try:
                data = list(reader)
                if button == 'No':
                    header_list = ['column' + str(x) for x in range(len(data[0]))]
            except:
                sg.popup_error('Error reading file')
                return
    sg.set_options(element_padding=(0, 0))
    layout = [[sg.Table(values=data, headings=header_list, max_col_width=25, auto_size_columns=True, justification='right', num_rows=min(len(data), 20))]]
    window = sg.Window('Table', layout, grab_anywhere=False)
    (event, values) = window.read()
    window.close()
table_example()