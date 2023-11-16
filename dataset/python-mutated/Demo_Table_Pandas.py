import PySimpleGUI as sg
import pandas as pd

def table_example():
    if False:
        for i in range(10):
            print('nop')
    sg.set_options(auto_size_buttons=True)
    filename = sg.popup_get_file('filename to open', no_window=True, file_types=(('CSV Files', '*.csv'),))
    if filename == '':
        return
    data = []
    header_list = []
    button = sg.popup_yes_no('Does this file have column names already?')
    if filename is not None:
        try:
            df = pd.read_csv(filename, sep=',', engine='python', header=None)
            data = df.values.tolist()
            if button == 'Yes':
                header_list = df.iloc[0].tolist()
                data = df[1:].values.tolist()
            elif button == 'No':
                header_list = ['column' + str(x) for x in range(len(data[0]))]
        except:
            sg.popup_error('Error reading file')
            return
    layout = [[sg.Table(values=data, headings=header_list, display_row_numbers=True, auto_size_columns=False, num_rows=min(25, len(data)))]]
    window = sg.Window('Table', layout, grab_anywhere=False)
    (event, values) = window.read()
    window.close()
table_example()