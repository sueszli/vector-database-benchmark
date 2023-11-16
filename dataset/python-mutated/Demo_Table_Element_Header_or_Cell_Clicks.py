import PySimpleGUI as sg
import random
import string
import operator
'\n    Table Element Demo With Sorting\n\n    The data for the table is assumed to have HEADERS across the first row.\n    This is often the case for CSV files or spreadsheets\n\n    In release 4.48.0 a new enable_click_events parameter was added to the Table Element\n    This enables you to click on Column Headers and individual cells as well as the standard Row selection\n\n    This demo shows how you can use these click events to sort your table by columns\n\n    Copyright 2022 PySimpleGUI\n'
sg.theme('Light green 6')

def word():
    if False:
        return 10
    return ''.join((random.choice(string.ascii_lowercase) for i in range(10)))

def number(max_val=1000):
    if False:
        return 10
    return random.randint(0, max_val)

def make_table(num_rows, num_cols):
    if False:
        for i in range(10):
            print('nop')
    data = [[j for j in range(num_cols)] for i in range(num_rows)]
    data[0] = [word() for _ in range(num_cols)]
    for i in range(1, num_rows):
        data[i] = [i, word(), *[number() for i in range(num_cols - 2)]]
    return data
data = make_table(num_rows=15, num_cols=6)
headings = [f'Col {col}' for col in range(1, len(data[0]) + 1)]

def sort_table(table, cols):
    if False:
        i = 10
        return i + 15
    ' sort a table by multiple columns\n        table: a list of lists (or tuple of tuples) where each inner list\n               represents a row\n        cols:  a list (or tuple) specifying the column numbers to sort by\n               e.g. (1,0) would sort by column 1, then by column 0\n    '
    for col in reversed(cols):
        try:
            table = sorted(table, key=operator.itemgetter(col))
        except Exception as e:
            sg.popup_error('Error in sort_table', 'Exception in sort_table', e)
    return table
layout = [[sg.Table(values=data[1:][:], headings=headings + ['Extra'], max_col_width=25, auto_size_columns=True, display_row_numbers=False, justification='right', right_click_selects=True, num_rows=20, alternating_row_color='lightyellow', key='-TABLE-', selected_row_colors='red on yellow', enable_events=True, expand_x=True, expand_y=True, enable_click_events=True, tooltip='This is a table')], [sg.Button('Read'), sg.Button('Double'), sg.Button('Change Colors')], [sg.Text('Cell clicked:'), sg.T(k='-CLICKED-')], [sg.Text('Read = read which rows are selected')], [sg.Text('Double = double the amount of data in the table')], [sg.Text('Change Colors = Changes the colors of rows 8 and 9'), sg.Sizegrip()]]
window = sg.Window('The Table Element', layout, resizable=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT, finalize=True)
window['-TABLE-'].bind('<Double-Button-1>', '+-double click-')
while True:
    (event, values) = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Edit Me':
        sg.execute_editor(__file__)
    elif event == 'Version':
        sg.popup_scrolled(__file__, sg.get_versions(), location=window.current_location(), keep_on_top=True, non_blocking=True)
    if event == 'Double':
        for i in range(1, len(data)):
            data.append(data[i])
        window['-TABLE-'].update(values=data[1:][:])
    elif event == 'Change Colors':
        window['-TABLE-'].update(row_colors=((8, 'white', 'red'), (9, 'green')))
    elif event == '-TABLE-+-double click-':
        print('Last cell clicked was', window['-TABLE-'].get_last_clicked_position())
    if isinstance(event, tuple):
        if event[0] == '-TABLE-':
            if event[2][0] == -1 and event[2][1] != -1:
                col_num_clicked = event[2][1]
                new_table = sort_table(data[1:][:], (col_num_clicked, 0))
                window['-TABLE-'].update(new_table)
                data = [data[0]] + new_table
            window['-CLICKED-'].update(f'{event[2][0]},{event[2][1]}')
window.close()