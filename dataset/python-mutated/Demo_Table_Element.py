import PySimpleGUI as sg
import random
import string
'\n    Basic use of the Table Element\n    \n    Copyright 2022 PySimpleGUI\n'

def word():
    if False:
        return 10
    return ''.join((random.choice(string.ascii_lowercase) for i in range(10)))

def number(max_val=1000):
    if False:
        print('Hello World!')
    return random.randint(0, max_val)

def make_table(num_rows, num_cols):
    if False:
        print('Hello World!')
    data = [[j for j in range(num_cols)] for i in range(num_rows)]
    data[0] = [word() for __ in range(num_cols)]
    for i in range(1, num_rows):
        data[i] = [word(), *[number() for i in range(num_cols - 1)]]
    return data
data = make_table(num_rows=15, num_cols=6)
headings = [str(data[0][x]) + ' ..' for x in range(len(data[0]))]
layout = [[sg.Table(values=data[1:][:], headings=headings, max_col_width=25, auto_size_columns=True, display_row_numbers=True, justification='center', num_rows=20, alternating_row_color='lightblue', key='-TABLE-', selected_row_colors='red on yellow', enable_events=True, expand_x=False, expand_y=True, vertical_scroll_only=False, enable_click_events=True, tooltip='This is a table')], [sg.Button('Read'), sg.Button('Double'), sg.Button('Change Colors')], [sg.Text('Read = read which rows are selected')], [sg.Text('Double = double the amount of data in the table')], [sg.Text('Change Colors = Changes the colors of rows 8 and 9'), sg.Sizegrip()]]
window = sg.Window('The Table Element', layout, resizable=True)
while True:
    (event, values) = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED:
        break
    if event == 'Double':
        for i in range(1, len(data)):
            data.append(data[i])
        window['-TABLE-'].update(values=data[1:][:])
    elif event == 'Change Colors':
        window['-TABLE-'].update(row_colors=((8, 'white', 'red'), (9, 'green')))
window.close()