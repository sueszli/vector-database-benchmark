import PySimpleGUI as sg
import random
import string
'\n    Demo Program - Table with checkboxes\n\n    This clever solution was sugged by GitHub user robochopbg.\n    The beauty of the simplicity is that the checkbox is simply another column in the table. When the checkbox changes\n        state, then the data in the table is changed and the table is updated in the Table element.\n    A big thank you again to user robochopbg!\n\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⠆⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⡿⠁⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⠟⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⡿⠃⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀\n⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠺⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠈⠻⣿⣿⣦⣄⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠈⠻⣿⣿⣷⣤⡀⠀⠀⣰⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣦⣼⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n\n    Copyright 2023 PySimpleGUI\n'
BLANK_BOX = '☐'
CHECKED_BOX = '☑'

def word():
    if False:
        print('Hello World!')
    return ''.join((random.choice(string.ascii_lowercase) for i in range(10)))

def number(max_val=1000):
    if False:
        i = 10
        return i + 15
    return random.randint(0, max_val)

def make_table(num_rows, num_cols):
    if False:
        return 10
    data = [[j for j in range(num_cols)] for i in range(num_rows)]
    data[0] = [word() for __ in range(num_cols)]
    for i in range(1, num_rows):
        data[i] = [BLANK_BOX if random.randint(0, 2) % 2 else CHECKED_BOX] + [word(), *[number() for i in range(num_cols - 1)]]
    return data
data = make_table(num_rows=15, num_cols=6)
headings = [str(data[0][x]) + ' ..' for x in range(len(data[0]))]
headings[0] = 'Checkbox'
selected = {i for (i, row) in enumerate(data[1:][:]) if row[0] == CHECKED_BOX}
layout = [[sg.Table(values=data[1:][:], headings=headings, max_col_width=25, auto_size_columns=False, col_widths=[10, 10, 20, 20, 30, 5], display_row_numbers=True, justification='center', num_rows=20, key='-TABLE-', selected_row_colors='red on yellow', expand_x=False, expand_y=True, vertical_scroll_only=False, enable_click_events=True, font='_ 14'), sg.Sizegrip()]]
window = sg.Window('Table with Checkbox', layout, resizable=True, finalize=True)
window['-TABLE-'].update(values=data[1:][:], select_rows=list(selected))
while True:
    (event, values) = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event[0] == '-TABLE-' and event[2][0] not in (None, -1):
        row = event[2][0] + 1
        if data[row][0] == CHECKED_BOX:
            selected.remove(row - 1)
            data[row][0] = BLANK_BOX
        else:
            selected.add(row - 1)
            data[row][0] = CHECKED_BOX
        window['-TABLE-'].update(values=data[1:][:], select_rows=list(selected))
window.close()