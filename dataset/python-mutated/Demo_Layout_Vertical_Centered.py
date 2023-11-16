import PySimpleGUI as sg
'\n    Center a column in a window\n\n    Solves a very specific kind of layout.\n    If you want to have something centered in a Window, this is a good way to do it\n    The "trick" here is:\n        * the first row of the layout has a Text element that expands vertically\n        * the row with the Column has a text element that expands horizontally\n    \n    This expanding Text element is what will cause the Column element to be centered\n\n    Copyright 2020 PySimpleGUI.org\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    column_to_be_centered = [[sg.Text('My Window')], [sg.Input(key='-IN-')], [sg.Text(size=(30, 1), key='-OUT-')], [sg.Button('Go'), sg.Button('Exit')]]
    layout = [[sg.Text(key='-EXPAND-', font='ANY 1', pad=(0, 0))], [sg.Text('', pad=(0, 0), key='-EXPAND2-'), sg.Column(column_to_be_centered, vertical_alignment='center', justification='center', k='-C-')]]
    window = sg.Window('Window Title', layout, resizable=True, finalize=True)
    window['-C-'].expand(True, True, True)
    window['-EXPAND-'].expand(True, True, True)
    window['-EXPAND2-'].expand(True, False, True)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go':
            window['-OUT-'].update(values['-IN-'])
    window.close()
if __name__ == '__main__':
    main()