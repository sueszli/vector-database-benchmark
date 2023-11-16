import PySimpleGUI as sg
'\n    Demo - Changing your window\'s theme at runtime\n    * Create your window using a "window create function"\n    * When your window\'s theme changes, close the window, call the "window create function"\n    \n    Copyright 2021 PySimpleGUI\n'

def make_window(theme=None):
    if False:
        i = 10
        return i + 15
    if theme:
        sg.theme(theme)
    layout = [[sg.T('This is your layout')], [sg.Button('Ok'), sg.Button('Change Theme'), sg.Button('Exit')]]
    return sg.Window('Pattern for changing theme', layout)

def main():
    if False:
        return 10
    window = make_window()
    while True:
        (event, values) = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        if event == 'Change Theme':
            (event, values) = sg.Window('Choose Theme', [[sg.Combo(sg.theme_list(), readonly=True, k='-THEME LIST-'), sg.OK(), sg.Cancel()]]).read(close=True)
            print(event, values)
            if event == 'OK':
                window.close()
                window = make_window(values['-THEME LIST-'])
    window.close()
if __name__ == '__main__':
    main()