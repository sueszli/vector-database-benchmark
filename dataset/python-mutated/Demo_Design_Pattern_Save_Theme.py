import PySimpleGUI as sg
'\n    Demo - Basic window design pattern\n    * Creates window in a separate function for easy "restart"\n    * Saves theme as a user variable\n    * Puts main code into a main function so that multiprocessing works if you later convert to use\n    \n    Copyright 2020 PySimpleGUI.org\n'

def make_window():
    if False:
        return 10
    sg.theme(sg.user_settings_get_entry('theme', None))
    layout = [[sg.T('This is your layout')], [sg.OK(), sg.Button('Theme', key='-THEME-'), sg.Button('Exit')]]
    return sg.Window('Pattern for theme saving', layout)

def main():
    if False:
        print('Hello World!')
    window = make_window()
    while True:
        (event, values) = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        if event == '-THEME-':
            (ev, vals) = sg.Window('Choose Theme', [[sg.Combo(sg.theme_list(), k='-THEME LIST-'), sg.OK(), sg.Cancel()]]).read(close=True)
            if ev == 'OK':
                window.close()
                sg.user_settings_set_entry('theme', vals['-THEME LIST-'])
                window = make_window()
    window.close()
if __name__ == '__main__':
    main()