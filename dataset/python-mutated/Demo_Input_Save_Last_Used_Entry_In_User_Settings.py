import PySimpleGUI as sg
'\n    Demo - Save previously entered value in Input element by using user_settings calls\n\n    Tired of typing in the same value or entering the same filename into an Input element?\n    If so, this may be exactly what you need.\n    \n    It simply saves the last value you entered so that the next time you start your program, that will be the default\n\n    Copyright 2022 PySimpleGUI.org\n'

def main():
    if False:
        i = 10
        return i + 15
    sg.user_settings_filename(path='.')
    layout = [[sg.T('This is your layout')], [sg.T('Remembers last value for this:'), sg.In(sg.user_settings_get_entry('-input-', ''), k='-INPUT-')], [sg.OK(), sg.Button('Exit')]]
    (event, values) = sg.Window('Save Input Element Last Value', layout).read(close=True)
    if event == 'OK':
        sg.user_settings_set_entry('-input-', values['-INPUT-'])
if __name__ == '__main__':
    main()