import PySimpleGUI as sg
'\n    Demo Restart Window (sorta reopen)\n    \n    Once a window is closed, you can\'t do anything with it.  You can\'t read it.  You can\'t "re-open" it.\n    The only choice is to recreate the window.  It\'s important that you use a "Fresh Layout" every time.\n    You can\'t pass the same layout from one indow to another.  You will get a popup error infomrning you\n    that you\'ve attempted to resuse a layout.\n\n    The purpose of this demo is to show you the simple "make window" design pattern.  It simply makes a \n    window using a layout that\'s defined in that function and returns the Window object.  It\'s not a bad\n    way to encapsulate windows if your applcation is gettinga little larger than the typical small data\n    entry window.\n\n    Copyright 2020 PySimpleGUI.org\n'

def make_window():
    if False:
        print('Hello World!')
    '\n    Defines a window layout and createws a indow using this layout.  The newly made Window\n    is returned to the caller.\n\n    :return: Window that is created using the layout defined in the function\n    :rtype: Window\n    '
    layout = [[sg.Text('My Window')], [sg.Input(key='-IN-'), sg.Text(size=(12, 1), key='-OUT-')], [sg.Button('Go'), sg.Button('Exit')]]
    return sg.Window('Window Title', layout)

def main():
    if False:
        while True:
            i = 10
    window = make_window()
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            window.close()
            window = make_window()
        elif event == 'Go':
            window['-OUT-'].update(values['-IN-'])
if __name__ == '__main__':
    main()