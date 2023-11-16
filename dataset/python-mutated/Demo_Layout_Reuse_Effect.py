import PySimpleGUI as sg
'\n    Demo - Layout "Reuse" (but NOT reusing the layout)\n    As cautioned in the PySimpleGUI documentation, layouts cannot be "reused".\n    \n    That said, there is a very simple design pattern that you\'ll find in many many\n    Demo Programs.  Any program that is capable of changing the theme uses this\n    same kind of pattern.\n    \n    Goal - write the layout code once and then use it multiple times\n        The layout is reused \n    \n    Solution - create the layout and window in a function and return it\n\n    Copyright 2021 PySimpleGUI\n'

def make_window():
    if False:
        print('Hello World!')
    '\n    Defines the layout and creates the window.\n\n    This will allow you to "reuse" the layout.\n    Of course, the layout isn\'t reused, it is creating a new copy of the layout\n    every time the function is called.\n\n    :return: newly created window\n    :rtype: sg.Window\n    '
    layout = [[sg.Text('This is your layout')], [sg.Input(key='-IN-')], [sg.Text('You typed:'), sg.Text(size=(20, 1), key='-OUT-')], [sg.Button('Go'), sg.Button('Dark Gray 13 Theme'), sg.Button('Exit')]]
    return sg.Window('Window Title', layout)

def main():
    if False:
        return 10
    '\n    Your main program that contains your event loop\n    Rather than creating the layout and Window in this function, you will\n    instead call the make_window function to make the layout and Window\n    '
    window = make_window()
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go':
            window['-OUT-'].update(values['-IN-'])
        elif event.startswith('Dark'):
            sg.theme('Dark Gray 13')
            window.close()
            window = make_window()
    window.close()
if __name__ == '__main__':
    main()