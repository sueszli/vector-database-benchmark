import PySimpleGUI as sg
"\n    Demo - Pin a window on top\n\n    Note that the PIN used requires Python 3.7+ due to a tkinter problem\n    This demo uses a Window call only recently added to GitHub in Aug 2021\n\n    4.46.0.7 of PySimpleGUI provides the methods:\n        Window.keep_on_top_set\n        Window.keep_on_top_clear\n        \n    A temporary implementation is included in case you don't have that version\n\n    Copyright 2021 PySimpleGUI\n"

def main():
    if False:
        i = 10
        return i + 15
    sg.theme('dark green 7')
    PIN = '📌'
    my_titlebar = [[sg.Text('Window title', expand_x=True, grab=True, text_color=sg.theme_background_color(), background_color=sg.theme_text_color(), font='_ 12', pad=(0, 0)), sg.Text(PIN, enable_events=True, k='-PIN-', font='_ 12', pad=(0, 0), metadata=False, text_color=sg.theme_background_color(), background_color=sg.theme_text_color())]]
    layout = my_titlebar + [[sg.Text('This is my window layout')], [sg.Input(key='-IN-')], [sg.Button('Go'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, no_titlebar=True, resizable=True, margins=(0, 0))
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-PIN-':
            window['-PIN-'].metadata = not window['-PIN-'].metadata
            if window['-PIN-'].metadata:
                window['-PIN-'].update(text_color='red')
                window.keep_on_top_set()
            else:
                window['-PIN-'].update(text_color=sg.theme_background_color())
                window.keep_on_top_clear()
    window.close()

def keep_on_top_set(window):
    if False:
        print('Hello World!')
    '\n    Sets keep_on_top after a window has been created.  Effect is the same\n    as if the window was created with this set.  The Window is also brought\n    to the front\n    '
    window.KeepOnTop = True
    window.bring_to_front()
    window.TKroot.wm_attributes('-topmost', 1)

def keep_on_top_clear(window):
    if False:
        return 10
    '\n    Clears keep_on_top after a window has been created.  Effect is the same\n    as if the window was created with this set.\n    '
    window.KeepOnTop = False
    window.TKroot.wm_attributes('-topmost', 0)
if __name__ == '__main__':
    if 'keep_on_top_set' not in dir(sg.Window):
        print('You do not have a PySimpleGUI version with required methods. Using the temp ones from this file.')
        sg.Window.keep_on_top_set = keep_on_top_set
        sg.Window.keep_on_top_clear = keep_on_top_clear
    main()