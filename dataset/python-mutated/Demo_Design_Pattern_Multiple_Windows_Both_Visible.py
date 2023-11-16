import PySimpleGUI as sg
'\n    Demo - 2 simultaneous windows using read_all_window\n\n    Both windows are immediately visible.  Each window updates the other.\n    \n    There\'s an added capability to "re-open" window 2 should it be closed.  This is done by simply calling the make_win2 function\n    again when the button is pressed in window 1.\n    \n    The program exits when both windows have been closed\n        \n    Copyright 2020 PySimpleGUI.org\n'

def make_win1():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('Window 1')], [sg.Text('Enter something to output to Window 2')], [sg.Input(key='-IN-', enable_events=True)], [sg.Text(size=(25, 1), key='-OUTPUT-')], [sg.Button('Reopen')], [sg.Button('Exit')]]
    return sg.Window('Window Title', layout, finalize=True)

def make_win2():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('Window 2')], [sg.Text('Enter something to output to Window 1')], [sg.Input(key='-IN-', enable_events=True)], [sg.Text(size=(25, 1), key='-OUTPUT-')], [sg.Button('Exit')]]
    return sg.Window('Window Title', layout, finalize=True)

def main():
    if False:
        print('Hello World!')
    (window1, window2) = (make_win1(), make_win2())
    window2.move(window1.current_location()[0], window1.current_location()[1] + 220)
    while True:
        (window, event, values) = sg.read_all_windows()
        if window == sg.WIN_CLOSED:
            break
        if event == sg.WIN_CLOSED or event == 'Exit':
            window.close()
            if window == window2:
                window2 = None
            elif window == window1:
                window1 = None
        elif event == 'Reopen':
            if not window2:
                window2 = make_win2()
                window2.move(window1.current_location()[0], window1.current_location()[1] + 220)
        elif event == '-IN-':
            output_window = window2 if window == window1 else window1
            if output_window:
                output_window['-OUTPUT-'].update(values['-IN-'])
            else:
                window['-OUTPUT-'].update('Other window is closed')
if __name__ == '__main__':
    main()