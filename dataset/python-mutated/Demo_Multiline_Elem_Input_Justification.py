"""
    Multline Element - Input Justification

    The justification of text for the Multiline element defaults to Left Justified
    Because of the way tkinter's widget works, setting the justification when creating the element
    is not enough to provide the correct justification.

    The demo shows you the technique to achieve justified input

    Key points:
        * Enable events on the multiline
        * Add 2 lines of code to your event loop
            * If get mline element event
                * Set the contents of the multline to be the correct justificaiton

    Copyright 2021 PySimpleGUI
"""
import PySimpleGUI as sg

def main():
    if False:
        print('Hello World!')
    justification = 'l'
    layout = [[sg.Text('Multiline Element Input Justification')], [sg.Multiline(size=(40, 10), key='-MLINE-', justification=justification, enable_events=True, autoscroll=True)], [sg.Radio('Left', 0, True, k='-L-'), sg.Radio('Center', 0, k='-C-'), sg.Radio('Right', 0, k='-R-')], [sg.Button('Go'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, keep_on_top=True, resizable=True, finalize=True)
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        justification = 'l' if values['-L-'] else 'r' if values['-R-'] else 'c'
        if event == '-MLINE-':
            window['-MLINE-'].update(values['-MLINE-'][:-1], justification=justification)
    window.close()
if __name__ == '__main__':
    main()