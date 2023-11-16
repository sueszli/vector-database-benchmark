import PySimpleGUI as sg
'\n    Extending PySimpleGUI using the tkinter event bindings\n\n    The idea here is to enable you to receive tkinter "Events" through the normal place you\n    get your events, the window.read() call.\n\n    Both elements and windows have a bind method.\n    window.bind(tkinter_event_string, key)   or   element.bind(tkinter_event_string, key_modifier)\n    First parameter is the tkinter event string.  These are things like <FocusIn> <Button-1> <Button-3> <Enter>\n    Second parameter for windows is an entire key, for elements is something added onto a key.  This key or modified key is what is returned when you read the window.\n    If the key modifier is text and the key is text, then the key returned from the read will be the 2 concatenated together.  Otherwise your event will be a tuple containing the key_modifier value you pass in and the key belonging to the element the event happened to.\n    \n    Copyright 2021 PySimpleGUI\n'
sg.theme('Dark Blue 3')

def main():
    if False:
        while True:
            i = 10
    layout = [[sg.Text('Move mouse over me', key='-TEXT-')], [sg.In(key='-IN-')], [sg.Button('Right Click Me', key='-BUTTON-'), sg.Button('Right Click Me2', key=(2, 3)), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, finalize=True)
    window.bind('<FocusOut>', '+FOCUS OUT+')
    window['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
    window[2, 3].bind('<Button-3>', '+RIGHT CLICK+')
    window['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
    window['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
    window['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+')
    window.bind('<Enter>', '* WINDOW ENTER *')
    while True:
        (event, values) = window.read()
        print(event, values)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
    window.close()
if __name__ == '__main__':
    main()