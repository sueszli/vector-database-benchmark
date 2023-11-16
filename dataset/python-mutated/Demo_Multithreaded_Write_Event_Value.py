import threading
import time
import PySimpleGUI as sg
'\n    Threaded Demo - Uses Window.write_event_value communications\n    \n    Requires PySimpleGUI.py version 4.25.0 and later\n    \n    This is a really important demo  to understand if you\'re going to be using multithreading in PySimpleGUI.\n    \n    Older mechanisms for multi-threading in PySimpleGUI relied on polling of a queue. The management of a communications\n    queue is now performed internally to PySimpleGUI.\n\n    The importance of using the new window.write_event_value call cannot be emphasized enough.  It will hav a HUGE impact, in\n    a positive way, on your code to move to this mechanism as your code will simply "pend" waiting for an event rather than polling.\n    \n    Copyright 2020 PySimpleGUI.org\n'
THREAD_EVENT = '-THREAD-'

def the_thread(window):
    if False:
        return 10
    "\n    The thread that communicates with the application through the window's events.\n\n    Once a second wakes and sends a new event and associated value to the window\n    "
    i = 0
    while True:
        time.sleep(1)
        window.write_event_value('-THREAD-', (threading.current_thread().name, i))
        i += 1

def main():
    if False:
        return 10
    '\n    The demo will display in the multiline info about the event and values dictionary as it is being\n    returned from window.read()\n    Every time "Start" is clicked a new thread is started\n    Try clicking "Dummy" to see that the window is active while the thread stuff is happening in the background\n    '
    layout = [[sg.Text("Output Area - cprint's route to here", font='Any 15')], [sg.Multiline(size=(65, 20), key='-ML-', autoscroll=True, reroute_stdout=True, write_only=True, reroute_cprint=True)], [sg.T('Input so you can see data in your dictionary')], [sg.Input(key='-IN-', size=(30, 1))], [sg.B('Start A Thread'), sg.B('Dummy'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout)
    while True:
        (event, values) = window.read()
        sg.cprint(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event.startswith('Start'):
            threading.Thread(target=the_thread, args=(window,), daemon=True).start()
        if event == THREAD_EVENT:
            sg.cprint(f'Data from the thread ', colors='white on purple', end='')
            sg.cprint(f'{values[THREAD_EVENT]}', colors='white on red')
    window.close()
if __name__ == '__main__':
    main()