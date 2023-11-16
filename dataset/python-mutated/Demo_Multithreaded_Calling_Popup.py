import threading
import time
import PySimpleGUI as sg
'\n    Threaded Demo - Uses Window.write_event_value to communicate from thread to GUI\n    \n    A demo specifically to show how to use write_event_value to\n        "show a popup from a thread"\n        \n    You cannot make any direct calls into PySimpleGUI from a thread\n    except for Window.write_event_value()\n    Cuation - This method still has a risk of tkinter crashing\n    \n    Copyright 2021 PySimpleGUI\n'

def the_thread(window: sg.Window, seconds):
    if False:
        i = 10
        return i + 15
    "\n    The thread that communicates with the application through the window's events.\n\n    Wakes every X seconds that are provided by user in the main GUI:\n        Sends an event to the main thread\n        Goes back to sleep\n    "
    i = 0
    while True:
        time.sleep(seconds)
        window.write_event_value('-POPUP-', ('Hello this is the thread...', f'My counter is {i}', f'Will send another message in {seconds} seconds'))
        i += 1

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    Every time "Start A Thread" is clicked a new thread is started\n    When the event is received from the thread, a popup is shown in its behalf\n    '
    layout = [[sg.Output(size=(60, 10))], [sg.T('How often a thread will show a popup in seconds'), sg.Spin((2, 5, 10, 20), initial_value=5, k='-SPIN-')], [sg.B('Start A Thread'), sg.B('Dummy'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, finalize=True, font='_ 15')
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-POPUP-':
            sg.popup_non_blocking('This is a popup that the thread wants to show', *values['-POPUP-'])
        elif event == 'Start A Thread':
            print(f"Starting thread.  You will see a new popup every {values['-SPIN-']} seconds")
            threading.Thread(target=the_thread, args=(window, values['-SPIN-']), daemon=True).start()
    window.close()
if __name__ == '__main__':
    main()