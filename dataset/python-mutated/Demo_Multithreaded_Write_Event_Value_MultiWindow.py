import threading
import time
import PySimpleGUI as sg
'\n    Threaded Demo - Multiwindow Version\n\n    This demo uses the write_event_value method in a multi-window environment.  Instead of window.read() returning the event to\n    the user, the call to read_all_windows is used which will return both the window that had the event along with the event.\n\n    Copyright 2020 PySimpleGUI.org\n'
THREAD_EVENT = '-THREAD-'
PROGRESS_EVENT = '-PROGRESS-'
cp = sg.cprint

def the_thread(window, window_prog):
    if False:
        for i in range(10):
            print('nop')
    "\n    The thread that communicates with the application through the window's events.\n\n    Once a second wakes and sends a new event and associated value to a window for 10 seconds.\n\n    Note that WHICH window the message is sent to doesn't really matter because the code\n    in the event loop is calling read_all_windows.  This means that any window with an event\n    will cause the call to return.\n    "
    for i in range(10):
        time.sleep(1)
        window.write_event_value(THREAD_EVENT, (threading.current_thread().name, i))
        window_prog.write_event_value(PROGRESS_EVENT, i)

def make_progbar_window():
    if False:
        print('Hello World!')
    layout = [[sg.Text('Progress Bar')], [sg.ProgressBar(10, orientation='h', size=(15, 20), k='-PROG-')]]
    return sg.Window('Progress Bar', layout, finalize=True, location=(800, 800))

def make_main_window():
    if False:
        return 10
    layout = [[sg.Text("Output Area - cprint's route to here", font='Any 15')], [sg.Multiline(size=(65, 20), key='-ML-', autoscroll=True, reroute_stdout=True, write_only=True, reroute_cprint=True)], [sg.T('Input so you can see data in your dictionary')], [sg.Input(key='-IN-', size=(30, 1))], [sg.B('Start A Thread'), sg.B('Dummy'), sg.Button('Exit')]]
    return sg.Window('Window Main', layout, finalize=True)

def main():
    if False:
        i = 10
        return i + 15
    '\n    The demo will display in the multiline info about the event and values dictionary as it is being\n    returned from window.read()\n    Every time "Start" is clicked a new thread is started\n    Try clicking "Dummy" to see that the window is active while the thread stuff is happening in the background\n    '
    main_window = make_main_window()
    window_prog = make_progbar_window()
    while True:
        (window, event, values) = sg.read_all_windows()
        print(window.Title, event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event.startswith('Start'):
            threading.Thread(target=the_thread, args=(main_window, window_prog), daemon=True).start()
        if event == THREAD_EVENT:
            cp(f'Thread Event ', colors='white on blue', end='')
            cp(f'{values[THREAD_EVENT]}', colors='white on red')
        if event == PROGRESS_EVENT:
            cp(f'Progress Event from thread ', colors='white on purple', end='')
            cp(f'{values[PROGRESS_EVENT]}', colors='white on red')
            window_prog['-PROG-'].update(values[event] % 10 + 1)
        if event == 'Dummy':
            window.write_event_value('-DUMMY-', 'pressed')
        if event == '-DUMMY-':
            cp('Dummy pressed')
    window.close()
if __name__ == '__main__':
    main()