import threading
import time
import PySimpleGUI as sg
'\n    You want to look for 3 points in this code, marked with comment "LOCATION X". \n    1. Where you put your call that takes a long time\n    2. Where the trigger to make the call takes place in the event loop\n    3. Where the completion of the call is indicated in the event loop\n\n    Demo on how to add a long-running item to your PySimpleGUI Event Loop\n    If you want to do something that takes a long time, and you do it in the \n    main event loop, you\'ll quickly begin to see messages from windows that your\n    program has hung, asking if you want to kill it.\n    \n    The problem is not that your problem is hung, the problem is that you are \n    not calling Read or Refresh often enough.\n    \n    One way through this, shown here, is to put your long work into a thread that\n    is spun off, allowed to work, and then gets back to the GUI when it\'s done working\n    on that task.\n    \n    Every time you start up one of these long-running functions, you\'ll give it an "ID".\n    When the function completes, it will send to the GUI Event Loop a message with \n    the format:\n        work_id ::: done\n    This makes it easy to parse out your original work ID\n    \n    You can hard code these IDs to make your code more readable.  For example, maybe\n    you have a function named "update_user_list()".  You can call the work ID "user list".\n    Then check for the message coming back later from the work task to see if it starts\n    with "user list".  If so, then that long-running task is over. \n    \n'

def long_function_wrapper(work_id, window):
    if False:
        i = 10
        return i + 15
    time.sleep(5)
    window.write_event_value('-THREAD DONE-', work_id)
    return

def the_gui():
    if False:
        print('Hello World!')
    sg.theme('Light Brown 3')
    layout = [[sg.Text('Multithreaded Work Example')], [sg.Text('Click Go to start a long-running function call')], [sg.Text(size=(25, 1), key='-OUTPUT-')], [sg.Text(size=(25, 1), key='-OUTPUT2-')], [sg.Text('⚫', text_color='blue', key=i, pad=(0, 0), font='Default 14') for i in range(20)], [sg.Button('Go'), sg.Button('Popup'), sg.Button('Exit')]]
    window = sg.Window('Multithreaded Window', layout)
    work_id = 0
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'Go':
            window['-OUTPUT-'].update('Starting long work %s' % work_id)
            window[work_id].update(text_color='red')
            thread_id = threading.Thread(target=long_function_wrapper, args=(work_id, window), daemon=True)
            thread_id.start()
            work_id = work_id + 1 if work_id < 19 else 0
        if event == '-THREAD DONE-':
            completed_work_id = values[event]
            window['-OUTPUT2-'].update('Complete Work ID "{}"'.format(completed_work_id))
            window[completed_work_id].update(text_color='green')
        if event == 'Popup':
            sg.popup_non_blocking('This is a popup showing that the GUI is running', grab_anywhere=True)
    window.close()
if __name__ == '__main__':
    the_gui()
    print('Exiting Program')