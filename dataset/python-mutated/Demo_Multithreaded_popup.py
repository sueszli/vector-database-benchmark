import threading
import time
import PySimpleGUI as sg
import queue
'\n    Threading Demo - "Call popup from a thread"\n\n    Can be extended to call any PySimpleGUI function by passing the function through the queue\n\n\n    Safest approach to threading is to use a Queue object to communicate\n    between threads and maintrhead.\n\n    The thread calls popup, a LOCAL function that should be called with the same\n    parameters that would be used to call opiup when called directly\n\n    The parameters passed to the local popup are passed through a queue to the main thread.\n    When a messages is received from the queue, sg.popup is called using the parms passed\n    through the queue\n    \n    Copyright 2021 PySimpleGUI.org\n'
mainthread_queue: queue.Queue = None

def popup(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    if mainthread_queue:
        mainthread_queue.put((args, kwargs))

def the_thread(count):
    if False:
        return 10
    "\n    The thread that communicates with the application through the window's events.\n\n    Once a second wakes and sends a new event and associated value to the window\n    "
    i = 0
    while True:
        time.sleep(2)
        popup(f'Hello, this is the thread #{count}', 'My counter value', i, text_color='white', background_color='red', non_blocking=True, keep_on_top=True, location=(1000 - 200 * count, 400))
        i += 1

def process_popup():
    if False:
        while True:
            i = 10
    try:
        queued_value = mainthread_queue.get_nowait()
        sg.popup_auto_close(*queued_value[0], **queued_value[1])
    except queue.Empty:
        pass

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    The demo will display in the multiline info about the event and values dictionary as it is being\n    returned from window.read()\n    Every time "Start" is clicked a new thread is started\n    Try clicking "Dummy" to see that the window is active while the thread stuff is happening in the background\n    '
    global mainthread_queue
    mainthread_queue = queue.Queue()
    layout = [[sg.Text("Output Area - cprint's route to here", font='Any 15')], [sg.Multiline(size=(65, 20), key='-ML-', autoscroll=True, reroute_stdout=True, write_only=True, reroute_cprint=True)], [sg.T('Input so you can see data in your dictionary')], [sg.Input(key='-IN-', size=(30, 1))], [sg.B('Start A Thread'), sg.B('Dummy'), sg.Button('Exit')]]
    window = sg.Window('Window Title', layout, finalize=True, keep_on_top=True)
    count = 0
    while True:
        (event, values) = window.read(timeout=500)
        sg.cprint(event, values) if event != sg.TIMEOUT_EVENT else None
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        process_popup()
        if event.startswith('Start'):
            threading.Thread(target=the_thread, args=(count,), daemon=True).start()
            count += 1
    window.close()
if __name__ == '__main__':
    main()