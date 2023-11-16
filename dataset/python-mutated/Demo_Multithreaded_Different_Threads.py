import threading
import time
import itertools
import PySimpleGUI as sg
'\n    DESIGN PATTERN - Multithreaded GUI\n    One method for running multiple threads in a PySimpleGUI environment.\n    The PySimpleGUI code, and thus the underlying GUI framework, runs as the primary, main thread\n    Other parts of the software are implemented as threads\n    \n    While users never know the implementation details within PySimpleGUI, the mechanism is that a queue.Queue\n    is used to communicate data between a thread and a PySimpleGUI window.\n    The PySimpleGUI code is structured just like a typical PySimpleGUI program.  A layout defined,\n        a Window is created, and an event loop is executed.\n\n    Copyright 2020 PySimpleGUI.org\n    \n'

def worker_thread1(thread_name, run_freq, window):
    if False:
        while True:
            i = 10
    '\n    A worker thread that communicates with the GUI\n    These threads can call functions that block without affecting the GUI (a good thing)\n    Note that this function is the code started as each thread. All threads are identical in this way\n    :param thread_name: Text name used  for displaying info\n    :param run_freq: How often the thread should run in milliseconds\n    :param window: window this thread will be conversing with\n    :type window: sg.Window\n    :return:\n    '
    print('Starting thread 1 - {} that runs every {} ms'.format(thread_name, run_freq))
    for i in itertools.count():
        time.sleep(run_freq / 1000)
        window.write_event_value(thread_name, f'count = {i}')

def worker_thread2(thread_name, run_freq, window):
    if False:
        for i in range(10):
            print('nop')
    '\n    A worker thread that communicates with the GUI\n    These threads can call functions that block without affecting the GUI (a good thing)\n    Note that this function is the code started as each thread. All threads are identical in this way\n    :param thread_name: Text name used  for displaying info\n    :param run_freq: How often the thread should run in milliseconds\n    :param window: window this thread will be conversing with\n    :type window: sg.Window\n    :return:\n    '
    print('Starting thread 2 - {} that runs every {} ms'.format(thread_name, run_freq))
    for i in itertools.count():
        time.sleep(run_freq / 1000)
        window.write_event_value(thread_name, f'count = {i}')

def worker_thread3(thread_name, run_freq, window):
    if False:
        while True:
            i = 10
    '\n    A worker thread that communicates with the GUI\n    These threads can call functions that block without affecting the GUI (a good thing)\n    Note that this function is the code started as each thread. All threads are identical in this way\n    :param thread_name: Text name used  for displaying info\n    :param run_freq: How often the thread should run in milliseconds\n    :param window: window this thread will be conversing with\n    :type window: sg.Window\n    :return:\n    '
    print('Starting thread 3 - {} that runs every {} ms'.format(thread_name, run_freq))
    for i in itertools.count():
        time.sleep(run_freq / 1000)
        window.write_event_value(thread_name, f'count = {i}')

def the_gui():
    if False:
        for i in range(10):
            print('nop')
    '\n    Starts and executes the GUI\n    Reads data from a Queue and displays the data to the window\n    Returns when the user exits / closes the window\n        (that means it does NOT return until the user exits the window)\n    :param gui_queue: Queue the GUI should read from\n    :return:\n    '
    layout = [[sg.Text('Multithreaded Window Example')], [sg.Text('', size=(15, 1), key='-OUTPUT-')], [sg.Multiline(size=(40, 26), key='-ML-', autoscroll=True)], [sg.Button('Exit')]]
    window = sg.Window('Multithreaded Window', layout, finalize=True)
    threading.Thread(target=worker_thread1, args=('Thread 1', 500, window), daemon=True).start()
    threading.Thread(target=worker_thread2, args=('Thread 2', 200, window), daemon=True).start()
    threading.Thread(target=worker_thread3, args=('Thread 3', 1000, window), daemon=True).start()
    sg.cprint_set_output_destination(window, '-ML-')
    colors = {'Thread 1': ('white', 'red'), 'Thread 2': ('white', 'purple'), 'Thread 3': ('white', 'blue')}
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        sg.cprint(event, values[event], c=colors[event])
    window.close()
if __name__ == '__main__':
    the_gui()