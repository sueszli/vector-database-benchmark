import PySimpleGUI as sg
import random
import time
import queue
'\n    Demo - Multi-threaded "Data Pump" Design Pattern\n\n    Send data to your PySimpleGUI program through a Python Queue, enabling integration with many\n    different types of data sources.\n\n    A thread gets data from a queue object and passes it over to the main event loop.\n    The external_thread is only used here to generaate random data.  It\'s not part of the\n    overall "Design Pattern".\n    \n    The thread the_thread IS part of the design pattern.  It reads data from the thread_queue and sends that\n    data over to the PySimpleGUI event loop.\n\n    Copyright 2022 PySimpleGUI\n'
gsize = (400, 400)
THREAD_KEY = '-THREAD-'
THREAD_INCOMING_DATA = '-INCOMING DATA-'
THREAD_EXITNG = '-THREAD EXITING-'
THREAD_EXTERNAL_EXITNG = '-EXTERNAL THREAD EXITING-'
thread_queue = queue.Queue()

def external_thread(thread_queue: queue.Queue):
    if False:
        i = 10
        return i + 15
    '\n    Represents some external source of data.\n    You would not include this code as a starting point with this Demo Program. Your data is assumed to\n    come from somewhere else. The important part is that you add data to the thread_queue\n    :param thread_queue:\n    :return:\n    '
    i = 0
    while True:
        time.sleep(0.01)
        point = (random.randint(0, gsize[0]), random.randint(0, gsize[1]))
        radius = random.randint(10, 40)
        thread_queue.put((point, radius))
        i += 1

def the_thread(window: sg.Window, thread_queue: queue.Queue):
    if False:
        return 10
    "\n    The thread that communicates with the application through the window's events.\n    Waits for data from a queue and sends that data on to the event loop\n    :param window:\n    :param thread_queue:\n    :return:\n    "
    while True:
        data = thread_queue.get()
        window.write_event_value((THREAD_KEY, THREAD_INCOMING_DATA), data)

def main():
    if False:
        print('Hello World!')
    layout = [[sg.Text('My Simulated Data Pump')], [sg.Multiline(size=(60, 20), k='-MLINE-')], [sg.Graph(gsize, (0, 0), gsize, k='-G-', background_color='gray')], [sg.Button('Go'), sg.Button('Exit')]]
    window = sg.Window('Simulated Data Pump', layout, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT)
    graph = window['-G-']
    while True:
        (event, values) = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go':
            window.start_thread(lambda : the_thread(window, thread_queue), (THREAD_KEY, THREAD_EXITNG))
            window.start_thread(lambda : external_thread(thread_queue), (THREAD_KEY, THREAD_EXTERNAL_EXITNG))
        elif event[0] == THREAD_KEY:
            if event[1] == THREAD_INCOMING_DATA:
                (point, radius) = values[event]
                graph.draw_circle(point, radius=radius, fill_color='green')
                window['-MLINE-'].print(f'Drawing at {point} radius {radius}', c='white on red')
            elif event[1] == THREAD_EXITNG:
                window['-MLINE-'].print('Thread has exited')
            elif event[1] == THREAD_EXTERNAL_EXITNG:
                window['-MLINE-'].print('Data Pump thread has exited')
        if event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Version':
            sg.popup_scrolled(__file__, sg.get_versions(), location=window.current_location(), keep_on_top=True, non_blocking=True)
    window.close()
if __name__ == '__main__':
    main()