import PySimpleGUI as sg
import queue
import logging
import threading
import time
'\n    This code originated in this project:\n    https://github.com/john144/MultiThreading\n    Thanks to John for writing this in the early days of PySimpleGUI\n    Demo program showing one way that a threaded application can function with PySimpleGUI\n    Events are sent from the ThreadedApp thread to the main thread, the GUI, by using a queue\n'
logger = logging.getLogger('mymain')

def externalFunction():
    if False:
        return 10
    logger.info('Hello from external app')
    logger.info('External app sleeping 5 seconds')
    time.sleep(5)
    logger.info('External app waking up and exiting')

class ThreadedApp(threading.Thread):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._stop_event = threading.Event()

    def run(self):
        if False:
            i = 10
            return i + 15
        externalFunction()

    def stop(self):
        if False:
            return 10
        self._stop_event.set()

class QueueHandler(logging.Handler):

    def __init__(self, log_queue):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        if False:
            i = 10
            return i + 15
        self.log_queue.put(record)

def main():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Multiline(size=(50, 15), key='-LOG-')], [sg.Button('Start', bind_return_key=True, key='-START-'), sg.Button('Exit')]]
    window = sg.Window('Log window', layout, default_element_size=(30, 2), font=('Helvetica', ' 10'), default_button_element_size=(8, 2))
    appStarted = False
    logging.basicConfig(level=logging.DEBUG)
    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    threadedApp = ThreadedApp()
    while True:
        (event, values) = window.read(timeout=100)
        if event == '-START-':
            if appStarted is False:
                threadedApp.start()
                logger.debug('App started')
                window['-START-'].update(disabled=True)
                appStarted = True
        elif event in (None, 'Exit'):
            break
        try:
            record = log_queue.get(block=False)
        except queue.Empty:
            pass
        else:
            msg = queue_handler.format(record)
            window['-LOG-'].update(msg + '\n', append=True)
    window.close()
if __name__ == '__main__':
    main()