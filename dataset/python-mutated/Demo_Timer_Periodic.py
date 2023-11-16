import PySimpleGUI as sg
import time
'\n    Demo Program - Periodic Timer Event\n\n    How to use a thread to generate an event every x seconds\n\n    One method of getting periodic timer event that\'s more predictable than using window.read(timeout=x)\n    The problem with using a timeout with window.read is that if any event happens prior to the timer\n    expiring, the timer event will not happen.  The timeout parameter is not designed to provide a "heartbeat"\n    type of timer but rather to guarantee you will get an event within that amount of time, be it a\n    user-caused event or a timeout.\n\n    Copyright 2022 PySimpleGUI\n'
timer_running = {}

def timer_status_change(timer_id, start=None, stop=None, delete=None):
    if False:
        return 10
    '\n    Encapsulates/manages the timers dictionary\n\n    :param timer_id: ID of timer to change status\n    :type timer_id: int\n    :param start:   Set to True when timer is started\n    :type start:    bool\n    :param stop:    Set to True when timer is stopped\n    :type stop:     bool\n    :param delete:  Set to True to delete a timer\n    :type delete    bool\n    '
    global timer_running
    if start:
        timer_running[timer_id] = True
    if stop:
        timer_running[timer_id] = False
    if delete:
        del timer_running[timer_id]

def timer_is_running(timer_id):
    if False:
        while True:
            i = 10
    '\n\n    :param timer_id:    The timer ID to check\n    :type timer_id:     int\n    :return:            True if the timer is running\n    :rtype:             bool\n    '
    if timer_running[timer_id]:
        return True
    return False

def periodic_timer_thread(window, interval, timer_id):
    if False:
        print('Hello World!')
    '\n    Thread that sends messages to the GUI after some interval of time\n\n    :param window:   Window the events will be sent to\n    :type window:    sg.Window\n    :param interval: How frequently to send an event\n    :type interval:  float\n    :param timer_id: A timer identifier\n    :type timer_id:  int\n    '
    while True:
        time.sleep(interval)
        window.write_event_value(('-THREAD-', '-TIMER EVENT-'), timer_id)
        if not timer_is_running(timer_id):
            timer_status_change(timer_id, delete=True)
            return

def main():
    if False:
        while True:
            i = 10
    layout = [[sg.Text('Window with periodic time events')], [sg.Text(key='-MESSAGE-')], [sg.Text('Timer Status:'), sg.Text(key='-TIMER STATUS-')], [sg.Text('Duration:'), sg.In(s=3, key='-DURATION-'), sg.Button('Start')], [sg.Text('Timer ID:'), sg.In(s=3, key='-STOP ID-'), sg.Button('Stop')], [sg.Button('Dummy'), sg.Button('Exit')]]
    window = sg.Window('Blinking LED Window', layout)
    timer_counter = 0
    while True:
        (event, values) = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        window['-MESSAGE-'].update(f'{event} {values}')
        window['-TIMER STATUS-'].update(f'{timer_running}')
        if event == 'Start':
            if values['-DURATION-']:
                timer_status_change(timer_counter, start=True)
                window.start_thread(lambda : periodic_timer_thread(window, float(values['-DURATION-']), timer_counter), ('-THREAD-', '-THREAD ENDED-'))
                timer_counter += 1
            else:
                window['-MESSAGE-'].update('Please enter a numeric duration')
        elif event == 'Stop':
            if values['-STOP ID-']:
                timer_status_change(int(values['-STOP ID-']), stop=True)
    window.close()
if __name__ == '__main__':
    main()