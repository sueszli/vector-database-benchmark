import sys
import sys
if sys.version_info[0] >= 3:
    import PySimpleGUIQt as sg
else:
    import PySimpleGUI27 as sg
import time
import random
import psutil
from threading import Thread
STEP_SIZE = 3
SAMPLES = 300
SAMPLE_MAX = 500
CANVAS_SIZE = (300, 200)
g_interval = 0.25
g_cpu_percent = 0
g_procs = None
g_exit = False

def CPU_thread(args):
    if False:
        return 10
    global g_interval, g_cpu_percent, g_procs, g_exit
    while not g_exit:
        try:
            g_cpu_percent = psutil.cpu_percent(interval=g_interval)
            g_procs = psutil.process_iter()
        except:
            pass

def main():
    if False:
        for i in range(10):
            print('nop')
    global g_exit, g_response_time
    sg.ChangeLookAndFeel('Black')
    sg.SetOptions(element_padding=(0, 0))
    layout = [[sg.Quit(button_color=('white', 'black')), sg.T('', font='Helvetica 25', key='output')], [sg.Graph(CANVAS_SIZE, (0, 0), (SAMPLES, SAMPLE_MAX), background_color='black', key='graph')]]
    window = sg.Window('CPU Graph', grab_anywhere=True, keep_on_top=True, background_color='black', no_titlebar=True, use_default_focus=False, location=(0, 0)).Layout(layout)
    graph = window.FindElement('graph')
    output = window.FindElement('output')
    thread = Thread(target=CPU_thread, args=(None,))
    thread.start()
    last_cpu = i = 0
    (prev_x, prev_y) = (0, 0)
    while True:
        (event, values) = window.Read(timeout=500)
        if event == 'Quit' or event is None:
            break
        current_cpu = int(g_cpu_percent * 10)
        if current_cpu == last_cpu:
            continue
        output.Update(current_cpu / 10)
        if current_cpu > SAMPLE_MAX:
            current_cpu = SAMPLE_MAX
        (new_x, new_y) = (i, current_cpu)
        if i >= SAMPLES:
            graph.Move(-STEP_SIZE, 0)
            prev_x = prev_x - STEP_SIZE
        graph.DrawLine((prev_x, prev_y), (new_x, new_y), color='white')
        (prev_x, prev_y) = (new_x, new_y)
        i += STEP_SIZE if i < SAMPLES else 0
        last_cpu = current_cpu
    g_exit = True
    window.Close()
if __name__ == '__main__':
    main()