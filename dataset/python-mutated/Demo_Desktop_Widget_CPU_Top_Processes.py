import PySimpleGUI as sg
import psutil
import operator
'\n    PSUTIL "Top" CPU Processes - Desktop Widget\n    \n    Creates a floating CPU utilization window running something similar to a "top" command.\n\n    Use the spinner to adjust the number of seconds between readings of the CPU utilizaiton\n    Rather than calling the threading module this program uses the PySimpleGUI perform_long_operation method.\n        The result is similar.  The function is run as a thread... the call is simply wrapped.\n            \n    Copyright 2022 PySimpleGUI\n'
g_interval = 1

def CPU_thread(window: sg.Window):
    if False:
        return 10
    while True:
        cpu_percent = psutil.cpu_percent(interval=g_interval)
        procs = psutil.process_iter()
        window.write_event_value('-CPU UPDATE FROM THREAD-', (cpu_percent, procs))

def main():
    if False:
        i = 10
        return i + 15
    global g_interval
    location = sg.user_settings_get_entry('-location-', (None, None))
    sg.theme('Black')
    layout = [[sg.Text(font=('Helvetica', 20), text_color=sg.YELLOWS[0], key='-CPU PERCENT-')], [sg.Text(size=(35, 12), font=('Courier New', 12), key='-PROCESSES-')], [sg.Text('Update every '), sg.Spin([x + 1 for x in range(10)], 3, key='-SPIN-'), sg.T('seconds')]]
    window = sg.Window('Top CPU Processes', layout, no_titlebar=True, keep_on_top=True, location=location, use_default_focus=False, alpha_channel=0.8, grab_anywhere=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT, enable_close_attempted_event=True)
    window.start_thread(lambda : CPU_thread(window), '-THREAD FINISHED-')
    g_interval = 1
    try:
        while True:
            (event, values) = window.read()
            if event in (sg.WIN_CLOSE_ATTEMPTED_EVENT, 'Exit'):
                sg.user_settings_set_entry('-location-', window.current_location())
                break
            if event == 'Edit Me':
                sp = sg.execute_editor(__file__)
            elif event == 'Version':
                sg.popup_scrolled(__file__, sg.get_versions(), keep_on_top=True, location=window.current_location())
            elif event == '-CPU UPDATE FROM THREAD-':
                (cpu_percent, procs) = values[event]
                if procs:
                    top = {}
                    for proc in procs:
                        try:
                            top[proc.name()] = proc.cpu_percent()
                        except Exception as e:
                            pass
                    top_sorted = sorted(top.items(), key=operator.itemgetter(1), reverse=True)
                    if top_sorted:
                        top_sorted.pop(0)
                    display_string = '\n'.join([f'{cpu / 10:2.2f} {proc:23}' for (proc, cpu) in top_sorted])
                    window['-CPU PERCENT-'].update(f'CPU {cpu_percent}')
                    window['-PROCESSES-'].update(display_string)
            g_interval = int(values['-SPIN-'])
    except Exception as e:
        sg.Print('*** GOT Exception in event loop ***', c='white on red', font='_ 18')
        sg.Print('Exception = ', e, wait=True)
    window.close()
if __name__ == '__main__':
    main()