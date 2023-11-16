import PySimpleGUI as sg
import time
"\n    Window that doesn't block\n    good for applications with an loop that polls hardware\n"

def StatusOutputExample():
    if False:
        while True:
            i = 10
    layout = [[sg.Text('Non-blocking GUI with updates')], [sg.Text('', size=(8, 2), font=('Helvetica', 20), justification='center', key='output')], [sg.Button('LED On'), sg.Button('LED Off'), sg.Button('Quit')]]
    window = sg.Window('Running Timer', layout, auto_size_text=True)
    i = 0
    while True:
        (event, values) = window.read(timeout=10)
        window['output'].update('{:02d}:{:02d}.{:02d}'.format(i // 100 // 60, i // 100 % 60, i % 100))
        if event in ('Quit', None):
            break
        if event == 'LED On':
            print('Turning on the LED')
        elif event == 'LED Off':
            print('Turning off the LED')
        i += 1
    window.close()

def RemoteControlExample():
    if False:
        return 10
    layout = [[sg.Text('Robotics Remote Control')], [sg.Text(' ' * 10), sg.RealtimeButton('Forward')], [sg.RealtimeButton('Left'), sg.Text(' ' * 15), sg.RealtimeButton('Right')], [sg.Text(' ' * 10), sg.RealtimeButton('Reverse')], [sg.Text('')], [sg.Quit(button_color=('black', 'orange'))]]
    window = sg.Window('Robotics Remote Control', layout, auto_size_text=True, finalize=True)
    while True:
        (event, values) = window.read(timeout=0, timeout_key='timeout')
        if event != 'timeout':
            print(event)
        if event in ('Quit', None):
            break
    window.close()

def main():
    if False:
        for i in range(10):
            print('nop')
    RemoteControlExample()
    StatusOutputExample()
    sg.popup('End of non-blocking demonstration')
if __name__ == '__main__':
    main()