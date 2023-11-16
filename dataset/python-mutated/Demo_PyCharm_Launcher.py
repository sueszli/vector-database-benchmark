import PySimpleGUI as sg
import subprocess
'\n    Demo mini-PyCharm "favorites" launcher\n    Open a python file for editing using a small window that sits in the corner of your desktop\n\n    Copyright 2020 PySimpleGUI.org\n'
LOCATION = (2340, 1240)
TRANSPARENCY = 0.7
PSG = 'C:\\Python\\PycharmProjects\\PySimpleGUI\\PySimpleGUI.py'
PSGQT = 'C:\\Python\\PycharmProjects\\PySimpleGUI\\PySimpleGUIQt.py'
PSGWX = 'C:\\Python\\PycharmProjects\\PySimpleGUI\\PySimpleGUIWx.py'
PSGWEB = 'C:\\Python\\PycharmProjects\\PySimpleGUI\\PySimpleGUIWeb.py'
PYCHARM = 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2019.1.1\\bin\\pycharm.bat'
button_dict = {'PySimpleGUI': PSG, 'PySimpleGUIQt': PSGQT, 'PySimpleGUIWx': PSGWX, 'PySimpleGUIWeb': PSGWEB, 'This Progam': __file__}

def mini_launcher():
    if False:
        return 10
    '\n    The main program.  Creates the Window and runs the event loop\n    '
    sg.theme('dark')
    sg.set_options(border_width=0)
    layout = [[sg.Text(' ' * 10, background_color='black')]]
    for button_text in button_dict:
        layout += [[sg.Button(button_text)]]
    layout += [[sg.T('❎', background_color='black', enable_events=True, key='Exit')]]
    window = sg.Window('Script launcher', layout, no_titlebar=True, grab_anywhere=True, keep_on_top=True, element_padding=(0, 0), default_button_element_size=(20, 1), location=LOCATION, auto_size_buttons=False, use_default_focus=False, alpha_channel=TRANSPARENCY, background_color='black')
    while True:
        (event, values) = window.read()
        if event == 'Exit' or event == sg.WINDOW_CLOSED:
            break
        file_to_edit = button_dict.get(event)
        try:
            execute_command_blocking(PYCHARM, file_to_edit)
        except Exception as e:
            sg.Print(f'Got an exception {e} trying to open in PyCharm this file:', file_to_edit)

def execute_command_blocking(command, *args):
    if False:
        i = 10
        return i + 15
    '\n    Creates a subprocess using supplied command and arguments.\n    Will not return until the process completes running\n    :param command: The command (full path) to execute\n    :param args: a tuple of arguments\n    :return: string with the output from the command\n\n    '
    print(f'Executing {command} with {args}')
    expanded_args = [a for a in args]
    try:
        sp = subprocess.Popen([command, expanded_args], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = sp.communicate()
        if out:
            print(out.decode('utf-8'))
        if err:
            print(err.decode('utf-8'))
    except Exception as e:
        sg.Print(f'execute got exception {e}')
        out = ''
    return out
if __name__ == '__main__':
    mini_launcher()