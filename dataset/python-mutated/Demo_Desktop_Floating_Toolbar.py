import sys
import PySimpleGUI as sg
import subprocess
import os
'\n    Demo_Toolbar - A floating toolbar with quick launcher\n    \n    One cool PySimpleGUI demo. Shows borderless windows, grab_anywhere, tight button layout\n    You can setup a specific program to launch when a button is clicked, or use the\n    Combobox to select a .py file found in the root folder, and run that file.\n    \n'
ROOT_PATH = './'

def Launcher():
    if False:
        while True:
            i = 10
    sg.theme('Dark')
    namesonly = [f for f in os.listdir(ROOT_PATH) if f.endswith('.py')]
    if len(namesonly) == 0:
        namesonly = ['test 1', 'test 2', 'test 3']
    sg.set_options(element_padding=(0, 0), button_element_size=(12, 1), auto_size_buttons=False)
    layout = [[sg.Combo(values=namesonly, size=(35, 30), key='demofile'), sg.Button('Run', button_color=('white', '#00168B')), sg.Button('Program 1'), sg.Button('Program 2'), sg.Button('Program 3', button_color=('white', '#35008B')), sg.Button('EXIT', button_color=('white', 'firebrick3'))], [sg.Text('', text_color='white', size=(50, 1), key='output')]]
    window = sg.Window('Floating Toolbar', layout, no_titlebar=True, grab_anywhere=True, keep_on_top=True)
    while True:
        (event, values) = window.read()
        if event == 'EXIT' or event == sg.WIN_CLOSED:
            break
        if event == 'Program 1':
            print('Run your program 1 here!')
        elif event == 'Program 2':
            print('Run your program 2 here!')
        elif event == 'Run':
            file = values['demofile']
            print('Launching %s' % file)
            ExecuteCommandSubprocess('python', os.path.join(ROOT_PATH, file))
        else:
            print(event)
    window.close()

def ExecuteCommandSubprocess(command, *args, wait=False):
    if False:
        i = 10
        return i + 15
    try:
        if sys.platform == 'linux':
            arg_string = ''
            arg_string = ' '.join([str(arg) for arg in args])
            print('python3 ' + arg_string)
            sp = subprocess.Popen(['python3 ', arg_string], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            arg_string = ' '.join([str(arg) for arg in args])
            sp = subprocess.Popen([command, arg_string], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if wait:
            (out, err) = sp.communicate()
            if out:
                print(out.decode('utf-8'))
            if err:
                print(err.decode('utf-8'))
    except:
        pass
if __name__ == '__main__':
    Launcher()