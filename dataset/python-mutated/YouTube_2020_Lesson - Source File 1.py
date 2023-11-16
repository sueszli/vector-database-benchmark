import PySimpleGUI as sg
import os
import time
import random

def test(x, y):
    if False:
        while True:
            i = 10
    return x
var = (1, 2, 3, 4)
sg.popup('This is a basic popup', 'I can have multiple item arguments', var)
while True:
    text = sg.popup_get_text('Enter "exit" to exit')
    if text == 'exit':
        break
    sg.popup('You entered:', text)
sg.popup_auto_close('Closing the program', background_color='red', text_color='white')
exit()
for i in range(1000):
    sg.one_line_progress_meter('My meter', i + 1, 1000, 'key', 'Message 1', 'Message 2')
for i in range(1000):
    if not sg.one_line_progress_meter('My meter', i + 1, 1000, 'key', 'Message 1', 'Message 2'):
        sg.popup('ABORTED')
        break
exit()
my_path = 'c:\\Python'
files = os.listdir(my_path)
print('\n'.join(files))
exit()
import PySimpleGUI as sg
layout = [[sg.Text('My layout')], [sg.Input(key='-INPUT-')], [sg.Button('OK'), sg.Button('Cancel')]]
window = sg.Window('Design Pattern 3 - Persistent Window', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Cancel'):
        break
window.close()
layout = [[sg.Text('Name:'), sg.Input(key='-NAME-')], [sg.Text('Favorite Color:'), sg.Combo(['Red', 'Blue', 'Green', 'Purple'], key='-COLOR-')], [sg.Button('Ok')]]
(event, values) = sg.Window('One shot', layout).read(close=True)
sg.popup(event, values)
exit()
import PySimpleGUI as sg
layout = [[sg.Text('My layout')], [sg.Input(key='-IN-')], [sg.Text('You entered:'), sg.Text(size=(20, 1), key='-OUT-')], [sg.Button('OK'), sg.Button('Cancel')]]
window = sg.Window('Update window with input value', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Cancel'):
        break
    window['-OUT-'].update(values['-IN-'])
window.close()
exit()
layout = [[sg.Text('Choose a file/Folder')], [sg.Text('Choose output folder'), sg.FolderBrowse(target='-IN3-')], [sg.Input(key='-IN-'), sg.FileBrowse()], [sg.Input(key='-IN2-', visible=False), sg.FileBrowse()], [sg.Input(key='-IN3-')], [sg.Button('Go'), sg.Button('Exit')]]
window = sg.Window('Window Title', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
window.close()
exit_button = b'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'
layout = [[sg.Text('My Window')], [sg.Button('Go'), sg.Button(image_data=exit_button, border_width=0, button_color=(sg.theme_background_color(), sg.theme_background_color()), key='Exit')]]
window = sg.Window('Window Title', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
window.close()
exit()
layout = [[sg.Listbox(list(range(10)), size=(10, 5), key='-LBOX-')], [sg.T('Name'), sg.In()], [sg.T('Address'), sg.In()], [sg.Button('Go'), sg.Button('Exit')]]
window = sg.Window('Window Title', layout, auto_size_text=False, default_element_size=(12, 1))
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
window.close()
exit()
import PySimpleGUI as sg
sg.theme('Dark Red')
menu_def = [['&File', ['&Open', '&Save', 'E&xit', 'Properties']], ['&Edit', ['Paste', ['Special', 'Normal'], 'Undo']], ['&Help', '&About...']]
column1 = [[sg.Text('Column 1', justification='center', size=(10, 1))], [sg.Spin(values=('Spin Box 1', 'Spin Box 2', 'Spin Box 3'), initial_value='Spin Box 1')], [sg.Spin(values=['Spin Box 1', 'Spin Box 2', 'Spin Box 3'], initial_value='Spin Box 2')], [sg.Spin(values=('Spin Box 1', 'Spin Box 2', 'Spin Box 3'), initial_value='Spin Box 3')]]
layout = [[sg.Menu(menu_def, tearoff=True)], [sg.Text('(Almost) All widgets in one Window!', size=(30, 1), justification='center', font=('Helvetica', 25), relief=sg.RELIEF_RIDGE)], [sg.Text('Here is some text.... and a place to enter text')], [sg.InputText('This is my text')], [sg.Frame(layout=[[sg.CBox('Checkbox', size=(10, 1)), sg.CBox('My second checkbox!', default=True)], [sg.Radio('My first Radio!     ', 'RADIO1', default=True, size=(10, 1)), sg.Radio('My second Radio!', 'RADIO1')]], title='Options', relief=sg.RELIEF_SUNKEN, tooltip='Use these to set flags')], [sg.MLine(default_text='This is the default Text should you decide not to type anything', size=(35, 3)), sg.MLine(default_text='A second multi-line', size=(35, 3))], [sg.Combo(('Combobox 1', 'Combobox 2'), default_value='Combobox 1', size=(20, 1)), sg.Slider(range=(1, 100), orientation='h', size=(34, 20), default_value=85)], [sg.OptionMenu(('Menu Option 1', 'Menu Option 2', 'Menu Option 3'))], [sg.Listbox(values=('Listbox 1', 'Listbox 2', 'Listbox 3'), size=(30, 3)), sg.Frame('Labelled Group', [[sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=25, tick_interval=25), sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=75), sg.Slider(range=(1, 100), orientation='v', size=(5, 20), default_value=10), sg.Col(column1)]])], [sg.Text('_' * 80)], [sg.Text('Choose A Folder', size=(35, 1))], [sg.Text('Your Folder', size=(15, 1), justification='right'), sg.InputText('Default Folder'), sg.FolderBrowse()], [sg.Submit(tooltip='Click to submit this form'), sg.Cancel()]]
window = sg.Window('Everything bagel', layout)
(event, values) = window.read()
sg.popup('Title', 'The results of the window.', 'The button clicked was "{}"'.format(event), 'The values are', values)
window.close()
exit()
layout = [[sg.Text('Enter your information', font='Any 20')], [sg.Text('Name', size=(8, 1), justification='r', font='Any 14'), sg.Input(key='-NAME-')], [sg.Text('Address', size=(8, 1), justification='r', font='Any 14'), sg.Input(key='-ADDRESS-')], [sg.Text('Phone', size=(8, 1), justification='r', font='Any 14'), sg.Input(key='-PHONE-')], [sg.Button('Go'), sg.Button('Exit')]]
window = sg.Window('Window Title', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
window.close()

def InfoIn(text, key):
    if False:
        for i in range(10):
            print('nop')
    return [sg.Text(text, size=(8, 1), justification='r', font='Any 14'), sg.Input(key=key)]
layout = [[sg.Text('Enter your information', font='Any 20')], InfoIn('Name', '-NAME-'), InfoIn('Address', '-ADDRESS-'), InfoIn('Phone', '-PHONE-'), [sg.Button('Go'), sg.Button('Exit')]]
window = sg.Window('Window Title', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
window.close()
exit()
import PySimpleGUI as sg
layout = [[sg.B(' X ', key=(r, c)) for r in range(3)] for c in range(4)]
layout += [[sg.OK()]]
window = sg.Window('title', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event is None:
        break
layout = [[]]
for row in range(5):
    row_layout = []
    for col in range(5):
        row_layout.append(sg.Button('X', key=(row, col)))
    layout.append(row_layout)
layout += [[sg.Button('OK'), sg.Button('Cancel')]]
window = sg.Window('Window Title', layout)
while True:
    (event, values) = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    if event == 'Go':
        window['-OUT-'].update(values['-IN-'])
window.close()
exit()
sg.Window('Sudoku', [[sg.Frame('', [[sg.I(random.randint(1, 9), justification='r', size=(3, 1), key=(frow * 3 + row, fcol * 3 + col)) for col in range(3)] for row in range(3)]) for fcol in range(3)] for frow in range(3)] + [[sg.B('Exit')]]).read()
exit()
import PySimpleGUI as sg
layout = [[sg.Text('My Timer')], [sg.Text(size=(12, 1), key='-OUT-')], [sg.Button('Exit')]]
window = sg.Window('Timer', layout, font='Any 20')
counter = 0
while True:
    (event, values) = window.read(timeout=100)
    if event in (None, 'Exit'):
        break
    window['-OUT-'].update(counter)
    counter += 1
window.close()
exit()
import PySimpleGUI as sg
tray = sg.SystemTray(menu=['UNUSED', ['My', 'Simple', '---', 'Menu', 'Exit']], data_base64=sg.DEFAULT_BASE64_ICON)
while True:
    event = tray.read()
    if event == 'Exit':
        tray.show_message('Exiting', 'Exiting the program', messageicon=sg.SYSTEM_TRAY_MESSAGE_ICON_INFORMATION)
        break
exit()
import PySimpleGUI as sg
layout = [[sg.Text('My Window')], [sg.Input(key='-IN-'), sg.Text(size=(12, 1), key='-OUT-')], [sg.Button('Go'), sg.Button('Exit')]]
window = sg.Window('Window Title', layout)
count = 0
while True:
    (event, values) = window.read(timeout=100)
    if event in (None, 'Exit'):
        break
    count += 1
window.close()
import PySimpleGUI as sg
window = sg.Window('test', layout=[[sg.ProgressBar(max_value=100, size=(30, 10), key='bar')]], finalize=True)
window['bar'].Widget.config(mode='indeterminate')
while True:
    (event, values) = window.read(timeout=100)
    if event is None:
        break
    window['bar'].Widget['value'] += 5