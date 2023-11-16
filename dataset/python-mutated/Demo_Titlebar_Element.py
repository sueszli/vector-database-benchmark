import PySimpleGUI as sg
'\n    The Custom Titlebar Demo\n    3 ways of getting a custom titlebar:\n    1. set_options - will create a titlebar that every window will have based on theme\n    2. Titlebar element - Adds custom titlebar to your window\n    3. use_custom_titlebar parameter - Add to your Window object\n'

def main():
    if False:
        i = 10
        return i + 15
    layout = [[sg.Text('My Window')], [sg.Input(k='-IN1-')], [sg.Input(k='-IN2-')], [sg.Input(k='-IN3-')], [sg.Button('Clear'), sg.Button('Popup'), sg.Button('Exit')]]
    window = sg.Window('My Custom Titlebar', layout, use_custom_titlebar=True)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Clear':
            [window[k]('') for k in ('-IN1-', '-IN2-', '-IN3-')]
        elif event == 'Popup':
            sg.popup('This is a popup')
    window.close()
main()