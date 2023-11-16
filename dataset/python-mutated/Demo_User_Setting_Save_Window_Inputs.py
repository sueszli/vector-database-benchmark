import PySimpleGUI as sg
'\n    Demo - User Setting API to save and load a window\'s contents\n\n    The PySimpleGUI "User Settings API" is a simple interface to JSON and Config Files.\n    If you\'re thinking of storying information in a JSON file, consider using the PySimpleGUI\n        User Settings API calls.  They make JSON files act like dictionaries.  There\'s no need\n        to load nor save as that\'s done for you.\n\n    There are 2 interfaces to the User Settings API.\n        1. Function calls - sg.user_settings\n        2. UserSettings Object - Uses a simple class interface\n\n    Note that using the Object/class interface does not require you to write a class.  If you\'re using\n    PySimpleGUI, you are already using many different objects.  The Elements & Window are objects.\n\n    In this demo, a UserSetting object is used to save the values from Input elements into a JSON file.\n    You can also re-loda the values from the JSON into your window.\n\n    Copyright 2022 PySimpleGUI\n'
window_contents = sg.UserSettings(path='.', filename='mysettings.json')

def main():
    if False:
        return 10
    layout = [[sg.Text('My Window')], [sg.Input(key='-IN1-')], [sg.Input(key='-IN2-')], [sg.Input(key='-IN3-')], [sg.Input(key='-IN4-')], [sg.Input(key='-IN5-')], [sg.Button('Save'), sg.Button('Load'), sg.Button('Exit')]]
    window = sg.Window('Save / Load Inputs Using User Settings API', layout)
    while True:
        (event, values) = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Save':
            for key in values:
                window_contents[key] = values[key]
        if event == 'Load':
            for key in values:
                saved_value = window_contents[key]
                window[key].update(saved_value)
    window.close()
if __name__ == '__main__':
    main()