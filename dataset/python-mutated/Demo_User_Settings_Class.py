import PySimpleGUI as sg
'\n    Demo - User Settings Using Class\n\n    There are 2 interfaces for the User Settings APIs in PySimpleGUI. \n    1. Function calls\n    2. The UserSettings class\n    \n    This demo focuses on using the class interface.  The advantage of using the class is that \n    lookups resemble the syntax used for Python dictionaries\n\n    This demo is very basic. The user_settings functions are used directly without a lookup table\n    or some other mechanism to map between PySimpleGUI keys and user settings keys. \n\n    Note that there are 2 coding conventions being used.  The PySimpleGUI Demo Programs all use\n    keys on the elements that are strings with the format \'-KEY-\'.  They are upper case.  The\n    coding convention being used in Demo Programs that use User Settings use keys that have\n    the same format, but are lower case.  A User Settings key is \'-key-\'.  The reason for this\n    convention is so that you will immediately know what the string you are looking at is.\n    By following this convention, someone reading the code that encounters \'-filename-\' will\n    immediately recognize that this is a User Setting.  \n\n    Two windows are shown.  One is a super-simple "save previously entered filename"\n    The other is a larger "settings window" where multiple settings are saved/loaded\n\n    Copyright 2020 PySimpleGUI.org\n'
SETTINGS_PATH = '.'
settings = sg.UserSettings(path=SETTINGS_PATH)

def make_window():
    if False:
        return 10
    '\n    Creates a new window.  The default values for some elements are pulled directly from the\n    "User Settings" without the use of temp variables.\n\n    Some get_entry calls don\'t have a default value, such as theme, because there was an initial call\n    that would have set the default value if the setting wasn\'t present.  Could still put the default\n    value if you wanted but it would be 2 places to change if you wanted a different default value.\n\n    Use of a lookup table to map between element keys and user settings could be aded. This demo\n    is intentionally done without one to show how to use the settings APIs in the most basic,\n    straightforward way.\n\n    If your application allows changing the theme, then a make_window function is good to have\n    so that you can close and re-create a window easily.\n\n    :return: (sg.Window)  The window that was created\n    '
    sg.theme(settings.get('-theme-', 'DarkBlue2'))
    layout = [[sg.Text('Settings Window')], [sg.Input(settings.get('-input-', ''), k='-IN-')], [sg.Listbox(sg.theme_list(), default_values=[settings['-theme-']], size=(15, 10), k='-LISTBOX-')], [sg.CB('Option 1', settings.get('-option1-', True), k='-CB1-')], [sg.CB('Option 2', settings.get('-option2-', False), k='-CB2-')], [sg.T('Settings file = ' + settings.get_filename())], [sg.Button('Save'), sg.Button('Settings Dictionary'), sg.Button('Exit without saving', k='Exit')]]
    window = sg.Window('A Settings Window', layout)

def settings_window():
    if False:
        for i in range(10):
            print('nop')
    '\n    Create and interact with a "settings window". You can a similar pair of functions to your\n    code to add a "settings" feature.\n    '
    window = make_window()
    current_theme = sg.theme()
    while True:
        (event, values) = window.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        if event == 'Save':
            settings['-input-'] = values['-IN-']
            settings['-theme-'] = values['-LISTBOX-'][0]
            settings['-option1-'] = values['-CB1-']
            settings['-option2-'] = values['-CB2-']
        elif event == 'Settings Dictionary':
            sg.popup(settings)
        if values['-LISTBOX-'] and values['-LISTBOX-'][0] != current_theme:
            current_theme = values['-LISTBOX-'][0]
            window.close()
            window = make_window()

def save_previous_filename_demo():
    if False:
        while True:
            i = 10
    '\n    Saving the previously selected filename....\n    A demo of one of the likely most popular use of user settings\n    * Use previous input as default for Input\n    * When a new filename is chosen, write the filename to user settings\n    '
    layout = [[sg.Text('Enter a filename:')], [sg.Input(settings.get('-filename-', ''), key='-IN-'), sg.FileBrowse()], [sg.B('Save'), sg.B('Exit Without Saving', key='Exit')]]
    window = sg.Window('Filename Example', layout)
    while True:
        (event, values) = window.read()
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        elif event == 'Save':
            settings['-filename-'] = values['-IN-']
    window.close()
if __name__ == '__main__':
    save_previous_filename_demo()
    settings_window()